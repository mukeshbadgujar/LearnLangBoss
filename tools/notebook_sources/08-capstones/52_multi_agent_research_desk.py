# %% [markdown]
# # 52 - Capstone: Multi-Agent Research Desk
#
# | | |
# |---|---|
# | **Level** | Capstone |
# | **Time** | 100 minutes |
# | **Prerequisites** | Track 07 (`39`-`50`), especially `41` and `50` |
# | **Checklist ID** | `52_multi_agent_research_desk` |
#
# ## Why this matters
#
# A researcher gathers, a writer drafts, a reviewer pushes back, an editor
# decides when it is good enough. That is how a real desk works, and it is a
# genuinely good fit for multi-agent design: the roles have different
# instructions, different tools, and - crucially - benefit from *not* seeing
# each other's working.
#
# It is also where multi-agent systems most often go wrong. Agents talk in
# circles, the reviewer approves everything, costs triple, and the output is
# worse than one good prompt. This capstone builds the desk **and** the
# controls that keep it honest: a supervisor with a budget, revision caps, a
# reviewer with a rubric it cannot wriggle out of, and a human gate before
# anything ships.

# %%
from pathlib import Path
import sys

_root = Path.cwd()
while not (_root / "shared").is_dir() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from shared.notebook_setup import setup, require  # noqa: E402

ctx = setup("52_multi_agent_research_desk")

# %%
import json
import operator
import time
from typing import Annotated, Literal, TypedDict

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from shared.llm import get_chat_model, get_embeddings

model = get_chat_model()
parser = StrOutputParser()

# %% [markdown]
# ## 1. The research substrate
#
# Real desks pull from the web. To keep this reproducible we use a local
# corpus, with a live search path that activates if you have a Tavily key.

# %%
KNOWLEDGE = {
    "vector-databases": (
        "Managed vector databases (Pinecone, Weaviate Cloud, Qdrant Cloud) charge per pod-hour or "
        "per million vectors. Typical mid-market spend is $500-2,000/month at 10M vectors. "
        "Self-hosted alternatives (pgvector, FAISS, Qdrant OSS) cost infrastructure only but "
        "require an engineer's ongoing attention. pgvector performance degrades noticeably past "
        "roughly 5M vectors without careful HNSW tuning. Migration between providers is painful "
        "because filter syntax and metadata models differ."
    ),
    "rag-quality": (
        "Retrieval quality dominates end-to-end RAG quality: in published benchmarks, improving "
        "recall@5 from 60% to 85% moves answer accuracy more than swapping a 7B model for a 70B "
        "one. Hybrid retrieval (dense + BM25) typically adds 8-15 points of recall over dense "
        "alone. Reranking adds a further 5-10 points at roughly 200ms and $0.001 per query. "
        "Chunk size between 400 and 800 tokens is the usual sweet spot for policy documents."
    ),
    "llm-costs": (
        "Frontier model pricing fell roughly 70% between 2024 and 2026. Small models "
        "(8B class) now handle classification, extraction and grading at 1-3% of frontier cost "
        "with negligible quality loss on those tasks. The dominant production cost is usually "
        "input tokens from retrieved context, not output tokens. Prompt caching cuts repeated "
        "system-prompt cost by 50-90% where the provider supports it."
    ),
    "agent-reliability": (
        "Agent reliability degrades sharply with step count: measured task success drops from "
        "roughly 90% at 3 steps to 40-60% at 15 steps without intervention. Explicit planning, "
        "state checkpointing, and per-step verification each recover part of that gap. "
        "Human-in-the-loop approval on irreversible actions is the single most effective "
        "mitigation, but approval fatigue sets in above about 1 interruption per 10 actions."
    ),
    "data-residency": (
        "Indian data residency requirements under the DPDP Act push analytics workloads to "
        "ap-south-1. Most managed LLM providers do not offer in-country inference, so teams "
        "either self-host open models or obtain explicit consent for cross-border transfer. "
        "This is now the most common procurement blocker for AI features in BFSI deals."
    ),
}


@tool
def search_knowledge(topic: str) -> str:
    """Search the research corpus for a topic.

    Available topics: vector-databases, rag-quality, llm-costs, agent-reliability,
    data-residency.
    """
    key = topic.lower().strip().replace(" ", "-")
    if key in KNOWLEDGE:
        return KNOWLEDGE[key]
    matches = [k for k in KNOWLEDGE if any(word in k for word in key.split("-"))]
    if matches:
        return f"No exact match. Did you mean: {', '.join(matches)}?"
    return f"Nothing found for {topic!r}. Available topics: {', '.join(KNOWLEDGE)}."


@tool
def list_topics() -> str:
    """List every topic available in the research corpus."""
    return ", ".join(sorted(KNOWLEDGE))


if require("TAVILY_API_KEY", feature="live web research"):
    from langchain_tavily import TavilySearch

    web_search = TavilySearch(max_results=3)
    research_tools = [search_knowledge, list_topics, web_search]
else:
    research_tools = [search_knowledge, list_topics]

print(f"research tools: {[t.name for t in research_tools]}")
print(f"corpus topics:  {sorted(KNOWLEDGE)}")

# %% [markdown]
# ## 2. Shared state
#
# Every agent reads and writes the same state object. This is the backbone of
# the design: the writer does not need the researcher's transcript, only its
# findings, and the reviewer does not need either - only the draft and the
# brief.

# %%
class Finding(BaseModel):
    """One researched claim with its source."""

    claim: str
    detail: str
    topic: str


class Critique(BaseModel):
    """A structured review against a fixed rubric."""

    grounded: bool = Field(description="Every claim traceable to a finding")
    specific: bool = Field(description="Uses concrete numbers rather than vague direction")
    actionable: bool = Field(description="A reader could make a decision from this")
    complete: bool = Field(description="Covers everything the brief asked for")
    required_changes: list[str] = Field(default_factory=list,
                                        description="Specific, actionable revisions. Empty if approved.")
    verdict: Literal["approve", "revise"]


class DeskState(TypedDict):
    brief: str
    plan: list[str]
    findings: Annotated[list[dict], operator.add]
    draft: str
    critiques: Annotated[list[dict], operator.add]
    revisions: Annotated[int, operator.add]
    researcher_calls: Annotated[int, operator.add]
    published: bool
    trace: Annotated[list[str], operator.add]


MAX_REVISIONS = 2
MAX_RESEARCH_CALLS = 6

# %% [markdown]
# ## 3. The supervisor
#
# The supervisor decides who works next. Two rules make the difference between
# a desk and a debating society:
#
# 1. It routes with **structured output**, so the decision is a value you can
#    log, test and override - not free text you have to parse.
# 2. **Budgets are enforced in code**, not by asking the model nicely.

# %%
class Assignment(BaseModel):
    """Whether the desk has enough evidence to start writing."""

    next_worker: Literal["researcher", "writer"]
    missing: str = Field(default="", description="What is still unresearched, if anything")
    reason: str = Field(description="Under 15 words")


supervisor_chain = ChatPromptTemplate.from_messages([
    ("system",
     "You run a research desk and there is no draft yet. Decide one thing: does the desk have "
     "enough evidence to start writing?\n\n"
     "- researcher: the brief asks about something with no findings yet, or the findings lack "
     "the numbers the brief needs.\n"
     "- writer: the findings cover every part of the brief. Choose this as soon as they do - "
     "more research past that point is wasted money.\n\n"
     "Judge coverage against the brief, not against how much you would like to know."),
    ("human",
     "Brief: {brief}\n\nFindings so far ({finding_count}):\n{findings}\n\n"
     "Research calls used: {researcher_calls}/{max_research_calls}"),
]) | model.with_structured_output(Assignment)


def supervisor(state: DeskState) -> Command[Literal["researcher", "writer", "reviewer", "publish"]]:
    """Route to the next worker, with hard budget enforcement around the model's choice."""
    # --- Hard rules. The model does not get a vote on these. ----------------
    if state["revisions"] >= MAX_REVISIONS and state["draft"]:
        return Command(goto="publish", update={"trace": ["supervisor: revision budget spent"]})

    latest = state["critiques"][-1] if state["critiques"] else None
    if latest and latest["verdict"] == "approve":
        return Command(goto="publish", update={"trace": ["supervisor: reviewer approved"]})

    if state["draft"]:
        # The review cycle is a strict alternation, so derive the next step from
        # the counts rather than asking the model. If the writer has not yet
        # answered the newest critique it owes a revision; otherwise the revised
        # draft needs re-reviewing.
        if not state["critiques"]:
            return Command(goto="reviewer", update={"trace": ["supervisor: first review"]})
        if state["revisions"] < len(state["critiques"]):
            return Command(goto="writer", update={"trace": ["supervisor: revise"]})
        return Command(goto="reviewer", update={"trace": ["supervisor: re-review"]})

    # --- The one genuinely ambiguous decision: is the research enough? ------
    assignment = supervisor_chain.invoke({
        "brief": state["brief"],
        "finding_count": len(state["findings"]),
        "findings": "\n".join(f"- [{f['topic']}] {f['claim']}" for f in state["findings"])
                    or "(none yet)",
        "researcher_calls": state["researcher_calls"],
        "max_research_calls": MAX_RESEARCH_CALLS,
    })

    destination = assignment.next_worker
    if destination == "researcher" and state["researcher_calls"] >= MAX_RESEARCH_CALLS:
        return Command(goto="writer",
                       update={"trace": ["supervisor: research budget spent, forcing writer"]})

    return Command(goto=destination,
                   update={"trace": [f"supervisor: {destination} ({assignment.reason})"]})

# %% [markdown]
# `Command(goto=...)` lets the supervisor update state and route in one step -
# no separate conditional edge to keep in sync with the node's logic. The
# `Command[Literal[...]]` return annotation is not decoration: it is how
# LangGraph learns the possible destinations, so the rendered graph shows real
# edges instead of a supervisor that appears to go nowhere.
#
# ### Count the decisions the model actually makes
#
# Look at that function again. Every route except one is hard-coded, and the
# only question that reaches the model is *is the research sufficient to start
# writing?*
#
# That is not a compromise; it is the finding. The first version of this desk
# routed everything through the model and hit the recursion limit, bouncing
# between the reviewer and itself. "The reviewer asked for changes, so the
# writer works next" is not a judgement call, and treating it as one invites
# the model to get it wrong. **Every rule you can state precisely belongs in
# code.** The model is for the decisions you genuinely cannot write down.
#
# Two practical consequences: the desk got dramatically cheaper (most
# supervisor turns no longer call a model at all), and it became testable -
# you can unit-test routing without a single API call.

# %% [markdown]
# ## 4. The researcher
#
# The researcher is a full agent with tools, running in its **own** message
# history. The desk never sees its intermediate tool calls - only structured
# findings. That isolation is what keeps the writer's context clean.

# %%
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware

researcher_agent = create_agent(
    model,
    tools=research_tools,
    system_prompt=(
        "You are a research analyst. Call list_topics first if you are unsure what exists. "
        "Gather concrete facts: numbers, percentages, costs, thresholds. "
        "Never state a fact you did not retrieve. Return dense findings, no preamble, "
        "no recommendations - that is the writer's job."
    ),
    middleware=[ModelCallLimitMiddleware(thread_limit=8, exit_behavior="end")],
    name="researcher",
)


class ExtractedFindings(BaseModel):
    """Findings pulled out of a research transcript."""

    findings: list[Finding]


finding_extractor = ChatPromptTemplate.from_messages([
    ("system", "Extract each distinct factual claim as a separate finding. Keep the numbers. "
               "Discard anything the researcher speculated about rather than retrieved."),
    ("human", "{transcript}"),
]) | model.with_structured_output(ExtractedFindings)


def researcher_node(state: DeskState) -> dict:
    known = {f["topic"] for f in state["findings"]}
    instruction = (
        f"Brief: {state['brief']}\n\n"
        f"Already researched: {sorted(known) or 'nothing'}. "
        "Research what is still missing. Use the tools."
    )
    outcome = researcher_agent.invoke({"messages": [HumanMessage(instruction)]})
    transcript = outcome["messages"][-1].content
    tool_calls = sum(len(getattr(m, "tool_calls", [])) for m in outcome["messages"])

    extracted = finding_extractor.invoke({"transcript": transcript}).findings
    return {
        "findings": [f.model_dump() for f in extracted],
        "researcher_calls": tool_calls,
        "trace": [f"researcher: {tool_calls} tool calls -> {len(extracted)} findings"],
    }

# %% [markdown]
# ## 5. The writer
#
# The writer sees the brief and the findings. Not the transcript, not the tool
# calls. On a revision it also sees exactly what the reviewer demanded.

# %%
writer_chain = ChatPromptTemplate.from_messages([
    ("system",
     "You are a senior analyst writing decision-grade briefs for an engineering leadership team.\n\n"
     "Structure:\n"
     "## Summary - three sentences a busy director can act on\n"
     "## What we found - the evidence, with the specific numbers\n"
     "## Recommendation - one clear course of action\n"
     "## Risks - what could make this wrong\n\n"
     "Rules: every claim must trace to a finding. Use the actual numbers. "
     "If the findings do not support a conclusion the brief asked for, say so explicitly "
     "rather than filling the gap. No filler, no hedging, no restating the question."),
    ("human",
     "Brief: {brief}\n\nFindings:\n{findings}\n\n"
     "{revision_instruction}"),
]) | model | parser


def writer_node(state: DeskState) -> dict:
    findings_text = "\n".join(
        f"- [{f['topic']}] {f['claim']}: {f['detail']}" for f in state["findings"]
    ) or "(no findings - say so in the brief)"

    if state["critiques"]:
        changes = state["critiques"][-1]["required_changes"]
        revision_instruction = (
            "This is a revision. Your previous draft:\n\n"
            f"{state['draft']}\n\n"
            "The reviewer requires these changes. Address every one:\n"
            + "\n".join(f"{i}. {c}" for i, c in enumerate(changes, 1))
        )
    else:
        revision_instruction = "Write the first draft."

    draft = writer_chain.invoke({"brief": state["brief"], "findings": findings_text,
                                 "revision_instruction": revision_instruction})
    return {"draft": draft,
            "revisions": 1 if state["critiques"] else 0,
            "trace": [f"writer: {'revision' if state['critiques'] else 'first draft'} "
                      f"({len(draft)} chars)"]}

# %% [markdown]
# ## 6. The reviewer
#
# The most common failure in multi-agent systems is a reviewer that approves
# everything. Three defences:
#
# 1. A **fixed rubric** as structured output - it must answer each criterion.
# 2. It sees the **findings**, so it can actually check grounding rather than
#    judging plausibility.
# 3. An **explicit instruction to be adversarial**, plus a code-level check
#    that an "approve" with required changes is treated as "revise".

# %%
reviewer_chain = ChatPromptTemplate.from_messages([
    ("system",
     "You are a demanding editor. Your job is to find what is wrong, not to be encouraging.\n\n"
     "Check each criterion honestly:\n"
     "- grounded: is every claim traceable to a finding? A plausible claim with no finding fails.\n"
     "- specific: does it use the actual numbers, or vague direction like 'significantly cheaper'?\n"
     "- actionable: could a director make a decision from this today?\n"
     "- complete: does it cover everything the brief asked for?\n\n"
     "Approve only if all four pass. If you list required changes, the verdict must be 'revise'. "
     "Each required change must be specific enough to act on - 'add more detail' is useless."),
    ("human", "Brief: {brief}\n\nFindings available to the writer:\n{findings}\n\nDraft:\n{draft}"),
]) | model.with_structured_output(Critique)


def reviewer_node(state: DeskState) -> dict:
    findings_text = "\n".join(f"- [{f['topic']}] {f['claim']}" for f in state["findings"])
    critique = reviewer_chain.invoke({"brief": state["brief"], "findings": findings_text,
                                      "draft": state["draft"]})

    # Code-level consistency: approval with outstanding changes is not approval.
    if critique.verdict == "approve" and critique.required_changes:
        critique.verdict = "revise"

    passed = sum([critique.grounded, critique.specific, critique.actionable, critique.complete])
    return {"critiques": [critique.model_dump()],
            "trace": [f"reviewer: {passed}/4 criteria, {critique.verdict}, "
                      f"{len(critique.required_changes)} changes"]}

# %% [markdown]
# ## 7. Publishing, with a human gate
#
# The desk produces something a human will send to leadership. That is exactly
# the kind of irreversible action that needs approval - `interrupt()` from
# notebook 36.

# %%
def publish_node(state: DeskState) -> dict:
    decision = interrupt({
        "action": "publish_brief",
        "brief": state["brief"],
        "draft": state["draft"],
        "revisions_used": state["revisions"],
        "reviewer_verdict": state["critiques"][-1]["verdict"] if state["critiques"] else "none",
        "outstanding_changes": state["critiques"][-1]["required_changes"] if state["critiques"] else [],
        "options": ["approve", "reject", "edit"],
    })

    if isinstance(decision, dict) and decision.get("action") == "edit":
        return {"draft": decision["draft"], "published": True,
                "trace": ["publish: human edited then approved"]}
    if decision == "reject" or (isinstance(decision, dict) and decision.get("action") == "reject"):
        return {"published": False, "trace": ["publish: rejected by human"]}
    return {"published": True, "trace": ["publish: approved by human"]}

# %% [markdown]
# ## 8. Wiring the desk
#
# Every worker returns to the supervisor. The supervisor is the only node that
# routes, which means there is exactly one place to look when the desk
# misbehaves.

# %%
builder = StateGraph(DeskState)
builder.add_node("supervisor", supervisor)
builder.add_node("researcher", researcher_node)
builder.add_node("writer", writer_node)
builder.add_node("reviewer", reviewer_node)
builder.add_node("publish", publish_node)

builder.add_edge(START, "supervisor")
builder.add_edge("researcher", "supervisor")
builder.add_edge("writer", "supervisor")
builder.add_edge("reviewer", "supervisor")
builder.add_edge("publish", END)

desk = builder.compile(checkpointer=InMemorySaver())
print(desk.get_graph().draw_ascii())

# %% [markdown]
# > The dashed lines out of the supervisor are the `Command(goto=...)` routes
# > declared by its return annotation. Solid lines are the static edges bringing
# > each worker back.

# %%
BRIEF = ("Should we move our RAG stack from self-hosted pgvector to a managed vector database? "
         "Cover cost, retrieval quality impact, and data residency. Recommend a course of action.")


class CostMeter(BaseCallbackHandler):
    def __init__(self):
        self.calls = self.input_tokens = self.output_tokens = 0

    def on_llm_end(self, response, **kwargs):
        self.calls += 1
        try:
            usage = response.generations[0][0].message.usage_metadata or {}
        except (AttributeError, IndexError):
            return
        self.input_tokens += usage.get("input_tokens", 0)
        self.output_tokens += usage.get("output_tokens", 0)


meter = CostMeter()
config = {"configurable": {"thread_id": "desk-run-1"}, "callbacks": [meter],
          "recursion_limit": 40}

started = time.perf_counter()
result = desk.invoke(
    {"brief": BRIEF, "plan": [], "findings": [], "draft": "", "critiques": [],
     "revisions": 0, "researcher_calls": 0, "published": False, "trace": []},
    config,
)
elapsed = time.perf_counter() - started

print(f"ran for {elapsed:.1f}s, {meter.calls} model calls, "
      f"{meter.input_tokens:,} in / {meter.output_tokens:,} out\n")
for step in desk.get_state(config).values["trace"]:
    print(f"   {step}")

# %% [markdown]
# ## 9. The approval gate

# %%
state = desk.get_state(config)
print(f"interrupted: {bool(state.interrupts)}")

if state.interrupts:
    request = state.interrupts[0].value
    print(f"\naction:            {request['action']}")
    print(f"revisions used:    {request['revisions_used']}")
    print(f"reviewer verdict:  {request['reviewer_verdict']}")
    print(f"outstanding:       {request['outstanding_changes']}")
    print(f"\n--- draft ({len(request['draft'])} chars) ---")
    print(request["draft"][:1400])

# %%
desk.invoke(Command(resume="approve"), config)
values = desk.get_state(config).values
print(f"published: {values['published']}")
print(f"findings:  {len(values['findings'])}")
print(f"revisions: {values['revisions']}")
print(f"critiques: {[c['verdict'] for c in values['critiques']]}")

# %% [markdown]
# ### Rejecting instead
#
# A rejection is not a failure of the desk - it is the gate doing its job.

# %%
reject_config = {"configurable": {"thread_id": "desk-run-reject"}, "recursion_limit": 40}
desk.invoke(
    {"brief": "Summarise what we know about agent reliability at high step counts.",
     "plan": [], "findings": [], "draft": "", "critiques": [], "revisions": 0,
     "researcher_calls": 0, "published": False, "trace": []},
    reject_config,
)
desk.invoke(Command(resume={"action": "reject"}), reject_config)
rejected = desk.get_state(reject_config).values
print(f"published: {rejected['published']}  (draft preserved: {len(rejected['draft'])} chars)")
print(f"trace tail: {rejected['trace'][-2:]}")

# %% [markdown]
# ## 10. Does the desk beat one good prompt?
#
# This is the question nobody asks and everybody should. Multi-agent costs 5-10x
# a single call. Score both against the reviewer's own rubric.

# %%
single_shot = ChatPromptTemplate.from_messages([
    ("system", "You are a senior analyst. Write a decision-grade brief with Summary, "
               "What we found, Recommendation and Risks sections. Use the research provided."),
    ("human", "Brief: {brief}\n\nResearch:\n{research}"),
]) | model | parser

single_meter = CostMeter()
single_started = time.perf_counter()
single_draft = single_shot.invoke(
    {"brief": BRIEF, "research": "\n\n".join(f"[{k}] {v}" for k, v in KNOWLEDGE.items())},
    {"callbacks": [single_meter]},
)
single_elapsed = time.perf_counter() - single_started

desk_draft = values["draft"]
findings_text = "\n".join(f"- [{f['topic']}] {f['claim']}" for f in values["findings"])

print(f"{'':10} {'calls':>6} {'in tokens':>10} {'seconds':>8} {'chars':>7}")
print(f"{'single':10} {single_meter.calls:>6} {single_meter.input_tokens:>10,} "
      f"{single_elapsed:>8.1f} {len(single_draft):>7}")
print(f"{'desk':10} {meter.calls:>6} {meter.input_tokens:>10,} {elapsed:>8.1f} {len(desk_draft):>7}")

# %%
print(f"\n{'criterion':16} {'single':>8} {'desk':>8}")
judgements = {
    "single": reviewer_chain.invoke({"brief": BRIEF, "findings": findings_text, "draft": single_draft}),
    "desk": reviewer_chain.invoke({"brief": BRIEF, "findings": findings_text, "draft": desk_draft}),
}
for criterion in ("grounded", "specific", "actionable", "complete"):
    print(f"{criterion:16} {str(getattr(judgements['single'], criterion)):>8} "
          f"{str(getattr(judgements['desk'], criterion)):>8}")

for label, judgement in judgements.items():
    print(f"\n{label}: {judgement.verdict}")
    for change in judgement.required_changes[:2]:
        print(f"   - {change}")

# %% [markdown]
# Be honest about what you see. With a small corpus already in context, the
# single call is often competitive - the desk earns its cost when research is
# genuinely open-ended, when the corpus does not fit in one window, or when the
# review step catches errors that matter.

# %% [markdown]
# ## 11. Memory across runs
#
# A desk that forgets is a desk that repeats itself. Store findings and editor
# preferences (notebook 49) so run two starts where run one finished.

# %%
embeddings = get_embeddings()
desk_store = InMemoryStore(index={"embed": embeddings,
                                  "dims": len(embeddings.embed_query("x")),
                                  "fields": ["text"]})


def cache_findings(state: DeskState) -> None:
    for finding in state["findings"]:
        desk_store.put(("desk", "findings"), f"{finding['topic']}:{hash(finding['claim']) & 0xffff}",
                       {"text": f"{finding['claim']}: {finding['detail']}",
                        "topic": finding["topic"]})


def recall_findings(brief: str, limit: int = 4) -> list[dict]:
    return [{"claim": hit.value["text"], "detail": "", "topic": hit.value["topic"]}
            for hit in desk_store.search(("desk", "findings"), query=brief, limit=limit)]


cache_findings(values)
print(f"cached {len(desk_store.search(('desk', 'findings')))} findings")
recalled = recall_findings("what do we know about the cost of managed vector search?")
for item in recalled[:2]:
    print(f"   [{item['topic']}] {item['claim'][:88]}")

# %%
# Editor preferences are procedural memory: the desk learns house style.
desk_store.put(("desk", "style"), "numbers",
               {"text": "Leadership wants INR figures alongside USD, and a payback period "
                        "whenever a migration is recommended."})
desk_store.put(("desk", "style"), "length",
               {"text": "Briefs must fit on one page. Cut the 'What we found' section first."})

house_style = "\n".join(f"- {i.value['text']}" for i in desk_store.search(("desk", "style")))
print(f"house style loaded into the writer prompt:\n{house_style}")

# %% [markdown]
# Wire `house_style` into `writer_chain`'s system message and the desk starts
# producing briefs in your organisation's voice without a code change.

# %% [markdown]
# ## 12. Streaming the desk to a UI
#
# A desk run takes a minute or more. Show the work.

# %%
WORKER_LABELS = {
    "supervisor": "Assigning work",
    "researcher": "Researching",
    "writer": "Drafting",
    "reviewer": "Reviewing",
    "publish": "Waiting for your approval",
}

stream_config = {"configurable": {"thread_id": "desk-stream"}, "recursion_limit": 40}
for chunk in desk.stream(
    {"brief": "What should we know about LLM cost trends before budgeting next year?",
     "plan": [], "findings": [], "draft": "", "critiques": [], "revisions": 0,
     "researcher_calls": 0, "published": False, "trace": []},
    stream_config,
    stream_mode="updates",
):
    for node, update in chunk.items():
        if node == "__interrupt__":
            print("   >> paused for approval")
            continue
        detail = (update or {}).get("trace", [""])[-1] if isinstance(update, dict) else ""
        print(f"   {WORKER_LABELS.get(node, node):28} {detail}")

# %% [markdown]
# ## 13. Failure modes and their fixes
#
# | Failure | Symptom | Fix used here |
# |---|---|---|
# | Infinite handoffs | Supervisor bounces between two workers | `recursion_limit` + hard stops before the model decides |
# | Rubber-stamp reviewer | Every draft approved first time | Fixed rubric, findings visible, "approve + changes" coerced to "revise" |
# | Runaway research | Twenty tool calls, no draft | `MAX_RESEARCH_CALLS`, enforced in `supervisor` |
# | Endless revision | Reviewer never satisfied | `MAX_REVISIONS`, then publish with outstanding changes shown to the human |
# | Context pollution | Writer sees raw tool output | Researcher is an isolated agent; only findings cross the boundary |
# | Silent cost blowout | Bill arrives | `CostMeter` on every run, logged per thread |
#
# The pattern: **the model proposes, the code disposes.** Every budget in this
# desk is enforced in Python, not requested in a prompt.

# %%
def desk_health(state_values: dict, meter: CostMeter) -> dict:
    """The metrics worth alerting on."""
    critiques = state_values.get("critiques", [])
    return {
        "model_calls": meter.calls,
        "input_tokens": meter.input_tokens,
        "revisions": state_values.get("revisions", 0),
        "hit_revision_cap": state_values.get("revisions", 0) >= MAX_REVISIONS,
        "reviewer_approved": bool(critiques) and critiques[-1]["verdict"] == "approve",
        "first_pass_approval": bool(critiques) and critiques[0]["verdict"] == "approve",
        "published": state_values.get("published", False),
    }


health = desk_health(values, meter)
print(json.dumps(health, indent=2))
if health["first_pass_approval"]:
    print("\nWARNING: reviewer approved the first draft. Check it is not rubber-stamping.")

# %% [markdown]
# ## Try it yourself
#
# 1. **Add a fact-checker.** A fourth worker that takes each claim in the draft
#    and verifies it against `findings`, returning the unsupported ones. Compare
#    its catch rate with the reviewer's `grounded` flag.
# 2. **Parallel research.** Use `Send` to fan out one researcher per topic and
#    fan in the findings. Measure the latency change.
# 3. **Make the reviewer cheaper.** Swap it to a small model and re-run the
#    comparison in section 10. Does review quality actually drop?
# 4. **Persist across runs.** Load `recall_findings` into the initial state and
#    confirm the supervisor skips straight to the writer on a repeat brief.
# 5. **Adversarial brief.** Ask for something the corpus cannot answer and
#    confirm the desk says so rather than inventing findings.

# %% [markdown]
# ## Recap
#
# | Decision | Why |
# |---|---|
# | Shared state, isolated agents | Findings cross the boundary; transcripts do not |
# | Supervisor as the only router | One place to debug when the desk misbehaves |
# | `Command(goto=...)` | Update state and route in a single step |
# | Structured routing | A loggable, testable decision instead of parsed text |
# | Budgets in code | `MAX_REVISIONS` and `MAX_RESEARCH_CALLS` before the model chooses |
# | Rubric-based review | The defence against a reviewer that approves everything |
# | Approve-with-changes coerced | Consistency the model cannot talk its way around |
# | `interrupt()` before publishing | Irreversible action, human gate |
# | Cost meter per run | Multi-agent is 5-10x; know the number |
# | Single-call baseline | Prove the desk is worth it before you keep it |
# | Store for findings and style | Run two starts where run one finished |
#
# ## Next
#
# -> [53_fastapi_langgraph_service.ipynb](53_fastapi_langgraph_service.ipynb)
