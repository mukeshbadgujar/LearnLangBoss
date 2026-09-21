# %% [markdown]
# # 50 - Deep Agents
#
# | | |
# |---|---|
# | **Level** | Expert (LangGraph) |
# | **Time** | 70 minutes |
# | **Prerequisites** | `41_multi_agent_handoffs`, `49_long_term_semantic_memory` |
# | **Checklist ID** | `50_deep_agents` |
#
# ## Why this matters
#
# A normal agent does one thing: call tools in a loop until it can answer. That
# works for "look up this ticket and summarise it". It falls apart on "research
# our three competitors and write a positioning brief" - the agent wanders,
# forgets what it already did, fills its context with raw tool output, and
# produces something shallow.
#
# A **deep agent** is an agent plus four capabilities that make long-horizon
# work possible:
#
# | Capability | Problem it solves |
# |---|---|
# | **Planning** | The agent loses the thread across 30 steps |
# | **Filesystem / workspace** | Findings do not fit in context |
# | **Sub-agents** | One agent's context gets polluted by every subtask |
# | **Detailed harness** | The model does not know *how* you want the work done |
#
# None of this is new machinery. It is the graph, state, tools, middleware and
# store you already know, assembled deliberately. This notebook builds a deep
# agent from those primitives, then shows the packaged version.

# %%
from pathlib import Path
import sys

_root = Path.cwd()
while not (_root / "shared").is_dir() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from shared.notebook_setup import setup, require  # noqa: E402

ctx = setup("50_deep_agents")

# %%
import operator
from typing import Annotated, TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    SummarizationMiddleware,
    TodoListMiddleware,
)
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from langchain_core.tools import InjectedToolCallId

from shared.llm import get_chat_model

model = get_chat_model()

# %% [markdown]
# ## 1. The baseline problem
#
# Give a plain agent a genuinely multi-step task and watch it flatten.

# %%
RESEARCH_NOTES = {
    "northwind": "Northwind Analytics. Founded 2019, Pune. 120 employees. Flagship product is a "
                 "self-serve BI tool. Pricing INR 2,400/seat/month. Strong in mid-market retail. "
                 "Weakness: no real-time streaming support. Raised Series B of $30M in 2024.",
    "acme": "Acme Data. Founded 2015, Bangalore. 400 employees. Enterprise data platform. "
            "Pricing is custom, typically INR 40 lakh+ annually. Strong in BFSI compliance. "
            "Weakness: 6-month implementation cycles, poor developer experience.",
    "kestrel": "Kestrel Systems. Founded 2022, remote-first. 35 employees. Developer-first "
               "streaming analytics. Usage-based pricing from INR 18,000/month. "
               "Strong real-time story. Weakness: thin BI/reporting layer, small support team.",
    "market": "Indian analytics market grew 28% in 2025. Buyers increasingly require data "
              "residency in-country. Procurement cycles lengthened from 3 to 5 months. "
              "Usage-based pricing now preferred by 61% of mid-market buyers.",
}


@tool
def research(topic: str) -> str:
    """Look up research notes. Topics: northwind, acme, kestrel, market."""
    return RESEARCH_NOTES.get(topic.lower().strip(), f"No notes found for {topic!r}.")


BRIEF = ("Write a competitive positioning brief for Northwind Analytics covering Acme and "
         "Kestrel, with a recommendation on pricing strategy.")

plain_agent = create_agent(model, tools=[research],
                           system_prompt="You are a market analyst. Use the research tool.")

plain = plain_agent.invoke({"messages": [HumanMessage(BRIEF)]})
plain_calls = [tc["args"]["topic"] for m in plain["messages"] for tc in getattr(m, "tool_calls", [])]
plain_answer = plain["messages"][-1].content

print(f"tool calls: {plain_calls}")
print(f"answer length: {len(plain_answer)} chars\n")
print(plain_answer[:500])

# %% [markdown]
# Typically: two or three lookups, then a generic brief. It never checked the
# market context, never revisited anything, and produced roughly what it could
# hold in one context window.

# %% [markdown]
# ## 2. Capability 1 - planning
#
# `TodoListMiddleware` adds a `write_todos` tool and a `todos` key to state. The
# plan is **visible state**, not something buried in the model's reasoning, so
# you can inspect it, and so can the model on every subsequent turn.

# %%
planning_agent = create_agent(
    model,
    tools=[research],
    system_prompt="You are a market analyst. Plan multi-step work before starting.",
    middleware=[TodoListMiddleware()],
    checkpointer=InMemorySaver(),
)

print(f"graph nodes: {list(planning_agent.get_graph().nodes)}")
print(f"state keys:  {planning_agent.stream_channels_list}")

# %%
config = {"configurable": {"thread_id": "planned-brief"}}
planned = planning_agent.invoke({"messages": [HumanMessage(BRIEF)]}, config)

todos = planning_agent.get_state(config).values.get("todos", [])
print(f"plan ({len(todos)} items):")
for todo in todos:
    print(f"   [{todo.get('status', '?'):11}] {todo.get('content', '')[:64]}")

planned_calls = [tc["args"].get("topic") for m in planned["messages"]
                 for tc in getattr(m, "tool_calls", []) if tc["name"] == "research"]
print(f"\nresearch calls: {planned_calls}")
print(f"answer length:  {len(planned['messages'][-1].content)} chars")

# %% [markdown]
# The plan lives in state, so you can render it in a UI as a progress bar, and
# a human can inspect exactly what the agent thinks it still has to do.

# %% [markdown]
# ## 3. Capability 2 - a workspace
#
# The second failure was context. A long task generates more findings than fit
# in the window. Give the agent a filesystem **in graph state**: it writes
# findings to files, and reads back only what it needs.

# %%
def merge_files(existing: dict | None, new: dict | None) -> dict:
    """Merge writes instead of replacing the workspace, so parallel writes coexist."""
    return {**(existing or {}), **(new or {})}


class WorkspaceState(TypedDict):
    messages: Annotated[list, operator.add]
    files: Annotated[dict[str, str], merge_files]


@tool
def write_file(path: str, content: str,
               tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Write content to a file in your workspace, replacing anything already there."""
    from langchain_core.messages import ToolMessage

    return Command(update={
        "files": {path: content},
        "messages": [ToolMessage(f"Wrote {len(content)} chars to {path}", tool_call_id=tool_call_id)],
    })


@tool
def read_file(path: str, state: Annotated[dict, InjectedState]) -> str:
    """Read a file from your workspace."""
    files = state.get("files") or {}
    if path not in files:
        return f"No such file: {path}. Available: {sorted(files) or 'none'}"
    return files[path]


@tool
def list_files(state: Annotated[dict, InjectedState]) -> str:
    """List the files in your workspace with their sizes."""
    files = state.get("files") or {}
    if not files:
        return "Workspace is empty."
    return "\n".join(f"{path} ({len(content)} chars)" for path, content in sorted(files.items()))


@tool
def edit_file(path: str, find: str, replace: str,
              state: Annotated[dict, InjectedState],
              tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Replace the first occurrence of `find` with `replace` in a file."""
    from langchain_core.messages import ToolMessage

    files = state.get("files") or {}
    if path not in files:
        return Command(update={"messages": [ToolMessage(f"No such file: {path}",
                                                        tool_call_id=tool_call_id)]})
    if find not in files[path]:
        return Command(update={"messages": [ToolMessage(f"{find!r} not found in {path}",
                                                        tool_call_id=tool_call_id)]})
    return Command(update={
        "files": {path: files[path].replace(find, replace, 1)},
        "messages": [ToolMessage(f"Edited {path}", tool_call_id=tool_call_id)],
    })


class WorkspaceMiddleware(AgentMiddleware):
    """A virtual filesystem living in graph state - checkpointed, inspectable, no disk access."""

    state_schema = WorkspaceState

    def __init__(self):
        super().__init__()
        self.tools = [write_file, read_file, list_files, edit_file]


print("workspace tools:", [t.name for t in WorkspaceMiddleware().tools])
print("model sees for write_file:", list(write_file.args.keys()))
print("merge reducer:", merge_files({"a.md": "1"}, {"b.md": "2"}))

# %% [markdown]
# Three details that matter:
#
# - **`InjectedState` / `InjectedToolCallId` are hidden from the model.** It sees
#   `path` and `content`, not the plumbing.
# - **A write needs `Command`**, not a return value, because it updates state as
#   well as answering the tool call.
# - **`files` needs a merging reducer.** With the default overwrite reducer, two
#   writes in the same superstep would leave you with one file.

# %% [markdown]
# ## 4. Capability 3 - sub-agents
#
# The third failure is context pollution. If one agent researches three
# competitors, its window fills with raw notes and the synthesis suffers.
#
# A sub-agent runs in its **own** message history and returns only a summary.
# The parent never sees the intermediate mess. This is context isolation, and
# it is the single biggest quality lever in a deep agent.

# %%
def make_subagent(name: str, instructions: str, tools: list):
    """A sub-agent is just an agent - the isolation comes from how we call it."""
    return create_agent(model, tools=tools, name=name,
                        system_prompt=instructions,
                        middleware=[ModelCallLimitMiddleware(thread_limit=8, exit_behavior="end")])


researcher = make_subagent(
    "researcher",
    "You research one topic thoroughly using the research tool. "
    "Return a dense factual summary: facts, figures, strengths, weaknesses. No preamble.",
    [research],
)


@tool
def delegate_research(topic: str, focus: str) -> str:
    """Delegate research on one topic to a specialist.

    The specialist works in its own context and returns only its findings,
    so raw material never enters your conversation.

    Args:
        topic: What to research (northwind, acme, kestrel, market).
        focus: What specifically you need out of it.
    """
    outcome = researcher.invoke({"messages": [
        HumanMessage(f"Research {topic}. Focus on: {focus}")
    ]})
    return outcome["messages"][-1].content


print(delegate_research.invoke({"topic": "kestrel", "focus": "pricing and weaknesses"})[:300])

# %% [markdown]
# ## 5. Capability 4 - the harness
#
# The last piece is the prompt, and it is the piece people underinvest in. A
# deep agent's system prompt is not a sentence - it is a **procedure**: when to
# plan, where to write things, when to delegate, what "done" means.

# %%
DEEP_AGENT_PROMPT = """You are a senior market analyst producing decision-grade briefs.

## Process

1. **Plan first.** Call `write_todos` before any other tool. Break the request into
   concrete steps. Mark each `in_progress` when you start it and `completed` the
   moment it is done - never batch completions.

2. **Delegate research.** Use `delegate_research` for each topic. Do not research
   directly; the specialist returns denser findings than you would gather inline.

3. **Write findings to files as you go.** After each piece of research, call
   `write_file` to save it (`research_<topic>.md`). Your conversation is not
   storage - files are. If you need something back, `read_file` it.

4. **Synthesise from files.** Once research is complete, read your files and write
   the brief to `brief.md`. Then reply with the brief itself.

## Quality bar

- Every claim cites a specific figure or fact from the research.
- The pricing recommendation names a number and justifies it against both competitors.
- Name at least one risk with the recommendation.
- If the research does not support a claim, say so rather than filling the gap.

## Rules

- Never write the final brief without having read your research files back.
- If a research call returns nothing useful, record that in the file rather than inventing content.
"""

print(f"harness: {len(DEEP_AGENT_PROMPT)} chars, {len(DEEP_AGENT_PROMPT.split())} words")

# %% [markdown]
# ## 6. Assembling the deep agent

# %%
deep_agent = create_agent(
    model,
    tools=[delegate_research, write_file, read_file, list_files, edit_file],
    system_prompt=DEEP_AGENT_PROMPT,
    middleware=[
        TodoListMiddleware(),
        WorkspaceMiddleware(),
        SummarizationMiddleware(model=model, trigger=("tokens", 6000), keep=("messages", 6)),
        ModelCallLimitMiddleware(thread_limit=30, exit_behavior="end"),
    ],
    checkpointer=InMemorySaver(),
)

print("nodes:", list(deep_agent.get_graph().nodes))
print("state:", deep_agent.stream_channels_list)

# %% [markdown]
# Four middleware, each solving one of the four failures:
#
# | Middleware | Capability |
# |---|---|
# | `TodoListMiddleware` | Planning |
# | `WorkspaceMiddleware` | Filesystem |
# | `SummarizationMiddleware` | Context survival on long runs |
# | `ModelCallLimitMiddleware` | A hard stop so a wandering agent cannot run forever |
#
# Sub-agents come in as a tool rather than middleware.

# %%
deep_config = {"configurable": {"thread_id": "deep-brief"}}

step_log: list[str] = []
for chunk in deep_agent.stream({"messages": [HumanMessage(BRIEF)]}, deep_config, stream_mode="updates"):
    for node, update in chunk.items():
        for message in update.get("messages", []) or []:
            for call in getattr(message, "tool_calls", []):
                label = call["args"].get("topic") or call["args"].get("path") or ""
                step_log.append(f"{call['name']}({label})".strip("()"))

print(f"{len(step_log)} tool calls:")
for step in step_log:
    print(f"   {step}")

# %%
final_state = deep_agent.get_state(deep_config).values

print("plan:")
for todo in final_state.get("todos", []):
    print(f"   [{todo.get('status', '?'):11}] {todo.get('content', '')[:62]}")

print("\nworkspace:")
for path, content in sorted((final_state.get("files") or {}).items()):
    print(f"   {path:26} {len(content):>6} chars")

# %%
brief = (final_state.get("files") or {}).get("brief.md") or final_state["messages"][-1].content
print(f"final brief: {len(brief)} chars (plain agent produced {len(plain_answer)})\n")
print(brief[:900])

# %% [markdown]
# ## 7. Did it actually get better?
#
# "Longer" is not "better". Score it against the quality bar in the harness.

# %%
from pydantic import BaseModel, Field


class BriefScore(BaseModel):
    """Score a competitive brief against a fixed rubric."""

    cites_specific_figures: bool = Field(description="Uses concrete numbers from the research")
    covers_both_competitors: bool = Field(description="Substantive coverage of Acme AND Kestrel")
    uses_market_context: bool = Field(description="References market trends, not just company facts")
    pricing_recommendation_is_specific: bool = Field(description="Names an actual number or model")
    names_a_risk: bool
    notes: str = Field(description="One sentence on the weakest part")


scorer = model.with_structured_output(BriefScore)
RUBRIC = "Score this competitive positioning brief strictly.\n\n{brief}"

print(f"{'criterion':38} {'plain':>7} {'deep':>7}")
scores = {label: scorer.invoke(RUBRIC.format(brief=text[:4000]))
          for label, text in [("plain", plain_answer), ("deep", brief)]}

for field in BriefScore.model_fields:
    if field == "notes":
        continue
    print(f"{field:38} {str(getattr(scores['plain'], field)):>7} {str(getattr(scores['deep'], field)):>7}")

for label, score in scores.items():
    print(f"\n{label}: {score.notes}")

# %% [markdown]
# ## 8. Human in the loop
#
# Long-horizon agents take consequential actions. Gate the ones that matter -
# this is `HumanInTheLoopMiddleware` from notebook 36, applied per tool.

# %%
@tool
def publish_brief(path: str, audience: str) -> str:
    """Publish a finished brief to an audience. This is irreversible."""
    return f"Published {path} to {audience}."


approving_agent = create_agent(
    model,
    tools=[delegate_research, write_file, read_file, list_files, publish_brief],
    system_prompt=DEEP_AGENT_PROMPT + "\nWhen the brief is finished, publish it to 'leadership'.",
    middleware=[
        TodoListMiddleware(),
        WorkspaceMiddleware(),
        HumanInTheLoopMiddleware(interrupt_on={
            "publish_brief": True,       # always ask
            "write_file": False,         # never ask - cheap and reversible
            "delegate_research": False,
        }),
        ModelCallLimitMiddleware(thread_limit=25, exit_behavior="end"),
    ],
    checkpointer=InMemorySaver(),
)

print("interrupt config: publish_brief requires approval, everything else runs freely")
print("nodes:", list(approving_agent.get_graph().nodes))

# %% [markdown]
# ```python
# approve_config = {"configurable": {"thread_id": "approved-brief"}}
# result = approving_agent.invoke({"messages": [HumanMessage(BRIEF)]}, approve_config)
#
# state = approving_agent.get_state(approve_config)
# if state.interrupts:                       # paused before publishing
#     request = state.interrupts[0].value
#     print(request)                         # shows the pending tool call and its args
#
#     # Approve, edit the arguments, or reject with feedback:
#     approving_agent.invoke(Command(resume=[{"type": "accept"}]), approve_config)
#     # {"type": "edit", "args": {...}}  |  {"type": "response", "args": "Not yet - add a risk section"}
# ```
#
# **Which tools to gate:** anything irreversible, anything that costs money,
# anything a customer sees. Gating a read-only lookup just teaches your users to
# click approve without reading.

# %% [markdown]
# ## 9. Skills - reusable procedures
#
# A "skill" is a named procedure the agent can load on demand: instructions plus
# the tools that procedure needs. Rather than one enormous prompt covering every
# task, keep them in the store (notebook 49) and load the relevant one.

# %%
from langgraph.store.memory import InMemoryStore

skill_store = InMemoryStore()

SKILLS = {
    "competitive_brief": {
        "name": "competitive_brief",
        "when": "The user asks for competitor analysis or positioning",
        "procedure": (
            "1. Research each named competitor plus the market.\n"
            "2. Build a comparison table: pricing, target segment, strength, weakness.\n"
            "3. Identify the gap only our product fills.\n"
            "4. Recommend positioning with a specific price point and one named risk."
        ),
        "tools": ["delegate_research", "write_file", "read_file"],
    },
    "incident_review": {
        "name": "incident_review",
        "when": "The user asks for a post-incident review or RCA",
        "procedure": (
            "1. Establish the timeline before analysing anything.\n"
            "2. Separate trigger, root cause, and contributing factors.\n"
            "3. Name detection and mitigation gaps.\n"
            "4. Propose actions with owners. Never assign blame to individuals."
        ),
        "tools": ["read_file", "write_file"],
    },
}
for key, skill in SKILLS.items():
    skill_store.put(("skills",), key, skill)


@tool
def list_skills() -> str:
    """List the procedures you know, and when each applies."""
    return "\n".join(f"- {i.value['name']}: {i.value['when']}" for i in skill_store.search(("skills",)))


@tool
def load_skill(name: str) -> str:
    """Load the step-by-step procedure for a named skill before starting work."""
    item = skill_store.get(("skills",), name)
    if item is None:
        return f"Unknown skill {name!r}. Call list_skills first."
    return f"Procedure for {name}:\n{item.value['procedure']}\n\nTools: {', '.join(item.value['tools'])}"


skilled_agent = create_agent(
    model,
    tools=[list_skills, load_skill, delegate_research, write_file, read_file, list_files],
    system_prompt=(
        "You are a senior analyst. Before starting any substantial task, call `list_skills` "
        "and `load_skill` to load the right procedure, then follow it exactly. "
        "Plan with `write_todos`, save findings with `write_file`."
    ),
    middleware=[TodoListMiddleware(), WorkspaceMiddleware(),
                ModelCallLimitMiddleware(thread_limit=25, exit_behavior="end")],
    checkpointer=InMemorySaver(),
)

skill_config = {"configurable": {"thread_id": "skilled"}}
skilled = skilled_agent.invoke(
    {"messages": [HumanMessage("Give me a positioning brief on Northwind versus Kestrel.")]},
    skill_config,
)
used = [tc["name"] for m in skilled["messages"] for tc in getattr(m, "tool_calls", [])]
print(f"tools used: {used}")
print(f"\n{skilled['messages'][-1].content[:400]}")

# %% [markdown]
# Skills keep the base prompt small (cheap, and the model follows short prompts
# better) while letting you add procedures without touching code. They live in
# the store, so a domain expert can edit one without a deploy.

# %% [markdown]
# ## 10. The packaged version
#
# `deepagents` bundles these four capabilities. It is optional - everything
# above works without it - but it is the reference implementation worth reading.

# %%
try:
    from deepagents import create_deep_agent

    packaged = create_deep_agent(
        tools=[research],
        instructions=DEEP_AGENT_PROMPT,
        subagents=[{
            "name": "researcher",
            "description": "Researches one topic deeply and returns dense findings",
            "prompt": "Research the topic thoroughly. Return facts and figures only.",
            "tools": ["research"],
        }],
    )
    print("deepagents installed - nodes:", list(packaged.get_graph().nodes))
except ImportError:
    print("`deepagents` is not installed (pip install deepagents).")
    print("It ships the same four capabilities we built by hand:")
    print("   planning      -> write_todos tool + todos state")
    print("   filesystem    -> ls / read_file / write_file / edit_file over state")
    print("   sub-agents    -> a task tool that spawns isolated agents")
    print("   harness       -> a long, opinionated default prompt")
    print("\nOur hand-built version is more code but every decision is visible and editable.")

# %% [markdown]
# ## 11. When a deep agent is the wrong answer
#
# This architecture costs 5-20x a plain agent and is far harder to debug. Use it
# only when the task genuinely needs it.
#
# | Signal | Build |
# |---|---|
# | Single lookup, single answer | A chain (notebook 08) |
# | A few tool calls, clear finish | A plain agent (notebook 17) |
# | Fixed steps, known order | A graph (notebook 31) |
# | Needs approval at one point | A graph with an interrupt (notebook 36) |
# | Open-ended, 10+ steps, output exceeds context | **A deep agent** |
#
# Ask: *would a competent human need a plan, notes and a way to hand off parts
# of this?* If not, you do not need a deep agent.

# %%
def recommend_architecture(steps: int, output_exceeds_context: bool, order_is_known: bool,
                           needs_approval: bool) -> str:
    if order_is_known and steps <= 6:
        return "graph with an interrupt" if needs_approval else "graph or chain"
    if steps >= 10 or output_exceeds_context:
        return "deep agent"
    return "plain agent"


for description, args in [
    ("summarise one ticket", (1, False, True, False)),
    ("triage then reply, with approval", (4, False, True, True)),
    ("look up 3 things and answer", (5, False, False, False)),
    ("full competitive brief", (15, True, False, False)),
]:
    print(f"{description:36} -> {recommend_architecture(*args)}")

# %% [markdown]
# ## 12. Production checklist
#
# - **Hard call limit.** `ModelCallLimitMiddleware` on every deep agent. An agent
#   without a ceiling is an unbounded bill.
# - **Checkpoint everything.** Long runs fail; resume beats restart.
# - **Stream progress.** A 3-minute silent run looks broken. Stream todos and
#   file writes (notebook 42).
# - **Gate irreversible tools only.** Approval fatigue is a real failure mode.
# - **Trace with LangSmith.** Sub-agent runs nest inside the parent trace; that
#   nesting is how you find out which sub-agent is burning the budget.
# - **Cap sub-agent depth.** A sub-agent that can spawn sub-agents needs a depth
#   counter in state, or it will recurse.
# - **Store artefacts outside state.** Checkpointing a 200KB workspace on every
#   step is slow; keep large files in object storage and paths in state.

# %%
from langchain_core.messages import ToolMessage


class DelegationBudget(AgentMiddleware):
    """Cap how many sub-agents a run may spawn, without raising."""

    def __init__(self, budget: int = 4):
        super().__init__()
        self.budget = budget

    def wrap_tool_call(self, request, handler):
        if not request.tool_call["name"].startswith("delegate"):
            return handler(request)

        spawned = sum(1 for message in (request.state.get("messages") or [])
                      for call in getattr(message, "tool_calls", [])
                      if call["name"].startswith("delegate"))
        if spawned >= self.budget:
            return ToolMessage(
                "Delegation budget exhausted. Synthesise from what you already have.",
                tool_call_id=request.tool_call["id"],
            )
        return handler(request)


print(f"DelegationBudget caps delegation at {DelegationBudget().budget} sub-agents per run")
print("It returns a ToolMessage rather than raising, so the agent can recover and finish.")

# %% [markdown]
# ## Try it yourself
#
# 1. **Remove one capability at a time** from `deep_agent` (planning, then the
#    workspace, then sub-agents) and re-score with the rubric. You will find one
#    matters far more than the others for this task.
# 2. **Add a critic sub-agent** that reads `brief.md` and returns required
#    revisions, then loop until it approves or three rounds pass.
# 3. **Persist the workspace.** Write files to `ctx.artifact(...)` on disk and
#    keep only paths in state, then confirm checkpoint size drops.
# 4. **Write a third skill** (`pricing_analysis`) into the store and check the
#    agent picks the right one without any code change.
# 5. **Instrument cost.** Attach notebook 27's `UsageTracker` and compare the
#    plain agent, the deep agent, and the deep agent without sub-agents.

# %% [markdown]
# ## Recap
#
# | Concept | Takeaway |
# |---|---|
# | Deep agent | Agent + planning + workspace + sub-agents + a real harness |
# | `TodoListMiddleware` | Plan as inspectable state, not hidden reasoning |
# | Workspace in state | `write_file` / `read_file` / `edit_file` over a state dict |
# | Merging reducer | Required so parallel file writes do not clobber each other |
# | Sub-agents | Context isolation - the parent sees summaries, not raw material |
# | Harness | A procedure, a quality bar and rules - not one sentence |
# | `HumanInTheLoopMiddleware` | Gate irreversible tools only |
# | Skills | Procedures in the store, loaded on demand, editable without a deploy |
# | `deepagents` | Packaged version of the same four capabilities |
# | Scoring | Measure against a rubric - longer output is not better output |
# | Call limits, depth guards | Non-negotiable on anything long-horizon |
# | When not to | Most tasks are a chain, an agent, or a small graph |
#
# ## Next
#
# -> [50a_deep_agent_backends_and_skills.ipynb](50a_deep_agent_backends_and_skills.ipynb)
