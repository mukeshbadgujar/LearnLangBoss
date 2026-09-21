# %% [markdown]
# # 32 - Conditional Routing
#
# | | |
# |---|---|
# | **Level** | Beginner (LangGraph) |
# | **Time** | 50 minutes |
# | **Prerequisites** | `31_stategraph_nodes_edges` |
# | **Checklist ID** | `32_conditional_routing` |
#
# ## Why this matters
#
# Unconditional edges give you a pipeline. **Conditional edges** give you a
# program: branches, loops, early exits and retry limits.
#
# This is also where you make the most important architectural choice in any
# LangGraph app: which decisions are made by **code** and which by the **model**.
# Get that boundary wrong and you either build something rigid or something
# unpredictable.

# %%
from pathlib import Path
import sys

_root = Path.cwd()
while not (_root / "shared").is_dir() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from shared.notebook_setup import setup, require  # noqa: E402

ctx = setup("32_conditional_routing")

# %%
import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from shared.llm import get_chat_model

model = get_chat_model()
parser = StrOutputParser()

# %% [markdown]
# ## 1. `add_conditional_edges`
#
# ```python
# builder.add_conditional_edges(source, path_function, path_map)
# ```
#
# The path function takes state and returns a **key**; `path_map` turns that key
# into a node name. The function is not a node - it produces no state update.

# %%
class TicketState(TypedDict):
    ticket: str
    category: str
    priority: str
    reply: str
    trace: Annotated[list[str], operator.add]


class Triage(BaseModel):
    """Classification of an inbound support ticket."""

    category: Literal["billing", "technical", "security", "howto"]
    priority: Literal["low", "medium", "high", "critical"]


triage_model = model.with_structured_output(Triage)


def triage_node(state: TicketState) -> dict:
    result = triage_model.invoke([("system", "Triage this Northwind support ticket."), ("human", state["ticket"])])
    return {"category": result.category, "priority": result.priority, "trace": ["triage"]}


def specialist(name: str, rules: str):
    def node(state: TicketState) -> dict:
        reply = (ChatPromptTemplate.from_messages([
            ("system", f"You are a {name} specialist. {rules} Answer in 3 sentences."),
            ("human", "{ticket}"),
        ]) | model | parser).invoke(state)
        return {"reply": reply, "trace": [name]}
    return node


# --- the routing function: plain Python, no LLM -------------------------- #
def route_by_category(state: TicketState) -> str:
    """Return a key that `path_map` resolves to a node name."""
    return state["category"]


builder = StateGraph(TicketState)
builder.add_node("triage", triage_node)
builder.add_node("billing_desk", specialist("billing", "Quote policy exactly. Never promise a refund."))
builder.add_node("technical_desk", specialist("support engineering", "Ask for symptoms and give diagnostics."))
builder.add_node("security_desk", specialist("security", "Never confirm anything unverified. Escalate contracts."))
builder.add_node("howto_desk", specialist("product guide", "Give short numbered steps."))

builder.add_edge(START, "triage")
builder.add_conditional_edges(
    "triage",
    route_by_category,
    {
        "billing": "billing_desk",
        "technical": "technical_desk",
        "security": "security_desk",
        "howto": "howto_desk",
    },
)
for desk in ("billing_desk", "technical_desk", "security_desk", "howto_desk"):
    builder.add_edge(desk, END)

router = builder.compile()
print(router.get_graph().draw_ascii())

# %%
TICKETS = [
    "We were billed twice for our 240 January seats.",
    "Dashboards have been timing out since this morning.",
    "Where is our data stored, and do you have SOC 2?",
    "How do I invite a read-only user to the workspace?",
]

for ticket in TICKETS:
    outcome = router.invoke({"ticket": ticket, "category": "", "priority": "", "reply": "", "trace": []})
    print(f"[{outcome['category']:9} {outcome['priority']:8}] {ticket[:48]}")
    print(f"    {outcome['reply'].strip()[:130]}\n")

# %% [markdown]
# ## 2. Routing to `END` - early exit
#
# `END` is a valid target. Use it to stop work that should not continue.

# %%
class GuardedState(TypedDict):
    ticket: str
    is_spam: bool
    reply: str
    trace: Annotated[list[str], operator.add]


SPAM_MARKERS = ["crypto", "seo services", "unsubscribe", "click here to claim"]


def spam_check(state: GuardedState) -> dict:
    text = state["ticket"].lower()
    return {"is_spam": any(m in text for m in SPAM_MARKERS), "trace": ["spam_check"]}


def route_spam(state: GuardedState) -> Literal["handle", "__end__"]:
    return END if state["is_spam"] else "handle"


builder = StateGraph(GuardedState)
builder.add_node("spam_check", spam_check)
builder.add_node("handle", lambda s: {"reply": "Thanks, we are on it.", "trace": ["handle"]})
builder.add_edge(START, "spam_check")
builder.add_conditional_edges("spam_check", route_spam, {"handle": "handle", END: END})
builder.add_edge("handle", END)
guarded = builder.compile()

for ticket in ["Our dashboards are down.", "Boost your SEO services - click here to claim!"]:
    outcome = guarded.invoke({"ticket": ticket, "is_spam": False, "reply": "", "trace": []})
    print(f"spam={outcome['is_spam']!s:5} trace={outcome['trace']} reply={outcome['reply']!r}")

# %% [markdown]
# Note the `Literal` return annotation on `route_spam`. LangGraph reads it to
# draw the diagram correctly even without a `path_map`, and it documents the
# function for the next reader.

# %% [markdown]
# ## 3. Loops with a counter
#
# The classic pattern: draft, critique, revise - until good enough or out of
# attempts. **Every loop needs both exits.**

# %%
class DraftState(TypedDict):
    request: str
    draft: str
    critique: str
    score: int
    revisions: Annotated[int, operator.add]
    trace: Annotated[list[str], operator.add]


class Critique(BaseModel):
    """A reviewer's assessment of a draft reply."""

    score: int = Field(description="Quality from 1 to 10", ge=1, le=10)
    issue: str = Field(description="The single most important thing to fix")


critic_model = model.with_structured_output(Critique)

MAX_REVISIONS = 3
GOOD_ENOUGH = 8


def write(state: DraftState) -> dict:
    prompt = (
        "Write a 3-sentence customer reply for: {request}"
        if not state["draft"]
        else "Rewrite this reply, fixing the issue.\n\nRequest: {request}\nDraft: {draft}\nIssue: {critique}"
    )
    draft = (ChatPromptTemplate.from_template(prompt) | model | parser).invoke(state)
    return {"draft": draft, "revisions": 1, "trace": [f"write#{state['revisions'] + 1}"]}


def critique(state: DraftState) -> dict:
    result = critic_model.invoke([
        ("system", "You review customer replies. Be demanding: empathy, specificity, no over-promising."),
        ("human", f"Request: {state['request']}\n\nReply: {state['draft']}"),
    ])
    return {"score": result.score, "critique": result.issue, "trace": [f"critique(score={result.score})"]}


def should_continue(state: DraftState) -> Literal["write", "__end__"]:
    if state["score"] >= GOOD_ENOUGH:
        return END
    if state["revisions"] >= MAX_REVISIONS:
        return END
    return "write"


builder = StateGraph(DraftState)
builder.add_node("write", write)
builder.add_node("critique", critique)
builder.add_edge(START, "write")
builder.add_edge("write", "critique")
builder.add_conditional_edges("critique", should_continue, {"write": "write", END: END})

loop = builder.compile()
print(loop.get_graph().draw_ascii())

# %%
outcome = loop.invoke({
    "request": "Customer was double-charged $18,400 and finance is closing the month.",
    "draft": "", "critique": "", "score": 0, "revisions": 0, "trace": [],
})
print("trace:", outcome["trace"])
print(f"final score {outcome['score']}/10 after {outcome['revisions']} revision(s)\n")
print(outcome["draft"].strip()[:320])

# %% [markdown]
# Two exit conditions: **quality met** and **budget exhausted**. Ship a loop with
# only the quality exit and you will eventually meet a model that never reaches
# the bar, and pay for it 25 times before `recursion_limit` saves you.

# %% [markdown]
# ## 4. Code decisions vs model decisions
#
# Routing functions can call an LLM. Usually they should not.

# %%
# Deterministic: free, instant, testable, auditable.
def route_by_priority(state: TicketState) -> str:
    if state["priority"] == "critical":
        return "page_oncall"
    if state["priority"] == "high":
        return "senior_queue"
    return "standard_queue"


# Model-driven: flexible, but costs a call and cannot be unit-tested the same way.
class Destination(BaseModel):
    """Where this ticket should go next."""

    next_step: Literal["refund", "investigate", "escalate", "close"]
    reason: str


decider = model.with_structured_output(Destination)


def route_by_judgement(state: TicketState) -> str:
    decision = decider.invoke([
        ("system", "Decide the next step for this ticket."),
        ("human", state["ticket"]),
    ])
    return decision.next_step


print("deterministic:", route_by_priority({"ticket": "", "category": "", "priority": "critical", "reply": "", "trace": []}))
print("model-driven :", route_by_judgement({"ticket": "We were double charged and want it back today.",
                                            "category": "", "priority": "", "reply": "", "trace": []}))

# %% [markdown]
# **The rule: if the decision can be written as an `if`, write it as an `if`.**
#
# | Decide in code | Decide with the model |
# |---|---|
# | Priority thresholds, retry counts, budgets | Intent and category of free text |
# | Anything a compliance auditor must verify | Whether a draft is good enough |
# | Anything with a legal or money consequence | Which of several research paths looks promising |
#
# A frequent good pattern is: **model classifies into a state field, code routes
# on that field.** That is exactly the `triage_node` + `route_by_category` split
# in section 1 - the LLM's judgement is captured in state where you can log it,
# test it and override it.

# %% [markdown]
# ## 5. `Command` - update state and route in one step
#
# Sometimes splitting "decide" from "update" is artificial. `Command` lets a node
# return both.

# %%
from langgraph.types import Command


class EscalationState(TypedDict):
    ticket: str
    priority: str
    assignee: str
    reply: str
    trace: Annotated[list[str], operator.add]


def assess(state: EscalationState) -> Command[Literal["oncall", "queue"]]:
    """Decide *and* write state in a single node."""
    urgent = any(word in state["ticket"].lower() for word in ["outage", "down", "data loss", "breach"])
    if urgent:
        return Command(update={"priority": "critical", "assignee": "oncall-rotation", "trace": ["assess:urgent"]},
                       goto="oncall")
    return Command(update={"priority": "normal", "assignee": "support-queue", "trace": ["assess:normal"]},
                   goto="queue")


builder = StateGraph(EscalationState)
builder.add_node("assess", assess)
builder.add_node("oncall", lambda s: {"reply": "Paging the on-call engineer now.", "trace": ["oncall"]})
builder.add_node("queue", lambda s: {"reply": "Queued for the next available agent.", "trace": ["queue"]})
builder.add_edge(START, "assess")
builder.add_edge("oncall", END)
builder.add_edge("queue", END)

command_graph = builder.compile()
for ticket in ["The whole platform is down.", "How do I export a report?"]:
    outcome = command_graph.invoke({"ticket": ticket, "priority": "", "assignee": "", "reply": "", "trace": []})
    print(f"{outcome['priority']:8} -> {outcome['assignee']:18} {outcome['trace']}")

# %% [markdown]
# `Command[Literal["oncall", "queue"]]` declares the possible destinations so the
# diagram stays accurate. Without the annotation LangGraph cannot draw the edges
# (pass `destinations=` to `add_node` as an alternative).
#
# | Use | When |
# |---|---|
# | `add_conditional_edges` | Routing logic is separate from the work; easiest to test |
# | `Command` | The node already computed what it needs to decide; avoids a redundant pass |

# %% [markdown]
# ## 6. Multiple destinations at once
#
# A path function may return a **list** of keys - all those nodes run in parallel.

# %%
class ReviewState(TypedDict):
    document: str
    checks: Annotated[list[str], operator.add]


def pick_checks(state: ReviewState) -> list[str]:
    """Run only the checks this document actually needs."""
    text = state["document"].lower()
    needed = ["grammar"]
    if any(w in text for w in ["$", "refund", "invoice", "price"]):
        needed.append("financial")
    if any(w in text for w in ["gdpr", "data", "personal", "consent"]):
        needed.append("legal")
    return needed


builder = StateGraph(ReviewState)
builder.add_node("grammar", lambda s: {"checks": ["grammar: ok"]})
builder.add_node("financial", lambda s: {"checks": ["financial: amounts verified"]})
builder.add_node("legal", lambda s: {"checks": ["legal: GDPR wording reviewed"]})
builder.add_node("done", lambda s: {"checks": [f"completed {len(s['checks'])} checks"]})
builder.add_conditional_edges(START, pick_checks, ["grammar", "financial", "legal"])
for check in ("grammar", "financial", "legal"):
    builder.add_edge(check, "done")
builder.add_edge("done", END)

checker = builder.compile()
for document in [
    "We will refund the $18,400 duplicate invoice today.",
    "Your data is processed under GDPR with documented consent.",
    "Thanks for reaching out, we will take a look.",
]:
    print(f"{checker.invoke({'document': document, 'checks': []})['checks']}")

# %% [markdown]
# Only the relevant checks ran. Note that `done` waited for whichever set fired -
# fan-in works the same regardless of how many branches were chosen.

# %% [markdown]
# ## 7. Putting it together: the full ticket router
#
# Triage, spam guard, specialist routing, quality loop, and escalation.

# %%
class FullState(TypedDict):
    ticket: str
    is_spam: bool
    category: str
    priority: str
    reply: str
    score: int
    revisions: Annotated[int, operator.add]
    trace: Annotated[list[str], operator.add]


def spam_guard(state: FullState) -> dict:
    return {"is_spam": any(m in state["ticket"].lower() for m in SPAM_MARKERS), "trace": ["spam_guard"]}


def full_triage(state: FullState) -> dict:
    result = triage_model.invoke([("system", "Triage this Northwind support ticket."), ("human", state["ticket"])])
    return {"category": result.category, "priority": result.priority, "trace": [f"triage:{result.category}"]}


def full_draft(state: FullState) -> dict:
    base = ("Write a 3-sentence reply as a {category} specialist. Priority {priority}. "
            "Never promise a refund or a fix date.\n\nTicket: {ticket}")
    if state["reply"]:
        base += "\n\nPrevious attempt: {reply}\nFix this: {trace}"
    reply = (ChatPromptTemplate.from_template(base) | model | parser).invoke(
        {**state, "trace": state["trace"][-1]}
    )
    return {"reply": reply, "revisions": 1, "trace": ["draft"]}


def full_review(state: FullState) -> dict:
    result = critic_model.invoke([
        ("system", "Review this support reply. Be demanding."),
        ("human", f"Ticket: {state['ticket']}\n\nReply: {state['reply']}"),
    ])
    return {"score": result.score, "trace": [f"review:{result.score}/10 - {result.issue}"]}


def after_spam(state: FullState) -> Literal["triage", "__end__"]:
    return END if state["is_spam"] else "triage"


def after_triage(state: FullState) -> Literal["escalate", "draft"]:
    return "escalate" if state["priority"] == "critical" else "draft"


def after_review(state: FullState) -> Literal["draft", "__end__"]:
    if state["score"] >= GOOD_ENOUGH or state["revisions"] >= MAX_REVISIONS:
        return END
    return "draft"


builder = StateGraph(FullState)
builder.add_node("spam_guard", spam_guard)
builder.add_node("triage", full_triage)
builder.add_node("escalate", lambda s: {"reply": "Escalated to the on-call engineer.", "trace": ["escalate"]})
builder.add_node("draft", full_draft)
builder.add_node("review", full_review)

builder.add_edge(START, "spam_guard")
builder.add_conditional_edges("spam_guard", after_spam, {"triage": "triage", END: END})
builder.add_conditional_edges("triage", after_triage, {"escalate": "escalate", "draft": "draft"})
builder.add_edge("escalate", END)
builder.add_edge("draft", "review")
builder.add_conditional_edges("review", after_review, {"draft": "draft", END: END})

full_router = builder.compile()
print(full_router.get_graph().draw_ascii())

# %%
for ticket in [
    "Boost your SEO services - click here to claim your free audit!",
    "The entire platform is down and we cannot process orders.",
    "How do I invite a read-only user?",
]:
    outcome = full_router.invoke({
        "ticket": ticket, "is_spam": False, "category": "", "priority": "",
        "reply": "", "score": 0, "revisions": 0, "trace": [],
    })
    print(f"\n{ticket[:56]}")
    for step in outcome["trace"]:
        print(f"    {step}")
    print(f"    -> {outcome['reply'].strip()[:120]}")

# %% [markdown]
# Three different paths through the same graph, each visible in `trace`. That
# observability is free once your decisions live in state.

# %% [markdown]
# ## 8. Debugging routing
#
# | Symptom | Likely cause |
# |---|---|
# | Wrong branch taken | Path function returned a key not in `path_map` - print the key |
# | `KeyError` in routing | Path function returned something `path_map` does not cover |
# | Loop never ends | Missing budget exit; add a counter |
# | `GraphRecursionError` | Same, and the safety net caught it |
# | Diagram missing edges | Add a `Literal` return type or a `path_map` |
# | Branch runs when it should not | Path function reading a state key written *later* in the same superstep |

# %%
def traced_route(state: FullState) -> str:
    """Wrap a router in logging while you debug it."""
    decision = after_triage(state)
    print(f"   [route] priority={state['priority']!r} -> {decision}")
    return decision


print(traced_route({"ticket": "", "is_spam": False, "category": "billing", "priority": "critical",
                    "reply": "", "score": 0, "revisions": 0, "trace": []}))

# %% [markdown]
# ## Try it yourself
#
# 1. **Add a `needs_human` branch** that routes to `END` with a flag when the
#    ticket mentions legal action - and check it runs before drafting.
# 2. **Convert `triage` to a `Command` node** that both writes the category and
#    jumps to the right desk, and compare the diagrams.
# 3. **Make the loop adaptive**: allow 5 revisions for critical tickets and 2 for
#    low priority, all in the path function.
# 4. **Break it on purpose.** Return an unmapped key from a path function and
#    read the error carefully - you will meet it again.

# %% [markdown]
# ## Recap
#
# | Concept | Takeaway |
# |---|---|
# | `add_conditional_edges` | source, path function, `path_map`; the function is not a node |
# | `Literal` return type | Lets LangGraph draw the diagram without a `path_map` |
# | Routing to `END` | Early exit for spam, refusals, completed work |
# | Loops | Always two exits: quality met **and** budget exhausted |
# | Code vs model decisions | If it can be an `if`, make it an `if` |
# | Classify-then-route | Model writes a state field; code routes on it - testable and loggable |
# | `Command(update=, goto=)` | Update state and choose the next node together |
# | List of destinations | Path function returning a list fans out in parallel |
# | Debugging | Print the routing key; most bugs are an unmapped return value |
#
# ## Next
#
# -> [33_compile_invoke_stream.ipynb](../06-langgraph-intermediate/33_compile_invoke_stream.ipynb)
