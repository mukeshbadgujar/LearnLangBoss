# %% [markdown]
# # 31 - StateGraph: Nodes and Edges
#
# | | |
# |---|---|
# | **Level** | Beginner (LangGraph) |
# | **Time** | 45 minutes |
# | **Prerequisites** | `30_state_schemas` |
# | **Checklist ID** | `31_stategraph_nodes_edges` |
#
# ## Why this matters
#
# State is the data; nodes and edges are the program. This notebook is the
# mechanics: every way to add a node, every way to connect one, and the handful
# of rules that decide what runs when.
#
# We build the running example for Track 05 - a **support ticket router** - one
# node at a time.

# %%
from pathlib import Path
import sys

_root = Path.cwd()
while not (_root / "shared").is_dir() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from shared.notebook_setup import setup, require  # noqa: E402

ctx = setup("31_stategraph_nodes_edges")

# %%
import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

from shared.llm import get_chat_model

model = get_chat_model()
parser = StrOutputParser()

# %% [markdown]
# ## 1. The node contract
#
# A node is any callable that takes the state and returns a **partial update**
# (a dict of the keys it changed) or `None`.

# %%
class State(TypedDict):
    ticket: str
    category: str
    trace: Annotated[list[str], operator.add]


def plain_function(state: State) -> dict:
    """The usual form: a function taking state, returning a partial dict."""
    return {"category": "billing", "trace": ["plain_function"]}


def returns_nothing(state: State) -> None:
    """Valid: a node that only has side effects (logging, metrics)."""
    print(f"   [side effect] saw ticket: {state['ticket'][:40]}")


class CallableNode:
    """Also valid: anything callable, which is handy for dependency injection."""

    def __init__(self, label: str):
        self.label = label

    def __call__(self, state: State) -> dict:
        return {"trace": [f"CallableNode({self.label})"]}


builder = StateGraph(State)
builder.add_node("plain", plain_function)
builder.add_node("side_effect", returns_nothing)
builder.add_node("callable", CallableNode("configured"))
builder.add_edge(START, "plain")
builder.add_edge("plain", "side_effect")
builder.add_edge("side_effect", "callable")
builder.add_edge("callable", END)

print(builder.compile().invoke({"ticket": "Charged twice for January", "category": "", "trace": []}))

# %% [markdown]
# ### Naming
#
# `add_node(fn)` infers the name from the function. `add_node("name", fn)` sets
# it explicitly. **Use explicit names** - node names appear in traces, stream
# output and edge definitions, and renaming a function should not silently break
# an edge.

# %%
def classify_ticket(state: State) -> dict:
    return {"trace": ["inferred name"]}


builder = StateGraph(State)
builder.add_node(classify_ticket)                 # name becomes "classify_ticket"
builder.add_node("triage", classify_ticket)       # same function, explicit name
builder.add_edge(START, "classify_ticket")
builder.add_edge("classify_ticket", "triage")
builder.add_edge("triage", END)
print("nodes:", list(builder.compile().get_graph().nodes))

# %% [markdown]
# ### Lambdas and partials
#
# Fine for glue, but they cannot be named automatically and they are painful in
# a stack trace. Prefer a real function for anything with logic.

# %%
from functools import partial


def set_priority(state: State, *, level: str) -> dict:
    return {"trace": [f"priority={level}"]}


builder = StateGraph(State)
builder.add_node("reset", lambda s: {"category": ""})
builder.add_node("high", partial(set_priority, level="high"))
builder.add_edge(START, "reset")
builder.add_edge("reset", "high")
builder.add_edge("high", END)
print(builder.compile().invoke({"ticket": "x", "category": "y", "trace": []}))

# %% [markdown]
# ## 2. Putting an LLM in a node
#
# A node is ordinary Python, so an LLM call is just a function call. Two rules
# make LLM nodes behave: **constrain the output** and **never trust it blindly**.

# %%
from pydantic import BaseModel, Field


class Triage(BaseModel):
    """Classification of an inbound support ticket."""

    category: Literal["billing", "technical", "security", "howto"] = Field(description="Best-fitting desk")
    priority: Literal["low", "medium", "high", "critical"] = Field(description="Urgency")
    summary: str = Field(description="One line, under 80 characters")


class TicketState(TypedDict):
    ticket: str
    category: str
    priority: str
    summary: str
    reply: str
    trace: Annotated[list[str], operator.add]


triage_model = model.with_structured_output(Triage)


def triage_node(state: TicketState) -> dict:
    """LLM node with a guaranteed shape - no parsing, no repair."""
    result = triage_model.invoke(
        [("system", "Triage this Northwind Analytics support ticket."), ("human", state["ticket"])]
    )
    return {
        "category": result.category,
        "priority": result.priority,
        "summary": result.summary,
        "trace": ["triage"],
    }


def reply_node(state: TicketState) -> dict:
    """LLM node whose prompt is specialised by earlier state."""
    reply = (ChatPromptTemplate.from_messages([
        ("system", "You are a {category} specialist. Priority is {priority}. "
                   "Write 3 sentences. Never promise a refund or a fix date."),
        ("human", "{ticket}"),
    ]) | model | parser).invoke(state)
    return {"reply": reply, "trace": ["reply"]}


builder = StateGraph(TicketState)
builder.add_node("triage", triage_node)
builder.add_node("reply", reply_node)
builder.add_edge(START, "triage")
builder.add_edge("triage", "reply")
builder.add_edge("reply", END)
ticket_graph = builder.compile()

TICKET = ("Subject: Double charged\nWe were billed twice for our 240 January seats. "
          "Finance is closing the month and needs the duplicate refunded.")

outcome = ticket_graph.invoke({"ticket": TICKET, "category": "", "priority": "", "summary": "", "reply": "", "trace": []})
print(f"{outcome['category']} / {outcome['priority']}: {outcome['summary']}")
print()
print(outcome["reply"].strip()[:250])

# %% [markdown]
# ## 3. Tools in nodes
#
# Two options. A plain node calling a tool function is deterministic; `ToolNode`
# executes whatever tool calls the model requested.

# %%
from langchain_core.tools import tool


@tool
def lookup_invoice(invoice_id: str) -> str:
    """Look up an invoice by ID."""
    return f"{invoice_id}: $18,400, paid 2026-01-02, status=DUPLICATE_SUSPECTED"


@tool
def lookup_account(company: str) -> str:
    """Look up a customer's plan and open disputes."""
    return f"{company}: enterprise, 240 seats, 2 open disputes"


# Option A: deterministic - you decide the tool runs, and with what arguments.
def enrich_node(state: TicketState) -> dict:
    facts = lookup_account.invoke({"company": "Globex"})
    return {"trace": [f"enriched: {facts}"]}


# Option B: ToolNode - runs whatever the model asked for, in parallel if several.
from langgraph.prebuilt import ToolNode

tool_node = ToolNode([lookup_invoice, lookup_account])
print("ToolNode handles:", [t.name for t in [lookup_invoice, lookup_account]])

# %% [markdown]
# `ToolNode` reads the last message's `tool_calls`, executes them, and appends
# `ToolMessage`s. It only makes sense in a graph whose state has a `messages`
# key, which is why the agent-shaped graph below uses `MessagesState`.

# %%
from langgraph.graph import MessagesState
from langgraph.prebuilt import tools_condition

tools = [lookup_invoice, lookup_account]
model_with_tools = model.bind_tools(tools)


def call_model(state: MessagesState) -> dict:
    return {"messages": [model_with_tools.invoke(state["messages"])]}


builder = StateGraph(MessagesState)
builder.add_node("model", call_model)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "model")
builder.add_conditional_edges("model", tools_condition, {"tools": "tools", END: END})
builder.add_edge("tools", "model")          # the loop
agent_graph = builder.compile()

print(agent_graph.get_graph().draw_ascii())

# %%
result = agent_graph.invoke({"messages": [("user", "Check invoice INV-2026-0102 for Globex and tell me if it looks like a duplicate.")]})
for message in result["messages"]:
    kind = message.__class__.__name__.replace("Message", "")
    calls = getattr(message, "tool_calls", None)
    suffix = f"  -> {[c['name'] for c in calls]}" if calls else ""
    print(f"{kind:9} {(message.content or '')[:95]}{suffix}")

# %% [markdown]
# That is `create_agent` rebuilt from parts: a model node, a tool node, and a
# conditional edge that loops. Understanding this diagram is understanding every
# agent you will ever debug.

# %% [markdown]
# ## 4. Edges
#
# ### Normal edges - always
#
# `add_edge("a", "b")` means *after `a`, run `b`*. Unconditional.

# %%
builder = StateGraph(State)
builder.add_node("a", lambda s: {"trace": ["a"]})
builder.add_node("b", lambda s: {"trace": ["b"]})
builder.add_node("c", lambda s: {"trace": ["c"]})
builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("b", "c")
builder.add_edge("c", END)
print(builder.compile().invoke({"ticket": "", "category": "", "trace": []})["trace"])

# %% [markdown]
# ### `add_sequence` - the shortcut for straight lines

# %%
builder = StateGraph(State)
builder.add_sequence([
    ("a", lambda s: {"trace": ["a"]}),
    ("b", lambda s: {"trace": ["b"]}),
    ("c", lambda s: {"trace": ["c"]}),
])
builder.add_edge(START, "a")
builder.add_edge("c", END)
print(builder.compile().invoke({"ticket": "", "category": "", "trace": []})["trace"])

# %% [markdown]
# ### Fan-out and fan-in
#
# Multiple edges *from* one node run those targets **in parallel**. Multiple
# edges *into* one node make it wait for all of them.

# %%
class FanState(TypedDict):
    findings: Annotated[list[str], operator.add]
    report: str


def merge(state: FanState) -> dict:
    return {"report": f"Report combining {len(state['findings'])} findings."}


builder = StateGraph(FanState)
builder.add_node("pricing", lambda s: {"findings": ["pricing competitive"]})
builder.add_node("churn", lambda s: {"findings": ["churn up 3%"]})
builder.add_node("usage", lambda s: {"findings": ["DAU flat"]})
builder.add_node("merge", merge)
for branch in ("pricing", "churn", "usage"):
    builder.add_edge(START, branch)          # fan out
    builder.add_edge(branch, "merge")        # fan in
builder.add_edge("merge", END)

fan_graph = builder.compile()
print(fan_graph.get_graph().draw_ascii())
print(fan_graph.invoke({"findings": [], "report": ""}))

# %% [markdown]
# `merge` ran **once**, after all three finished. That is the superstep model:
# LangGraph runs everything ready in the current step, merges the writes, then
# moves on. Notebook 38 goes deeper.

# %% [markdown]
# ### Multiple entry points
#
# `START` can have several edges too - useful when initialisation work is
# independent of the main path.

# %%
builder = StateGraph(State)
builder.add_node("load_config", lambda s: {"trace": ["config"]})
builder.add_node("load_ticket", lambda s: {"trace": ["ticket"]})
builder.add_node("process", lambda s: {"trace": ["process"]})
builder.add_edge(START, "load_config")
builder.add_edge(START, "load_ticket")
builder.add_edge("load_config", "process")
builder.add_edge("load_ticket", "process")
builder.add_edge("process", END)
print(builder.compile().invoke({"ticket": "", "category": "", "trace": []})["trace"])

# %% [markdown]
# ## 5. Visualising
#
# Always draw the graph before debugging it. Half of all LangGraph bugs are
# visible in the diagram.

# %%
print(ticket_graph.get_graph().draw_ascii())

# %%
print(ticket_graph.get_graph().draw_mermaid())

# %% [markdown]
# Paste that Mermaid into any Markdown renderer, or in a notebook use:
#
# ```python
# from IPython.display import Image, display
# display(Image(graph.get_graph().draw_mermaid_png()))
# ```
#
# (`draw_mermaid_png` calls a remote renderer, so it needs network access.)

# %% [markdown]
# ## 6. Compile-time errors
#
# `compile()` validates the graph. These failures are cheap - they happen before
# a single token is spent.

# %%
# Dangling edge: target does not exist.
bad = StateGraph(State)
bad.add_node("a", lambda s: {"trace": ["a"]})
bad.add_edge(START, "a")
bad.add_edge("a", "nonexistent")
try:
    bad.compile()
except Exception as exc:
    print(f"1. {type(exc).__name__}: {str(exc)[:110]}")

# Unreachable node: nothing points at it.
bad = StateGraph(State)
bad.add_node("a", lambda s: {"trace": ["a"]})
bad.add_node("orphan", lambda s: {"trace": ["orphan"]})
bad.add_edge(START, "a")
bad.add_edge("a", END)
try:
    compiled = bad.compile()
    print(f"2. compiled; 'orphan' is simply never reached: {compiled.invoke({'ticket': '', 'category': '', 'trace': []})['trace']}")
except Exception as exc:
    print(f"2. {type(exc).__name__}: {str(exc)[:110]}")

# No entry point.
bad = StateGraph(State)
bad.add_node("a", lambda s: {"trace": ["a"]})
bad.add_edge("a", END)
try:
    bad.compile()
except Exception as exc:
    print(f"3. {type(exc).__name__}: {str(exc)[:110]}")

# %% [markdown]
# ## 7. Runtime behaviour worth recognising
#
# ### Unknown keys are dropped silently
#
# This one bites. A node returning a key that is not in the state schema does
# **not** raise - the value is discarded and the graph carries on.

# %%
class Strict(TypedDict):
    known: str


builder = StateGraph(Strict)
builder.add_node("typo", lambda s: {"knwon": "oops"})     # note the typo
builder.add_edge(START, "typo")
builder.add_edge("typo", END)
print(builder.compile().invoke({"known": "original"}))
print("  ^ 'knwon' vanished; 'known' was never updated, and nothing complained")

# %% [markdown]
# A typo in a returned key therefore looks like "my node did nothing". When a
# node's effect seems to disappear, check the spelling of the key against the
# schema first - it is the most common cause.
#
# A cheap defence for graphs you care about:

# %%
def checked(schema, fn):
    """Wrap a node so unknown keys fail loudly instead of vanishing."""
    allowed = set(schema.__annotations__)

    def wrapper(state):
        update = fn(state)
        if update:
            unknown = set(update) - allowed
            if unknown:
                raise KeyError(f"{fn.__name__} returned keys not in the schema: {sorted(unknown)}")
        return update

    wrapper.__name__ = fn.__name__
    return wrapper


def typo_node(state: Strict) -> dict:
    return {"knwon": "oops"}


builder = StateGraph(Strict)
builder.add_node("typo", checked(Strict, typo_node))
builder.add_edge(START, "typo")
builder.add_edge("typo", END)
try:
    builder.compile().invoke({"known": "original"})
except Exception as exc:
    print(f"{type(exc).__name__}: {exc}")

# %%
# An infinite loop, stopped by the recursion limit.
from langgraph.errors import GraphRecursionError


class LoopState(TypedDict):
    count: Annotated[int, operator.add]


builder = StateGraph(LoopState)
builder.add_node("forever", lambda s: {"count": 1})
builder.add_edge(START, "forever")
builder.add_edge("forever", "forever")
try:
    builder.compile().invoke({"count": 0}, {"recursion_limit": 8})
except GraphRecursionError as exc:
    print(f"GraphRecursionError: {str(exc)[:130]}")

# %% [markdown]
# `recursion_limit` (default 25) counts **supersteps**, not node executions. It
# is a safety net, not a design tool - a cyclic graph should have its own exit
# condition, which is what notebook 32 is about.

# %% [markdown]
# ## 8. The complete ticket router so far

# %%
class RouterState(TypedDict):
    ticket: str
    category: str
    priority: str
    summary: str
    account: str
    reply: str
    trace: Annotated[list[str], operator.add]


def enrich(state: RouterState) -> dict:
    return {"account": lookup_account.invoke({"company": "Globex"}), "trace": ["enrich"]}


def triage(state: RouterState) -> dict:
    result = triage_model.invoke([("system", "Triage this ticket."), ("human", state["ticket"])])
    return {"category": result.category, "priority": result.priority,
            "summary": result.summary, "trace": ["triage"]}


def draft(state: RouterState) -> dict:
    reply = (ChatPromptTemplate.from_messages([
        ("system", "You are a {category} specialist. Priority {priority}. Account context: {account}. "
                   "Write 3 sentences. Never promise a refund."),
        ("human", "{ticket}"),
    ]) | model | parser).invoke(state)
    return {"reply": reply, "trace": ["draft"]}


builder = StateGraph(RouterState)
builder.add_node("enrich", enrich)
builder.add_node("triage", triage)
builder.add_node("draft", draft)
builder.add_edge(START, "enrich")
builder.add_edge(START, "triage")            # enrich and triage run in parallel
builder.add_edge("enrich", "draft")
builder.add_edge("triage", "draft")          # draft waits for both
builder.add_edge("draft", END)

router = builder.compile()
print(router.get_graph().draw_ascii())

final = router.invoke({"ticket": TICKET, "category": "", "priority": "", "summary": "",
                       "account": "", "reply": "", "trace": []})
print(f"\n{final['category']}/{final['priority']} | trace={final['trace']}")
print(final["reply"].strip()[:240])

# %% [markdown]
# ## Try it yourself
#
# 1. **Add a `log_node`** that returns `None` and prints the category, and verify
#    it does not disturb state.
# 2. **Make `draft` fail** when `category == "security"`, then read the traceback
#    and note how the node name appears in it.
# 3. **Fan out to four** classifier nodes voting on the category, and write a
#    `merge` node that takes the majority. What reducer does the votes field need?
# 4. **Draw before you build.** Sketch a graph with an approval step and a
#    rejection loop, then implement it - notebook 32 gives you the missing piece.

# %% [markdown]
# ## Recap
#
# | Concept | Takeaway |
# |---|---|
# | Node contract | Takes state, returns a partial dict or `None` |
# | Explicit names | `add_node("name", fn)` - names appear in traces and edges |
# | LLM nodes | Constrain with `with_structured_output`; specialise the prompt from state |
# | `ToolNode` | Executes the model's requested tool calls; needs `messages` state |
# | `tools_condition` | Prebuilt router: tools if tool calls present, else `END` |
# | `add_edge` | Unconditional "then run" |
# | `add_sequence` | Shortcut for straight-line pipelines |
# | Fan-out / fan-in | Several edges out = parallel; several in = wait for all |
# | `draw_ascii` / `draw_mermaid` | Draw it before you debug it |
# | Compile errors | Dangling edges and missing entry points caught for free |
# | Unknown keys | **Silently dropped** - a typo looks like "my node did nothing" |
# | `recursion_limit` | Superstep safety net, not a design tool |
#
# ## Next
#
# -> [32_conditional_routing.ipynb](32_conditional_routing.ipynb)
