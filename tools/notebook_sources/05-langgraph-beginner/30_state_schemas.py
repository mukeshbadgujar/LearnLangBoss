# %% [markdown]
# # 30 - State Schemas and Reducers
#
# | | |
# |---|---|
# | **Level** | Beginner (LangGraph) |
# | **Time** | 45 minutes |
# | **Prerequisites** | `29_graphs_vs_agents` |
# | **Checklist ID** | `30_state_schemas` |
#
# ## Why this matters
#
# State is the single most important design decision in a LangGraph application.
# Get it right and nodes stay small and independent. Get it wrong and you spend
# your time debugging why one node silently erased another node's work.
#
# The specific thing to understand is the **reducer**: the rule that decides how
# a node's return value merges into the existing state. Default is "overwrite".
# That default causes most beginner bugs.

# %%
from pathlib import Path
import sys

_root = Path.cwd()
while not (_root / "shared").is_dir() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from shared.notebook_setup import setup, require  # noqa: E402

ctx = setup("30_state_schemas")

# %%
from langgraph.graph import END, START, StateGraph

from shared.llm import get_chat_model

model = get_chat_model()

# %% [markdown]
# ## 1. `TypedDict` - the default choice

# %%
from typing import TypedDict


class BasicState(TypedDict):
    ticket: str
    category: str
    confidence: float


def classify(state: BasicState) -> dict:
    return {"category": "billing", "confidence": 0.92}


builder = StateGraph(BasicState)
builder.add_node("classify", classify)
builder.add_edge(START, "classify")
builder.add_edge("classify", END)
graph = builder.compile()

print(graph.invoke({"ticket": "Charged twice", "category": "", "confidence": 0.0}))

# %% [markdown]
# `TypedDict` is a **static** annotation - Python does not enforce it at runtime,
# and neither does LangGraph. Wrong types pass straight through.

# %%
print(graph.invoke({"ticket": "Charged twice", "category": "", "confidence": "not a number"}))
print("  ^ no error: TypedDict is a hint, not a validator")

# %% [markdown]
# ### Optional keys with `total=False`
#
# Useful when a field only exists after a particular node has run.

# %%
class PartialState(TypedDict, total=False):
    ticket: str          # every key is now optional
    category: str
    escalation_reason: str


builder = StateGraph(PartialState)
builder.add_node("classify", lambda s: {"category": "billing"})
builder.add_edge(START, "classify")
builder.add_edge("classify", END)

print(builder.compile().invoke({"ticket": "Charged twice"}))
print("  ^ 'escalation_reason' is simply absent, not None")

# %% [markdown]
# ## 2. Pydantic - validation at the boundary
#
# Use it when bad input must fail loudly, or when you want defaults.

# %%
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ValidatedState(BaseModel):
    ticket: str
    category: Literal["billing", "technical", "security", "unknown"] = "unknown"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("ticket")
    @classmethod
    def ticket_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("ticket must not be empty")
        return value


builder = StateGraph(ValidatedState)
builder.add_node("classify", lambda s: {"category": "billing", "confidence": 0.92})
builder.add_edge(START, "classify")
builder.add_edge("classify", END)
validated_graph = builder.compile()

print(validated_graph.invoke({"ticket": "Charged twice"}))

# %%
for bad_input, why in [
    ({"ticket": ""}, "empty ticket"),
    ({"ticket": "x", "confidence": 1.5}, "confidence out of range"),
    ({"ticket": "x", "category": "invoicing"}, "category not in the Literal"),
]:
    try:
        validated_graph.invoke(bad_input)
        print(f"  accepted (unexpected): {why}")
    except Exception as exc:
        print(f"  rejected  {why:32} {type(exc).__name__}")

# %% [markdown]
# ### Exactly when validation runs
#
# This detail catches people out: the model is constructed **when state is built
# as input for a node**. So a node that writes a bad value is caught by the
# *next* node - and not at all if nothing runs after it.

# %%
class Scored(BaseModel):
    ticket: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


def writes_bad_value(state: Scored) -> dict:
    return {"confidence": 5.0}       # out of range


# Last node in the graph: the bad write is never validated.
builder = StateGraph(Scored)
builder.add_node("bad", writes_bad_value)
builder.add_edge(START, "bad")
builder.add_edge("bad", END)
print("no downstream node:", builder.compile().invoke({"ticket": "x"}))

# Add a node after it and the same write is rejected.
builder = StateGraph(Scored)
builder.add_node("bad", writes_bad_value)
builder.add_node("next", lambda s: {})
builder.add_edge(START, "bad")
builder.add_edge("bad", "next")
builder.add_edge("next", END)
try:
    builder.compile().invoke({"ticket": "x"})
except Exception as exc:
    print(f"with a downstream node: {type(exc).__name__} - {str(exc).splitlines()[1].strip()}")

# %% [markdown]
# Also note that `invoke` returns a **plain dict**, not a model instance, even
# with a Pydantic schema. Validation is on the way in, not on the way out.
#
# | | `TypedDict` | Pydantic |
# |---|---|---|
# | Runtime validation | No | On input to each node |
# | Defaults | No | Yes |
# | Overhead | None | Small, per node |
# | Returns | dict | dict (not a model instance) |
# | Best for | Internal graphs, prototypes | Untrusted input, public APIs |
#
# `TypedDict` is the right default. Reach for Pydantic at the trust boundary.

# %% [markdown]
# ## 3. The overwrite bug
#
# Here is the mistake everyone makes once.

# %%
class NotesState(TypedDict):
    topic: str
    notes: list[str]


def researcher_a(state: NotesState) -> dict:
    return {"notes": ["A: pricing data from the CRM"]}


def researcher_b(state: NotesState) -> dict:
    return {"notes": ["B: churn data from the warehouse"]}


builder = StateGraph(NotesState)
builder.add_node("a", researcher_a)
builder.add_node("b", researcher_b)
builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("b", END)

print(builder.compile().invoke({"topic": "renewals", "notes": []}))

# %% [markdown]
# Researcher A's work is gone. The default reducer is **"last write wins"**, so
# `b` replaced the list rather than adding to it.
#
# ## 4. Reducers with `Annotated`
#
# Attach a merge function to the field and the problem disappears.

# %%
import operator
from typing import Annotated


class AccumulatingState(TypedDict):
    topic: str
    notes: Annotated[list[str], operator.add]   # concatenate instead of replace


builder = StateGraph(AccumulatingState)
builder.add_node("a", researcher_a)
builder.add_node("b", researcher_b)
builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("b", END)

print(builder.compile().invoke({"topic": "renewals", "notes": []}))

# %% [markdown]
# `Annotated[T, reducer]` reads as: *the field has type `T`, and when a node
# returns a value for it, merge with `reducer(existing, new)`.*
#
# `operator.add` works for lists (concatenate), numbers (sum) and strings
# (concatenate), which covers a surprising amount of real state.

# %%
class CounterState(TypedDict):
    calls: Annotated[int, operator.add]
    log: Annotated[str, operator.add]


builder = StateGraph(CounterState)
builder.add_node("step1", lambda s: {"calls": 1, "log": "step1;"})
builder.add_node("step2", lambda s: {"calls": 1, "log": "step2;"})
builder.add_edge(START, "step1")
builder.add_edge("step1", "step2")
builder.add_edge("step2", END)

print(builder.compile().invoke({"calls": 0, "log": ""}))

# %% [markdown]
# ### Custom reducers
#
# Any `(existing, new) -> merged` function works. This is where domain logic
# belongs - deduplication, ranking, capping.

# %%
def merge_unique(existing: list[str], new: list[str]) -> list[str]:
    """Append, preserving order, skipping duplicates."""
    seen = set(existing)
    return existing + [item for item in new if not (item in seen or seen.add(item))]


def keep_highest(existing: float, new: float) -> float:
    """Confidence only ever goes up."""
    return max(existing, new)


def merge_dict(existing: dict, new: dict) -> dict:
    """Shallow dict merge - new keys win."""
    return {**existing, **new}


def last_n(n: int):
    """Reducer factory: keep only the most recent n items (bounded memory)."""
    def reducer(existing: list, new: list) -> list:
        return (existing + new)[-n:]
    return reducer


class RichState(TypedDict):
    sources: Annotated[list[str], merge_unique]
    confidence: Annotated[float, keep_highest]
    metadata: Annotated[dict, merge_dict]
    recent_events: Annotated[list[str], last_n(3)]


builder = StateGraph(RichState)
builder.add_node("first", lambda s: {
    "sources": ["handbook.md", "faq.md"],
    "confidence": 0.6,
    "metadata": {"stage": "retrieval", "k": 4},
    "recent_events": ["retrieved"],
})
builder.add_node("second", lambda s: {
    "sources": ["faq.md", "leave_policy.txt"],     # faq.md is a duplicate
    "confidence": 0.4,                              # lower, so ignored
    "metadata": {"stage": "generation"},            # overrides 'stage', keeps 'k'
    "recent_events": ["reranked", "generated", "verified"],
})
builder.add_edge(START, "first")
builder.add_edge("first", "second")
builder.add_edge("second", END)

outcome = builder.compile().invoke(
    {"sources": [], "confidence": 0.0, "metadata": {}, "recent_events": []}
)
for key, value in outcome.items():
    print(f"  {key:14} {value}")

# %% [markdown]
# Note `recent_events`: three nodes wrote five events and the state holds the
# last three. Bounded state is how you stop long-running graphs growing without
# limit.

# %% [markdown]
# ## 5. `add_messages` - the reducer you will use most
#
# Chat state needs more than concatenation: messages have IDs, and an update with
# an existing ID should **replace** rather than append. `add_messages` does that.

# %%
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.graph import add_messages


class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


existing = [HumanMessage("What is the leave policy?", id="m1"), AIMessage("24 days annually.", id="m2")]

print("append:")
for m in add_messages(existing, [HumanMessage("And sick leave?", id="m3")]):
    print(f"   {m.id} {m.__class__.__name__:13} {m.content}")

print("\nreplace by id (m2 corrected):")
for m in add_messages(existing, [AIMessage("24 days annually, plus 6 casual.", id="m2")]):
    print(f"   {m.id} {m.__class__.__name__:13} {m.content}")

print("\ndelete by id:")
for m in add_messages(existing, [RemoveMessage(id="m1")]):
    print(f"   {m.id} {m.__class__.__name__:13} {m.content}")

# %% [markdown]
# Replace-by-id is what makes streaming token updates and message editing work.
# Deletion via `RemoveMessage` is covered properly in notebook 37.
#
# ### `MessagesState` - the shortcut

# %%
from langgraph.graph import MessagesState

print("MessagesState is just:", MessagesState.__annotations__)


class SupportState(MessagesState):
    """Subclass to add your own fields alongside `messages`."""

    category: str
    escalated: bool


def respond(state: SupportState) -> dict:
    reply = model.invoke(state["messages"])
    return {"messages": [reply], "category": "billing"}


builder = StateGraph(SupportState)
builder.add_node("respond", respond)
builder.add_edge(START, "respond")
builder.add_edge("respond", END)

outcome = builder.compile().invoke(
    {"messages": [HumanMessage("In one sentence: what is a vector database?")], "category": "", "escalated": False}
)
print(f"\ncategory={outcome['category']}  messages={len(outcome['messages'])}")
print(outcome["messages"][-1].content.strip()[:180])

# %% [markdown]
# ## 6. Reducers are what make parallelism safe
#
# Without a reducer, two nodes running in parallel and writing the same key is an
# error - LangGraph cannot know which write should win.

# %%
class ParallelUnsafe(TypedDict):
    findings: list[str]


builder = StateGraph(ParallelUnsafe)
builder.add_node("pricing", lambda s: {"findings": ["pricing looks competitive"]})
builder.add_node("churn", lambda s: {"findings": ["churn up 3% QoQ"]})
builder.add_edge(START, "pricing")
builder.add_edge(START, "churn")          # both run in the same step
builder.add_edge("pricing", END)
builder.add_edge("churn", END)

try:
    print(builder.compile().invoke({"findings": []}))
except Exception as exc:
    print(f"{type(exc).__name__}: {str(exc)[:200]}")

# %%
class ParallelSafe(TypedDict):
    findings: Annotated[list[str], operator.add]


builder = StateGraph(ParallelSafe)
builder.add_node("pricing", lambda s: {"findings": ["pricing looks competitive"]})
builder.add_node("churn", lambda s: {"findings": ["churn up 3% QoQ"]})
builder.add_edge(START, "pricing")
builder.add_edge(START, "churn")
builder.add_edge("pricing", END)
builder.add_edge("churn", END)

print(builder.compile().invoke({"findings": []}))

# %% [markdown]
# With a reducer, LangGraph has a rule for combining concurrent writes and the
# fan-out just works. Notebook 38 builds on exactly this.

# %% [markdown]
# ## 7. Input and output schemas
#
# Your internal state usually contains scratch fields callers should never see.
# `input_schema` and `output_schema` keep the public contract clean.

# %%
class Input(TypedDict):
    question: str


class Output(TypedDict):
    answer: str


class Internal(TypedDict):
    question: str
    retrieved: list[str]
    draft: str
    critique: str
    answer: str


builder = StateGraph(Internal, input_schema=Input, output_schema=Output)
builder.add_node("retrieve", lambda s: {"retrieved": ["24 days annual leave", "6 days casual leave"]})
builder.add_node("draft", lambda s: {"draft": f"Based on {len(s['retrieved'])} sources..."})
builder.add_node("finalise", lambda s: {"answer": s["draft"] + " you get 24 days of annual leave."})
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "draft")
builder.add_edge("draft", "finalise")
builder.add_edge("finalise", END)

scoped = builder.compile()
print(scoped.invoke({"question": "How much leave do I get?"}))
print("  ^ only 'answer' is returned; retrieved/draft/critique stayed internal")

# %% [markdown]
# ## 8. Designing state well
#
# 1. **Start minimal.** Add a field when a node actually needs it.
# 2. **Flat beats nested.** Reducers merge top-level keys; deep nesting fights that.
# 3. **A reducer on every list.** If two nodes might write it, it needs one.
# 4. **Bound anything that grows.** Use `last_n` or summarisation (notebook 47).
# 5. **Keep blobs out.** Store an ID or path, not a 40 MB PDF - state is
#    serialised into every checkpoint.
# 6. **Name for meaning, not mechanism.** `needs_human_review`, not `flag2`.

# %%
# Anti-pattern: everything in one nested blob - reducers cannot help you here.
class BadState(TypedDict):
    data: dict          # {"user": {...}, "docs": [...], "scratch": {...}}


# Better: flat, typed, each field with the right merge behaviour.
class GoodState(TypedDict):
    question: str
    user_id: str
    doc_ids: Annotated[list[str], merge_unique]
    findings: Annotated[list[str], operator.add]
    recent_events: Annotated[list[str], last_n(10)]
    needs_human_review: bool
    answer: str


print("fields:", list(GoodState.__annotations__))

# %% [markdown]
# ## Try it yourself
#
# 1. **Write a `merge_scores` reducer** for `dict[str, float]` that keeps the
#    maximum per key, and prove it with two parallel nodes.
# 2. **Build a bounded conversation.** Combine `add_messages` with a cap so the
#    state never holds more than 10 messages. What breaks if a tool call message
#    gets separated from its result?
# 3. **Swap `TypedDict` for Pydantic** on the ticket graph from notebook 29 and
#    make an invalid priority impossible.
# 4. **Reproduce the concurrent-write error**, then fix it three ways: a reducer,
#    sequencing the nodes, and giving each node its own key.

# %% [markdown]
# ## Recap
#
# | Concept | Takeaway |
# |---|---|
# | `TypedDict` | Default schema; no runtime validation |
# | Pydantic state | Validates on input to each node; a bad write is caught by the *next* node |
# | Default reducer | **Overwrite** - the source of most beginner bugs |
# | `Annotated[T, reducer]` | Attach a merge rule to a field |
# | `operator.add` | Concatenate lists/strings, sum numbers |
# | Custom reducers | Dedup, max, dict merge, bounded `last_n` |
# | `add_messages` | Append, replace-by-id, delete via `RemoveMessage` |
# | `MessagesState` | Subclass it to add your own fields |
# | Parallel writes | Require a reducer or the graph raises |
# | `input_schema` / `output_schema` | Hide internal scratch fields from callers |
#
# ## Next
#
# -> [31_stategraph_nodes_edges.ipynb](31_stategraph_nodes_edges.ipynb)
