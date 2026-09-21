"""Second LangGraph probe: exact signatures for the features notebooks 33-50 use."""

from __future__ import annotations

import inspect
from importlib.metadata import version as pkg_version

for dist in ["langgraph", "langgraph-checkpoint", "langgraph-checkpoint-sqlite", "langgraph-prebuilt", "langgraph-sdk"]:
    try:
        print(f"{dist:32} {pkg_version(dist)}")
    except Exception:
        print(f"{dist:32} not installed")

print("\n=== SqliteSaver ===")
from langgraph.checkpoint.sqlite import SqliteSaver

print("  from_conn_string:", hasattr(SqliteSaver, "from_conn_string"),
      inspect.signature(SqliteSaver.from_conn_string) if hasattr(SqliteSaver, "from_conn_string") else "")
print("  methods:", [m for m in dir(SqliteSaver) if not m.startswith("_")][:20])

print("\n=== InMemoryStore ===")
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()
print("  put:", inspect.signature(store.put))
print("  search:", inspect.signature(store.search))
print("  get:", inspect.signature(store.get))
store.put(("users", "u1"), "pref1", {"text": "prefers concise answers"})
print("  search result:", store.search(("users", "u1")))

print("\n=== RetryPolicy / CachePolicy fields ===")
from langgraph.types import CachePolicy, RetryPolicy

print("  RetryPolicy:", inspect.signature(RetryPolicy))
print("  CachePolicy:", inspect.signature(CachePolicy))

print("\n=== add_node kwargs ===")
from langgraph.graph import StateGraph

print(" ", inspect.signature(StateGraph.add_node))

print("\n=== stream modes ===")
try:
    from langgraph.types import StreamMode

    print("  StreamMode:", StreamMode)
except Exception as exc:
    print("  ", exc)

print("\n=== full interrupt/resume roundtrip ===")
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class S(TypedDict):
    draft: str
    approved: bool


def write(state: S):
    return {"draft": "Dear customer, we will refund you."}


def review(state: S):
    decision = interrupt({"draft": state["draft"], "question": "Approve?"})
    return {"approved": decision == "approve"}


g = StateGraph(S)
g.add_node("write", write)
g.add_node("review", review)
g.add_edge(START, "write")
g.add_edge("write", "review")
g.add_edge("review", END)
app = g.compile(checkpointer=InMemorySaver())

cfg = {"configurable": {"thread_id": "t1"}}
first = app.invoke({"draft": "", "approved": False}, cfg)
print("  first:", first)
print("  __interrupt__ present:", "__interrupt__" in first)
snap = app.get_state(cfg)
print("  next:", snap.next, "| interrupts:", snap.tasks[0].interrupts if snap.tasks else None)
resumed = app.invoke(Command(resume="approve"), cfg)
print("  resumed:", resumed)
print("  history len:", len(list(app.get_state_history(cfg))))

print("\n=== custom stream writer ===")
from langgraph.config import get_stream_writer


class S2(TypedDict):
    n: int


def worker(state: S2):
    writer = get_stream_writer()
    writer({"progress": "halfway"})
    return {"n": state["n"] + 1}


g2 = StateGraph(S2)
g2.add_node("worker", worker)
g2.add_edge(START, "worker")
g2.add_edge("worker", END)
app2 = g2.compile()
print("  custom chunks:", list(app2.stream({"n": 0}, stream_mode="custom")))
print("  multi-mode:", list(app2.stream({"n": 0}, stream_mode=["updates", "custom"])))

print("\n=== Send / map-reduce ===")
import operator
from typing import Annotated

from langgraph.types import Send


class S3(TypedDict):
    topics: list[str]
    notes: Annotated[list[str], operator.add]


def fan_out(state: S3):
    return [Send("research", {"topic": t}) for t in state["topics"]]


def research(state: dict):
    return {"notes": [f"note on {state['topic']}"]}


g3 = StateGraph(S3)
g3.add_node("research", research)
g3.add_conditional_edges(START, fan_out, ["research"])
g3.add_edge("research", END)
app3 = g3.compile()
print("  ", app3.invoke({"topics": ["a", "b", "c"], "notes": []}))

print("\n=== subgraph ===")
class Inner(TypedDict):
    x: int


inner = StateGraph(Inner)
inner.add_node("double", lambda s: {"x": s["x"] * 2})
inner.add_edge(START, "double")
inner.add_edge("double", END)
inner_app = inner.compile()

outer = StateGraph(Inner)
outer.add_node("sub", inner_app)
outer.add_node("inc", lambda s: {"x": s["x"] + 1})
outer.add_edge(START, "sub")
outer.add_edge("sub", "inc")
outer.add_edge("inc", END)
outer_app = outer.compile()
print("  ", outer_app.invoke({"x": 3}))
print("  subgraph stream:", list(outer_app.stream({"x": 3}, stream_mode="updates", subgraphs=True)))

print("\n=== middleware signatures ===")
import langchain.agents.middleware as mw

for name in [
    "HumanInTheLoopMiddleware", "TodoListMiddleware", "SummarizationMiddleware",
    "ModelCallLimitMiddleware", "ToolCallLimitMiddleware", "ToolRetryMiddleware",
    "ModelFallbackMiddleware", "PIIMiddleware", "ContextEditingMiddleware",
    "LLMToolSelectorMiddleware", "FilesystemFileSearchMiddleware", "ShellToolMiddleware",
    "InterruptOnConfig", "AgentMiddleware",
]:
    obj = getattr(mw, name, None)
    if obj is None:
        print(f"  X  {name}")
        continue
    try:
        print(f"  ok {name}{str(inspect.signature(obj))[:160]}")
    except (TypeError, ValueError):
        print(f"  ok {name} (no signature)")

print("\n=== AgentMiddleware hooks ===")
print("  ", [m for m in dir(mw.AgentMiddleware) if not m.startswith("_")])

print("\n=== langchain.agents exports ===")
import langchain.agents as la

print("  ", sorted(n for n in dir(la) if n[0].isupper() or n.startswith("create")))
