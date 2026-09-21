"""Probe the advanced LangGraph surface used by notebooks 39-50."""

import inspect
import operator
from typing import Annotated, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

print("=== cache backends ===")
for path in ["langgraph.cache.memory.InMemoryCache", "langgraph.cache.sqlite.SqliteCache",
             "langgraph.cache.base.BaseCache"]:
    module, _, attr = path.rpartition(".")
    try:
        mod = __import__(module, fromlist=[attr])
        obj = getattr(mod, attr)
        print(f"  ok {path:44} {str(inspect.signature(obj))[:70]}")
    except Exception as exc:
        print(f"  X  {path:44} {type(exc).__name__}: {exc}")

print("\n=== node cache_policy roundtrip ===")
try:
    from langgraph.cache.memory import InMemoryCache
    from langgraph.types import CachePolicy

    hits = []

    class C(TypedDict):
        x: int
        y: int

    def slow(state: C) -> dict:
        hits.append(1)
        return {"y": state["x"] * 2}

    b = StateGraph(C)
    b.add_node("slow", slow, cache_policy=CachePolicy(ttl=60))
    b.add_edge(START, "slow")
    b.add_edge("slow", END)
    g = b.compile(cache=InMemoryCache())
    print("  run1:", g.invoke({"x": 3, "y": 0}))
    print("  run2:", g.invoke({"x": 3, "y": 0}))
    print("  node executions:", len(hits), "(1 means the cache hit)")
except Exception as exc:
    import traceback
    traceback.print_exc()

print("\n=== time travel: fork from a past checkpoint ===")
try:
    class T(TypedDict):
        value: Annotated[list[str], operator.add]

    b = StateGraph(T)
    b.add_node("one", lambda s: {"value": ["one"]})
    b.add_node("two", lambda s: {"value": ["two"]})
    b.add_node("three", lambda s: {"value": ["three"]})
    b.add_sequence([])if False else None
    b.add_edge(START, "one")
    b.add_edge("one", "two")
    b.add_edge("two", "three")
    b.add_edge("three", END)
    g = b.compile(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "tt"}}
    print("  original:", g.invoke({"value": []}, cfg))
    hist = list(g.get_state_history(cfg))
    print("  checkpoints:", len(hist))
    for h in hist:
        print(f"    step={h.metadata.get('step')} next={h.next} value={h.values['value']}")
    # pick the checkpoint whose next is ('two',)
    target = next(h for h in hist if h.next == ("two",))
    replay_cfg = target.config
    print("  replaying from:", replay_cfg["configurable"]["checkpoint_id"][:8])
    print("  replay result:", g.invoke(None, replay_cfg))
    forked = g.update_state(target.config, {"value": ["EDITED"]})
    print("  fork config:", forked["configurable"]["checkpoint_id"][:8])
    print("  fork result:", g.invoke(None, forked))
    print("  history after fork:", len(list(g.get_state_history(cfg))))
except Exception:
    import traceback
    traceback.print_exc()

print("\n=== subgraph with different schema (wrapper node) ===")
try:
    class Inner(TypedDict):
        text: str
        score: int

    class Outer(TypedDict):
        document: str
        rating: int

    ib = StateGraph(Inner)
    ib.add_node("grade", lambda s: {"score": len(s["text"])})
    ib.add_edge(START, "grade")
    ib.add_edge("grade", END)
    inner_app = ib.compile()

    def call_inner(state: Outer) -> dict:
        result = inner_app.invoke({"text": state["document"], "score": 0})
        return {"rating": result["score"]}

    ob = StateGraph(Outer)
    ob.add_node("inner", call_inner)
    ob.add_edge(START, "inner")
    ob.add_edge("inner", END)
    print("  ", ob.compile().invoke({"document": "hello", "rating": 0}))
except Exception as exc:
    print("  ", exc)

print("\n=== subgraph interrupt propagates to parent ===")
try:
    from langgraph.types import interrupt

    class S(TypedDict):
        v: str

    ib = StateGraph(S)
    ib.add_node("ask", lambda s: {"v": str(interrupt("inner question"))})
    ib.add_edge(START, "ask")
    ib.add_edge("ask", END)
    inner_app = ib.compile()

    ob = StateGraph(S)
    ob.add_node("sub", inner_app)
    ob.add_edge(START, "sub")
    ob.add_edge("sub", END)
    g = ob.compile(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "sub-int"}}
    out = g.invoke({"v": ""}, cfg)
    print("  interrupted:", "__interrupt__" in out, out.get("__interrupt__"))
    print("  resumed:", g.invoke(Command(resume="answer"), cfg))
    snap = g.get_state(cfg, subgraphs=True)
    print("  get_state(subgraphs=True) ok")
except Exception:
    import traceback
    traceback.print_exc()

print("\n=== Command(goto=..., graph=Command.PARENT) handoff ===")
try:
    class H(TypedDict):
        log: Annotated[list[str], operator.add]

    def agent_a(state: H) -> Command:
        return Command(update={"log": ["a"]}, goto="agent_b")

    def agent_b(state: H) -> Command:
        return Command(update={"log": ["b"]}, goto=END)

    b = StateGraph(H)
    b.add_node("agent_a", agent_a, destinations=("agent_b",))
    b.add_node("agent_b", agent_b, destinations=(END,))
    b.add_edge(START, "agent_a")
    print("  ", b.compile().invoke({"log": []}))
except Exception:
    import traceback
    traceback.print_exc()

print("\n=== InMemoryStore with semantic index ===")
try:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from langgraph.store.memory import InMemoryStore

    from shared.llm import get_embeddings

    emb = get_embeddings()
    store = InMemoryStore(index={"embed": emb, "dims": 384, "fields": ["text"]})
    ns = ("memories", "user-1")
    store.put(ns, "m1", {"text": "prefers concise bullet-point answers"})
    store.put(ns, "m2", {"text": "works in platform engineering on the Kestrel project"})
    store.put(ns, "m3", {"text": "dislikes being asked to repeat context"})
    hits = store.search(ns, query="what team does the user work on?", limit=2)
    for item in hits:
        print(f"   score={item.score:.3f} {item.value['text']}")
except Exception:
    import traceback
    traceback.print_exc()

print("\n=== store injected into a node ===")
try:
    from langgraph.store.base import BaseStore
    from langgraph.store.memory import InMemoryStore

    class M(TypedDict):
        user_id: str
        recalled: list[str]

    def remember(state: M, *, store: BaseStore) -> dict:
        ns = ("memories", state["user_id"])
        store.put(ns, "fact", {"text": "the user's project is Kestrel"})
        items = store.search(ns)
        return {"recalled": [i.value["text"] for i in items]}

    b = StateGraph(M)
    b.add_node("remember", remember)
    b.add_edge(START, "remember")
    b.add_edge("remember", END)
    g = b.compile(store=InMemoryStore())
    print("  ", g.invoke({"user_id": "u1", "recalled": []}))
except Exception:
    import traceback
    traceback.print_exc()

print("\n=== error_handler on add_node ===")
try:
    class E(TypedDict):
        out: Annotated[list[str], operator.add]

    def boom(state: E) -> dict:
        raise ValueError("kaboom")

    def rescue(state: E) -> dict:
        return {"out": ["rescued"]}

    b = StateGraph(E)
    b.add_node("boom", boom, error_handler=rescue)
    b.add_edge(START, "boom")
    b.add_edge("boom", END)
    print("  ", b.compile().invoke({"out": []}))
except Exception:
    import traceback
    traceback.print_exc()

print("\n=== deepagents / langgraph_supervisor availability ===")
for name in ["deepagents", "langgraph_supervisor", "langgraph_swarm"]:
    try:
        __import__(name)
        print(f"  ok {name}")
    except Exception as exc:
        print(f"  X  {name}: {type(exc).__name__}")

print("\n=== middleware hooks signature ===")
try:
    import langchain.agents.middleware as mw

    for hook in ["before_model", "after_model", "wrap_model_call", "wrap_tool_call", "before_agent", "after_agent"]:
        fn = getattr(mw.AgentMiddleware, hook, None)
        if fn:
            print(f"  {hook:18} {str(inspect.signature(fn))[:110]}")
except Exception as exc:
    print("  ", exc)

print("\n=== langgraph_sdk / platform surface ===")
try:
    import langgraph_sdk

    print("  langgraph_sdk:", [n for n in dir(langgraph_sdk) if not n.startswith("_")][:10])
except Exception as exc:
    print("  ", exc)
