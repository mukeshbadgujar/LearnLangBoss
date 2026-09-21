"""Probe the store / long-term-memory and deep-agent APIs used by notebooks 49-50."""
from __future__ import annotations

import inspect


def line(label, value=""):
    print(f"{label:52} {value}")


print("=== store imports ===")
for mod, name in [
    ("langgraph.store.memory", "InMemoryStore"),
    ("langgraph.store.base", "BaseStore"),
    ("langgraph.store.base", "Item"),
    ("langgraph.store.base", "SearchItem"),
    ("langgraph.store.sqlite", "SqliteStore"),
    ("langgraph.config", "get_store"),
    ("langgraph.runtime", "Runtime"),
    ("langgraph.runtime", "get_runtime"),
    ("langgraph.prebuilt", "InjectedStore"),
]:
    try:
        obj = getattr(__import__(mod, fromlist=[name]), name)
        line(f"{mod}.{name}", "OK")
    except Exception as exc:
        line(f"{mod}.{name}", f"FAIL {type(exc).__name__}: {exc}")

print("\n=== InMemoryStore signatures ===")
from langgraph.store.memory import InMemoryStore

for meth in ("__init__", "put", "get", "search", "delete", "list_namespaces"):
    try:
        line(f"InMemoryStore.{meth}", str(inspect.signature(getattr(InMemoryStore, meth))))
    except Exception as exc:
        line(f"InMemoryStore.{meth}", f"FAIL {exc}")

print("\n=== store basic roundtrip (no index) ===")
store = InMemoryStore()
store.put(("users", "u1", "facts"), "f1", {"text": "prefers dark mode", "kind": "preference"})
store.put(("users", "u1", "facts"), "f2", {"text": "works on the Kestrel team", "kind": "profile"})
item = store.get(("users", "u1", "facts"), "f1")
line("get -> type", type(item).__name__)
line("item fields", [f for f in dir(item) if not f.startswith("_")][:14])
line("item.value", getattr(item, "value", None))
line("item.key/namespace", (getattr(item, "key", None), getattr(item, "namespace", None)))
line("created_at/updated_at", (getattr(item, "created_at", None), getattr(item, "updated_at", None)))
res = store.search(("users", "u1", "facts"))
line("search(ns) -> n", len(res))
line("search first type", type(res[0]).__name__)
line("search filter", len(store.search(("users", "u1", "facts"), filter={"kind": "preference"})))
line("list_namespaces", store.list_namespaces())
store.delete(("users", "u1", "facts"), "f2")
line("after delete", len(store.search(("users", "u1", "facts"))))

print("\n=== semantic index ===")
try:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from shared.llm import get_embeddings

    emb = get_embeddings()
    sem = InMemoryStore(index={"embed": emb, "dims": len(emb.embed_query("probe")), "fields": ["text"]})
    sem.put(("u1", "memories"), "m1", {"text": "Mukesh leads the Kestrel platform team"})
    sem.put(("u1", "memories"), "m2", {"text": "The team deploys to ap-south-1 on AWS"})
    sem.put(("u1", "memories"), "m3", {"text": "Favourite lunch spot is the dosa place downstairs"})
    hits = sem.search(("u1", "memories"), query="where do they deploy?", limit=2)
    line("semantic search -> n", len(hits))
    for h in hits:
        line("  hit", (h.key, getattr(h, "score", None), h.value["text"][:40]))
except Exception as exc:
    line("semantic index", f"FAIL {type(exc).__name__}: {exc}")

print("\n=== store injected into a node ===")
try:
    from typing import Annotated, TypedDict

    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.config import get_store
    from langgraph.graph import END, START, StateGraph

    class S(TypedDict):
        text: str
        recalled: list

    def writer(state: S, *, store) -> dict:
        store.put(("probe",), "k", {"text": state["text"]})
        return {}

    def reader(state: S) -> dict:
        st = get_store()
        return {"recalled": [i.value["text"] for i in st.search(("probe",))]}

    b = StateGraph(S)
    b.add_node("writer", writer)
    b.add_node("reader", reader)
    b.add_edge(START, "writer")
    b.add_edge("writer", "reader")
    b.add_edge("reader", END)
    g = b.compile(store=InMemoryStore(), checkpointer=InMemorySaver())
    out = g.invoke({"text": "hello store", "recalled": []}, {"configurable": {"thread_id": "t"}})
    line("store= kwarg on node", "OK")
    line("get_store() in node", out["recalled"])
except Exception as exc:
    line("store injection", f"FAIL {type(exc).__name__}: {exc}")

print("\n=== create_agent with store ===")
try:
    from langchain.agents import create_agent
    line("create_agent sig", str(inspect.signature(create_agent))[:400])
except Exception as exc:
    line("create_agent", f"FAIL {exc}")

print("\n=== InjectedStore in a tool ===")
try:
    from typing import Annotated as Ann

    from langchain_core.tools import tool
    from langgraph.prebuilt import InjectedStore
    from langgraph.store.base import BaseStore

    @tool
    def remember(fact: str, store: Ann[BaseStore, InjectedStore()]) -> str:
        """Save a fact."""
        store.put(("probe",), fact[:8], {"text": fact})
        return "saved"

    line("InjectedStore tool schema", list(remember.args.keys()))
except Exception as exc:
    line("InjectedStore", f"FAIL {type(exc).__name__}: {exc}")

print("\n=== deepagents / middleware for 50 ===")
for mod, name in [
    ("deepagents", "create_deep_agent"),
    ("langchain.agents.middleware", "SummarizationMiddleware"),
    ("langchain.agents.middleware", "HumanInTheLoopMiddleware"),
    ("langchain.agents.middleware", "ModelCallLimitMiddleware"),
    ("langchain.agents.middleware", "AgentMiddleware"),
    ("langchain.agents.middleware", "TodoListMiddleware"),
    ("langchain.agents.middleware", "PlanningMiddleware"),
    ("langchain.agents.middleware", "FilesystemMiddleware"),
    ("langchain.agents.middleware", "AnthropicPromptCachingMiddleware"),
    ("langchain.agents.middleware", "ContextEditingMiddleware"),
    ("langchain.tools", "tool"),
]:
    try:
        getattr(__import__(mod, fromlist=[name]), name)
        line(f"{mod}.{name}", "OK")
    except Exception as exc:
        line(f"{mod}.{name}", f"FAIL {type(exc).__name__}")

print("\n=== middleware module contents ===")
try:
    import langchain.agents.middleware as mw

    line("middleware exports", sorted(n for n in dir(mw) if n[0].isupper()))
except Exception as exc:
    line("middleware", f"FAIL {exc}")

print("\n=== AgentMiddleware hooks ===")
try:
    from langchain.agents.middleware import AgentMiddleware

    line("AgentMiddleware methods", sorted(n for n in dir(AgentMiddleware) if not n.startswith("_")))
except Exception as exc:
    line("AgentMiddleware", f"FAIL {exc}")
