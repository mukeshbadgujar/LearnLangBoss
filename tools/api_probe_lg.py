"""Probe the installed LangGraph surface so the notebooks use real APIs."""

from __future__ import annotations

import importlib
import inspect


def version(name: str) -> None:
    try:
        mod = importlib.import_module(name)
        print(f"{name:32} {getattr(mod, '__version__', '?')}")
    except Exception as exc:
        print(f"{name:32} MISSING ({type(exc).__name__})")


def check(path: str) -> None:
    module, _, attr = path.rpartition(".")
    try:
        mod = importlib.import_module(module)
    except Exception as exc:
        print(f"  X  {path:62} module import failed: {type(exc).__name__}: {exc}")
        return
    if not hasattr(mod, attr):
        print(f"  X  {path:62} attribute missing")
        return
    obj = getattr(mod, attr)
    try:
        sig = str(inspect.signature(obj))
    except (TypeError, ValueError):
        sig = ""
    print(f"  ok {path:62} {sig[:110]}")


print("=== versions ===")
for name in [
    "langgraph", "langgraph.checkpoint.sqlite", "langgraph.checkpoint.memory",
    "langchain", "langchain_core", "langgraph.prebuilt", "langgraph_supervisor",
    "langgraph.store.memory", "deepagents",
]:
    version(name)

print("\n=== core graph ===")
for path in [
    "langgraph.graph.StateGraph",
    "langgraph.graph.START",
    "langgraph.graph.END",
    "langgraph.graph.MessagesState",
    "langgraph.graph.add_messages",
    "langgraph.graph.message.add_messages",
    "langgraph.graph.state.CompiledStateGraph",
    "langgraph.types.Command",
    "langgraph.types.Send",
    "langgraph.types.interrupt",
    "langgraph.types.RetryPolicy",
    "langgraph.types.CachePolicy",
    "langgraph.types.StreamWriter",
    "langgraph.types.StateSnapshot",
    "langgraph.config.get_stream_writer",
    "langgraph.config.get_config",
    "langgraph.errors.GraphRecursionError",
    "langgraph.errors.NodeInterrupt",
    "langgraph.constants.START",
]:
    check(path)

print("\n=== checkpointers / stores ===")
for path in [
    "langgraph.checkpoint.memory.MemorySaver",
    "langgraph.checkpoint.memory.InMemorySaver",
    "langgraph.checkpoint.sqlite.SqliteSaver",
    "langgraph.checkpoint.sqlite.aio.AsyncSqliteSaver",
    "langgraph.checkpoint.base.BaseCheckpointSaver",
    "langgraph.store.memory.InMemoryStore",
    "langgraph.store.base.BaseStore",
]:
    check(path)

print("\n=== prebuilt / agents ===")
for path in [
    "langgraph.prebuilt.create_react_agent",
    "langgraph.prebuilt.ToolNode",
    "langgraph.prebuilt.tools_condition",
    "langgraph.prebuilt.InjectedState",
    "langgraph.prebuilt.InjectedStore",
    "langchain.agents.create_agent",
    "langchain.agents.AgentState",
    "langchain.tools.tool",
    "langchain_core.tools.tool",
    "langchain_core.tools.InjectedToolCallId",
]:
    check(path)

print("\n=== middleware ===")
try:
    import langchain.agents.middleware as mw

    names = sorted(n for n in dir(mw) if n[0].isupper())
    print("  ", ", ".join(names))
except Exception as exc:
    print("   middleware import failed:", exc)

print("\n=== StateGraph methods ===")
try:
    from langgraph.graph import StateGraph

    for name in ["add_node", "add_edge", "add_conditional_edges", "add_sequence", "compile"]:
        if hasattr(StateGraph, name):
            print(f"  ok StateGraph.{name}{str(inspect.signature(getattr(StateGraph, name)))[:150]}")
        else:
            print(f"  X  StateGraph.{name}")
except Exception as exc:
    print("  ", exc)

print("\n=== compiled graph methods ===")
try:
    from typing import TypedDict

    from langgraph.graph import END, START, StateGraph

    class S(TypedDict):
        x: int

    g = StateGraph(S)
    g.add_node("n", lambda s: {"x": s["x"] + 1})
    g.add_edge(START, "n")
    g.add_edge("n", END)
    app = g.compile()
    for name in [
        "invoke", "stream", "astream", "astream_events", "batch",
        "get_state", "get_state_history", "update_state", "get_graph", "with_config",
    ]:
        print(f"  {'ok' if hasattr(app, name) else 'X '} {name}")
    print("  result:", app.invoke({"x": 1}))
    print("  stream_modes:", str(inspect.signature(app.stream))[:200])
    print("\n  mermaid:\n", app.get_graph().draw_mermaid())
    print("  ascii ok:", bool(app.get_graph().draw_ascii()))
except Exception as exc:
    import traceback

    traceback.print_exc()

print("\n=== compile() signature ===")
try:
    from langgraph.graph import StateGraph

    print(" ", inspect.signature(StateGraph.compile))
except Exception as exc:
    print("  ", exc)

print("\n=== interrupt/Command ===")
try:
    from langgraph.types import Command, interrupt

    print("  Command fields:", getattr(Command, "__annotations__", {}))
    print("  interrupt sig:", inspect.signature(interrupt))
except Exception as exc:
    print("  ", exc)
