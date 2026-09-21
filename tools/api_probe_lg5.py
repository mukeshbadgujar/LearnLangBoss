"""Probe remaining details for notebooks 49-50."""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def line(label, value=""):
    print(f"{label:46} {value}")


print("=== create_agent full signature ===")
from langchain.agents import create_agent

print(inspect.signature(create_agent))

print("\n=== SqliteStore ===")
try:
    from langgraph.store.sqlite import SqliteStore

    line("from_conn_string", str(inspect.signature(SqliteStore.from_conn_string)))
    with SqliteStore.from_conn_string(":memory:") as st:
        st.setup()
        st.put(("probe",), "a", {"text": "hello"})
        line("roundtrip", st.get(("probe",), "a").value)
except Exception as exc:
    line("SqliteStore", f"FAIL {type(exc).__name__}: {exc}")

print("\n=== TodoListMiddleware ===")
try:
    from langchain.agents.middleware import TodoListMiddleware

    line("__init__", str(inspect.signature(TodoListMiddleware.__init__)))
    m = TodoListMiddleware()
    line("tools", [t.name for t in getattr(m, "tools", [])])
    line("state_schema keys", list(getattr(m.state_schema, "__annotations__", {}).keys()))
except Exception as exc:
    line("TodoListMiddleware", f"FAIL {type(exc).__name__}: {exc}")

print("\n=== other middleware signatures ===")
import langchain.agents.middleware as mw

for name in ("FilesystemFileSearchMiddleware", "ShellToolMiddleware", "LLMToolSelectorMiddleware",
             "ContextEditingMiddleware", "ClearToolUsesEdit", "PIIMiddleware",
             "ToolCallLimitMiddleware", "LLMToolEmulator", "SummarizationMiddleware",
             "HumanInTheLoopMiddleware", "ModelCallLimitMiddleware"):
    try:
        line(name, str(inspect.signature(getattr(mw, name).__init__))[:230])
    except Exception as exc:
        line(name, f"FAIL {exc}")

print("\n=== agent with store + checkpointer ===")
try:
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.store.memory import InMemoryStore

    agent = create_agent(
        GenericFakeChatModel(messages=iter([])),
        tools=[],
        checkpointer=InMemorySaver(),
        store=InMemoryStore(),
    )
    line("create_agent(store=, checkpointer=)", "OK")
    line("nodes", list(agent.get_graph().nodes))
except Exception as exc:
    line("create_agent store", f"FAIL {type(exc).__name__}: {exc}")

print("\n=== agent with TodoListMiddleware ===")
try:
    agent = create_agent(GenericFakeChatModel(messages=iter([])), tools=[],
                         middleware=[mw.TodoListMiddleware()])
    line("nodes", list(agent.get_graph().nodes))
    line("state schema keys", list(agent.stream_channels_list)[:12])
except Exception as exc:
    line("todo agent", f"FAIL {type(exc).__name__}: {exc}")

print("\n=== custom AgentMiddleware hook signatures ===")
from langchain.agents.middleware import AgentMiddleware

for hook in ("before_model", "after_model", "before_agent", "after_agent",
             "wrap_model_call", "wrap_tool_call"):
    try:
        line(hook, str(inspect.signature(getattr(AgentMiddleware, hook)))[:180])
    except Exception as exc:
        line(hook, f"FAIL {exc}")

print("\n=== InjectedState / InjectedToolCallId ===")
for mod, name in [("langgraph.prebuilt", "InjectedState"),
                  ("langchain_core.tools", "InjectedToolCallId"),
                  ("langchain.tools", "ToolRuntime"),
                  ("langchain_core.tools", "ToolRuntime")]:
    try:
        getattr(__import__(mod, fromlist=[name]), name)
        line(f"{mod}.{name}", "OK")
    except Exception as exc:
        line(f"{mod}.{name}", f"FAIL {type(exc).__name__}")
