"""Runtime smoke test for the LangGraph patterns used in notebooks 33-38.

No model calls, so it runs offline.
"""

import operator
import sqlite3
import tempfile
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver, MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.config import get_config, get_stream_writer
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command, interrupt

TMP = Path(tempfile.mkdtemp(prefix="lgsmoke_"))
ok = 0
fail = 0


def case(label: str):
    def wrap(fn):
        global ok, fail
        try:
            fn()
            print(f"  ok   {label}")
            ok += 1
        except Exception as exc:
            print(f"  FAIL {label}: {type(exc).__name__}: {exc}")
            fail += 1
    return wrap


class Echo(MessagesState):
    turn_count: int


def echo_node(state: Echo) -> dict:
    return {"messages": [AIMessage(f"echo:{state['messages'][-1].content}")],
            "turn_count": state.get("turn_count", 0) + 1}


def echo_builder() -> StateGraph:
    b = StateGraph(Echo)
    b.add_node("chat", echo_node)
    b.add_edge(START, "chat")
    b.add_edge("chat", END)
    return b


# --------------------------------------------------------------------------- #
# 33 - compile / invoke / stream
# --------------------------------------------------------------------------- #
class Seq(TypedDict):
    topic: str
    a: str
    b: str
    trace: Annotated[list[str], operator.add]


def seq_builder() -> StateGraph:
    b = StateGraph(Seq)
    b.add_sequence([
        ("one", lambda s: {"a": "A", "trace": ["one"]}),
        ("two", lambda s: {"b": "B", "trace": ["two"]}),
    ])
    b.add_edge(START, "one")
    b.add_edge("two", END)
    return b


@case("one builder, several compile() variants")
def _():
    b = seq_builder()
    assert b.compile().name is not None
    assert b.compile(name="custom").name == "custom"
    b.compile(checkpointer=InMemorySaver())
    b.compile(checkpointer=InMemorySaver(), interrupt_before=["two"])


@case("stream_mode updates / values")
def _():
    g = seq_builder().compile()
    init = {"topic": "t", "a": "", "b": "", "trace": []}
    updates = list(g.stream(dict(init), stream_mode="updates"))
    assert [list(c)[0] for c in updates] == ["one", "two"], updates
    values = list(g.stream(dict(init), stream_mode="values"))
    assert len(values) == len(updates) + 1, (len(values), len(updates))


@case("stream_mode tasks / debug / checkpoints")
def _():
    g = seq_builder().compile(checkpointer=InMemorySaver())
    init = {"topic": "t", "a": "", "b": "", "trace": []}
    cfg = {"configurable": {"thread_id": "s1"}}
    for mode in ("tasks", "debug", "checkpoints"):
        chunks = list(g.stream(dict(init), {**cfg, "configurable": {"thread_id": f"s-{mode}"}}, stream_mode=mode))
        assert chunks, f"no chunks for {mode}"


@case("custom stream writer + no-op under invoke")
def _():
    class Ix(TypedDict):
        docs: list[str]
        indexed: int

    def index(state: Ix) -> dict:
        writer = get_stream_writer()
        for i, d in enumerate(state["docs"], 1):
            writer({"done": i, "total": len(state["docs"]), "current": d})
        return {"indexed": len(state["docs"])}

    b = StateGraph(Ix)
    b.add_node("index", index)
    b.add_edge(START, "index")
    b.add_edge("index", END)
    g = b.compile()
    chunks = list(g.stream({"docs": ["a", "b", "c"], "indexed": 0}, stream_mode="custom"))
    assert len(chunks) == 3 and chunks[-1]["done"] == 3, chunks
    assert g.invoke({"docs": ["a"], "indexed": 0})["indexed"] == 1     # no-op writer


@case("combined stream modes yield (mode, payload)")
def _():
    class Ix(TypedDict):
        n: int

    def work(state: Ix) -> dict:
        get_stream_writer()({"progress": 1})
        return {"n": state["n"] + 1}

    b = StateGraph(Ix)
    b.add_node("work", work)
    b.add_edge(START, "work")
    b.add_edge("work", END)
    chunks = list(b.compile().stream({"n": 0}, stream_mode=["updates", "custom"]))
    modes = {m for m, _ in chunks}
    assert modes == {"updates", "custom"}, modes


@case("get_config() reads configurable")
def _():
    class C(TypedDict):
        tone: str

    def node(state: C) -> dict:
        return {"tone": get_config().get("configurable", {}).get("tone", "none")}

    b = StateGraph(C)
    b.add_node("n", node)
    b.add_edge(START, "n")
    b.add_edge("n", END)
    out = b.compile().invoke({"tone": ""}, config={"configurable": {"tone": "formal"}})
    assert out["tone"] == "formal", out


@case("batch with max_concurrency")
def _():
    g = seq_builder().compile()
    outs = g.batch([{"topic": t, "a": "", "b": "", "trace": []} for t in ("x", "y")],
                   config={"max_concurrency": 2})
    assert len(outs) == 2 and all(o["b"] == "B" for o in outs)


# --------------------------------------------------------------------------- #
# 34 - checkpointing
# --------------------------------------------------------------------------- #
@case("no checkpointer forgets, checkpointer remembers")
def _():
    b = echo_builder()
    forgetful = b.compile()
    forgetful.invoke({"messages": [HumanMessage("one")], "turn_count": 0})
    out = forgetful.invoke({"messages": [HumanMessage("two")], "turn_count": 0})
    assert len(out["messages"]) == 2, out

    remembering = b.compile(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "m1"}}
    remembering.invoke({"messages": [HumanMessage("one")], "turn_count": 0}, cfg)
    out = remembering.invoke({"messages": [HumanMessage("two")]}, cfg)
    assert len(out["messages"]) == 4, len(out["messages"])
    assert out["turn_count"] == 2


@case("thread isolation")
def _():
    g = echo_builder().compile(checkpointer=InMemorySaver())
    for t in ("a", "b"):
        g.invoke({"messages": [HumanMessage(t)], "turn_count": 0}, {"configurable": {"thread_id": t}})
    g.invoke({"messages": [HumanMessage("a again")]}, {"configurable": {"thread_id": "a"}})
    a = g.get_state({"configurable": {"thread_id": "a"}}).values
    bb = g.get_state({"configurable": {"thread_id": "b"}}).values
    assert a["turn_count"] == 2 and bb["turn_count"] == 1, (a["turn_count"], bb["turn_count"])


@case("StateSnapshot fields")
def _():
    g = echo_builder().compile(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "snap"}}
    g.invoke({"messages": [HumanMessage("hi")], "turn_count": 0}, cfg)
    s = g.get_state(cfg)
    assert s.next == (), s.next
    assert s.config["configurable"]["checkpoint_id"]
    assert s.created_at
    assert "step" in s.metadata and "source" in s.metadata, s.metadata
    assert s.tasks == ()


@case("update_state applies reducers")
def _():
    class T(MessagesState):
        priority: str

    b = StateGraph(T)
    b.add_node("n", lambda s: {"messages": [AIMessage("ok")]})
    b.add_edge(START, "n")
    b.add_edge("n", END)
    g = b.compile(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "u1"}}
    g.invoke({"messages": [HumanMessage("x")], "priority": "low"}, cfg)
    g.update_state(cfg, {"priority": "critical"})
    assert g.get_state(cfg).values["priority"] == "critical"


@case("update_state(as_node=) keeps the pause point")
def _():
    class T(MessagesState):
        priority: str

    b = StateGraph(T)
    b.add_node("triage", lambda s: {"priority": "low"})
    b.add_node("handle", lambda s: {"messages": [AIMessage(f"p={s['priority']}")]})
    b.add_edge(START, "triage")
    b.add_edge("triage", "handle")
    b.add_edge("handle", END)
    g = b.compile(checkpointer=InMemorySaver(), interrupt_before=["handle"])
    cfg = {"configurable": {"thread_id": "an1"}}
    g.invoke({"messages": [HumanMessage("x")], "priority": ""}, cfg)
    assert g.get_state(cfg).next == ("handle",)
    g.update_state(cfg, {"priority": "critical"}, as_node="triage")
    assert g.get_state(cfg).next == ("handle",), g.get_state(cfg).next
    out = g.invoke(None, cfg)
    assert out["messages"][-1].content == "p=critical", out["messages"][-1].content


@case("state history sources and checkpoint replay")
def _():
    g = echo_builder().compile(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "h1"}}
    g.invoke({"messages": [HumanMessage("one")], "turn_count": 0}, cfg)
    g.invoke({"messages": [HumanMessage("two")]}, cfg)
    history = list(g.get_state_history(cfg))
    assert len(history) >= 4, len(history)
    sources = {s.metadata.get("source") for s in history}
    assert "input" in sources and "loop" in sources, sources

    old = history[2]
    replay = {"configurable": {"thread_id": "h1",
                               "checkpoint_id": old.config["configurable"]["checkpoint_id"]}}
    past = g.get_state(replay)
    assert len(past.values["messages"]) <= len(g.get_state(cfg).values["messages"])


@case("MemorySaver is InMemorySaver, delete_thread works")
def _():
    assert MemorySaver is InMemorySaver
    saver = InMemorySaver()
    g = echo_builder().compile(checkpointer=saver)
    for i in range(2):
        g.invoke({"messages": [HumanMessage(str(i))], "turn_count": 0},
                 {"configurable": {"thread_id": f"d{i}"}})
    saver.delete_thread("d0")
    assert g.get_state({"configurable": {"thread_id": "d0"}}).values in ({}, None)
    assert g.get_state({"configurable": {"thread_id": "d1"}}).values["turn_count"] == 1


# --------------------------------------------------------------------------- #
# 35 - sqlite savers
# --------------------------------------------------------------------------- #
@case("SqliteSaver survives a simulated restart")
def _():
    path = TMP / "ckpt.sqlite"
    conn = sqlite3.connect(path, check_same_thread=False)
    g = echo_builder().compile(checkpointer=SqliteSaver(conn))
    cfg = {"configurable": {"thread_id": "durable"}}
    g.invoke({"messages": [HumanMessage("remember me")], "turn_count": 0}, cfg)
    conn.close()

    conn2 = sqlite3.connect(path, check_same_thread=False)
    g2 = echo_builder().compile(checkpointer=SqliteSaver(conn2))
    state = g2.get_state(cfg)
    assert state.values["turn_count"] == 1, state.values
    g2.invoke({"messages": [HumanMessage("again")]}, cfg)
    assert g2.get_state(cfg).values["turn_count"] == 2
    conn2.close()


@case("SqliteSaver.from_conn_string closes on exit")
def _():
    path = str(TMP / "scoped.sqlite")
    with SqliteSaver.from_conn_string(path) as saver:
        g = echo_builder().compile(checkpointer=saver)
        g.invoke({"messages": [HumanMessage("x")], "turn_count": 0},
                 {"configurable": {"thread_id": "sc"}})
    try:
        g.get_state({"configurable": {"thread_id": "sc"}})
        raise AssertionError("expected a closed-connection error")
    except AssertionError:
        raise
    except Exception:
        pass


@case("sqlite :memory: works")
def _():
    with SqliteSaver.from_conn_string(":memory:") as saver:
        g = echo_builder().compile(checkpointer=saver)
        cfg = {"configurable": {"thread_id": "mem"}}
        g.invoke({"messages": [HumanMessage("x")], "turn_count": 0}, cfg)
        assert g.get_state(cfg).values["turn_count"] == 1


@case("sqlite schema has the documented tables and columns")
def _():
    path = TMP / "schema.sqlite"
    conn = sqlite3.connect(path, check_same_thread=False)
    g = echo_builder().compile(checkpointer=SqliteSaver(conn))
    g.invoke({"messages": [HumanMessage("x")], "turn_count": 0}, {"configurable": {"thread_id": "sch"}})
    conn.close()

    c = sqlite3.connect(path)
    tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "checkpoints" in tables, tables
    cols = {r[1] for r in c.execute("PRAGMA table_info(checkpoints)")}
    for needed in ("thread_id", "checkpoint_id", "type", "checkpoint"):
        assert needed in cols, (needed, cols)
    rows = c.execute("SELECT thread_id, checkpoint_id, type, LENGTH(checkpoint) FROM checkpoints").fetchall()
    assert rows
    c.close()


@case("sqlite PRAGMA tuning")
def _():
    conn = sqlite3.connect(TMP / "tuned.sqlite", check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA synchronous=NORMAL")
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    g = echo_builder().compile(checkpointer=SqliteSaver(conn))
    g.invoke({"messages": [HumanMessage("x")], "turn_count": 0}, {"configurable": {"thread_id": "tn"}})
    conn.close()


# --------------------------------------------------------------------------- #
# 36 - human in the loop
# --------------------------------------------------------------------------- #
class R(TypedDict):
    amount: float
    approved: bool
    note: str
    trace: Annotated[list[str], operator.add]


@case("static interrupt_before: pause, inspect, edit, resume")
def _():
    b = StateGraph(R)
    b.add_node("assess", lambda s: {"trace": ["assess"]})
    b.add_node("pay", lambda s: {"trace": [f"PAID {s['amount']}"]})
    b.add_edge(START, "assess")
    b.add_edge("assess", "pay")
    b.add_edge("pay", END)
    g = b.compile(checkpointer=InMemorySaver(), interrupt_before=["pay"])
    cfg = {"configurable": {"thread_id": "r1"}}
    out = g.invoke({"amount": 100.0, "approved": False, "note": "", "trace": []}, cfg)
    assert out["trace"] == ["assess"], out["trace"]
    assert g.get_state(cfg).next == ("pay",)
    g.update_state(cfg, {"amount": 50.0})
    final = g.invoke(None, cfg)
    assert final["trace"] == ["assess", "PAID 50.0"], final["trace"]


@case("dynamic interrupt(): conditional pause with a payload")
def _():
    def review(state: R) -> dict:
        if state["amount"] < 1000:
            return {"approved": True, "trace": ["auto"]}
        decision = interrupt({"kind": "refund_approval", "amount": state["amount"],
                              "question": "Approve?", "options": ["approve", "reject", "reduce"]})
        if isinstance(decision, dict) and decision.get("action") == "reduce":
            return {"approved": True, "amount": decision["amount"],
                    "note": decision.get("reason", ""), "trace": ["reduced"]}
        if decision == "approve":
            return {"approved": True, "trace": ["approved"]}
        return {"approved": False, "trace": ["rejected"]}

    b = StateGraph(R)
    b.add_node("review", review)
    b.add_node("exec", lambda s: {"trace": [f"PAID {s['amount']}" if s["approved"] else "no action"]})
    b.add_edge(START, "review")
    b.add_edge("review", "exec")
    b.add_edge("exec", END)
    g = b.compile(checkpointer=InMemorySaver())

    small = g.invoke({"amount": 49.0, "approved": False, "note": "", "trace": []},
                     {"configurable": {"thread_id": "small"}})
    assert "__interrupt__" not in small and small["trace"] == ["auto", "PAID 49.0"], small["trace"]

    cfg = {"configurable": {"thread_id": "big"}}
    paused = g.invoke({"amount": 18400.0, "approved": False, "note": "", "trace": []}, cfg)
    assert "__interrupt__" in paused
    payload = paused["__interrupt__"][0].value
    assert payload["kind"] == "refund_approval" and payload["amount"] == 18400.0, payload
    done = g.invoke(Command(resume="approve"), cfg)
    assert done["trace"][-1] == "PAID 18400.0", done["trace"]

    cfg2 = {"configurable": {"thread_id": "red"}}
    g.invoke({"amount": 18400.0, "approved": False, "note": "", "trace": []}, cfg2)
    red = g.invoke(Command(resume={"action": "reduce", "amount": 9200.0, "reason": "partial"}), cfg2)
    assert red["trace"][-1] == "PAID 9200.0" and red["note"] == "partial", red

    cfg3 = {"configurable": {"thread_id": "rej"}}
    g.invoke({"amount": 18400.0, "approved": False, "note": "", "trace": []}, cfg3)
    rej = g.invoke(Command(resume="reject"), cfg3)
    assert rej["trace"][-1] == "no action", rej["trace"]


@case("node re-runs from the top on resume (the double side-effect bug)")
def _():
    class C(TypedDict):
        effects: Annotated[list[str], operator.add]
        answer: str

    hits = []

    def careless(state: C) -> dict:
        hits.append(1)
        d = interrupt("Approve?")
        return {"effects": ["done"], "answer": str(d)}

    b = StateGraph(C)
    b.add_node("careless", careless)
    b.add_edge(START, "careless")
    b.add_edge("careless", END)
    g = b.compile(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "c1"}}
    g.invoke({"effects": [], "answer": ""}, cfg)
    g.invoke(Command(resume="yes"), cfg)
    assert len(hits) == 2, f"expected the node body to run twice, got {len(hits)}"

    # The fix: interrupt alone in its own node.
    hits2 = []

    def gate(state: C) -> dict:
        return {"answer": str(interrupt("Approve?"))}

    def effect(state: C) -> dict:
        hits2.append(1)
        return {"effects": ["done"]}

    b = StateGraph(C)
    b.add_node("gate", gate)
    b.add_node("effect", effect)
    b.add_edge(START, "gate")
    b.add_edge("gate", "effect")
    b.add_edge("effect", END)
    g = b.compile(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "c2"}}
    g.invoke({"effects": [], "answer": ""}, cfg)
    out = g.invoke(Command(resume="yes"), cfg)
    assert len(hits2) == 1 and out["effects"] == ["done"], (hits2, out)


@case("approval queue reads snapshot.tasks[*].interrupts")
def _():
    def review(state: R) -> dict:
        decision = interrupt({"amount": state["amount"], "question": "Approve?"})
        return {"approved": decision == "approve", "trace": [str(decision)]}

    b = StateGraph(R)
    b.add_node("review", review)
    b.add_node("exec", lambda s: {"trace": ["paid" if s["approved"] else "skipped"]})
    b.add_edge(START, "review")
    b.add_edge("review", "exec")
    b.add_edge("exec", END)
    g = b.compile(checkpointer=InMemorySaver())

    threads = []
    for i, amount in enumerate([49.0, 18400.0, 5200.0], start=1):
        tid = f"q-{i}"
        threads.append(tid)
        g.invoke({"amount": amount, "approved": False, "note": "", "trace": []},
                 {"configurable": {"thread_id": tid}})

    pending = []
    for tid in threads:
        snap = g.get_state({"configurable": {"thread_id": tid}})
        for task in snap.tasks:
            for itr in task.interrupts:
                pending.append({"thread_id": tid, "node": task.name, "payload": itr.value})
    assert len(pending) == 3, pending
    assert pending[0]["node"] == "review", pending[0]

    for item in pending:
        g.invoke(Command(resume="approve"), {"configurable": {"thread_id": item["thread_id"]}})

    remaining = [t for t in threads if g.get_state({"configurable": {"thread_id": t}}).tasks]
    assert not remaining, remaining


@case("tool approval: reject produces ToolMessages the agent can read")
def _():
    from langchain_core.messages import AIMessage, ToolMessage
    from langchain_core.tools import tool
    from langgraph.prebuilt import ToolNode

    @tool
    def close_ticket(ticket_id: str, reason: str) -> str:
        """Close a support ticket."""
        return f"{ticket_id} closed: {reason}"

    def guarded(state: MessagesState) -> dict:
        last = state["messages"][-1]
        decision = interrupt({"calls": [{"name": c["name"], "args": c["args"]} for c in last.tool_calls]})
        if decision != "approve":
            return {"messages": [ToolMessage(content="Rejected by reviewer.", tool_call_id=c["id"])
                                 for c in last.tool_calls]}
        return ToolNode([close_ticket]).invoke(state)

    b = StateGraph(MessagesState)
    b.add_node("tools", guarded)
    b.add_edge(START, "tools")
    b.add_edge("tools", END)
    g = b.compile(checkpointer=InMemorySaver())

    ai = AIMessage(content="", tool_calls=[{"name": "close_ticket", "id": "tc1",
                                            "args": {"ticket_id": "T-101", "reason": "resolved"}}])
    cfg = {"configurable": {"thread_id": "tool-rej"}}
    paused = g.invoke({"messages": [HumanMessage("close it"), ai]}, cfg)
    assert paused["__interrupt__"][0].value["calls"][0]["name"] == "close_ticket"
    out = g.invoke(Command(resume="reject"), cfg)
    assert isinstance(out["messages"][-1], ToolMessage) and "Rejected" in out["messages"][-1].content

    cfg2 = {"configurable": {"thread_id": "tool-ok"}}
    g.invoke({"messages": [HumanMessage("close it"), ai]}, cfg2)
    out = g.invoke(Command(resume="approve"), cfg2)
    assert "closed" in out["messages"][-1].content, out["messages"][-1].content


@case("editing proposed tool_calls before execution")
def _():
    from langchain_core.messages import AIMessage
    from langchain_core.tools import tool
    from langgraph.prebuilt import ToolNode

    @tool
    def close_ticket(ticket_id: str, reason: str) -> str:
        """Close a support ticket."""
        return f"{ticket_id} closed: {reason}"

    def editable(state: MessagesState) -> dict:
        last = state["messages"][-1]
        decision = interrupt({"calls": [dict(c) for c in last.tool_calls]})
        if isinstance(decision, dict) and decision.get("action") == "edit":
            edited = last.model_copy(update={"tool_calls": decision["tool_calls"]})
            return ToolNode([close_ticket]).invoke({"messages": state["messages"][:-1] + [edited]})
        return ToolNode([close_ticket]).invoke(state)

    b = StateGraph(MessagesState)
    b.add_node("tools", editable)
    b.add_edge(START, "tools")
    b.add_edge("tools", END)
    g = b.compile(checkpointer=InMemorySaver())

    ai = AIMessage(content="", tool_calls=[{"name": "close_ticket", "id": "tc1",
                                            "args": {"ticket_id": "T-101", "reason": "resolved"}}])
    cfg = {"configurable": {"thread_id": "edit"}}
    paused = g.invoke({"messages": [HumanMessage("close it"), ai]}, cfg)
    proposed = paused["__interrupt__"][0].value["calls"]
    corrected = [dict(c, args={**c["args"], "reason": "verified with finance"}) for c in proposed]
    out = g.invoke(Command(resume={"action": "edit", "tool_calls": corrected}), cfg)
    assert "verified with finance" in out["messages"][-1].content, out["messages"][-1].content


@case("HumanInTheLoopMiddleware builds an agent with an approval node")
def _():
    # A real tool-calling model is needed to actually trigger the interrupt, so
    # here we only verify the wiring that the notebook relies on.
    from langchain.agents import create_agent
    from langchain.agents.middleware import HumanInTheLoopMiddleware, InterruptOnConfig
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.tools import tool

    @tool
    def close_ticket(ticket_id: str) -> str:
        """Close a support ticket."""
        return f"{ticket_id} closed"

    @tool
    def search_tickets(query: str) -> str:
        """Search tickets."""
        return "none"

    mw = HumanInTheLoopMiddleware(interrupt_on={"close_ticket": True, "search_tickets": False})
    assert InterruptOnConfig is not None

    fake = GenericFakeChatModel(messages=iter([AIMessage("Done.")]))
    agent = create_agent(fake, [close_ticket, search_tickets],
                         middleware=[mw], checkpointer=InMemorySaver())
    nodes = set(agent.get_graph().nodes)
    assert any("human" in n.lower() or "interrupt" in n.lower() for n in nodes), nodes


# --------------------------------------------------------------------------- #
# 37 - message history
# --------------------------------------------------------------------------- #
@case("RemoveMessage deletes, REMOVE_ALL_MESSAGES clears")
def _():
    from langchain_core.messages import RemoveMessage
    from langgraph.graph.message import REMOVE_ALL_MESSAGES

    class M(MessagesState):
        pass

    def trim(state: M) -> dict:
        keep = state["messages"][-2:]
        return {"messages": [RemoveMessage(id=m.id) for m in state["messages"][:-2]]}

    b = StateGraph(M)
    b.add_node("trim", trim)
    b.add_edge(START, "trim")
    b.add_edge("trim", END)
    msgs = [HumanMessage(f"m{i}", id=f"id{i}") for i in range(6)]
    out = b.compile().invoke({"messages": msgs})
    assert len(out["messages"]) == 2, len(out["messages"])

    def wipe(state: M) -> dict:
        return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}

    b = StateGraph(M)
    b.add_node("wipe", wipe)
    b.add_edge(START, "wipe")
    b.add_edge("wipe", END)
    out = b.compile().invoke({"messages": msgs})
    assert out["messages"] == [], out["messages"]


@case("trim_messages keeps tool pairs intact")
def _():
    from langchain_core.messages import AIMessage, SystemMessage, ToolMessage, trim_messages

    history = [
        SystemMessage("sys"),
        HumanMessage("q1"),
        AIMessage(content="", tool_calls=[{"name": "t", "id": "1", "args": {}}]),
        ToolMessage(content="r", tool_call_id="1"),
        AIMessage("a1"),
        HumanMessage("q2"),
    ]
    kept = trim_messages(history, max_tokens=4, token_counter=len, strategy="last",
                         include_system=True, start_on="human")
    assert isinstance(kept[0], SystemMessage), kept
    ids = [type(m).__name__ for m in kept]
    assert "ToolMessage" not in ids or "AIMessage" in ids, ids


# --------------------------------------------------------------------------- #
# 38 - parallel
# --------------------------------------------------------------------------- #
@case("Send() map-reduce fan-out")
def _():
    from langgraph.types import Send

    class S3(TypedDict):
        topics: list[str]
        notes: Annotated[list[str], operator.add]

    b = StateGraph(S3)
    b.add_node("research", lambda s: {"notes": [f"note:{s['topic']}"]})
    b.add_conditional_edges(START, lambda s: [Send("research", {"topic": t}) for t in s["topics"]], ["research"])
    b.add_edge("research", END)
    out = b.compile().invoke({"topics": ["a", "b", "c"], "notes": []})
    assert sorted(out["notes"]) == ["note:a", "note:b", "note:c"], out


@case("defer=True waits for all branches")
def _():
    class D(TypedDict):
        seen: Annotated[list[str], operator.add]

    b = StateGraph(D)
    b.add_node("fast", lambda s: {"seen": ["fast"]})
    b.add_node("slow_a", lambda s: {"seen": ["a"]})
    b.add_node("slow_b", lambda s: {"seen": ["b"]})
    b.add_node("merge", lambda s: {"seen": [f"merged{len(s['seen'])}"]}, defer=True)
    b.add_edge(START, "fast")
    b.add_edge(START, "slow_a")
    b.add_edge("slow_a", "slow_b")
    b.add_edge("fast", "merge")
    b.add_edge("slow_b", "merge")
    b.add_edge("merge", END)
    out = b.compile().invoke({"seen": []})
    assert out["seen"][-1] == "merged3", out["seen"]


@case("RetryPolicy on a flaky node")
def _():
    from langgraph.types import RetryPolicy

    attempts = []

    class F(TypedDict):
        n: int

    def flaky(state: F) -> dict:
        attempts.append(1)
        if len(attempts) < 3:
            raise ConnectionError("transient")
        return {"n": state["n"] + 1}

    b = StateGraph(F)
    b.add_node("flaky", flaky, retry_policy=RetryPolicy(max_attempts=5, initial_interval=0.01,
                                                        retry_on=ConnectionError))
    b.add_edge(START, "flaky")
    b.add_edge("flaky", END)
    assert b.compile().invoke({"n": 0})["n"] == 1
    assert len(attempts) == 3, attempts


@case("max_concurrency config accepted on a fan-out")
def _():
    class P(TypedDict):
        out: Annotated[list[str], operator.add]

    b = StateGraph(P)
    for name in ("a", "b", "c", "d"):
        b.add_node(name, lambda s, n=name: {"out": [n]})
        b.add_edge(START, name)
        b.add_edge(name, END)
    out = b.compile().invoke({"out": []}, config={"max_concurrency": 2})
    assert sorted(out["out"]) == ["a", "b", "c", "d"], out


print(f"\n{ok} passed, {fail} failed")
raise SystemExit(1 if fail else 0)
