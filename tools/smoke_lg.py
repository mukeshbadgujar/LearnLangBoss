"""Runtime smoke test for the LangGraph patterns used in notebooks 29-32.

Only exercises the API-shape-sensitive parts; no model calls, so it runs
offline and in CI.
"""

import operator
from typing import Annotated, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, MessagesState, StateGraph, add_messages
from langgraph.types import Command

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


class S(TypedDict):
    ticket: str
    category: str
    trace: Annotated[list[str], operator.add]


@case("interrupt_before + update_state + resume")
def _():
    class T(TypedDict):
        a: str
        b: str

    b = StateGraph(T)
    b.add_node("one", lambda s: {"a": "A"})
    b.add_node("two", lambda s: {"b": "B" + s["a"]})
    b.add_edge(START, "one")
    b.add_edge("one", "two")
    b.add_edge("two", END)
    g = b.compile(checkpointer=InMemorySaver(), interrupt_before=["two"])
    cfg = {"configurable": {"thread_id": "x"}}
    g.invoke({"a": "", "b": ""}, cfg)
    assert g.get_state(cfg).next == ("two",), g.get_state(cfg).next
    g.update_state(cfg, {"a": "EDITED"})
    out = g.invoke(None, cfg)
    assert out["b"] == "BEDITED", out
    assert len(list(g.get_state_history(cfg))) >= 3


@case("total=False TypedDict state")
def _():
    class P(TypedDict, total=False):
        ticket: str
        category: str

    b = StateGraph(P)
    b.add_node("c", lambda s: {"category": "billing"})
    b.add_edge(START, "c")
    b.add_edge("c", END)
    assert b.compile().invoke({"ticket": "t"}) == {"ticket": "t", "category": "billing"}


@case("pydantic state validation rejects bad input, returns a dict")
def _():
    from pydantic import BaseModel, Field, field_validator

    class V(BaseModel):
        ticket: str
        category: Literal["billing", "unknown"] = "unknown"
        confidence: float = Field(default=0.0, ge=0.0, le=1.0)

        @field_validator("ticket")
        @classmethod
        def not_empty(cls, v: str) -> str:
            if not v.strip():
                raise ValueError("empty")
            return v

    b = StateGraph(V)
    b.add_node("c", lambda s: {"category": "billing", "confidence": 0.9})
    b.add_edge(START, "c")
    b.add_edge("c", END)
    g = b.compile()
    out = g.invoke({"ticket": "hi"})
    assert isinstance(out, dict) and out["category"] == "billing", out
    for bad in [{"ticket": ""}, {"ticket": "x", "confidence": 1.5}, {"ticket": "x", "category": "nope"}]:
        try:
            g.invoke(bad)
            raise AssertionError(f"should have rejected {bad}")
        except AssertionError:
            raise
        except Exception:
            pass


@case("pydantic bad node write caught by the NEXT node only")
def _():
    from pydantic import BaseModel, Field

    class Sc(BaseModel):
        ticket: str
        confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    def bad(s):
        return {"confidence": 5.0}

    b = StateGraph(Sc)
    b.add_node("bad", bad)
    b.add_edge(START, "bad")
    b.add_edge("bad", END)
    assert b.compile().invoke({"ticket": "x"})["confidence"] == 5.0   # not validated

    b = StateGraph(Sc)
    b.add_node("bad", bad)
    b.add_node("next", lambda s: {})
    b.add_edge(START, "bad")
    b.add_edge("bad", "next")
    b.add_edge("next", END)
    try:
        b.compile().invoke({"ticket": "x"})
        raise AssertionError("expected ValidationError")
    except AssertionError:
        raise
    except Exception:
        pass


@case("overwrite vs operator.add reducer")
def _():
    class N(TypedDict):
        notes: list[str]

    b = StateGraph(N)
    b.add_node("a", lambda s: {"notes": ["A"]})
    b.add_node("b", lambda s: {"notes": ["B"]})
    b.add_edge(START, "a")
    b.add_edge("a", "b")
    b.add_edge("b", END)
    assert b.compile().invoke({"notes": []})["notes"] == ["B"]

    class M(TypedDict):
        notes: Annotated[list[str], operator.add]

    b = StateGraph(M)
    b.add_node("a", lambda s: {"notes": ["A"]})
    b.add_node("b", lambda s: {"notes": ["B"]})
    b.add_edge(START, "a")
    b.add_edge("a", "b")
    b.add_edge("b", END)
    assert b.compile().invoke({"notes": []})["notes"] == ["A", "B"]


@case("custom reducers: unique / max / dict / last_n")
def _():
    def merge_unique(existing: list, new: list) -> list:
        seen = set(existing)
        return existing + [i for i in new if not (i in seen or seen.add(i))]

    def keep_highest(a: float, b: float) -> float:
        return max(a, b)

    def merge_dict(a: dict, b: dict) -> dict:
        return {**a, **b}

    def last_n(n: int):
        return lambda a, b: (a + b)[-n:]

    class R(TypedDict):
        sources: Annotated[list[str], merge_unique]
        confidence: Annotated[float, keep_highest]
        metadata: Annotated[dict, merge_dict]
        recent: Annotated[list[str], last_n(3)]

    b = StateGraph(R)
    b.add_node("f", lambda s: {"sources": ["a", "b"], "confidence": 0.6,
                               "metadata": {"stage": "r", "k": 4}, "recent": ["1"]})
    b.add_node("g", lambda s: {"sources": ["b", "c"], "confidence": 0.4,
                               "metadata": {"stage": "g"}, "recent": ["2", "3", "4"]})
    b.add_edge(START, "f")
    b.add_edge("f", "g")
    b.add_edge("g", END)
    out = b.compile().invoke({"sources": [], "confidence": 0.0, "metadata": {}, "recent": []})
    assert out["sources"] == ["a", "b", "c"], out
    assert out["confidence"] == 0.6, out
    assert out["metadata"] == {"stage": "g", "k": 4}, out
    assert out["recent"] == ["2", "3", "4"], out


@case("add_messages append / replace / remove")
def _():
    from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage

    base = [HumanMessage("q", id="m1"), AIMessage("a", id="m2")]
    assert len(add_messages(base, [HumanMessage("q2", id="m3")])) == 3
    replaced = add_messages(base, [AIMessage("a2", id="m2")])
    assert len(replaced) == 2 and replaced[1].content == "a2"
    assert len(add_messages(base, [RemoveMessage(id="m1")])) == 1


@case("MessagesState subclass with extra fields")
def _():
    from langchain_core.messages import AIMessage, HumanMessage

    class Sup(MessagesState):
        category: str

    b = StateGraph(Sup)
    b.add_node("r", lambda s: {"messages": [AIMessage("hi")], "category": "billing"})
    b.add_edge(START, "r")
    b.add_edge("r", END)
    out = b.compile().invoke({"messages": [HumanMessage("hello")], "category": ""})
    assert out["category"] == "billing" and len(out["messages"]) == 2


@case("parallel write without reducer raises, with reducer works")
def _():
    class U(TypedDict):
        findings: list[str]

    b = StateGraph(U)
    b.add_node("p", lambda s: {"findings": ["p"]})
    b.add_node("c", lambda s: {"findings": ["c"]})
    b.add_edge(START, "p")
    b.add_edge(START, "c")
    b.add_edge("p", END)
    b.add_edge("c", END)
    try:
        b.compile().invoke({"findings": []})
        raise AssertionError("expected InvalidUpdateError")
    except AssertionError:
        raise
    except Exception:
        pass

    class Sa(TypedDict):
        findings: Annotated[list[str], operator.add]

    b = StateGraph(Sa)
    b.add_node("p", lambda s: {"findings": ["p"]})
    b.add_node("c", lambda s: {"findings": ["c"]})
    b.add_edge(START, "p")
    b.add_edge(START, "c")
    b.add_edge("p", END)
    b.add_edge("c", END)
    assert sorted(b.compile().invoke({"findings": []})["findings"]) == ["c", "p"]


@case("input_schema / output_schema")
def _():
    class In(TypedDict):
        question: str

    class Out(TypedDict):
        answer: str

    class Full(TypedDict):
        question: str
        retrieved: list[str]
        answer: str

    b = StateGraph(Full, input_schema=In, output_schema=Out)
    b.add_node("r", lambda s: {"retrieved": ["x"]})
    b.add_node("a", lambda s: {"answer": f"from {len(s['retrieved'])}"})
    b.add_edge(START, "r")
    b.add_edge("r", "a")
    b.add_edge("a", END)
    out = b.compile().invoke({"question": "q"})
    assert set(out) == {"answer"}, out


@case("add_sequence")
def _():
    b = StateGraph(S)
    b.add_sequence([("a", lambda s: {"trace": ["a"]}),
                    ("b", lambda s: {"trace": ["b"]}),
                    ("c", lambda s: {"trace": ["c"]})])
    b.add_edge(START, "a")
    b.add_edge("c", END)
    assert b.compile().invoke({"ticket": "", "category": "", "trace": []})["trace"] == ["a", "b", "c"]


@case("node returning None")
def _():
    b = StateGraph(S)
    b.add_node("noop", lambda s: None)
    b.add_edge(START, "noop")
    b.add_edge("noop", END)
    assert b.compile().invoke({"ticket": "t", "category": "", "trace": []})["ticket"] == "t"


@case("callable class node + functools.partial node")
def _():
    from functools import partial

    class C:
        def __init__(self, label): self.label = label
        def __call__(self, state): return {"trace": [self.label]}

    def setp(state, *, level): return {"trace": [level]}

    b = StateGraph(S)
    b.add_node("c", C("cls"))
    b.add_node("p", partial(setp, level="high"))
    b.add_edge(START, "c")
    b.add_edge("c", "p")
    b.add_edge("p", END)
    assert b.compile().invoke({"ticket": "", "category": "", "trace": []})["trace"] == ["cls", "high"]


@case("add_node(fn) infers name")
def _():
    def classify_ticket(state): return {"trace": ["x"]}

    b = StateGraph(S)
    b.add_node(classify_ticket)
    b.add_edge(START, "classify_ticket")
    b.add_edge("classify_ticket", END)
    assert "classify_ticket" in b.compile().get_graph().nodes


@case("conditional edge to END with Literal annotation")
def _():
    def route(state: S) -> Literal["handle", "__end__"]:
        return END if state["category"] == "spam" else "handle"

    b = StateGraph(S)
    b.add_node("check", lambda s: {"trace": ["check"]})
    b.add_node("handle", lambda s: {"trace": ["handle"]})
    b.add_edge(START, "check")
    b.add_conditional_edges("check", route, {"handle": "handle", END: END})
    b.add_edge("handle", END)
    g = b.compile()
    assert g.invoke({"ticket": "", "category": "spam", "trace": []})["trace"] == ["check"]
    assert g.invoke({"ticket": "", "category": "ok", "trace": []})["trace"] == ["check", "handle"]


@case("conditional edges from START with a list of destinations")
def _():
    class R(TypedDict):
        document: str
        checks: Annotated[list[str], operator.add]

    def pick(state: R) -> list[str]:
        needed = ["grammar"]
        if "$" in state["document"]:
            needed.append("financial")
        return needed

    b = StateGraph(R)
    b.add_node("grammar", lambda s: {"checks": ["g"]})
    b.add_node("financial", lambda s: {"checks": ["f"]})
    b.add_node("legal", lambda s: {"checks": ["l"]})
    b.add_node("done", lambda s: {"checks": [f"n={len(s['checks'])}"]})
    b.add_conditional_edges(START, pick, ["grammar", "financial", "legal"])
    for n in ("grammar", "financial", "legal"):
        b.add_edge(n, "done")
    b.add_edge("done", END)
    g = b.compile()
    assert g.invoke({"document": "plain", "checks": []})["checks"] == ["g", "n=1"]
    out = g.invoke({"document": "$100", "checks": []})["checks"]
    assert set(out[:2]) == {"g", "f"} and out[-1] == "n=2", out


@case("Command(update=, goto=) node")
def _():
    class E(TypedDict):
        ticket: str
        priority: str
        trace: Annotated[list[str], operator.add]

    def assess(state: E) -> Command[Literal["oncall", "queue"]]:
        if "down" in state["ticket"]:
            return Command(update={"priority": "critical", "trace": ["urgent"]}, goto="oncall")
        return Command(update={"priority": "normal", "trace": ["normal"]}, goto="queue")

    b = StateGraph(E)
    b.add_node("assess", assess)
    b.add_node("oncall", lambda s: {"trace": ["oncall"]})
    b.add_node("queue", lambda s: {"trace": ["queue"]})
    b.add_edge(START, "assess")
    b.add_edge("oncall", END)
    b.add_edge("queue", END)
    g = b.compile()
    assert g.invoke({"ticket": "all down", "priority": "", "trace": []})["trace"] == ["urgent", "oncall"]
    assert g.invoke({"ticket": "question", "priority": "", "trace": []})["trace"] == ["normal", "queue"]
    assert "oncall" in g.get_graph().draw_ascii()


@case("loop with counter exits")
def _():
    class L(TypedDict):
        score: int
        rounds: Annotated[int, operator.add]

    def work(state: L) -> dict:
        return {"score": state["score"] + 2, "rounds": 1}

    def again(state: L) -> Literal["work", "__end__"]:
        if state["score"] >= 8 or state["rounds"] >= 3:
            return END
        return "work"

    b = StateGraph(L)
    b.add_node("work", work)
    b.add_edge(START, "work")
    b.add_conditional_edges("work", again, {"work": "work", END: END})
    out = b.compile().invoke({"score": 0, "rounds": 0})
    assert out["rounds"] == 3 and out["score"] == 6, out


@case("unknown state key is silently dropped, and the checked() guard catches it")
def _():
    class Strict(TypedDict):
        known: str

    b = StateGraph(Strict)
    b.add_node("typo", lambda s: {"knwon": "oops"})
    b.add_edge(START, "typo")
    b.add_edge("typo", END)
    assert b.compile().invoke({"known": "original"}) == {"known": "original"}

    def checked(schema, fn):
        allowed = set(schema.__annotations__)

        def wrapper(state):
            update = fn(state)
            if update:
                unknown = set(update) - allowed
                if unknown:
                    raise KeyError(f"unknown keys: {sorted(unknown)}")
            return update

        wrapper.__name__ = getattr(fn, "__name__", "node")
        return wrapper

    b = StateGraph(Strict)
    b.add_node("typo", checked(Strict, lambda s: {"knwon": "oops"}))
    b.add_edge(START, "typo")
    b.add_edge("typo", END)
    try:
        b.compile().invoke({"known": "original"})
        raise AssertionError("expected KeyError")
    except AssertionError:
        raise
    except Exception:
        pass


@case("dangling edge and missing entry point fail at compile")
def _():
    b = StateGraph(S)
    b.add_node("a", lambda s: {"trace": ["a"]})
    b.add_edge(START, "a")
    b.add_edge("a", "nonexistent")
    try:
        b.compile()
        raise AssertionError("expected compile failure for dangling edge")
    except AssertionError:
        raise
    except Exception:
        pass

    b = StateGraph(S)
    b.add_node("a", lambda s: {"trace": ["a"]})
    b.add_edge("a", END)
    try:
        b.compile()
        raise AssertionError("expected compile failure for missing entry point")
    except AssertionError:
        raise
    except Exception:
        pass


@case("orphan node compiles and is skipped")
def _():
    b = StateGraph(S)
    b.add_node("a", lambda s: {"trace": ["a"]})
    b.add_node("orphan", lambda s: {"trace": ["orphan"]})
    b.add_edge(START, "a")
    b.add_edge("a", END)
    assert b.compile().invoke({"ticket": "", "category": "", "trace": []})["trace"] == ["a"]


@case("GraphRecursionError on unbounded loop")
def _():
    class L(TypedDict):
        count: Annotated[int, operator.add]

    b = StateGraph(L)
    b.add_node("forever", lambda s: {"count": 1})
    b.add_edge(START, "forever")
    b.add_edge("forever", "forever")
    try:
        b.compile().invoke({"count": 0}, {"recursion_limit": 8})
        raise AssertionError("expected GraphRecursionError")
    except GraphRecursionError:
        pass


@case("multiple START edges (parallel init)")
def _():
    b = StateGraph(S)
    b.add_node("cfg", lambda s: {"trace": ["cfg"]})
    b.add_node("tkt", lambda s: {"trace": ["tkt"]})
    b.add_node("proc", lambda s: {"trace": ["proc"]})
    b.add_edge(START, "cfg")
    b.add_edge(START, "tkt")
    b.add_edge("cfg", "proc")
    b.add_edge("tkt", "proc")
    b.add_edge("proc", END)
    trace = b.compile().invoke({"ticket": "", "category": "", "trace": []})["trace"]
    assert trace[-1] == "proc" and len(trace) == 3, trace


@case("draw_ascii and draw_mermaid")
def _():
    b = StateGraph(S)
    b.add_node("a", lambda s: {"trace": ["a"]})
    b.add_edge(START, "a")
    b.add_edge("a", END)
    g = b.compile()
    assert "a" in g.get_graph().draw_ascii()
    assert "graph TD" in g.get_graph().draw_mermaid()


print(f"\n{ok} passed, {fail} failed")
raise SystemExit(1 if fail else 0)
