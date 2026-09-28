"""Q42. Static interrupts vs dynamic interrupt().

THE PROBLEM
The bookstore always wants a pause before `issue_refund`, but only wants a
dynamic question when the refund amount is over $20.

WHAT WE ARE GOING TO SOLVE
Contrast interrupt_before (compile-time) with interrupt() inside a node
(runtime, state-dependent).

WHAT THIS EXAMPLE IS ABOUT
Two tiny refund graphs on Order A100 ($18) and a $45 hardcover. Static pause
always stops before issue; dynamic pause only asks when amount > 20.

WHAT IT SOLVES
You know when to name a node in compile() versus calling interrupt() from code.

KEYWORDS
- interrupt_before: compile-time list of node names; graph always pauses before them.
- interrupt(): runtime pause from inside a node; payload and whether to pause can depend on state.
- Static vs dynamic: decided when you build the graph vs decided while the node runs.
"""

from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class RefundState(TypedDict):
    order_id: str
    amount: float
    status: str


def prepare(state: RefundState) -> dict:
    return {"status": f"prepared ${state['amount']} for {state['order_id']}"}


def maybe_ask(state: RefundState) -> dict:
    # Dynamic: only pause when amount is high-risk (> $20).
    if state["amount"] > 20:
        ok = interrupt({"question": f"Approve refund of ${state['amount']}?"})
        return {"status": "approved" if ok else "denied"}
    return {"status": "auto-approved under $20"}


def issue_refund(state: RefundState) -> dict:
    return {"status": f"issued ({state['status']})"}


def build_static():
    g = StateGraph(RefundState)
    g.add_node("prepare", prepare)
    g.add_node("issue_refund", issue_refund)
    g.add_edge(START, "prepare")
    g.add_edge("prepare", "issue_refund")
    g.add_edge("issue_refund", END)
    # Static: always pause before issue_refund, regardless of amount.
    return g.compile(
        checkpointer=MemorySaver(),
        interrupt_before=["issue_refund"],
    )


def build_dynamic():
    g = StateGraph(RefundState)
    g.add_node("prepare", prepare)
    g.add_node("maybe_ask", maybe_ask)
    g.add_node("issue_refund", issue_refund)
    g.add_edge(START, "prepare")
    g.add_edge("prepare", "maybe_ask")
    g.add_edge("maybe_ask", "issue_refund")
    g.add_edge("issue_refund", END)
    return g.compile(checkpointer=MemorySaver())


def main() -> None:
    static = build_static()
    cfg_s = {"configurable": {"thread_id": "static-A100"}}
    mid = static.invoke(
        {"order_id": "A100", "amount": 18.0, "status": ""}, cfg_s
    )
    print("STATIC paused before issue:", mid["status"])
    print(static.invoke(None, cfg_s)["status"])  # resume past interrupt_before

    dynamic = build_dynamic()
    cfg_low = {"configurable": {"thread_id": "dyn-18"}}
    print(
        "DYNAMIC $18 (no interrupt):",
        dynamic.invoke({"order_id": "A100", "amount": 18.0, "status": ""}, cfg_low),
    )

    cfg_hi = {"configurable": {"thread_id": "dyn-45"}}
    paused = dynamic.invoke(
        {"order_id": "B200", "amount": 45.0, "status": ""}, cfg_hi
    )
    print("DYNAMIC $45 payload:", paused["__interrupt__"][0].value)
    print("DYNAMIC resume:", dynamic.invoke(Command(resume=True), cfg_hi)["status"])


if __name__ == "__main__":
    print(__doc__)
    main()
