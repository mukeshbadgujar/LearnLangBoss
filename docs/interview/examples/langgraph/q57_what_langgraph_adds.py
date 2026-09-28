"""Q57. What LangGraph adds over chains/agents.

THE PROBLEM
A LangChain chain can answer bookstore FAQ text, but cannot pause for a human
refund approval or resume tomorrow on the same ticket.

WHAT WE ARE GOING TO SOLVE
Show durable state + a branch + a checkpointed pause - capabilities that need
a graph runtime.

WHAT THIS EXAMPLE IS ABOUT
Order A100 refund path with MemorySaver and interrupt when amount > $20
(here $18 auto-continues; comment shows the pause path).

WHAT IT SOLVES
LangGraph adds cycles, shared state, persistence, and HITL that plain chains
do not provide natively.

KEYWORDS
- Durable execution: checkpoints let a run survive restarts.
- Cycles: edges can loop (retries, polls) under recursion_limit.
- HITL: interrupt/resume built into the runtime.
"""

from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt


class RefundState(TypedDict):
    order_id: str
    amount: float
    status: str


def check_policy(state: RefundState) -> dict:
    return {"status": "policy ok: unread print 30-day"}


def maybe_human(state: RefundState) -> dict:
    if state["amount"] > 20:
        ok = interrupt({"question": f"Approve ${state['amount']}?"})
        return {"status": "human-approved" if ok else "denied"}
    return {"status": f"{state['status']}; auto-ok ${state['amount']}"}


def build_graph():
    g = StateGraph(RefundState)
    g.add_node("policy", check_policy)
    g.add_node("human_gate", maybe_human)
    g.add_edge(START, "policy")
    g.add_edge("policy", "human_gate")
    g.add_edge("human_gate", END)
    return g.compile(checkpointer=MemorySaver())


def main() -> None:
    graph = build_graph()
    cfg = {"configurable": {"thread_id": "adds-A100"}}
    out = graph.invoke(
        {"order_id": "A100", "amount": 18.0, "status": ""}, cfg
    )
    print(out["status"])
    print("LangGraph added: shared state, checkpointer, optional interrupt().")


if __name__ == "__main__":
    print(__doc__)
    main()
