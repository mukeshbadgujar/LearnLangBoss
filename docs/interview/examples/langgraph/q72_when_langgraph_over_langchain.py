"""Q72. When to choose LangGraph over LangChain.

THE PROBLEM
The bookstore FAQ bot is a straight retrieve->answer chain, but refunds need
approval pauses and multi-desk routing - people still reach for one tool for both.

WHAT WE ARE GOING TO SOLVE
Show a simple chain-shaped function for FAQ and a StateGraph for the refund
HITL path, then state the decision rule.

WHAT THIS EXAMPLE IS ABOUT
FAQ: return window text. Refund: graph with interrupt gate for amounts > $20
(A100 at $18 auto-passes).

WHAT IT SOLVES
Use LangChain chains/prebuilts for linear work; LangGraph when you need state,
branches, persistence, or HITL.

KEYWORDS
- Linear workflow: fixed steps - LangChain/LCEL is enough.
- Stateful workflow: loops, pauses, multi-agent - LangGraph.
- Decision rule: pick the lowest layer that still expresses the control flow.
"""

from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt


def faq_chain(question: str) -> str:
    """LangChain-shaped: no shared state machine required."""
    return "Unread print books can be returned within 30 days. (A100 $18 scene)"


class RefundState(TypedDict):
    amount: float
    status: str


def gate(state: RefundState) -> dict:
    if state["amount"] > 20:
        ok = interrupt({"question": f"Approve ${state['amount']}?"})
        return {"status": "approved" if ok else "denied"}
    return {"status": f"auto-approved ${state['amount']} for A100"}


def build_refund_graph():
    g = StateGraph(RefundState)
    g.add_node("gate", gate)
    g.add_edge(START, "gate")
    g.add_edge("gate", END)
    return g.compile(checkpointer=MemorySaver())


def main() -> None:
    print("FAQ via chain:", faq_chain("How long to return a print book?"))
    graph = build_refund_graph()
    cfg = {"configurable": {"thread_id": "when-A100"}}
    print(
        "Refund via graph:",
        graph.invoke({"amount": 18.0, "status": ""}, cfg)["status"],
    )
    print("Choose LangGraph when control flow is the product.")


if __name__ == "__main__":
    print(__doc__)
    main()
