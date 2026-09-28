"""Q61. HITL breakpoints with interrupt_before.

THE PROBLEM
Finance wants a hard stop before any refund is issued, even when the amount is
small - a compile-time breakpoint, not ad-hoc logic.

WHAT WE ARE GOING TO SOLVE
Use interrupt_before=["issue"] with a checkpointer, then resume the A100 refund.

WHAT THIS EXAMPLE IS ABOUT
Bookstore refund for The Little Prince $18 pauses at the breakpoint before
issue; a reviewer resumes.

WHAT IT SOLVES
Static HITL breakpoints for compliance gates on named nodes.

KEYWORDS
- Breakpoint: compile-time pause via interrupt_before / interrupt_after.
- HITL: human reviews before the side-effect node runs.
- Resume: invoke again on the same thread_id (input None) to continue.
"""

from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph


class RefundState(TypedDict):
    order_id: str
    amount: float
    stage: str


def draft(state: RefundState) -> dict:
    return {"stage": f"drafted ${state['amount']} for {state['order_id']}"}


def issue(state: RefundState) -> dict:
    return {"stage": f"issued refund ${state['amount']} (30-day unread print)"}


def build_graph():
    g = StateGraph(RefundState)
    g.add_node("draft", draft)
    g.add_node("issue", issue)
    g.add_edge(START, "draft")
    g.add_edge("draft", "issue")
    g.add_edge("issue", END)
    return g.compile(
        checkpointer=MemorySaver(),
        interrupt_before=["issue"],
    )


def main() -> None:
    graph = build_graph()
    cfg = {"configurable": {"thread_id": "hitl-A100"}}
    paused = graph.invoke(
        {"order_id": "A100", "amount": 18.0, "stage": ""}, cfg
    )
    print("Breakpoint after draft:", paused["stage"])
    print("Next node:", graph.get_state(cfg).next)
    done = graph.invoke(None, cfg)
    print("After human resume:", done["stage"])


if __name__ == "__main__":
    print(__doc__)
    main()
