"""Q18. What is a checkpointer?

THE PROBLEM
Without persistence, closing the chat mid-refund loses A100 progress. Crashes
restart from zero. Human approval cannot pause safely.

WHAT WE ARE GOING TO SOLVE
Compile with MemorySaver and show the three capabilities: thread memory, pause
points (interrupt_before), and resume after a 'crash' (new invoke same thread).

WHAT THIS EXAMPLE IS ABOUT
Graph: classify -> issue_refund. interrupt_before issue_refund. Operator approves
by continuing the thread for The Little Prince $18.

WHAT IT SOLVES
Checkpointer unlocks (1) conversational memory, (2) human-in-the-loop, (3) fault
tolerance via snapshots after each superstep.

KEYWORDS
- Checkpointer: backend that stores state snapshots per thread.
- thread_id: key that selects which checkpoint history to load.
- interrupt_before: compile-time pause before named nodes.
"""

from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph


class RefundState(TypedDict):
    order_id: str
    stage: str
    approved: bool


def classify(state: RefundState) -> dict:
    return {"stage": "classified", "order_id": "A100"}


def issue_refund(state: RefundState) -> dict:
    return {"stage": "refunded_$18_The_Little_Prince", "approved": True}


def build_graph(checkpointer: MemorySaver):
    g = StateGraph(RefundState)
    g.add_node("classify", classify)
    g.add_node("issue_refund", issue_refund)
    g.add_edge(START, "classify")
    g.add_edge("classify", "issue_refund")
    g.add_edge("issue_refund", END)
    return g.compile(checkpointer=checkpointer, interrupt_before=["issue_refund"])


def main() -> None:
    saver = MemorySaver()
    graph = build_graph(saver)
    config = {"configurable": {"thread_id": "refund-a100"}}

    # 1) Conversational / run memory + 2) HITL pause
    graph.invoke(
        {"order_id": "", "stage": "", "approved": False},
        config,
    )
    paused = graph.get_state(config)
    print("paused before:", paused.next, "| stage:", paused.values["stage"])

    # 3) Fault tolerance: 'resume' same thread as if a new worker picked up
    resumed = graph.invoke(None, config)
    print("after resume:", resumed["stage"], "| approved:", resumed["approved"])


if __name__ == "__main__":
    print(__doc__)
    main()
