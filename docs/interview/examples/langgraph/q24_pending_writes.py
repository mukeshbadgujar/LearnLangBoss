"""Q24. Pending writes

THE PROBLEM
A worker crashes after a node finished but before the next step was scheduled.
On resume you must neither drop nor double-apply that node's update.

WHAT WE ARE GOING TO SOLVE
Explain pending writes with a real checkpointed graph: pause mid-flow and show
that completed node output is durable on the thread before the next node runs.

WHAT THIS EXAMPLE IS ABOUT
classify -> issue for A100. interrupt_before issue. After classify, state already
holds classified stage - that durable write is what pending-write bookkeeping
protects around crash windows.

WHAT IT SOLVES
Pending writes in checkpoint metadata distinguish committed node outputs from
work still to schedule - avoiding lost or duplicated updates on resume.

KEYWORDS
- Pending writes: checkpoint metadata about node outputs not yet fully applied
  into the next scheduled step.
- Durability: surviving process crash without corrupting the thread.
- Resume: continue a thread from the last consistent checkpoint.
"""

from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph


class PendState(TypedDict):
    order_id: str
    stage: str
    refunded: bool


def classify(state: PendState) -> dict:
    return {"order_id": "A100", "stage": "classified_The_Little_Prince"}


def issue(state: PendState) -> dict:
    return {"stage": "refunded", "refunded": True}


def build_graph(checkpointer: MemorySaver):
    g = StateGraph(PendState)
    g.add_node("classify", classify)
    g.add_node("issue", issue)
    g.add_edge(START, "classify")
    g.add_edge("classify", "issue")
    g.add_edge("issue", END)
    return g.compile(checkpointer=checkpointer, interrupt_before=["issue"])


def main() -> None:
    graph = build_graph(MemorySaver())
    config = {"configurable": {"thread_id": "pending-a100"}}
    graph.invoke({"order_id": "", "stage": "", "refunded": False}, config)
    snap = graph.get_state(config)
    print("after classify (paused):", snap.values)
    print("next:", snap.next)
    # tasks / pending writes live on the checkpoint - resume must reuse them
    print("tasks:", snap.tasks)
    resumed = graph.invoke(None, config)
    print("resumed without re-losing classify output:", resumed)


if __name__ == "__main__":
    print(__doc__)
    main()
