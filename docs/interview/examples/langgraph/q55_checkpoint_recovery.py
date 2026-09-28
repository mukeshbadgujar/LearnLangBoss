"""Q55. Checkpoint recovery after a mid-refund crash.

THE PROBLEM
The refund worker dies after drafting Order A100's refund but before issuing
it. Restarting from scratch would double-charge the ledger.

WHAT WE ARE GOING TO SOLVE
Crash mid-graph, then resume the *same* thread_id from the MemorySaver
checkpoint so issue_refund runs once.

WHAT THIS EXAMPLE IS ABOUT
A100 The Little Prince $18 refund: draft succeeds, crash flag trips, process
"restarts" and continues from the checkpoint.

WHAT IT SOLVES
Fault tolerance: persistence lets you resume unfinished work safely.

KEYWORDS
- Checkpointer: stores graph state after supersteps.
- thread_id: identity of one durable conversation / ticket run.
- Recovery: invoke again with the same thread_id to continue from the last checkpoint.
"""

from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

# Shared across "process restart" in this demo (simulates durable MemorySaver/DB).
SAVER = MemorySaver()
CRASH_ONCE = {"armed": True}


class RefundState(TypedDict):
    order_id: str
    amount: float
    draft: str
    issued: str


def draft_refund(state: RefundState) -> dict:
    return {"draft": f"Draft ${state['amount']} for {state['order_id']} The Little Prince"}


def issue_refund(state: RefundState) -> dict:
    if CRASH_ONCE["armed"]:
        CRASH_ONCE["armed"] = False
        raise RuntimeError("process crashed after draft, before issue")
    return {"issued": f"Issued {state['draft']} (30-day unread print policy)"}


def build_graph():
    g = StateGraph(RefundState)
    g.add_node("draft", draft_refund)
    g.add_node("issue", issue_refund)
    g.add_edge(START, "draft")
    g.add_edge("draft", "issue")
    g.add_edge("issue", END)
    return g.compile(checkpointer=SAVER)


def main() -> None:
    graph = build_graph()
    cfg = {"configurable": {"thread_id": "refund-crash-A100"}}
    seed = {"order_id": "A100", "amount": 18.0, "draft": "", "issued": ""}

    try:
        graph.invoke(seed, cfg)
    except RuntimeError as exc:
        print("Crash:", exc)
        snap = graph.get_state(cfg)
        print("Checkpoint kept draft:", snap.values.get("draft"))

    # Same thread_id -> resume from checkpoint (draft already done).
    done = graph.invoke(None, cfg)
    print("Recovered:", done["issued"])


if __name__ == "__main__":
    print(__doc__)
    main()
