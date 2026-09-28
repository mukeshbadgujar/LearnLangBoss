"""Q63. Time-travel replay of a bad refund draft.

THE PROBLEM
An agent drafted a wrong refund blurb for Order A100 ("ebook, no return"). A
supervisor wants to rewind to the checkpoint *before* that draft and rewrite it
- not a generic time-travel intro.

WHAT WE ARE GOING TO SOLVE
List checkpoints, pick an earlier config, and update_state / re-run from there
so the refund draft is corrected.

WHAT THIS EXAMPLE IS ABOUT
Bad draft checkpoint -> replay from prior step with a corrected draft line for
The Little Prince print return.

WHAT IT SOLVES
Time travel is for fixing a specific bad node output on a known thread.

KEYWORDS
- Time travel: load an earlier checkpoint and continue from there.
- get_state_history: lists prior states for a thread_id.
- update_state: write a corrected channel value before re-invoking.
"""

from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph


class DraftState(TypedDict):
    order_id: str
    draft: str
    final: str


STEP = {"n": 0}


def write_draft(state: DraftState) -> dict:
    STEP["n"] += 1
    if STEP["n"] == 1:
        # First pass: bad draft we will rewind from.
        return {"draft": "BAD: A100 is an ebook - no return"}
    return {"draft": state["draft"]}


def finalize(state: DraftState) -> dict:
    return {"final": f"Customer-facing: {state['draft']}"}


def build_graph(saver: MemorySaver):
    g = StateGraph(DraftState)
    g.add_node("write_draft", write_draft)
    g.add_node("finalize", finalize)
    g.add_edge(START, "write_draft")
    g.add_edge("write_draft", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=saver)


def main() -> None:
    saver = MemorySaver()
    graph = build_graph(saver)
    cfg = {"configurable": {"thread_id": "refund-draft-A100"}}

    bad = graph.invoke(
        {"order_id": "A100", "draft": "", "final": ""}, cfg
    )
    print("Bad final:", bad["final"])

    history = list(graph.get_state_history(cfg))
    # Pick a checkpoint before finalize polluted the customer-facing text.
    earlier = next(h for h in history if h.next == ("finalize",) or h.next == ("write_draft",))
    # Prefer the state still at write_draft / before finalize when possible.
    candidates = [h for h in history if "finalize" not in (h.next or ()) or h.values.get("final") == ""]
    rewind = candidates[-1] if candidates else history[-1]
    print("Rewinding from checkpoint id:", rewind.config["configurable"].get("checkpoint_id"))

    # Correct the bad draft in place, then resume finalize from that checkpoint.
    graph.update_state(
        rewind.config,
        {
            "draft": "GOOD: A100 The Little Prince print - unread, 30-day return, $18"
        },
    )
    fixed = graph.invoke(None, rewind.config)
    print("Replayed final:", fixed["final"])


if __name__ == "__main__":
    print(__doc__)
    main()
