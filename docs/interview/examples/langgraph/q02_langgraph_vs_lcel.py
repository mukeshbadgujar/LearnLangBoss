"""Q2. LangGraph vs LCEL

THE PROBLEM
An LCEL-style pipeline for shipping questions only moves forward. When tracking
for A100 returns 'carrier delay', the chain has no way to ask again or branch
back - it just ends with an error string.

WHAT WE ARE GOING TO SOLVE
Contrast a one-way pipeline function with a small LangGraph that can cycle when
tracking is incomplete.

WHAT THIS EXAMPLE IS ABOUT
Customer asks where order A100 (The Little Prince) is. First lookup says
'delayed'; graph loops to enrich the note, then answers. The LCEL-shaped
helper cannot loop.

WHAT IT SOLVES
LCEL = forward-only composition. LangGraph = stateful graph that may cycle.

KEYWORDS
- LCEL: LangChain Expression Language - pipe steps with | in one direction.
- DAG: directed acyclic graph - no loops back.
- Stateful: shared state remembered across steps.
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


def lcel_style_pipeline(order_id: str, status: str) -> str:
    """Stand-in for an LCEL chain: format -> 'call' -> answer. No cycle."""
    formatted = f"Track {order_id}"
    if status == "delayed":
        return f"{formatted}: carrier delay - chain ends here (no retry)."
    return f"{formatted}: out for delivery."


class TrackState(TypedDict):
    order_id: str
    title: str
    status: str
    attempts: int
    answer: str


def lookup_carrier(state: TrackState) -> dict:
    attempts = state["attempts"] + 1
    if state["status"] == "delayed":
        # First pass: note the delay, loop once for a real ETA.
        return {
            "attempts": attempts,
            "status": "need_eta",
            "answer": "Delay noted; fetching ETA...",
        }
    if state["status"] == "need_eta":
        return {
            "attempts": attempts,
            "status": "done",
            "answer": f"{state['title']} (A100): ETA tomorrow after delay.",
        }
    return {
        "attempts": attempts,
        "status": "done",
        "answer": f"{state['title']}: on the way.",
    }


def need_another_lookup(state: TrackState) -> str:
    if state["status"] == "done":
        return END
    return "lookup_carrier"


def build_graph():
    g = StateGraph[TrackState, None, TrackState, TrackState](TrackState)
    g.add_node("lookup_carrier", lookup_carrier)
    g.add_edge(START, "lookup_carrier")
    g.add_conditional_edges("lookup_carrier", need_another_lookup)
    return g.compile()


def main() -> None:
    print("LCEL-style:", lcel_style_pipeline("A100", "delayed"))
    graph = build_graph()
    out = graph.invoke(
        {
            "order_id": "A100",
            "title": "The Little Prince",
            "status": "delayed",
            "attempts": 0,
            "answer": "",
        }
    )
    print("LangGraph:", out["answer"], "| attempts:", out["attempts"])


if __name__ == "__main__":
    print(__doc__)
    main()
