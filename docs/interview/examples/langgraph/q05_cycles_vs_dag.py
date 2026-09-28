"""Q5. Cycles vs DAG (Airflow-style)

THE PROBLEM
Nightly CSV sync is a fixed DAG. Refund eligibility for A100 is not - the desk
may re-ask 'is it unread?' until the answer is clear. Airflow forces you to
pre-list every loop; agents cannot.

WHAT WE ARE GOING TO SOLVE
Build a LangGraph cycle that re-asks until unread is confirmed or max tries.

WHAT THIS EXAMPLE IS ABOUT
Customer claims The Little Prince is unread. First two passes leave unread
unknown; third confirms. Cycle edge goes back to ask_again.

WHAT IT SOLVES
DAG tools fit static pipelines. LangGraph cycles fit runtime agent decisions.

KEYWORDS
- DAG: directed acyclic graph - no cycles (Airflow-style pipelines).
- Cycle: edge back to a prior node until a condition is met.
- Pre-enumeration: listing every failure path before runtime (impractical for LLMs).
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class CycleState(TypedDict):
    order_id: str
    unread_known: bool
    unread: bool
    tries: int
    status: str


def ask_again(state: CycleState) -> dict:
    tries = state["tries"] + 1
    # Simulate customer clarifying on the third ask.
    if tries >= 3:
        return {
            "tries": tries,
            "unread_known": True,
            "unread": True,
            "status": "confirmed unread",
        }
    return {
        "tries": tries,
        "unread_known": False,
        "status": f"ask #{tries}: is The Little Prince unread?",
    }


def decide(state: CycleState) -> dict:
    if state["unread"]:
        return {"status": f"A100 eligible under 30-day unread print policy"}
    return {"status": "deny - not unread"}


def route(state: CycleState) -> str:
    if not state["unread_known"]:
        return "ask_again"
    return "decide"


def build_graph():
    g = StateGraph(CycleState)
    g.add_node("ask_again", ask_again)
    g.add_node("decide", decide)
    g.add_edge(START, "ask_again")
    g.add_conditional_edges(
        "ask_again",
        route,
        {"ask_again": "ask_again", "decide": "decide"},
    )
    g.add_edge("decide", END)
    return g.compile()


def main() -> None:
    graph = build_graph()
    out = graph.invoke(
        {
            "order_id": "A100",
            "unread_known": False,
            "unread": False,
            "tries": 0,
            "status": "",
        }
    )
    print("status:", out["status"])
    print("tries:", out["tries"])


if __name__ == "__main__":
    print(__doc__)
    main()
