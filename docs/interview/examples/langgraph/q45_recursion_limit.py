"""Q45. recursion_limit and runaway loops.

THE PROBLEM
A buggy "check again" edge on a shipping status poll can loop forever and burn
API budget before anyone notices.

WHAT WE ARE GOING TO SOLVE
Set recursion_limit so LangGraph halts after N supersteps instead of spinning.

WHAT THIS EXAMPLE IS ABOUT
Order A100 status poll that keeps routing back to itself. With a low
recursion_limit the run fails loudly.

WHAT IT SOLVES
A production guardrail against infinite conditional loops.

KEYWORDS
- recursion_limit: max supersteps per invoke; set in config or at compile time.
- Superstep: one parallel wave of node execution in the graph runtime.
- Runaway loop: a conditional edge that never reaches END.
"""

from typing import TypedDict

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph


class PollState(TypedDict):
    order_id: str
    checks: int
    status: str


def check_shipping(state: PollState) -> dict:
    # Intentionally never flips to "delivered" - simulates a bad exit condition.
    return {"checks": state["checks"] + 1, "status": "in_transit"}


def route(state: PollState) -> str:
    if state["status"] == "delivered":
        return "done"
    return "again"  # buggy: always comes back


def finish(state: PollState) -> dict:
    return {"status": f"done after {state['checks']} checks"}


def build_graph():
    g = StateGraph(PollState)
    g.add_node("check", check_shipping)
    g.add_node("finish", finish)
    g.add_edge(START, "check")
    g.add_conditional_edges(
        "check",
        route,
        {"again": "check", "done": "finish"},
    )
    g.add_edge("finish", END)
    return g.compile()


def main() -> None:
    graph = build_graph()
    seed = {"order_id": "A100", "checks": 0, "status": ""}
    try:
        graph.invoke(seed, config={"recursion_limit": 8})
    except GraphRecursionError as exc:
        print("Guarded: recursion_limit stopped the A100 poll loop")
        print(type(exc).__name__)


if __name__ == "__main__":
    print(__doc__)
    main()
