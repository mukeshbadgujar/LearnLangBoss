"""Q31. Stop infinite retry

THE PROBLEM
A conditional edge retries payment capture forever when the card declines for A100.

WHAT WE ARE GOING TO SOLVE
Put retries in state; after 3 attempts route to END (escalate). Also note
recursion_limit as a safety net.

WHAT THIS EXAMPLE IS ABOUT
capture_payment fails until retries hit 3, then escalate stops the loop.

WHAT IT SOLVES
Design-level counter + terminal edge; recursion_limit as hard backstop.

KEYWORDS
- Retry counter: state field incremented each attempt.
- Terminal route: edge to END / escalate instead of looping.
- recursion_limit: max supersteps per invoke in config.
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class RetryState(TypedDict):
    order_id: str
    retries: int
    status: str


def capture_payment(state: RetryState) -> dict:
    retries = state["retries"] + 1
    # Always fails in this demo until we stop retrying.
    return {
        "retries": retries,
        "status": f"decline_attempt_{retries}",
        "order_id": "A100",
    }


def escalate(state: RetryState) -> dict:
    return {"status": "escalate_to_human_after_3 - The Little Prince $18"}


def should_retry(state: RetryState) -> str:
    if state["retries"] >= 3:
        return "escalate"
    return "capture_payment"


def build_graph():
    g = StateGraph(RetryState)
    g.add_node("capture_payment", capture_payment)
    g.add_node("escalate", escalate)
    g.add_edge(START, "capture_payment")
    g.add_conditional_edges(
        "capture_payment",
        should_retry,
        {"capture_payment": "capture_payment", "escalate": "escalate"},
    )
    g.add_edge("escalate", END)
    return g.compile()


def main() -> None:
    graph = build_graph()
    # Safety net also available: config={"recursion_limit": 10}
    out = graph.invoke(
        {"order_id": "", "retries": 0, "status": ""},
        {"recursion_limit": 20},
    )
    print("final:", out)


if __name__ == "__main__":
    print(__doc__)
    main()
