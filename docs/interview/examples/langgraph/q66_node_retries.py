"""Q66. Node retries with RetryPolicy.

THE PROBLEM
The carrier API flakes when looking up Order A100. One blip should not fail the
whole support ticket.

WHAT WE ARE GOING TO SOLVE
Attach RetryPolicy to the shipping node so transient errors retry automatically.

WHAT THIS EXAMPLE IS ABOUT
First call to carrier raises; retry succeeds and returns A100 shipped $18.

WHAT IT SOLVES
Per-node retries without wrapping every call in manual try/except loops.

KEYWORDS
- RetryPolicy: langgraph.types policy (max_attempts, backoff, retry_on).
- Transient error: temporary failure worth retrying.
- Node retry: runtime re-invokes the same node until success or attempts exhausted.
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy


class ShipState(TypedDict):
    order_id: str
    status: str


ATTEMPTS = {"n": 0}


class CarrierGlitch(Exception):
    """Custom exception - default_retry_on retries non-builtin-ish errors."""


def lookup_carrier(state: ShipState) -> dict:
    ATTEMPTS["n"] += 1
    if ATTEMPTS["n"] < 2:
        raise CarrierGlitch("UPS timeout")
    return {
        "status": f"{state['order_id']} The Little Prince shipped $18 (attempt {ATTEMPTS['n']})"
    }


def build_graph():
    g = StateGraph(ShipState)
    g.add_node(
        "lookup_carrier",
        lookup_carrier,
        retry_policy=RetryPolicy(max_attempts=3, initial_interval=0.01, jitter=False),
    )
    g.add_edge(START, "lookup_carrier")
    g.add_edge("lookup_carrier", END)
    return g.compile()


def main() -> None:
    ATTEMPTS["n"] = 0
    out = build_graph().invoke({"order_id": "A100", "status": ""})
    print(out["status"])
    print("Carrier attempts:", ATTEMPTS["n"])


if __name__ == "__main__":
    print(__doc__)
    main()
