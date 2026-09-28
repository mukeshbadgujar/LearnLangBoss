"""Q4. What is a superstep?

THE PROBLEM
Support wants fraud, inventory, and policy checks on refund A100 at once.
Without turn-based merging, parallel writes fight over state.

WHAT WE ARE GOING TO SOLVE
Show fan-out in one superstep: three nodes run, then a reducer merges notes
before the next step (approve) starts.

WHAT THIS EXAMPLE IS ABOUT
Order A100 The Little Prince $18. Parallel nodes append check notes; after the
superstep completes, approve reads the combined list.

WHAT IT SOLVES
A superstep = one round where ready nodes run; next round waits until all finish
and reducers merge updates into one checkpoint.

KEYWORDS
- Superstep: one execution round of concurrent ready nodes.
- Fan-out: start several nodes in the same superstep.
- Reducer: rule that combines parallel updates to one key.
"""

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph


class CheckState(TypedDict):
    order_id: str
    title: str
    notes: Annotated[list[str], operator.add]
    verdict: str


def fraud_check(state: CheckState) -> dict:
    return {"notes": [f"fraud:ok for {state['order_id']}"]}


def inventory_check(state: CheckState) -> dict:
    return {"notes": [f"inventory: {state['title']} marked returnable"]}


def policy_check(state: CheckState) -> dict:
    return {"notes": ["policy: unread print, within 30 days"]}


def approve(state: CheckState) -> dict:
    return {"verdict": "approve $18 - " + "; ".join(state["notes"])}


def build_graph():
    g = StateGraph(CheckState)
    g.add_node("fraud_check", fraud_check)
    g.add_node("inventory_check", inventory_check)
    g.add_node("policy_check", policy_check)
    g.add_node("approve", approve)
    # Fan-out: START fans to three nodes in one superstep.
    g.add_edge(START, "fraud_check")
    g.add_edge(START, "inventory_check")
    g.add_edge(START, "policy_check")
    # Fan-in: all three must finish before approve (next superstep).
    g.add_edge("fraud_check", "approve")
    g.add_edge("inventory_check", "approve")
    g.add_edge("policy_check", "approve")
    g.add_edge("approve", END)
    return g.compile()


def main() -> None:
    graph = build_graph()
    out = graph.invoke(
        {
            "order_id": "A100",
            "title": "The Little Prince",
            "notes": [],
            "verdict": "",
        }
    )
    print("notes after superstep:", out["notes"])
    print("verdict:", out["verdict"])


if __name__ == "__main__":
    print(__doc__)
    main()
