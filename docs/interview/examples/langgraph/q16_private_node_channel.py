"""Q16. Private node / subgraph channel vs graph-level state

THE PROBLEM
Refund math uses temp tax scratch vars. If they live on global state, other
nodes can collide on names like `scratch`.

WHAT WE ARE GOING TO SOLVE
Parent keeps order_id / decision; child subgraph keeps private tax_scratch.

WHAT THIS EXAMPLE IS ABOUT
Parent routes A100 into a refund subgraph. Subgraph computes fee privately and
returns only amount_due to the parent.

WHAT IT SOLVES
Graph-level channels are shared; subgraph state can stay isolated.

KEYWORDS
- Shared channel: state key visible to all nodes in that graph.
- Subgraph: compiled graph used as a node, optionally with its own schema.
- Name collision: two modules overwriting the same global key.
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class ParentState(TypedDict):
    order_id: str
    title: str
    amount_due: float
    decision: str


class ChildState(TypedDict):
    order_id: str
    tax_scratch: float
    amount_due: float


def compute_fee(state: ChildState) -> dict:
    # Private scratch - parent never sees tax_scratch.
    tax = 0.0  # print books often untaxed in this toy shop
    return {"tax_scratch": tax, "amount_due": 18.0 + tax}


def build_child():
    g = StateGraph(ChildState)
    g.add_node("compute_fee", compute_fee)
    g.add_edge(START, "compute_fee")
    g.add_edge("compute_fee", END)
    return g.compile()


def approve(state: ParentState) -> dict:
    return {"decision": f"refund ${state['amount_due']:.2f} for {state['title']}"}


def build_parent(child):
    g = StateGraph(ParentState)
    g.add_node("refund_math", child)
    g.add_node("approve", approve)
    g.add_edge(START, "refund_math")
    g.add_edge("refund_math", "approve")
    g.add_edge("approve", END)
    return g.compile()


def main() -> None:
    parent = build_parent(build_child())
    out = parent.invoke(
        {
            "order_id": "A100",
            "title": "The Little Prince",
            "amount_due": 0.0,
            "decision": "",
        }
    )
    print(out)
    print("Parent state has no tax_scratch key - that stayed in the subgraph.")


if __name__ == "__main__":
    print(__doc__)
    main()
