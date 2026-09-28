"""Q64. Why nest subgraphs.

THE PROBLEM
Shipping logic for Order A100 is a mini-workflow (carrier -> ETA -> format) but
lives inline in the parent desk graph, making the parent unreadable.

WHAT WE ARE GOING TO SOLVE
Nest a compiled shipping subgraph and call it from one parent node.

WHAT THIS EXAMPLE IS ABOUT
Parent desk graph delegates shipping details to a subgraph; parent only sees
the short status string.

WHAT IT SOLVES
Encapsulation: reusable workflows, cleaner parent topology, optional private state.

KEYWORDS
- Subgraph: a compiled graph used as a node inside another graph.
- Nesting: parent invokes child; child can hide internal channels.
- Encapsulation: keep specialist multi-step logic out of the supervisor graph.
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class ShippingSub(TypedDict):
    order_id: str
    carrier: str
    eta: str
    status: str


def carrier_lookup(state: ShippingSub) -> dict:
    return {"carrier": "UPS"}


def eta_lookup(state: ShippingSub) -> dict:
    return {"eta": "2 days"}


def format_status(state: ShippingSub) -> dict:
    return {
        "status": (
            f"{state['order_id']} The Little Prince via {state['carrier']}, "
            f"ETA {state['eta']}, $18"
        )
    }


def build_shipping_subgraph():
    g = StateGraph(ShippingSub)
    g.add_node("carrier", carrier_lookup)
    g.add_node("eta", eta_lookup)
    g.add_node("format", format_status)
    g.add_edge(START, "carrier")
    g.add_edge("carrier", "eta")
    g.add_edge("eta", "format")
    g.add_edge("format", END)
    return g.compile()


_SHIP = build_shipping_subgraph()


class DeskState(TypedDict):
    order_id: str
    shipping_status: str
    policy: str


def run_shipping(state: DeskState) -> dict:
    out = _SHIP.invoke(
        {
            "order_id": state["order_id"],
            "carrier": "",
            "eta": "",
            "status": "",
        }
    )
    return {"shipping_status": out["status"]}


def attach_policy(state: DeskState) -> dict:
    return {"policy": "Unread print books: 30-day return"}


def build_parent():
    g = StateGraph(DeskState)
    g.add_node("shipping", run_shipping)
    g.add_node("policy", attach_policy)
    g.add_edge(START, "shipping")
    g.add_edge("shipping", "policy")
    g.add_edge("policy", END)
    return g.compile()


def main() -> None:
    out = build_parent().invoke(
        {"order_id": "A100", "shipping_status": "", "policy": ""}
    )
    print(out["shipping_status"])
    print(out["policy"])


if __name__ == "__main__":
    print(__doc__)
    main()
