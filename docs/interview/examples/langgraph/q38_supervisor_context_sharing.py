"""Q38. Supervisor context sharing vs isolation.

THE PROBLEM
A bookstore supervisor routes Order A100 tickets to shipping and refund specialists,
but the refund specialist keeps seeing the shipping agent's scratch notes.

WHAT WE ARE GOING TO SOLVE
Show shared state (same messages channel) versus a private subgraph that only
returns a short result.

WHAT THIS EXAMPLE IS ABOUT
Order A100 - The Little Prince, shipped, $18. Supervisor asks shipping status,
then refund policy, without leaking shipping tool notes into the refund path.

WHAT IT SOLVES
Shared messages work for a short handoff. Private subgraph state keeps specialist
scratch work out of the other agent's context.

KEYWORDS
- Supervisor: a node that decides which specialist runs next.
- Shared state: every specialist reads and writes the same channels (often messages).
- Subgraph isolation: a nested graph with its own schema; only selected fields cross the boundary.
- Handoff: routing control from one agent (or node) to another.
"""

from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage


class DeskState(TypedDict):
    messages: Annotated[list, add_messages]
    order_id: str
    shipping_note: str
    refund_note: str


class ShippingPrivate(TypedDict):
    order_id: str
    scratch: str
    result: str


def supervisor(state: DeskState) -> dict:
    # Shared channel: every specialist will see this human turn.
    return {"messages": [HumanMessage(content=f"Help with order {state['order_id']}")]}


def shipping_shared(state: DeskState) -> dict:
    # Writes into the shared messages list - refund will see this too.
    return {
        "shipping_note": "A100 left warehouse yesterday",
        "messages": [AIMessage(content="[shipping scratch] carrier=UPS, label=X9")],
    }


def build_shipping_subgraph():
    def lookup(state: ShippingPrivate) -> dict:
        return {"scratch": "carrier=UPS label=X9", "result": f"{state['order_id']} shipped"}

    def clean(state: ShippingPrivate) -> dict:
        # Only `result` is meant to leave the private graph.
        return {"result": state["result"]}

    g = StateGraph(ShippingPrivate)
    g.add_node("lookup", lookup)
    g.add_node("clean", clean)
    g.add_edge(START, "lookup")
    g.add_edge("lookup", "clean")
    g.add_edge("clean", END)
    return g.compile()


_SHIP = build_shipping_subgraph()


def shipping_isolated(state: DeskState) -> dict:
    out = _SHIP.invoke({"order_id": state["order_id"], "scratch": "", "result": ""})
    # Scrub: push only the short result into shared desk state.
    return {
        "shipping_note": out["result"],
        "messages": [AIMessage(content=out["result"])],
    }


def refund_specialist(state: DeskState) -> dict:
    prior = [m.content for m in state["messages"]]
    return {
        "refund_note": "Unread print books: 30-day return. Amount $18.",
        "messages": [AIMessage(content=f"Refund sees {len(prior)} prior messages")],
    }


def build_shared() -> object:
    g = StateGraph(DeskState)
    g.add_node("supervisor", supervisor)
    g.add_node("shipping", shipping_shared)
    g.add_node("refund", refund_specialist)
    g.add_edge(START, "supervisor")
    g.add_edge("supervisor", "shipping")
    g.add_edge("shipping", "refund")
    g.add_edge("refund", END)
    return g.compile()


def build_isolated() -> object:
    g = StateGraph(DeskState)
    g.add_node("supervisor", supervisor)
    g.add_node("shipping", shipping_isolated)
    g.add_node("refund", refund_specialist)
    g.add_edge(START, "supervisor")
    g.add_edge("supervisor", "shipping")
    g.add_edge("shipping", "refund")
    g.add_edge("refund", END)
    return g.compile()


def main() -> None:
    seed = {
        "messages": [],
        "order_id": "A100",
        "shipping_note": "",
        "refund_note": "",
    }
    shared = build_shared().invoke(seed)
    print("SHARED shipping msgs leaked to refund:")
    for m in shared["messages"]:
        print(" ", m.content)

    isolated = build_isolated().invoke(seed)
    print("ISOLATED - refund only sees cleaned shipping result:")
    for m in isolated["messages"]:
        print(" ", m.content)


if __name__ == "__main__":
    print(__doc__)
    main()
