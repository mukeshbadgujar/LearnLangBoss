"""Q29. Send API

THE PROBLEM
A normal edge always targets fixed nodes with shared state. You need N parallel
workers for N line items, and N is only known at runtime.

WHAT WE ARE GOING TO SOLVE
Router returns list[Send] to spawn one worker per item.

WHAT THIS EXAMPLE IS ABOUT
Cart has The Little Prince plus a bookmark. Fan-out one price_check Send per SKU.

WHAT IT SOLVES
Send = dynamic parallel invocations with custom per-call input; unlike a static edge.

KEYWORDS
- Send: object naming a node and the state/input for that invocation.
- Fan-out: N parallel node runs decided at runtime.
- Map-reduce: map with Send, reduce with a reducer-backed channel.
"""

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send


class MapState(TypedDict):
    items: list[str]
    prices: Annotated[list[str], operator.add]


def plan_items(state: MapState) -> dict:
    return {
        "items": ["The Little Prince", "Bookmark - rose"],
    }


def fan_out(state: MapState) -> list[Send]:
    return [Send("price_check", {"item": name}) for name in state["items"]]


def price_check(payload: dict) -> dict:
    price = "18.00" if "Little Prince" in payload["item"] else "3.00"
    return {"prices": [f"{payload['item']}=${price}"]}


def build_graph():
    g = StateGraph(MapState)
    g.add_node("plan_items", plan_items)
    g.add_node("price_check", price_check)
    g.add_edge(START, "plan_items")
    g.add_conditional_edges("plan_items", fan_out, ["price_check"])
    g.add_edge("price_check", END)
    return g.compile()


def main() -> None:
    out = build_graph().invoke({"items": [], "prices": []})
    print("prices:", out["prices"])


if __name__ == "__main__":
    print(__doc__)
    main()
