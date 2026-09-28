"""Q59. Why reducers exist.

THE PROBLEM
Parallel bookstore workers both write `events`; last writer silently drops the
other worker's event.

WHAT WE ARE GOING TO SOLVE
Explain last-write-wins vs Annotated[..., operator.add] with a fan-out that
would otherwise lose data.

WHAT THIS EXAMPLE IS ABOUT
Shipping and billing workers emit events for Order A100; reducer keeps both.

WHAT IT SOLVES
Reducers make parallel updates safe and intentional.

KEYWORDS
- Last-write-wins: default channel behavior without a reducer.
- operator.add: common list/int reducer that appends or sums.
- Channel: one named field in graph state with its own merge rule.
"""

from typing import Annotated, TypedDict
import operator

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send


class EventState(TypedDict):
    order_id: str
    workers: list[str]
    events: Annotated[list[str], operator.add]


def fan(state: EventState):
    return [Send("emit", {"name": w, "order_id": state["order_id"]}) for w in state["workers"]]


def emit(state: dict) -> dict:
    return {"events": [f"{state['name']}:{state['order_id']}"]}


def sink(state: EventState) -> dict:
    return {}


def build_graph():
    g = StateGraph(EventState)
    g.add_node("emit", emit)
    g.add_node("sink", sink)
    g.add_conditional_edges(START, fan, ["emit"])
    g.add_edge("emit", "sink")
    g.add_edge("sink", END)
    return g.compile()


def main() -> None:
    out = build_graph().invoke(
        {
            "order_id": "A100",
            "workers": ["shipping", "billing"],
            "events": [],
        }
    )
    print("Both events kept:", sorted(out["events"]))
    print("Without a reducer, one of these would be overwritten.")


if __name__ == "__main__":
    print(__doc__)
    main()
