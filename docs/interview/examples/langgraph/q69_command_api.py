"""Q69. Command API for routing and state updates together.

THE PROBLEM
After triage, the bookstore graph must both set `queue` and jump to the right
desk in one return value.

WHAT WE ARE GOING TO SOLVE
Return Command(update=..., goto=...) from a node instead of a plain dict plus
a separate conditional edge.

WHAT THIS EXAMPLE IS ABOUT
Triage Order A100 return vs tracking with Command.

WHAT IT SOLVES
Dynamic routing and state update in a single Command object.

KEYWORDS
- Command: LangGraph object carrying state update and/or goto target.
- goto: next node name (or END) chosen at runtime.
- update: partial state dict merged like a normal node return.
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command


class DeskState(TypedDict):
    text: str
    queue: str
    answer: str


def triage(state: DeskState) -> Command:
    if "return" in state["text"].lower() or "refund" in state["text"].lower():
        return Command(update={"queue": "refunds"}, goto="refunds")
    return Command(update={"queue": "shipping"}, goto="shipping")


def refunds(state: DeskState) -> dict:
    return {"answer": f"[{state['queue']}] A100 $18 - 30-day unread print return"}


def shipping(state: DeskState) -> dict:
    return {"answer": f"[{state['queue']}] A100 The Little Prince shipped"}


def build_graph():
    g = StateGraph(DeskState)
    g.add_node("triage", triage)
    g.add_node("refunds", refunds)
    g.add_node("shipping", shipping)
    g.add_edge(START, "triage")
    # Command.goto targets must be declared as nodes; END edges from leaves.
    g.add_edge("refunds", END)
    g.add_edge("shipping", END)
    return g.compile()


def main() -> None:
    graph = build_graph()
    refund = graph.invoke({"text": "return A100", "queue": "", "answer": ""})
    ship = graph.invoke({"text": "track A100", "queue": "", "answer": ""})
    print(refund)
    print(ship)
    print("Command set queue and goto in one return from triage.")


if __name__ == "__main__":
    print(__doc__)
    main()
