"""Q52. Node and edge types in LangGraph.

THE PROBLEM
Newcomers treat every bookstore step as "just a function" and miss that edges
can be fixed, conditional, or fan-out Sends.

WHAT WE ARE GOING TO SOLVE
Name the main node kinds (plain function, tool-ish worker) and edge kinds
(normal, conditional) on one A100 desk graph.

WHAT THIS EXAMPLE IS ABOUT
Triage -> shipping or refund -> END, with a normal edge into triage and
conditional edges out.

WHAT IT SOLVES
Clear vocabulary: nodes do work; edges decide what runs next.

KEYWORDS
- Node: unit of work that returns a partial state update.
- Normal edge: always go from A to B.
- Conditional edge: a routing function chooses the next node name.
- START / END: graph entry and terminal sentinels.
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class DeskState(TypedDict):
    text: str
    specialist: str
    note: str


def triage(state: DeskState) -> dict:
    return {"note": f"triaged: {state['text'][:40]}"}


def pick(state: DeskState) -> str:
    return "refund" if "return" in state["text"].lower() else "shipping"


def shipping(state: DeskState) -> dict:
    return {"specialist": "shipping", "note": "A100 shipped $18"}


def refund(state: DeskState) -> dict:
    return {"specialist": "refund", "note": "30-day unread print return ok"}


def build_graph():
    g = StateGraph(DeskState)
    g.add_node("triage", triage)  # plain Python node
    g.add_node("shipping", shipping)
    g.add_node("refund", refund)
    g.add_edge(START, "triage")  # normal edge
    g.add_conditional_edges(  # conditional edges
        "triage",
        pick,
        {"shipping": "shipping", "refund": "refund"},
    )
    g.add_edge("shipping", END)
    g.add_edge("refund", END)
    return g.compile()


def main() -> None:
    graph = build_graph()
    print(graph.invoke({"text": "return The Little Prince A100", "specialist": "", "note": ""}))
    print(graph.invoke({"text": "track shipment A100", "specialist": "", "note": ""}))


if __name__ == "__main__":
    print(__doc__)
    main()
