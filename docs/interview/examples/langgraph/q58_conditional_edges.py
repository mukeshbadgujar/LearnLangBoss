"""Q58. Nodes, edges, and conditional edges.

THE PROBLEM
Support tickets for Order A100 need different desks; a single linear chain
always hits refund even for tracking questions.

WHAT WE ARE GOING TO SOLVE
Use a routing function on a conditional edge to choose shipping vs refund.

WHAT THIS EXAMPLE IS ABOUT
Bookstore triage graph: text -> conditional edge -> specialist node.

WHAT IT SOLVES
Conditional edges encode business routing without stuffing if/else into one
giant node.

KEYWORDS
- Node: performs work and returns state updates.
- Edge: connection that queues the next node.
- Conditional edge: path chosen by a function of current state.
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class TicketState(TypedDict):
    text: str
    desk: str
    reply: str


def classify(state: TicketState) -> dict:
    return {"desk": "pending"}


def choose_desk(state: TicketState) -> str:
    t = state["text"].lower()
    if "refund" in t or "return" in t:
        return "refund_desk"
    return "shipping_desk"


def refund_desk(state: TicketState) -> dict:
    return {
        "desk": "refund",
        "reply": "A100: eligible under unread print 30-day policy ($18)",
    }


def shipping_desk(state: TicketState) -> dict:
    return {
        "desk": "shipping",
        "reply": "A100 The Little Prince: shipped",
    }


def build_graph():
    g = StateGraph(TicketState)
    g.add_node("classify", classify)
    g.add_node("refund_desk", refund_desk)
    g.add_node("shipping_desk", shipping_desk)
    g.add_edge(START, "classify")
    g.add_conditional_edges(
        "classify",
        choose_desk,
        {"refund_desk": "refund_desk", "shipping_desk": "shipping_desk"},
    )
    g.add_edge("refund_desk", END)
    g.add_edge("shipping_desk", END)
    return g.compile()


def main() -> None:
    graph = build_graph()
    print(graph.invoke({"text": "return A100 please", "desk": "", "reply": ""}))
    print(graph.invoke({"text": "where is A100", "desk": "", "reply": ""}))


if __name__ == "__main__":
    print(__doc__)
    main()
