"""Q50. LangGraph vs CrewAI vs AutoGen - one-line differentiators.

THE PROBLEM
Someone proposes rewriting the bookstore desk in CrewAI or AutoGen without
saying what control each framework actually gives you.

WHAT WE ARE GOING TO SOLVE
Run a real small LangGraph graph for A100 triage, and document CrewAI / AutoGen
in comments only (no imports).

WHAT THIS EXAMPLE IS ABOUT
Explicit graph control for shipping vs refund routing on Order A100.

WHAT IT SOLVES
One-liners: LangGraph = explicit state/flow; CrewAI = role crews fast;
AutoGen = conversation-driven multi-agent experiments.

KEYWORDS
- LangGraph: low-level explicit graph and state control (persistence, HITL).
- CrewAI: high-level role-based crews (not imported here).
- AutoGen: high-level conversation-driven multi-agent patterns (not imported here).
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

# CrewAI (not imported): define Agent roles + Crew; framework owns most control flow.
# AutoGen (not imported): agents chat with each other; great for research dialogues.
# LangGraph (this file): you own nodes, edges, state, and checkpoints.


class DeskState(TypedDict):
    text: str
    path: str
    answer: str


def route(state: DeskState) -> str:
    if "refund" in state["text"].lower():
        return "refund"
    return "shipping"


def shipping_node(state: DeskState) -> dict:
    return {
        "path": "shipping",
        "answer": "A100 The Little Prince: shipped $18",
    }


def refund_node(state: DeskState) -> dict:
    return {
        "path": "refund",
        "answer": "A100 refund: unread print books, 30-day return, $18",
    }


def build_graph():
    g = StateGraph(DeskState)
    g.add_node("shipping", shipping_node)
    g.add_node("refund", refund_node)
    g.add_conditional_edges(START, route, {"shipping": "shipping", "refund": "refund"})
    g.add_edge("shipping", END)
    g.add_edge("refund", END)
    return g.compile()


def main() -> None:
    graph = build_graph()
    print(graph.invoke({"text": "refund for A100", "path": "", "answer": ""}))
    print(graph.invoke({"text": "track A100", "path": "", "answer": ""}))
    print("Pick LangGraph when state and flow are the hard part.")


if __name__ == "__main__":
    print(__doc__)
    main()
