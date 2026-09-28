"""Q8. Minimal two-node graph

THE PROBLEM
You need the shortest real LangGraph: define state, two nodes, compile, invoke.

WHAT WE ARE GOING TO SOLVE
Triage then resolve a bookstore ticket for A100 without extra APIs.

WHAT THIS EXAMPLE IS ABOUT
Ticket text mentions The Little Prince refund. triage writes notes; resolve
appends the $18 policy outcome.

WHAT IT SOLVES
Minimal StateGraph pattern: TypedDict -> add_node -> add_edge -> compile -> invoke.

KEYWORDS
- START / END: built-in markers for graph entry and exit.
- compile(): validates and builds the runnable graph.
- invoke(): runs the graph once with an input dict.
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    ticket: str
    triage_notes: str


def triage(state: State) -> dict:
    return {"triage_notes": f"Reviewed refund ticket: {state['ticket']}"}


def resolve(state: State) -> dict:
    return {
        "triage_notes": state["triage_notes"]
        + " -> approve $18 if unread print within 30 days (A100)."
    }


def build_graph():
    builder = StateGraph(State)
    builder.add_node("triage", triage)
    builder.add_node("resolve", resolve)
    builder.add_edge(START, "triage")
    builder.add_edge("triage", "resolve")
    builder.add_edge("resolve", END)
    return builder.compile()


def main() -> None:
    graph = build_graph()
    result = graph.invoke(
        {
            "ticket": "Refund The Little Prince on order A100",
            "triage_notes": "",
        }
    )
    print(result["triage_notes"])


if __name__ == "__main__":
    print(__doc__)
    main()
