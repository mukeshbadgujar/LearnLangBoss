"""Q25. Conditional edges vs Command

THE PROBLEM
Routing can live in a separate function (add_conditional_edges) or inside the
node via Command. Interviewers ask when each fits.

WHAT WE ARE GOING TO SOLVE
Same A100 triage twice: once with an external router, once with Command(goto=...).

WHAT THIS EXAMPLE IS ABOUT
Refund vs shipping for The Little Prince. First graph uses conditional edges;
second returns Command with update + goto from the node.

WHAT IT SOLVES
Conditional edges = external router. Command = node updates state and routes
in one return (and can cross subgraph boundaries - see q27).

KEYWORDS
- add_conditional_edges: register a routing function outside the node.
- Command: return value that carries update and/or goto.
- Dynamic routing: next node chosen at runtime from inside the node.
"""

from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command


class RouteState(TypedDict):
    message: str
    team: str
    result: str


def classify_only(state: RouteState) -> dict:
    if "refund" in state["message"].lower():
        return {"team": "refunds"}
    return {"team": "shipping"}


def external_router(state: RouteState) -> str:
    return "refund_desk" if state["team"] == "refunds" else "ship_desk"


def refund_desk(state: RouteState) -> dict:
    return {"result": "A100 refund path - unread print 30 days"}


def ship_desk(state: RouteState) -> dict:
    return {"result": "A100 shipping path"}


def classify_command(state: RouteState) -> Command[Literal["refund_desk", "ship_desk"]]:
    if "refund" in state["message"].lower():
        return Command(update={"team": "refunds"}, goto="refund_desk")
    return Command(update={"team": "shipping"}, goto="ship_desk")


def build_conditional():
    g = StateGraph(RouteState)
    g.add_node("classify_only", classify_only)
    g.add_node("refund_desk", refund_desk)
    g.add_node("ship_desk", ship_desk)
    g.add_edge(START, "classify_only")
    g.add_conditional_edges(
        "classify_only",
        external_router,
        {"refund_desk": "refund_desk", "ship_desk": "ship_desk"},
    )
    g.add_edge("refund_desk", END)
    g.add_edge("ship_desk", END)
    return g.compile()


def build_command():
    g = StateGraph(RouteState)
    g.add_node("classify_command", classify_command)
    g.add_node("refund_desk", refund_desk)
    g.add_node("ship_desk", ship_desk)
    g.add_edge(START, "classify_command")
    g.add_edge("refund_desk", END)
    g.add_edge("ship_desk", END)
    return g.compile()


def main() -> None:
    msg = {
        "message": "Refund The Little Prince on A100",
        "team": "",
        "result": "",
    }
    print("conditional:", build_conditional().invoke(msg)["result"])
    print("command:    ", build_command().invoke(msg)["result"])


if __name__ == "__main__":
    print(__doc__)
    main()
