"""Q1. What is LangGraph?

THE PROBLEM
A plain prompt chain can answer one refund FAQ, then stops. Order A100 needs
classify check, then issue or deny - and if the check fails, try again. A
straight chain cannot loop or pause safely.

WHAT WE ARE GOING TO SOLVE
Show LangGraph as a graph of nodes and edges over shared state, with a cycle
when policy re-check is needed.

WHAT THIS EXAMPLE IS ABOUT
Shopper wants a refund for order A100 (The Little Prince, shipped, $18).
Policy: unread print books, 30 days. One node checks policy; if unclear, loop
back once with a note, then decide.

WHAT IT SOLVES
You see state, nodes, edges, and a cycle - the pieces plain chaining lacks.

KEYWORDS
- LangGraph: library that runs AI workflows as a graph of Python nodes and edges.
- State: shared dict every node reads and partially updates.
- Cycle: an edge that sends control back to an earlier node.
- Node: a plain Python function that does one step of work.
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class RefundState(TypedDict):
    order_id: str
    title: str
    amount: float
    days_since_delivery: int
    unread: bool
    checks: int
    decision: str
    note: str


def check_policy(state: RefundState) -> dict:
    """Apply the bookstore return policy to A100."""
    checks = state["checks"] + 1
    if state["days_since_delivery"] > 30:
        return {"checks": checks, "decision": "deny", "note": "Outside 30-day window."}
    if not state["unread"]:
        return {"checks": checks, "decision": "deny", "note": "Book shows as read/used."}
    if checks == 1 and state["note"] == "":
        # First pass left a blank note - loop once to re-check with context.
        return {
            "checks": checks,
            "decision": "recheck",
            "note": "Need confirm unread flag for The Little Prince.",
        }
    return {
        "checks": checks,
        "decision": "approve",
        "note": f"Refund ${state['amount']:.2f} for {state['title']}.",
    }


def route_after_check(state: RefundState) -> str:
    if state["decision"] == "recheck" and state["checks"] < 2:
        return "check_policy"
    return "finalize"


def finalize(state: RefundState) -> dict:
    return {"decision": state["decision"]}


def build_graph():
    g = StateGraph(RefundState)
    g.add_node("check_policy", check_policy)
    g.add_node("finalize", finalize)
    g.add_edge(START, "check_policy")
    g.add_conditional_edges(
        "check_policy",
        route_after_check,
        {"check_policy": "check_policy", "finalize": "finalize"},
    )
    g.add_edge("finalize", END)
    return g.compile()


def main() -> None:
    graph = build_graph()
    result = graph.invoke(
        {
            "order_id": "A100",
            "title": "The Little Prince",
            "amount": 18.0,
            "days_since_delivery": 12,
            "unread": True,
            "checks": 0,
            "decision": "",
            "note": "",
        }
    )
    print("decision:", result["decision"])
    print("note:", result["note"])
    print("checks (cycle ran):", result["checks"])


if __name__ == "__main__":
    print(__doc__)
    main()
