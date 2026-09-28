"""Q28. Tool returns Command

THE PROBLEM
A lookup tool finds the order and should both write state and send the graph to
the refund node - without a separate router guessing afterward.

WHAT WE ARE GOING TO SOLVE
A tool-shaped function returns Command(update=..., goto=...) and is used as a node
(same mechanism tools use when the agent runtime allows Command returns).

WHAT THIS EXAMPLE IS ABOUT
lookup_order tool loads A100 The Little Prince $18 and Commands goto apply_policy.

WHAT IT SOLVES
Tools can return Command for update + route - useful for tool-triggered handoffs.

KEYWORDS
- Tool: callable the agent/graph invokes for side effects or data.
- Command from tool: update state and goto after the tool finishes.
- Handoff: jump to a specialist node based on tool results.
"""

from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command


class ToolState(TypedDict):
    query: str
    order_id: str
    amount: float
    title: str
    result: str


def lookup_order_tool(state: ToolState) -> Command[Literal["apply_policy", "not_found"]]:
    """Behaves like a tool: fetch order, then Command to the next node."""
    q = state["query"].lower()
    if "a100" in q or "little prince" in q:
        return Command(
            update={
                "order_id": "A100",
                "amount": 18.0,
                "title": "The Little Prince",
            },
            goto="apply_policy",
        )
    return Command(update={"order_id": ""}, goto="not_found")


def apply_policy(state: ToolState) -> dict:
    return {
        "result": (
            f"Refund ${state['amount']:.2f} for {state['title']} "
            "if unread print within 30 days."
        )
    }


def not_found(state: ToolState) -> dict:
    return {"result": "Order not found"}


def build_graph():
    g = StateGraph(ToolState)
    g.add_node("lookup_order_tool", lookup_order_tool)
    g.add_node("apply_policy", apply_policy)
    g.add_node("not_found", not_found)
    g.add_edge(START, "lookup_order_tool")
    g.add_edge("apply_policy", END)
    g.add_edge("not_found", END)
    return g.compile()


def main() -> None:
    graph = build_graph()
    out = graph.invoke(
        {
            "query": "refund A100",
            "order_id": "",
            "amount": 0.0,
            "title": "",
            "result": "",
        }
    )
    print(out["result"])


if __name__ == "__main__":
    print(__doc__)
    main()
