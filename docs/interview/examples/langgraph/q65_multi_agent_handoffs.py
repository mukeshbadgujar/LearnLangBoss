"""Q65. Multi-agent handoffs.

THE PROBLEM
A shipping specialist finishes Order A100 tracking and must hand the customer
to refunds without losing the order context.

WHAT WE ARE GOING TO SOLVE
Model an explicit handoff edge: shipping -> refund with shared state fields.

WHAT THIS EXAMPLE IS ABOUT
Two specialist nodes; shipping writes a handoff note; refund continues on A100.

WHAT IT SOLVES
Handoffs are just routing + shared state (or Command.goto), not magic.

KEYWORDS
- Handoff: transferring control from one specialist node/agent to another.
- Shared state: order_id and notes survive the handoff.
- Multi-agent: multiple specialists coordinated by a graph (or supervisor).
"""

from typing import Annotated, TypedDict
import operator

from langgraph.graph import END, START, StateGraph


class MultiAgentState(TypedDict):
    order_id: str
    active_agent: str
    notes: Annotated[list[str], operator.add]
    reply: str


def shipping_agent(state: MultiAgentState) -> dict:
    return {
        "active_agent": "shipping",
        "notes": [f"{state['order_id']} shipped $18 The Little Prince"],
        "reply": "Handing you to refunds for the return question.",
    }


def should_handoff(state: MultiAgentState) -> str:
    return "refund_agent"


def refund_agent(state: MultiAgentState) -> dict:
    prior = state["notes"][-1] if state["notes"] else ""
    return {
        "active_agent": "refund",
        "notes": ["policy: unread print 30-day"],
        "reply": f"Refund desk received handoff. Prior note: {prior}",
    }


def build_graph():
    g = StateGraph(MultiAgentState)
    g.add_node("shipping_agent", shipping_agent)
    g.add_node("refund_agent", refund_agent)
    g.add_edge(START, "shipping_agent")
    g.add_conditional_edges(
        "shipping_agent",
        should_handoff,
        {"refund_agent": "refund_agent"},
    )
    g.add_edge("refund_agent", END)
    return g.compile()


def main() -> None:
    out = build_graph().invoke(
        {
            "order_id": "A100",
            "active_agent": "",
            "notes": [],
            "reply": "",
        }
    )
    print("Active:", out["active_agent"])
    print("Notes:", out["notes"])
    print(out["reply"])


if __name__ == "__main__":
    print(__doc__)
    main()
