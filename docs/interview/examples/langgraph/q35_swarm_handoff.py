"""Q35. Swarm handoff

THE PROBLEM
Shipping agent discovers the real ask is a refund and should hand off directly
to refunds - no bounce through a supervisor.

WHAT WE ARE GOING TO SOLVE
Peer Command handoff (stand-in for create_handoff_tool + create_swarm).

WHAT THIS EXAMPLE IS ABOUT
Active agent starts as shipping on A100, then handoff_to_refunds via Command.

WHAT IT SOLVES
Swarm = peer-to-peer handoff; active agent chooses the next peer.

KEYWORDS
- Swarm: decentralized multi-agent topology.
- Handoff tool: tool that transfers control to a named peer agent.
- create_swarm / create_handoff_tool: langgraph-swarm helpers.
"""

from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command


class SwarmState(TypedDict):
    message: str
    active: str
    answer: str


def shipping_agent(state: SwarmState) -> Command[Literal["refunds_agent", "__end__"]]:
    text = state["message"].lower()
    if "refund" in text or "return" in text:
        # Peer handoff - like create_handoff_tool(agent_name="refunds_agent")
        return Command(
            update={"active": "handoff->refunds"},
            goto="refunds_agent",
        )
    return Command(
        update={"active": "shipping", "answer": "A100 is out for delivery."},
        goto=END,
    )


def refunds_agent(state: SwarmState) -> dict:
    return {
        "active": "refunds",
        "answer": "Unread print books: 30 days - refund $18 for The Little Prince.",
    }


def build_swarm():
    # Packaged API: langgraph_swarm.create_swarm / create_handoff_tool
    g = StateGraph(SwarmState)
    g.add_node("shipping_agent", shipping_agent)
    g.add_node("refunds_agent", refunds_agent)
    g.add_edge(START, "shipping_agent")  # default_active_agent
    g.add_edge("refunds_agent", END)
    return g.compile()


def main() -> None:
    out = build_swarm().invoke(
        {
            "message": "tracking says delivered - I want a refund for A100",
            "active": "shipping",
            "answer": "",
        }
    )
    print("active:", out["active"])
    print("answer:", out["answer"])


if __name__ == "__main__":
    print(__doc__)
    main()
