"""Q33. Multi-agent topologies

THE PROBLEM
Bookstore support needs refunds, shipping, and catalog specialists. How are they
wired - supervisor, swarm, or custom network?

WHAT WE ARE GOING TO SOLVE
Implement a tiny custom network with Command routing (the third pattern), and
comment supervisor vs swarm.

WHAT THIS EXAMPLE IS ABOUT
Front desk routes A100 refund to refunds specialist; shipping question to ships.
No LLM - plain routers standing in for agents.

WHAT IT SOLVES
Name three topologies: Supervisor, Swarm, Network/custom graph - show custom.

KEYWORDS
- Supervisor: central orchestrator picks the next specialist each turn.
- Swarm: peer-to-peer handoffs via tools.
- Network / custom: you wire edges and Command yourself.
"""

from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command


class MultiState(TypedDict):
    message: str
    path: str
    answer: str


def front_desk(state: MultiState) -> Command[Literal["refunds", "ships"]]:
    if "refund" in state["message"].lower() or "return" in state["message"].lower():
        return Command(update={"path": "supervisor-like->refunds"}, goto="refunds")
    return Command(update={"path": "supervisor-like->ships"}, goto="ships")


def refunds(state: MultiState) -> dict:
    return {"answer": "A100 The Little Prince: unread print, 30 days, $18."}


def ships(state: MultiState) -> dict:
    return {"answer": "A100 is shipped - track via carrier portal."}


def build_network():
    # Custom network topology (hand-wired). Supervisor/swarm are prebuilt packages.
    g = StateGraph(MultiState)
    g.add_node("front_desk", front_desk)
    g.add_node("refunds", refunds)
    g.add_node("ships", ships)
    g.add_edge(START, "front_desk")
    g.add_edge("refunds", END)
    g.add_edge("ships", END)
    return g.compile()


def main() -> None:
    graph = build_network()
    r = graph.invoke({"message": "refund A100", "path": "", "answer": ""})
    s = graph.invoke({"message": "where is A100?", "path": "", "answer": ""})
    print("refunds:", r["path"], r["answer"])
    print("ships:  ", s["path"], s["answer"])
    print(
        "Also know: create_supervisor (central) and create_swarm (peer handoff) "
        "as packaged topologies."
    )


if __name__ == "__main__":
    print(__doc__)
    main()
