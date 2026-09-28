"""Q36. Supervisor vs Swarm - when to pick which

THE PROBLEM
System design interview: bookstore needs refunds + shipping agents. Do you
centralize routing or allow peer handoffs?

WHAT WE ARE GOING TO SOLVE
Run both topologies on the same ticket and print the justification checklist.

WHAT THIS EXAMPLE IS ABOUT
Ticket mentions shipping then refund for A100. Supervisor always goes
desk->worker->desk. Swarm shipping hands off straight to refunds.

WHAT IT SOLVES
Supervisor = auditable single throat to choke. Swarm = specialist judges handoff.

KEYWORDS
- Centralized control: one router decides every turn (supervisor).
- Distributed judgment: active specialist picks the next peer (swarm).
- Auditability: easier when one node owns routing decisions.
"""

from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command


class S(TypedDict):
    message: str
    trace: str
    answer: str


def supervisor(state: S) -> Command[Literal["ref", "ship", "__end__"]]:
    if state["answer"]:
        return Command(goto=END)
    if "refund" in state["message"].lower():
        return Command(update={"trace": "sup->ref"}, goto="ref")
    return Command(update={"trace": "sup->ship"}, goto="ship")


def ref(state: S) -> Command[Literal["supervisor"]]:
    return Command(
        update={"answer": "refund $18 A100 via supervisor"},
        goto="supervisor",
    )


def ship(state: S) -> Command[Literal["supervisor"]]:
    return Command(
        update={"answer": "ship status via supervisor"},
        goto="supervisor",
    )


def swarm_ship(state: S) -> Command[Literal["swarm_ref", "__end__"]]:
    if "refund" in state["message"].lower():
        return Command(update={"trace": "ship->ref"}, goto="swarm_ref")
    return Command(update={"answer": "shipping only"}, goto=END)


def swarm_ref(state: S) -> dict:
    return {"answer": "refund $18 A100 via swarm handoff"}


def build_supervisor():
    g = StateGraph(S)
    g.add_node("supervisor", supervisor)
    g.add_node("ref", ref)
    g.add_node("ship", ship)
    g.add_edge(START, "supervisor")
    return g.compile()


def build_swarm():
    g = StateGraph(S)
    g.add_node("swarm_ship", swarm_ship)
    g.add_node("swarm_ref", swarm_ref)
    g.add_edge(START, "swarm_ship")
    g.add_edge("swarm_ref", END)
    return g.compile()


def main() -> None:
    ticket = {
        "message": "package arrived, refund The Little Prince A100",
        "trace": "",
        "answer": "",
    }
    s = build_supervisor().invoke(ticket)
    w = build_swarm().invoke(dict(ticket))
    print("supervisor:", s["trace"], "->", s["answer"])
    print("swarm:     ", w["trace"], "->", w["answer"])
    print(
        "Pick supervisor for compliance/audit; swarm when the active specialist "
        "best knows who should take over next."
    )


if __name__ == "__main__":
    print(__doc__)
    main()
