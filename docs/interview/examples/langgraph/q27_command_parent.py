"""Q27. Command(graph=Command.PARENT)

THE PROBLEM
A refund subgraph finishes and must hand control to a parent supervisor node,
not to another node inside the child.

WHAT WE ARE GOING TO SOLVE
Child node returns Command(graph=Command.PARENT, goto=...).

WHAT THIS EXAMPLE IS ABOUT
Parent: intake -> refund_child subgraph -> summarize. Child computes eligibility
for A100 then Command.PARENT to summarize.

WHAT IT SOLVES
goto defaults to the current graph; Command.PARENT resolves goto in the parent.

KEYWORDS
- Command.PARENT: target the closest parent graph for goto.
- Subgraph: child graph nested as a node.
- Handoff: specialist signals completion back to supervisor.
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command


class ParentState(TypedDict):
    order_id: str
    eligible: bool
    summary: str


class ChildState(TypedDict):
    order_id: str
    eligible: bool


def check_child(state: ChildState) -> Command:
    # Done inside child - route to parent's summarize node.
    # Do not annotate Command[Literal["summarize"]]: that registers an edge
    # on the child graph. PARENT goto is resolved on the parent instead.
    return Command(
        update={"eligible": True, "order_id": "A100"},
        goto="summarize",
        graph=Command.PARENT,
    )


def build_child():
    g = StateGraph(ChildState)
    g.add_node("check_child", check_child)
    g.add_edge(START, "check_child")
    return g.compile()


def intake(state: ParentState) -> dict:
    return {"order_id": "A100"}


def summarize(state: ParentState) -> dict:
    flag = "eligible" if state["eligible"] else "ineligible"
    return {
        "summary": f"{state['order_id']} The Little Prince $18 marked {flag} (30-day unread)."
    }


def build_parent(child):
    g = StateGraph(ParentState)
    g.add_node("intake", intake)
    g.add_node("refund_child", child)
    g.add_node("summarize", summarize)
    g.add_edge(START, "intake")
    g.add_edge("intake", "refund_child")
    # summarize is reached via Command.PARENT from the child
    g.add_edge("summarize", END)
    return g.compile()


def main() -> None:
    parent = build_parent(build_child())
    out = parent.invoke(
        {"order_id": "", "eligible": False, "summary": ""}
    )
    print(out["summary"])


if __name__ == "__main__":
    print(__doc__)
    main()
