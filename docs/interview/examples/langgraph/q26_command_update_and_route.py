"""Q26. Command update and route together

THE PROBLEM
After reading a ticket, you must set assigned_team and jump to the right desk
in one step - not update, then separately route.

WHAT WE ARE GOING TO SOLVE
A triage node returns Command(update=..., goto=...).

WHAT THIS EXAMPLE IS ABOUT
A100 mentions connection to payment for a refund vs a damaged-book claim.
Triage sets team and goto in one Command.

WHAT IT SOLVES
Command bundles partial state update and dynamic goto in a single return.

KEYWORDS
- Command.update: dict merged into state.
- Command.goto: next node name (or Send).
- Literal[...]: annotate possible goto targets for graph validation.
"""

from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command


class IncidentState(TypedDict):
    ticket: str
    assigned_team: str
    outcome: str


def triage(state: IncidentState) -> Command[Literal["refund_agent", "damage_agent"]]:
    text = state["ticket"].lower()
    if "refund" in text or "return" in text:
        return Command(update={"assigned_team": "refunds"}, goto="refund_agent")
    return Command(update={"assigned_team": "quality"}, goto="damage_agent")


def refund_agent(state: IncidentState) -> dict:
    return {"outcome": "Approve $18 if unread within 30 days (A100)."}


def damage_agent(state: IncidentState) -> dict:
    return {"outcome": "Open damage claim for The Little Prince."}


def build_graph():
    g = StateGraph(IncidentState)
    g.add_node("triage", triage)
    g.add_node("refund_agent", refund_agent)
    g.add_node("damage_agent", damage_agent)
    g.add_edge(START, "triage")
    g.add_edge("refund_agent", END)
    g.add_edge("damage_agent", END)
    return g.compile()


def main() -> None:
    graph = build_graph()
    r = graph.invoke(
        {
            "ticket": "Please refund order A100 The Little Prince",
            "assigned_team": "",
            "outcome": "",
        }
    )
    d = graph.invoke(
        {
            "ticket": "Book arrived with a torn cover",
            "assigned_team": "",
            "outcome": "",
        }
    )
    print("refund:", r["assigned_team"], "->", r["outcome"])
    print("damage:", d["assigned_team"], "->", d["outcome"])


if __name__ == "__main__":
    print(__doc__)
    main()
