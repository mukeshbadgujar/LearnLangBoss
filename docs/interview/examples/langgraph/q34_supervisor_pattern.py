"""Q34. Supervisor pattern

THE PROBLEM
Refunds and shipping specialists must not call each other. One supervisor decides
who acts next for every bookstore ticket.

WHAT WE ARE GOING TO SOLVE
Hand-build a supervisor-style loop: supervisor node Commands to a worker, worker
returns to supervisor, supervisor may END. (create_supervisor package noted.)

WHAT THIS EXAMPLE IS ABOUT
Supervisor reads 'refund A100' -> refunds worker -> back to supervisor -> END.

WHAT IT SOLVES
Supervisor = single auditable router; specialists only talk through it.

KEYWORDS
- Supervisor pattern: central agent chooses the next specialist.
- create_supervisor: helper in langgraph-supervisor package.
- Chain of command: workers never peer-route.
"""

from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command


class SupState(TypedDict):
    message: str
    worker: str
    done: bool
    answer: str


def supervisor(state: SupState) -> Command[Literal["refunds", "shipping", "__end__"]]:
    if state["done"]:
        return Command(goto=END)
    if state["answer"]:
        # Worker finished - close the turn (supervisor ends the run).
        return Command(update={"done": True}, goto=END)
    if "refund" in state["message"].lower():
        return Command(update={"worker": "refunds"}, goto="refunds")
    return Command(update={"worker": "shipping"}, goto="shipping")


def refunds(state: SupState) -> Command[Literal["supervisor"]]:
    return Command(
        update={
            "answer": "Approve $18 if unread within 30 days (The Little Prince / A100)."
        },
        goto="supervisor",
    )


def shipping(state: SupState) -> Command[Literal["supervisor"]]:
    return Command(
        update={"answer": "A100 shipped - ETA on tracking page."},
        goto="supervisor",
    )


def build_graph():
    # Real deployments often use: from langgraph_supervisor import create_supervisor
    g = StateGraph(SupState)
    g.add_node("supervisor", supervisor)
    g.add_node("refunds", refunds)
    g.add_node("shipping", shipping)
    g.add_edge(START, "supervisor")
    return g.compile()


def main() -> None:
    out = build_graph().invoke(
        {
            "message": "Please refund order A100",
            "worker": "",
            "done": False,
            "answer": "",
        }
    )
    print("worker:", out["worker"])
    print("answer:", out["answer"])


if __name__ == "__main__":
    print(__doc__)
    main()
