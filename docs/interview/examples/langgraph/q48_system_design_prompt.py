"""Q48. Realistic LangGraph system-design prompt structure.

THE PROBLEM
Interview prompts like "design a support agent with human approval" get answered
as a pile of tools with no state, persistence, or HITL plan.

WHAT WE ARE GOING TO SOLVE
Walk a bookstore incident-style design top-down: state, topology, persistence,
HITL point, failure modes - then sketch the graph skeleton.

WHAT THIS EXAMPLE IS ABOUT
Design a desk that triages Order A100, investigates shipping/policy, and
requires human approval before issuing a refund over $20.

WHAT IT SOLVES
A repeatable answer structure interviewers grade: checkpointing and HITL are
first-class, not afterthoughts.

KEYWORDS
- State schema: shared fields every node reads/writes.
- Topology: supervisor, sequence, or swarm shape of the graph.
- Persistence: checkpointer + thread_id mapping to a ticket.
- Failure modes: recursion_limit, reducers, retries decided up front.
"""

from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class SupportDesignState(TypedDict):
    ticket: str
    order_id: str
    amount: float
    findings: str
    approved: bool
    outcome: str


def triage(state: SupportDesignState) -> dict:
    return {"order_id": "A100", "amount": 18.0}


def investigate(state: SupportDesignState) -> dict:
    return {
        "findings": (
            f"{state['order_id']} The Little Prince shipped $18; "
            "policy unread print 30-day return"
        )
    }


def approve_refund(state: SupportDesignState) -> dict:
    if state["amount"] > 20:
        ok = interrupt({"question": f"Approve refund ${state['amount']}?"})
        return {"approved": bool(ok)}
    return {"approved": True}


def close_ticket(state: SupportDesignState) -> dict:
    if state["approved"]:
        return {"outcome": "refund queued"}
    return {"outcome": "refund blocked"}


def build_graph():
    # (1) state schema above
    # (2) topology: linear investigate -> HITL -> close (supervisor optional)
    # (3) persistence: MemorySaver; thread_id = ticket id
    # (4) HITL: interrupt in approve_refund when amount > 20
    # (5) failure: set recursion_limit in production config
    g = StateGraph(SupportDesignState)
    g.add_node("triage", triage)
    g.add_node("investigate", investigate)
    g.add_node("approve_refund", approve_refund)
    g.add_node("close_ticket", close_ticket)
    g.add_edge(START, "triage")
    g.add_edge("triage", "investigate")
    g.add_edge("investigate", "approve_refund")
    g.add_edge("approve_refund", "close_ticket")
    g.add_edge("close_ticket", END)
    return g.compile(checkpointer=MemorySaver())


def main() -> None:
    print("Design checklist: state, topology, persistence, HITL, failure modes")
    graph = build_graph()
    cfg = {"configurable": {"thread_id": "ticket-A100"}, "recursion_limit": 25}
    out = graph.invoke(
        {
            "ticket": "Customer wants refund for A100",
            "order_id": "",
            "amount": 0.0,
            "findings": "",
            "approved": False,
            "outcome": "",
        },
        cfg,
    )
    print(out["findings"])
    print(out["outcome"], "| approved=", out["approved"])
    # If amount were > 20: graph.invoke(Command(resume=True), cfg)


if __name__ == "__main__":
    print(__doc__)
    main()
