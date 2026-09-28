"""Q10. What is a reducer?

THE PROBLEM
Without a reducer, each node that returns notes overwrites the previous note.
Refund A100 loses the fraud note when policy runs.

WHAT WE ARE GOING TO SOLVE
Compare overwrite (no reducer) on a scalar status vs operator.add on a notes list.

WHAT THIS EXAMPLE IS ABOUT
Two sequential nodes append audit notes for The Little Prince refund; status
overwrites from pending -> approved.

WHAT IT SOLVES
Default = overwrite (good for scalars). Annotated[list, operator.add] appends.

KEYWORDS
- Reducer: function that merges new channel values with existing ones.
- Overwrite: default when no reducer is set - last write wins.
- Scalar: single value (str/int/bool), usually fine to overwrite.
"""

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph


class AuditState(TypedDict):
    order_id: str
    status: str  # no reducer -> overwrite
    notes: Annotated[list[str], operator.add]


def fraud_note(state: AuditState) -> dict:
    return {"status": "pending", "notes": ["fraud cleared"]}


def policy_note(state: AuditState) -> dict:
    return {
        "status": "approved",
        "notes": ["policy: unread print 30 days - $18"],
    }


def build_graph():
    g = StateGraph(AuditState)
    g.add_node("fraud_note", fraud_note)
    g.add_node("policy_note", policy_note)
    g.add_edge(START, "fraud_note")
    g.add_edge("fraud_note", "policy_note")
    g.add_edge("policy_note", END)
    return g.compile()


def main() -> None:
    graph = build_graph()
    out = graph.invoke({"order_id": "A100", "status": "new", "notes": []})
    print("status (overwritten):", out["status"])
    print("notes (reduced with add):", out["notes"])


if __name__ == "__main__":
    print(__doc__)
    main()
