"""Q3. Three building blocks: State, Nodes, Edges

THE PROBLEM
Interviewers ask what every LangGraph app is made of. Vague answers lose points.

WHAT WE ARE GOING TO SOLVE
Name and show State, Nodes, and Edges on a tiny refund triage graph.

WHAT THIS EXAMPLE IS ABOUT
Ticket: refund for A100 The Little Prince ($18). State holds order fields.
Nodes: classify intent, then apply policy. Edges: START -> classify -> policy -> END.
Conditional edge sends 'not_refund' tickets to a deny node.

WHAT IT SOLVES
You can point at one concrete State, two Nodes, and fixed + conditional Edges.

KEYWORDS
- State: schema for the shared backpack of data.
- Node: Python function returning a partial state update.
- Edge: rule for which node runs next (fixed or conditional).
- Partial update: return only the keys you changed; LangGraph merges them.
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class DeskState(TypedDict):
    message: str
    intent: str
    order_id: str
    amount: float
    outcome: str


def classify(state: DeskState) -> dict:
    text = state["message"].lower()
    if "refund" in text:
        return {"intent": "refund", "order_id": "A100", "amount": 18.0}
    return {"intent": "other", "order_id": "", "amount": 0.0}


def apply_policy(state: DeskState) -> dict:
    # Unread print books, 30 days - assumed eligible for this ticket.
    return {
        "outcome": f"Approve ${state['amount']:.2f} refund for {state['order_id']}."
    }


def deny_other(state: DeskState) -> dict:
    return {"outcome": "Not a refund ticket - hand to FAQ bot."}


def after_classify(state: DeskState) -> str:
    return "apply_policy" if state["intent"] == "refund" else "deny_other"


def build_graph():
    # 1) State schema
    g = StateGraph(DeskState)
    # 2) Nodes
    g.add_node("classify", classify)
    g.add_node("apply_policy", apply_policy)
    g.add_node("deny_other", deny_other)
    # 3) Edges
    g.add_edge(START, "classify")
    g.add_conditional_edges(
        "classify",
        after_classify,
        {"apply_policy": "apply_policy", "deny_other": "deny_other"},
    )
    g.add_edge("apply_policy", END)
    g.add_edge("deny_other", END)
    return g.compile()


def main() -> None:
    graph = build_graph()
    refund = graph.invoke(
        {
            "message": "Please refund The Little Prince on A100",
            "intent": "",
            "order_id": "",
            "amount": 0.0,
            "outcome": "",
        }
    )
    other = graph.invoke(
        {
            "message": "What are store hours?",
            "intent": "",
            "order_id": "",
            "amount": 0.0,
            "outcome": "",
        }
    )
    print("refund path:", refund["outcome"])
    print("other path:", other["outcome"])


if __name__ == "__main__":
    print(__doc__)
    main()
