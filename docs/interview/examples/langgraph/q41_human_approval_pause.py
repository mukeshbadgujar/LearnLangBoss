"""Q41. Pause a graph for human approval before a high-risk action.

THE PROBLEM
A refund for Order A100 is $18 today, but the desk must never auto-issue a
refund when the amount is high-risk without a human click.

WHAT WE ARE GOING TO SOLVE
Call interrupt() inside a node, checkpoint with MemorySaver, then resume with
Command(resume=True).

WHAT THIS EXAMPLE IS ABOUT
Bookstore refund approval for The Little Prince ($18). The graph pauses with a
question payload a UI or Slack bot would show.

WHAT IT SOLVES
High-risk steps wait for a human; the same thread_id resumes exactly where
interrupt() left off.

KEYWORDS
- interrupt(): pauses the graph from inside a node and surfaces a payload to the client.
- Command(resume=...): resumes a paused graph and feeds the human decision back into interrupt().
- Checkpointer: stores state so a pause can last across process restarts.
- HITL: human-in-the-loop review before a risky side effect.
"""

from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class RefundState(TypedDict):
    order_id: str
    title: str
    amount: float
    approved: bool
    receipt: str


def draft_refund(state: RefundState) -> dict:
    return {"receipt": f"Draft refund ${state['amount']} for {state['order_id']}"}


def request_approval(state: RefundState) -> dict:
    # Dynamic pause: payload is what the frontend / Slack message shows.
    decision = interrupt(
        {"question": f"Approve refund of ${state['amount']} for {state['order_id']}?"}
    )
    return {"approved": bool(decision)}


def issue_refund(state: RefundState) -> dict:
    if not state["approved"]:
        return {"receipt": "Refund denied by reviewer"}
    return {
        "receipt": (
            f"Issued ${state['amount']} for {state['title']} "
            f"(policy: unread print, 30-day return)"
        )
    }


def build_graph():
    g = StateGraph(RefundState)
    g.add_node("draft", draft_refund)
    g.add_node("approve", request_approval)
    g.add_node("issue", issue_refund)
    g.add_edge(START, "draft")
    g.add_edge("draft", "approve")
    g.add_edge("approve", "issue")
    g.add_edge("issue", END)
    return g.compile(checkpointer=MemorySaver())


def main() -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": "refund-A100"}}
    paused = graph.invoke(
        {
            "order_id": "A100",
            "title": "The Little Prince",
            "amount": 18.0,
            "approved": False,
            "receipt": "",
        },
        config,
    )
    print("Paused payload:", paused["__interrupt__"][0].value)

    # Human clicks Approve in the UI -> app sends Command(resume=True).
    done = graph.invoke(Command(resume=True), config)
    print(done["receipt"], "| approved=", done["approved"])


if __name__ == "__main__":
    print(__doc__)
    main()
