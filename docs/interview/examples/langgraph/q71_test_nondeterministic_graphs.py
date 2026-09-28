"""Q71. Testing non-deterministic graphs.

THE PROBLEM
A bookstore agent that sometimes phrases answers differently breaks brittle
string equality tests.

WHAT WE ARE GOING TO SOLVE
Test invariants and structure (queue, required fields, policy facts) instead of
exact prose - plus a deterministic router node under unit test.

WHAT THIS EXAMPLE IS ABOUT
Triage + reply graph for A100; assert routing and required substrings, not the
full sentence.

WHAT IT SOLVES
A practical eval style for flaky language: contract tests + golden facts.

KEYWORDS
- Non-deterministic: model wording varies across runs.
- Invariant: property that must hold regardless of phrasing.
- Contract test: assert schema/routing/facts, not full LLM text.
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class TicketState(TypedDict):
    text: str
    queue: str
    answer: str


def triage(state: TicketState) -> dict:
    q = "refunds" if "return" in state["text"].lower() else "shipping"
    return {"queue": q}


def reply(state: TicketState) -> dict:
    # Wording could vary if this were an LLM; facts must remain.
    if state["queue"] == "refunds":
        return {"answer": "You may return unread print books within 30 days (A100 $18)."}
    return {"answer": "Order A100 The Little Prince has shipped."}


def build_graph():
    g = StateGraph(TicketState)
    g.add_node("triage", triage)
    g.add_node("reply", reply)
    g.add_edge(START, "triage")
    g.add_edge("triage", "reply")
    g.add_edge("reply", END)
    return g.compile()


def test_return_ticket_invariants() -> None:
    out = build_graph().invoke(
        {"text": "I want to return A100", "queue": "", "answer": ""}
    )
    assert out["queue"] == "refunds"
    assert "30" in out["answer"]
    assert "A100" in out["answer"]


def test_shipping_ticket_invariants() -> None:
    out = build_graph().invoke(
        {"text": "track my package", "queue": "", "answer": ""}
    )
    assert out["queue"] == "shipping"
    assert "shipped" in out["answer"].lower()


def main() -> None:
    test_return_ticket_invariants()
    test_shipping_ticket_invariants()
    print("Invariant tests passed (no exact-string LLM assertions).")


if __name__ == "__main__":
    print(__doc__)
    main()
