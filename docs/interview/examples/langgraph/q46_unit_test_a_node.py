"""Q46. Unit test a single LangGraph node in isolation.

THE PROBLEM
Engineers wrap every refund helper in a full graph.invoke just to assert one
routing decision, so tests are slow and flaky.

WHAT WE ARE GOING TO SOLVE
Call the node function directly with a hand-built state dict and assert the
return value (and Command fields when used).

WHAT THIS EXAMPLE IS ABOUT
triage_ticket routes Order A100 refund vs shipping questions without compiling
a graph.

WHAT IT SOLVES
Fast unit tests for node logic; reserve graph.invoke for integration tests.

KEYWORDS
- Node: a plain Python function (or runnable) that maps state to a partial update.
- Unit test: call the function directly; no compile, no checkpointer.
- Command: optional return type with .update and .goto for dynamic routing.
- Isolation: keep node logic free of hidden globals so tests stay fast and honest.
"""

from typing import TypedDict

from langgraph.types import Command


class TicketState(TypedDict):
    text: str
    queue: str


def triage_ticket(state: TicketState) -> Command:
    """Route a bookstore ticket. This is the unit under test."""
    lower = state["text"].lower()
    if "refund" in lower or "return" in lower:
        return Command(update={"queue": "refunds"}, goto="refund_desk")
    if "ship" in lower or "track" in lower:
        return Command(update={"queue": "shipping"}, goto="shipping_desk")
    return Command(update={"queue": "general"}, goto="general_desk")


def test_refund_routes_to_refunds() -> None:
    result = triage_ticket(
        {"text": "Please refund order A100 The Little Prince", "queue": ""}
    )
    assert result.update == {"queue": "refunds"}
    assert result.goto == "refund_desk"


def test_shipping_routes_to_shipping() -> None:
    result = triage_ticket({"text": "Where did A100 ship?", "queue": ""})
    assert result.update == {"queue": "shipping"}
    assert result.goto == "shipping_desk"


def main() -> None:
    # Integration tests would call build_graph().invoke(...); keep those separate.
    test_refund_routes_to_refunds()
    test_shipping_routes_to_shipping()
    sample = triage_ticket({"text": "Hello bookstore", "queue": ""})
    print("General sample goto:", sample.goto)
    print("All node unit tests passed (no graph.compile needed).")


if __name__ == "__main__":
    print(__doc__)
    main()
