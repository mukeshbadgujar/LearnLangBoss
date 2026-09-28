"""Q74. Testing AI agents (practical layers).

THE PROBLEM
The team only smoke-tests the full bookstore agent in the UI, so regressions
in triage or policy facts slip through.

WHAT WE ARE GOING TO SOLVE
Show three layers: unit-test a node, integration-test the graph path, and a
tiny golden-fact eval on the answer.

WHAT THIS EXAMPLE IS ABOUT
A100 return ticket through triage -> answer, with assertions at each layer.

WHAT IT SOLVES
A testing pyramid for agents that does not require calling Groq.

KEYWORDS
- Unit test: node function in isolation.
- Integration test: compiled graph path and state.
- Golden fact: required substring / structured field that must appear.
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class AgentState(TypedDict):
    text: str
    intent: str
    answer: str


def classify(state: AgentState) -> dict:
    intent = "return" if "return" in state["text"].lower() else "other"
    return {"intent": intent}


def answer(state: AgentState) -> dict:
    if state["intent"] == "return":
        return {
            "answer": "Order A100 The Little Prince: unread print returns within 30 days ($18)."
        }
    return {"answer": "Please share your order id."}


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("classify", classify)
    g.add_node("answer", answer)
    g.add_edge(START, "classify")
    g.add_edge("classify", "answer")
    g.add_edge("answer", END)
    return g.compile()


def test_unit_classify() -> None:
    assert classify({"text": "return A100", "intent": "", "answer": ""}) == {
        "intent": "return"
    }


def test_integration_path() -> None:
    out = build_graph().invoke(
        {"text": "I want to return The Little Prince", "intent": "", "answer": ""}
    )
    assert out["intent"] == "return"
    assert out["answer"]


def test_golden_facts() -> None:
    out = build_graph().invoke(
        {"text": "return please", "intent": "", "answer": ""}
    )
    for fact in ("A100", "30", "print"):
        assert fact in out["answer"]


def main() -> None:
    test_unit_classify()
    test_integration_path()
    test_golden_facts()
    print("Unit + integration + golden-fact layers passed.")


if __name__ == "__main__":
    print(__doc__)
    main()
