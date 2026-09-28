"""Q32. Subgraph with isolated state

THE PROBLEM
Fraud scoring uses temporary feature vectors. The parent desk only needs a
pass/fail - shared state would leak scratch fields.

WHAT WE ARE GOING TO SOLVE
Parent StateGraph embeds a compiled child with a different schema.

WHAT THIS EXAMPLE IS ABOUT
Parent holds A100 ticket text. Child fraud subgraph uses private features list
and returns only risk_pass mapped into parent.

WHAT IT SOLVES
Isolated subgraph state when scratch work must not pollute the parent channels.

KEYWORDS
- Subgraph: compiled graph used as a node in a parent.
- Isolated state: child schema independent of parent.
- Shared state: alternate mode where child reads/writes parent channels.
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class ParentState(TypedDict):
    ticket: str
    risk_pass: bool
    decision: str


class FraudState(TypedDict):
    ticket: str
    features: list[str]
    risk_pass: bool


def extract_features(state: FraudState) -> dict:
    return {"features": ["order:A100", "amount:18", "title:LittlePrince"]}


def score(state: FraudState) -> dict:
    return {"risk_pass": True}


def build_fraud_subgraph():
    g = StateGraph(FraudState)
    g.add_node("extract_features", extract_features)
    g.add_node("score", score)
    g.add_edge(START, "extract_features")
    g.add_edge("extract_features", "score")
    g.add_edge("score", END)
    return g.compile()


def decide(state: ParentState) -> dict:
    if state["risk_pass"]:
        return {"decision": "continue refund for unread print within 30 days"}
    return {"decision": "hold for fraud review"}


def build_parent(fraud):
    g = StateGraph(ParentState)
    g.add_node("fraud", fraud)
    g.add_node("decide", decide)
    g.add_edge(START, "fraud")
    g.add_edge("fraud", "decide")
    g.add_edge("decide", END)
    return g.compile()


def main() -> None:
    parent = build_parent(build_fraud_subgraph())
    out = parent.invoke(
        {
            "ticket": "Refund The Little Prince A100",
            "risk_pass": False,
            "decision": "",
        }
    )
    print(out)
    print("Parent never stored `features` - isolated in the subgraph.")


if __name__ == "__main__":
    print(__doc__)
    main()
