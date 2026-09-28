"""Q13. input_schema and output_schema

THE PROBLEM
Internal refund graph tracks scratch fields (retry_count, fraud_flag). Callers
should only send ticket text and get a decision - not every internal key.

WHAT WE ARE GOING TO SOLVE
Define Input / Output schemas separate from full State on StateGraph.

WHAT THIS EXAMPLE IS ABOUT
Caller submits a refund request for A100. Internally we set fraud_flag and
retries; output schema exposes only decision and amount.

WHAT IT SOLVES
Public API contract stays small while internal state can be rich.

KEYWORDS
- input_schema: keys accepted from outside on invoke.
- output_schema: keys returned to the caller.
- Encapsulation: hide internal scratch channels from clients.
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class Input(TypedDict):
    ticket: str


class Output(TypedDict):
    decision: str
    amount: float


class State(TypedDict):
    ticket: str
    fraud_flag: bool
    retries: int
    decision: str
    amount: float


def enrich(state: State) -> dict:
    return {"fraud_flag": False, "retries": 0, "amount": 18.0}


def decide(state: State) -> dict:
    if "little prince" in state["ticket"].lower() or "a100" in state["ticket"].lower():
        return {"decision": "approve"}
    return {"decision": "review"}


def build_graph():
    g = StateGraph(State, input_schema=Input, output_schema=Output)
    g.add_node("enrich", enrich)
    g.add_node("decide", decide)
    g.add_edge(START, "enrich")
    g.add_edge("enrich", "decide")
    g.add_edge("decide", END)
    return g.compile()


def main() -> None:
    graph = build_graph()
    # Only ticket in; only decision + amount out.
    out = graph.invoke({"ticket": "Refund A100 The Little Prince"})
    print("public output:", out)
    assert set(out.keys()) <= {"decision", "amount"}


if __name__ == "__main__":
    print(__doc__)
    main()
