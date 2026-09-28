"""Q30. Fan-out / fan-in with Send

THE PROBLEM
Refund A100 may touch several policy docs. Check each in parallel, then merge
findings once before deciding.

WHAT WE ARE GOING TO SOLVE
Send map over doc names, operator.add findings, then aggregate node (fan-in).

WHAT THIS EXAMPLE IS ABOUT
Docs: return window, unread rule, shipping exception. Parallel checkers append
findings; aggregate prints the $18 decision.

WHAT IT SOLVES
Full map-reduce: Send fan-out -> reducer merge -> single aggregate after the superstep.

KEYWORDS
- Fan-in: wait for all parallel branches, then one downstream node.
- operator.add: concatenate parallel list updates.
- aggregate: node that runs after the mapped workers finish.
"""

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send


class ReduceState(TypedDict):
    docs: list[str]
    findings: Annotated[list[str], operator.add]
    decision: str


def list_docs(state: ReduceState) -> dict:
    return {
        "docs": ["return_window", "unread_rule", "shipping_exception"],
    }


def fan_out_docs(state: ReduceState) -> list[Send]:
    return [Send("check_doc", {"doc": d}) for d in state["docs"]]


def check_doc(payload: dict) -> dict:
    doc = payload["doc"]
    snippets = {
        "return_window": "30 days for print",
        "unread_rule": "must be unread",
        "shipping_exception": "A100 shipped - still returnable if unread",
    }
    return {"findings": [f"{doc}:{snippets[doc]}"]}


def aggregate(state: ReduceState) -> dict:
    joined = "; ".join(state["findings"])
    return {
        "decision": f"Approve $18 The Little Prince - {joined}",
    }


def build_graph():
    g = StateGraph(ReduceState)
    g.add_node("list_docs", list_docs)
    g.add_node("check_doc", check_doc)
    g.add_node("aggregate", aggregate)
    g.add_edge(START, "list_docs")
    g.add_conditional_edges("list_docs", fan_out_docs, ["check_doc"])
    g.add_edge("check_doc", "aggregate")
    g.add_edge("aggregate", END)
    return g.compile()


def main() -> None:
    out = build_graph().invoke({"docs": [], "findings": [], "decision": ""})
    print("findings:", out["findings"])
    print("decision:", out["decision"])


if __name__ == "__main__":
    print(__doc__)
    main()
