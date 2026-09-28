"""Q15. Parallel write conflict (InvalidUpdateError)

THE PROBLEM
Two parallel validators both write `report` with no reducer. LangGraph must not
silently pick a winner.

WHAT WE ARE GOING TO SOLVE
Demonstrate InvalidUpdateError, then fix with Annotated[list, operator.add].

WHAT THIS EXAMPLE IS ABOUT
Policy and warehouse checks for A100 The Little Prince run in one superstep and
both try to set the same key.

WHAT IT SOLVES
Unreduced parallel writes raise InvalidUpdateError; a reducer merges both.

KEYWORDS
- InvalidUpdateError: raised on conflicting parallel writes without a reducer.
- Non-deterministic: last-writer-wins would depend on timing - LangGraph forbids it.
"""

import operator
from typing import Annotated, TypedDict

from langgraph.errors import InvalidUpdateError
from langgraph.graph import END, START, StateGraph


class BadState(TypedDict):
    order_id: str
    report: str  # no reducer


class GoodState(TypedDict):
    order_id: str
    report: Annotated[list[str], operator.add]


def policy_check_bad(state: BadState) -> dict:
    return {"report": "policy: unread ok"}


def warehouse_check_bad(state: BadState) -> dict:
    return {"report": "warehouse: ship label scanned"}


def policy_check_good(state: GoodState) -> dict:
    return {"report": ["policy: unread ok"]}


def warehouse_check_good(state: GoodState) -> dict:
    return {"report": ["warehouse: ship label scanned"]}


def build_bad():
    g = StateGraph(BadState)
    g.add_node("policy_check_bad", policy_check_bad)
    g.add_node("warehouse_check_bad", warehouse_check_bad)
    g.add_edge(START, "policy_check_bad")
    g.add_edge(START, "warehouse_check_bad")
    g.add_edge("policy_check_bad", END)
    g.add_edge("warehouse_check_bad", END)
    return g.compile()


def build_good():
    g = StateGraph(GoodState)
    g.add_node("policy_check_good", policy_check_good)
    g.add_node("warehouse_check_good", warehouse_check_good)
    g.add_edge(START, "policy_check_good")
    g.add_edge(START, "warehouse_check_good")
    g.add_edge("policy_check_good", END)
    g.add_edge("warehouse_check_good", END)
    return g.compile()


def main() -> None:
    print("--- broken: parallel writes to `report` without reducer ---")
    try:
        build_bad().invoke({"order_id": "A100", "report": ""})
    except InvalidUpdateError as exc:
        print("caught InvalidUpdateError:", type(exc).__name__)
        print(str(exc)[:200])

    print("--- fixed: operator.add on report list ---")
    out = build_good().invoke({"order_id": "A100", "report": []})
    print("merged report:", out["report"])


if __name__ == "__main__":
    print(__doc__)
    main()
