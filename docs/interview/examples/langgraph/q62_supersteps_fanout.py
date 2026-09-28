"""Q62. Supersteps and fan-out/fan-in.

THE PROBLEM
People say "parallel nodes" without knowing LangGraph runs them in supersteps
and joins with reducers.

WHAT WE ARE GOING TO SOLVE
Fan out two bookstore checks in one superstep and fan in their notes.

WHAT THIS EXAMPLE IS ABOUT
Order A100: inventory check and address check run together, then a join node
prints the merged list.

WHAT IT SOLVES
A concrete superstep: multiple workers, one join, reducer-merged state.

KEYWORDS
- Superstep: one synchronized wave of node executions.
- Fan-out / fan-in: split work then merge results.
- Send: schedules a worker with its own input for this superstep.
"""

from typing import Annotated, TypedDict
import operator

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send


class CheckState(TypedDict):
    order_id: str
    checks: list[str]
    results: Annotated[list[str], operator.add]


def split(state: CheckState):
    return [Send("check", {"kind": c, "order_id": state["order_id"]}) for c in state["checks"]]


def check(state: dict) -> dict:
    kind = state["kind"]
    oid = state["order_id"]
    if kind == "inventory":
        return {"results": [f"{oid}: The Little Prince in warehouse"]}
    return {"results": [f"{oid}: ship-to address verified"]}


def join(state: CheckState) -> dict:
    return {}


def build_graph():
    g = StateGraph(CheckState)
    g.add_node("check", check)
    g.add_node("join", join)
    g.add_conditional_edges(START, split, ["check"])
    g.add_edge("check", "join")
    g.add_edge("join", END)
    return g.compile()


def main() -> None:
    out = build_graph().invoke(
        {
            "order_id": "A100",
            "checks": ["inventory", "address"],
            "results": [],
        }
    )
    print("Superstep merged:", sorted(out["results"]))


if __name__ == "__main__":
    print(__doc__)
    main()
