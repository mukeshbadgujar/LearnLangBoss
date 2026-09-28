"""Q54. Fan-out and fan-in execution.

THE PROBLEM
Checking shipping status and return policy for Order A100 one after another
adds latency the desk does not need.

WHAT WE ARE GOING TO SOLVE
Fan out with Send to run workers in parallel, then fan in with a reducer on
notes.

WHAT THIS EXAMPLE IS ABOUT
One ticket fans out to shipping_worker and policy_worker; join sees both notes.

WHAT IT SOLVES
Parallel specialist work in one superstep, merged safely via operator.add.

KEYWORDS
- Fan-out: start several node instances (often via Send) in one step.
- Fan-in: join results back into shared state (usually with a reducer).
- Send: packet that schedules a node with a custom input payload.
"""

from typing import Annotated, TypedDict
import operator

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send


class DeskState(TypedDict):
    order_id: str
    tasks: list[str]
    notes: Annotated[list[str], operator.add]


def fan_out(state: DeskState) -> list[Send]:
    return [Send("worker", {"task": t, "order_id": state["order_id"]}) for t in state["tasks"]]


def worker(state: dict) -> dict:
    task = state["task"]
    oid = state["order_id"]
    if task == "shipping":
        return {"notes": [f"{oid} The Little Prince shipped $18"]}
    return {"notes": ["unread print books: 30-day return"]}


def join(state: DeskState) -> dict:
    return {}  # notes already merged by reducer


def build_graph():
    g = StateGraph(DeskState)
    g.add_node("worker", worker)
    g.add_node("join", join)
    g.add_conditional_edges(START, fan_out, ["worker"])
    g.add_edge("worker", "join")
    g.add_edge("join", END)
    return g.compile()


def main() -> None:
    out = build_graph().invoke(
        {"order_id": "A100", "tasks": ["shipping", "policy"], "notes": []}
    )
    print("Fan-in notes:", sorted(out["notes"]))


if __name__ == "__main__":
    print(__doc__)
    main()
