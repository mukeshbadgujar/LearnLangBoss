"""Q22. Time travel

THE PROBLEM
Desk mis-tagged A100 as deny. You want to rewind to before decide, flip the
flag, and replay - not re-run the whole intake.

WHAT WE ARE GOING TO SOLVE
MemorySaver + get_state_history + update_state + invoke(None, earlier_config).

WHAT THIS EXAMPLE IS ABOUT
Intake -> decide for The Little Prince. History shows both checkpoints. We fork
from intake with eligible=True and re-decide.

WHAT IT SOLVES
Time travel = inspect history, mutate an earlier checkpoint, resume a branch.

KEYWORDS
- get_state_history: list checkpoints for a thread (newest first).
- update_state: inject values into a checkpoint.
- Branching: continue from a past checkpoint on a new path.
"""

from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph


class TravelState(TypedDict):
    order_id: str
    eligible: bool
    decision: str


def intake(state: TravelState) -> dict:
    # Bug in first run: marked not eligible.
    return {"order_id": "A100", "eligible": False}


def decide(state: TravelState) -> dict:
    if state["eligible"]:
        return {"decision": "approve $18 The Little Prince"}
    return {"decision": "deny"}


def build_graph(checkpointer: MemorySaver):
    g = StateGraph(TravelState)
    g.add_node("intake", intake)
    g.add_node("decide", decide)
    g.add_edge(START, "intake")
    g.add_edge("intake", "decide")
    g.add_edge("decide", END)
    return g.compile(checkpointer=checkpointer)


def main() -> None:
    graph = build_graph(MemorySaver())
    config = {"configurable": {"thread_id": "time-a100"}}
    first = graph.invoke({"order_id": "", "eligible": False, "decision": ""}, config)
    print("first decision:", first["decision"])

    history = list(graph.get_state_history(config))
    # history[0] is newest; find checkpoint whose next step is decide
    earlier = next(h for h in history if h.next == ("decide",))
    print("rewind to before decide, next=", earlier.next)

    graph.update_state(earlier.config, {"eligible": True})
    replayed = graph.invoke(None, earlier.config)
    print("after time travel:", replayed["decision"])


if __name__ == "__main__":
    print(__doc__)
    main()
