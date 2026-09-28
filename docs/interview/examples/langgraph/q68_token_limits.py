"""Q68. Token limits and production scaling habits.

THE PROBLEM
Long bookstore threads grow until model calls hit context limits and latency
spikes.

WHAT WE ARE GOING TO SOLVE
Show practical graph-side habits: trim history channel, keep tool outputs short,
and bound recursion_limit - without calling a model.

WHAT THIS EXAMPLE IS ABOUT
A ticket graph that keeps only the last N notes about Order A100 before
composing a reply.

WHAT IT SOLVES
Operational levers for token pressure: trim state, short tool results, recursion caps.

KEYWORDS
- Context window: max tokens a model can read at once.
- Trim: drop old messages/notes before the model step.
- recursion_limit: caps graph steps (and thus repeated model calls).
"""

from typing import Annotated, TypedDict
import operator

from langgraph.graph import END, START, StateGraph


class ScaleState(TypedDict):
    order_id: str
    notes: Annotated[list[str], operator.add]
    kept: list[str]
    reply: str


def ingest(state: ScaleState) -> dict:
    # Simulate noisy tool spam the desk accumulates.
    spam = [f"note-{i} for {state['order_id']}" for i in range(8)]
    spam[0] = f"{state['order_id']} The Little Prince shipped $18"
    spam[1] = "policy: unread print 30-day return"
    return {"notes": spam}


def trim_notes(state: ScaleState) -> dict:
    # Keep only the last 3 notes - stand-in for message trimming before an LLM.
    return {"kept": state["notes"][-3:]}


def compose(state: ScaleState) -> dict:
    return {"reply": " | ".join(state["kept"])}


def build_graph():
    g = StateGraph(ScaleState)
    g.add_node("ingest", ingest)
    g.add_node("trim", trim_notes)
    g.add_node("compose", compose)
    g.add_edge(START, "ingest")
    g.add_edge("ingest", "trim")
    g.add_edge("trim", "compose")
    g.add_edge("compose", END)
    return g.compile()


def main() -> None:
    out = build_graph().invoke(
        {"order_id": "A100", "notes": [], "kept": [], "reply": ""},
        config={"recursion_limit": 25},
    )
    print("Trimmed kept:", out["kept"])
    print("Reply:", out["reply"])


if __name__ == "__main__":
    print(__doc__)
    main()
