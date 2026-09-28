"""Q53. State management and reducers.

THE PROBLEM
Two bookstore nodes both return a `notes` list; without a reducer the second
write overwrites the first.

WHAT WE ARE GOING TO SOLVE
Annotate a channel with operator.add so updates append instead of replace.

WHAT THIS EXAMPLE IS ABOUT
Shipping and policy nodes each append a note about Order A100; final state
keeps both.

WHAT IT SOLVES
Reducers define how concurrent or sequential updates merge into one channel.

KEYWORDS
- State: the TypedDict (or schema) channels the graph persists and passes around.
- Reducer: a merge function for a channel (e.g. operator.add for lists).
- Overwrite vs append: default last-write-wins; reducers change that.
- Annotated: typing wrapper that attaches the reducer to a state field.
- Channel: one named field whose merge rule the runtime applies on every write.
"""

from typing import Annotated, TypedDict
import operator

from langgraph.graph import END, START, StateGraph


class DeskState(TypedDict):
    order_id: str
    notes: Annotated[list[str], operator.add]


def shipping_note(state: DeskState) -> dict:
    # First writer into `notes`.
    return {"notes": [f"{state['order_id']} shipped $18"]}


def policy_note(state: DeskState) -> dict:
    # Second writer: reducer appends instead of replacing the shipping note.
    return {"notes": ["policy: unread print, 30-day return"]}


def build_graph():
    g = StateGraph(DeskState)
    g.add_node("shipping", shipping_note)
    g.add_node("policy", policy_note)
    g.add_edge(START, "shipping")
    g.add_edge("shipping", "policy")
    g.add_edge("policy", END)
    return g.compile()


def main() -> None:
    # Without Annotated[..., operator.add], the policy note would replace shipping.
    out = build_graph().invoke({"order_id": "A100", "notes": ["seed"]})
    print("Merged notes:", out["notes"])
    assert out["notes"][0] == "seed"
    assert "shipped" in out["notes"][1]
    assert "30-day" in out["notes"][2]
    print("Reducer kept seed + shipping + policy (3 entries).")
    print("Interview line: reducers are how parallel/sequential writes merge.")
    print("Bookstore scene: shipping + policy notes on Order A100 stay together.")


if __name__ == "__main__":
    print(__doc__)
    main()
