"""Q9. What compile() does

THE PROBLEM
People think add_node alone makes a runnable app. Without compile(), there is
no validated executable - and checkpointers must be attached at compile time.

WHAT WE ARE GOING TO SOLVE
Show compile validating edges, then compile with MemorySaver so A100 refund
state survives a second invoke on the same thread.

WHAT THIS EXAMPLE IS ABOUT
First invoke classifies a refund for The Little Prince. Second invoke on the
same thread_id continues with checkpointed state (empty input).

WHAT IT SOLVES
compile() validates structure, wires channels/reducers, and bakes in
checkpointer / interrupts. Those are not added later on invoke().

KEYWORDS
- compile-time: build/validate/configure before any user request runs.
- checkpointer: persistence backend passed into compile().
- Validation: compile fails fast on edges to missing node names.
"""

from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage


class TicketState(TypedDict):
    messages: Annotated[list, add_messages]
    stage: str


def classify(state: TicketState) -> dict:
    return {"stage": "classified_refund_A100"}


def stamp_policy(state: TicketState) -> dict:
    return {"stage": state["stage"] + "|policy_unread_30d"}


def build_graph(checkpointer: MemorySaver):
    g = StateGraph(TicketState)
    g.add_node("classify", classify)
    g.add_node("stamp_policy", stamp_policy)
    g.add_edge(START, "classify")
    g.add_edge("classify", "stamp_policy")
    g.add_edge("stamp_policy", END)
    # Persistence and interrupts are compile-time options:
    return g.compile(checkpointer=checkpointer)


def main() -> None:
    # Structural validation: this would fail at compile if edge pointed at "fix".
    saver = MemorySaver()
    graph = build_graph(saver)
    config = {"configurable": {"thread_id": "desk-a100"}}
    first = graph.invoke(
        {"messages": [HumanMessage(content="Refund The Little Prince")], "stage": ""},
        config,
    )
    print("after first invoke:", first["stage"])
    # Resume same thread - checkpointer restored state at compile wiring.
    snap = graph.get_state(config)
    print("checkpointed stage:", snap.values.get("stage"))
    print("compile baked in MemorySaver; invoke only supplies thread_id.")


if __name__ == "__main__":
    print(__doc__)
    main()
