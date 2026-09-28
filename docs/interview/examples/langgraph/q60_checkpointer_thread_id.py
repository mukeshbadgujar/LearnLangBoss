"""Q60. Checkpointer and thread_id on a shipping ticket.

THE PROBLEM
A shopper asks "where is my book?" twice in one day. Without a thread_id the
second message looks brand new and the desk re-looks-up Order A100 from scratch.

WHAT WE ARE GOING TO SOLVE
First-time explanation: MemorySaver + configurable thread_id ties turns to one
shipping ticket conversation.

WHAT THIS EXAMPLE IS ABOUT
Shipping ticket T-SHIP-A100. Turn 1 stores carrier status; turn 2 reads it
back from the same thread.

WHAT IT SOLVES
Persistence via thread_id - the foundational checkpointer story (not crash
recovery; see q55 for that).

KEYWORDS
- Checkpointer: component that saves/loads graph state.
- thread_id: string key for one durable session (here a shipping ticket id).
- MemorySaver: in-memory checkpointer for demos and tests.
"""

from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage


class ShippingTicket(TypedDict):
    messages: Annotated[list, add_messages]
    last_status: str


def lookup_carrier(state: ShippingTicket) -> dict:
    last_human = state["messages"][-1].content
    status = "A100 The Little Prince: shipped via UPS, $18"
    return {
        "last_status": status,
        "messages": [AIMessage(content=f"Heard '{last_human}'. {status}")],
    }


def build_graph():
    g = StateGraph(ShippingTicket)
    g.add_node("lookup", lookup_carrier)
    g.add_edge(START, "lookup")
    g.add_edge("lookup", END)
    return g.compile(checkpointer=MemorySaver())


def main() -> None:
    graph = build_graph()
    # thread_id = the shipping ticket - same id continues the same checkpointed chat.
    cfg = {"configurable": {"thread_id": "T-SHIP-A100"}}

    turn1 = graph.invoke(
        {"messages": [HumanMessage(content="Where is order A100?")], "last_status": ""},
        cfg,
    )
    print("Turn 1:", turn1["last_status"])

    turn2 = graph.invoke(
        {"messages": [HumanMessage(content="Any update since this morning?")]},
        cfg,
    )
    print("Turn 2 messages:", len(turn2["messages"]))
    print("Same thread kept last_status:", turn2["last_status"])


if __name__ == "__main__":
    print(__doc__)
    main()
