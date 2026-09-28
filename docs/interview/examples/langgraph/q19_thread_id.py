"""Q19. thread_id

THE PROBLEM
Two shoppers chat at once. Without thread_id, a checkpointer cannot tell whose
A100 refund history to load - persistence is useless.

WHAT WE ARE GOING TO SOLVE
Run two threads with MemorySaver and show isolated state.

WHAT THIS EXAMPLE IS ABOUT
Thread desk-a100 tracks The Little Prince refund. Thread desk-b200 is a shipping
question. Same graph, different configurable thread_id.

WHAT IT SOLVES
thread_id is the primary key for checkpoint history and multi-tenant isolation.

KEYWORDS
- thread_id: string id for one conversation/run history.
- configurable: config dict slot for runtime options like thread_id.
- Multi-tenancy: many users on one app with isolated data.
"""

from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph


class ChatState(TypedDict):
    topic: str
    last: str


def stamp(state: ChatState) -> dict:
    return {"last": f"handled:{state['topic']}"}


def build_graph(checkpointer: MemorySaver):
    g = StateGraph(ChatState)
    g.add_node("stamp", stamp)
    g.add_edge(START, "stamp")
    g.add_edge("stamp", END)
    return g.compile(checkpointer=checkpointer)


def main() -> None:
    graph = build_graph(MemorySaver())
    a = {"configurable": {"thread_id": "desk-a100"}}
    b = {"configurable": {"thread_id": "desk-b200"}}
    graph.invoke({"topic": "refund The Little Prince $18", "last": ""}, a)
    graph.invoke({"topic": "where is my shipment?", "last": ""}, b)
    print("A100 thread:", graph.get_state(a).values)
    print("B200 thread:", graph.get_state(b).values)
    print("Without thread_id, the checkpointer has nowhere to hang history.")


if __name__ == "__main__":
    print(__doc__)
    main()
