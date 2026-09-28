"""Q21. Checkpointer vs Store

THE PROBLEM
Thread memory remembers today's A100 ticket. Customer preference 'email receipts'
must survive new threads next month.

WHAT WE ARE GOING TO SOLVE
Use MemorySaver for the active ticket and InMemoryStore for long-term profile.

WHAT THIS EXAMPLE IS ABOUT
Checkpointer holds the open refund stage. Store keeps shopper preference across
thread ids.

WHAT IT SOLVES
Checkpointer = short-term thread-scoped. Store = long-term cross-thread memory.

KEYWORDS
- Checkpointer: snapshots for one thread's run history.
- Store / BaseStore: cross-thread durable memories (namespaced keys).
- InMemoryStore: in-process store implementation for demos.
"""

from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore


class TicketState(TypedDict):
    order_id: str
    stage: str
    receipt_pref: str


def load_pref(state: TicketState, *, store: InMemoryStore) -> dict:
    item = store.get(("shoppers",), "shopper-42")
    pref = item.value["receipts"] if item else "unknown"
    return {"receipt_pref": pref}


def mark_stage(state: TicketState) -> dict:
    return {"stage": "refund_review_A100", "order_id": "A100"}


def build_graph(checkpointer: MemorySaver, store: InMemoryStore):
    g = StateGraph(TicketState)

    def load_pref_node(state: TicketState) -> dict:
        return load_pref(state, store=store)

    g.add_node("load_pref", load_pref_node)
    g.add_node("mark_stage", mark_stage)
    g.add_edge(START, "load_pref")
    g.add_edge("load_pref", "mark_stage")
    g.add_edge("mark_stage", END)
    return g.compile(checkpointer=checkpointer)


def main() -> None:
    store = InMemoryStore()
    store.put(("shoppers",), "shopper-42", {"receipts": "email", "title_fav": "The Little Prince"})

    graph = build_graph(MemorySaver(), store)
    # Thread 1 - today's ticket
    t1 = {"configurable": {"thread_id": "ticket-today"}}
    out1 = graph.invoke({"order_id": "", "stage": "", "receipt_pref": ""}, t1)
    # Thread 2 - new conversation, same store profile
    t2 = {"configurable": {"thread_id": "ticket-next-month"}}
    out2 = graph.invoke({"order_id": "", "stage": "", "receipt_pref": ""}, t2)

    print("thread today stage:", out1["stage"], "| pref:", out1["receipt_pref"])
    print("new thread stage:", out2["stage"], "| pref still:", out2["receipt_pref"])
    print("Checkpoints differ per thread; Store preference is shared.")


if __name__ == "__main__":
    print(__doc__)
    main()
