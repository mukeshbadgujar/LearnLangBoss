"""Q67. Stream modes values vs updates vs messages on one graph.

THE PROBLEM
Docs list many stream modes; the team wants to *see* values, updates, and
messages side by side on one small bookstore graph.

WHAT WE ARE GOING TO SOLVE
Print all three modes for the same A100 two-node graph. Messages mode will be
empty/absent without a chat model - we still request it and explain why.

WHAT THIS EXAMPLE IS ABOUT
lookup -> policy on Order A100; compare chunk shapes.

WHAT IT SOLVES
Muscle memory for which mode feeds which UI (state panel vs chat tokens).

KEYWORDS
- values: full state snapshot after each superstep.
- updates: per-node partial dicts.
- messages: LLM token stream (needs a model in a node).
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class TinyState(TypedDict):
    order_id: str
    shipping: str
    policy: str


def lookup(state: TinyState) -> dict:
    return {"shipping": f"{state['order_id']} shipped $18"}


def policy(state: TinyState) -> dict:
    return {"policy": "unread print 30-day return"}


def build_graph():
    g = StateGraph(TinyState)
    g.add_node("lookup", lookup)
    g.add_node("policy", policy)
    g.add_edge(START, "lookup")
    g.add_edge("lookup", "policy")
    g.add_edge("policy", END)
    return g.compile()


def main() -> None:
    graph = build_graph()
    seed = {"order_id": "A100", "shipping": "", "policy": ""}

    print("=== values ===")
    for c in graph.stream(seed, stream_mode="values"):
        print(c)

    print("=== updates ===")
    for c in graph.stream(seed, stream_mode="updates"):
        print(c)

    print("=== messages (expect nothing: no chat model in nodes) ===")
    chunks = list(graph.stream(seed, stream_mode="messages"))
    print(chunks if chunks else "(no message tokens - add ChatGroq in a node)")


if __name__ == "__main__":
    print(__doc__)
    main()
