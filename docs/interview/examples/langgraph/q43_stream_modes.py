"""Q43. Which stream mode - messages vs values (short version).

THE PROBLEM
The support UI needs live updates while a ticket graph runs, but engineers keep
asking which stream_mode to pick.

WHAT WE ARE GOING TO SOLVE
Show values (full state after each superstep) versus updates (only what each
node returned). Comment when messages (token stream) is the right choice.

WHAT THIS EXAMPLE IS ABOUT
A two-node Order A100 lookup -> policy note graph streamed both ways.

WHAT IT SOLVES
Use values/updates for state UIs; use messages when a chat model inside a node
must stream tokens.

KEYWORDS
- values: emit the full graph state after every superstep.
- updates: emit only the partial dict each node returned.
- messages: token-level LLM output with node metadata (needs a chat model in a node).
- stream_mode: argument to graph.stream / astream selecting what chunks look like.
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class TicketState(TypedDict):
    order_id: str
    shipping: str
    policy: str


def lookup_shipping(state: TicketState) -> dict:
    return {"shipping": f"{state['order_id']} The Little Prince shipped $18"}


def attach_policy(state: TicketState) -> dict:
    return {"policy": "Unread print books: 30-day return"}


def build_graph():
    g = StateGraph(TicketState)
    g.add_node("shipping", lookup_shipping)
    g.add_node("policy", attach_policy)
    g.add_edge(START, "shipping")
    g.add_edge("shipping", "policy")
    g.add_edge("policy", END)
    return g.compile()


def main() -> None:
    graph = build_graph()
    seed = {"order_id": "A100", "shipping": "", "policy": ""}

    print("--- stream_mode='values' (full state each step) ---")
    for chunk in graph.stream(seed, stream_mode="values"):
        print(chunk)

    print("--- stream_mode='updates' (node diffs only) ---")
    for chunk in graph.stream(seed, stream_mode="updates"):
        print(chunk)

    # messages mode streams LLM tokens; this file has no chat model in a node.
    print("Tip: use stream_mode='messages' when a node calls a chat model.")


if __name__ == "__main__":
    print(__doc__)
    main()
