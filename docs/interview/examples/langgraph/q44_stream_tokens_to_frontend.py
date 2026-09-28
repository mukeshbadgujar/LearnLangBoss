"""Q44. Stream graph output toward a frontend.

THE PROBLEM
The bookstore chat widget should show live progress as the multi-step ticket
graph runs (which specialist spoke), not wait for the final JSON blob.

WHAT WE ARE GOING TO SOLVE
Demonstrate graph.stream with updates (and note messages for real tokens).
Map each chunk to a fake send_to_frontend callback.

WHAT THIS EXAMPLE IS ABOUT
Supervisor-style two nodes: shipping status then refund eligibility for A100.
We stream node updates the UI would render as status lines.

WHAT IT SOLVES
A clear pattern for pushing graph progress to a frontend. Token streaming
needs a chat model in the node; here we stream real node updates so the file runs.

KEYWORDS
- stream_mode updates: each chunk is {node_name: partial_state}.
- stream_mode messages: (token, metadata) pairs; metadata['langgraph_node'] names the speaker.
- Frontend bridge: map stream chunks to WebSocket / SSE events.
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class DeskState(TypedDict):
    order_id: str
    line: str


def shipping_agent(state: DeskState) -> dict:
    return {"line": f"Shipping: {state['order_id']} The Little Prince left warehouse"}


def refund_agent(state: DeskState) -> dict:
    return {"line": "Refund: $18 eligible under 30-day unread print policy"}


def build_graph():
    g = StateGraph(DeskState)
    g.add_node("shipping_agent", shipping_agent)
    g.add_node("refund_agent", refund_agent)
    g.add_edge(START, "shipping_agent")
    g.add_edge("shipping_agent", "refund_agent")
    g.add_edge("refund_agent", END)
    return g.compile()


def send_to_frontend(text: str, source: str) -> None:
    # Stand-in for WebSocket / Server-Sent Events to the chat UI.
    print(f"[UI <- {source}] {text}")


def main() -> None:
    graph = build_graph()
    seed = {"order_id": "A100", "line": ""}

    # Real runnable demo: stream node updates to the UI.
    for update in graph.stream(seed, stream_mode="updates"):
        for node_name, partial in update.items():
            send_to_frontend(partial["line"], source=node_name)

    # Token-by-token path (needs a chat model inside the node):
    # for mode, payload in graph.stream(seed, stream_mode=["messages", "updates"]):
    #     if mode == "messages":
    #         token, metadata = payload
    #         send_to_frontend(token.content, source=metadata.get("langgraph_node"))
    print("Comment: add ChatGroq in a node to unlock stream_mode='messages' tokens.")


if __name__ == "__main__":
    print(__doc__)
    main()
