"""Q70. Observability with a callback handler on a graph.

THE PROBLEM
Without callbacks, the bookstore desk graph is a black box when a refund path
misbehaves.

WHAT WE ARE GOING TO SOLVE
Attach a simple BaseCallbackHandler that logs node starts - the same hook
surface LangSmith tracing builds on.

WHAT THIS EXAMPLE IS ABOUT
A100 shipping -> policy graph invoked with a callback handler recording events.

WHAT IT SOLVES
Local proof that graphs emit callback events you (or LangSmith) can observe.
See q73 for what to inspect inside LangSmith itself.

KEYWORDS
- Callback handler: object notified on chain/graph/model events.
- LangSmith: hosted tracer that consumes the same callback/tracing pipeline.
- Tracing: recording nested runs (spans) for later debug.
"""

from typing import Any, TypedDict
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langgraph.graph import END, START, StateGraph


class DeskState(TypedDict):
    order_id: str
    text: str


class BookstoreTraceHandler(BaseCallbackHandler):
    """Minimal handler - in production LangSmith's tracer fills this role."""

    def __init__(self) -> None:
        self.events: list[str] = []

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        name = (serialized or {}).get("name") or (metadata or {}).get("langgraph_node")
        self.events.append(f"start:{name}")


def shipping(state: DeskState) -> dict:
    return {"text": f"{state['order_id']} The Little Prince shipped $18"}


def policy(state: DeskState) -> dict:
    return {"text": state["text"] + " | 30-day unread print return"}


def build_graph():
    g = StateGraph(DeskState)
    g.add_node("shipping", shipping)
    g.add_node("policy", policy)
    g.add_edge(START, "shipping")
    g.add_edge("shipping", "policy")
    g.add_edge("policy", END)
    return g.compile()


def main() -> None:
    handler = BookstoreTraceHandler()
    out = build_graph().invoke(
        {"order_id": "A100", "text": ""},
        config={"callbacks": [handler]},
    )
    print("Result:", out["text"])
    print("Callback events:", handler.events)
    print("Ship these same runs to LangSmith with LANGCHAIN_TRACING_V2=true.")


if __name__ == "__main__":
    print(__doc__)
    main()
