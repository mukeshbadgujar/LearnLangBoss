"""Q6. MessagesState

THE PROBLEM
A chat-only FAQ bot only needs conversation history. Hand-rolling TypedDict +
add_messages every time is boilerplate. Business fields need a custom schema.

WHAT WE ARE GOING TO SOLVE
Use built-in MessagesState for a short refund chat about A100, appending real
message objects (no model call - we construct AIMessage to show the reducer).

WHAT THIS EXAMPLE IS ABOUT
Human asks about returning The Little Prince. Desk node appends a policy AI
reply. MessagesState keeps the list via add_messages.

WHAT IT SOLVES
MessagesState = ready-made messages channel. Custom TypedDict when you also
need order_id, priority, etc.

KEYWORDS
- MessagesState: built-in state with a messages list and add_messages reducer.
- add_messages: merges chat turns by id (append or replace), not blind concat.
- HumanMessage / AIMessage: LangChain message types for user and assistant turns.
"""

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph


def desk_reply(state: MessagesState) -> dict:
    # A real node would call ChatGroq here. We build AIMessage to show the reducer.
    last = state["messages"][-1].content
    reply = (
        "Order A100 (The Little Prince, $18): unread print books may be "
        "returned within 30 days."
        if "return" in last.lower() or "refund" in last.lower()
        else "How can the bookstore help with your order?"
    )
    return {"messages": [AIMessage(content=reply)]}


def build_graph():
    g = StateGraph(MessagesState)
    g.add_node("desk_reply", desk_reply)
    g.add_edge(START, "desk_reply")
    g.add_edge("desk_reply", END)
    return g.compile()


def main() -> None:
    graph = build_graph()
    out = graph.invoke(
        {"messages": [HumanMessage(content="Can I return The Little Prince on A100?")]}
    )
    for m in out["messages"]:
        print(f"{m.type}: {m.content}")


if __name__ == "__main__":
    print(__doc__)
    main()
