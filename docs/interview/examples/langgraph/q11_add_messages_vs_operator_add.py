"""Q11. add_messages vs operator.add

THE PROBLEM
operator.add on a list always concatenates - same message twice means duplicates.
Chat history for A100 needs id-aware merge when a partial reply is updated.

WHAT WE ARE GOING TO SOLVE
Show operator.add duplicating a log line, and add_messages replacing a message
with the same id instead of appending a twin.

WHAT THIS EXAMPLE IS ABOUT
Desk logs shipping events with operator.add. Chat channel uses add_messages:
first AI stub, then a full reply with the same id for The Little Prince FAQ.

WHAT IT SOLVES
operator.add = blind concat. add_messages = merge/replace by message id.

KEYWORDS
- operator.add: Python add - for lists, concatenation.
- add_messages: LangGraph reducer that upserts messages by id.
- Message id: identity used to update an in-progress assistant message.
"""

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


class DualState(TypedDict):
    logs: Annotated[list[str], operator.add]
    messages: Annotated[list, add_messages]


# Same id on both AI turns -> add_messages replaces instead of duplicating.
_MSG_ID = "ai-a100-1"


def draft_reply(state: DualState) -> dict:
    return {
        "logs": ["opened A100"],
        "messages": [AIMessage(content="Checking policy...", id=_MSG_ID)],
    }


def finalize_reply(state: DualState) -> dict:
    # Same id -> add_messages replaces; logs get a second identical append.
    return {
        "logs": ["opened A100"],
        "messages": [
            AIMessage(
                content="Unread print books: 30 days (order A100, $18).",
                id=_MSG_ID,
            )
        ],
    }


def build_graph():
    g = StateGraph(DualState)
    g.add_node("draft_reply", draft_reply)
    g.add_node("finalize_reply", finalize_reply)
    g.add_edge(START, "draft_reply")
    g.add_edge("draft_reply", "finalize_reply")
    g.add_edge("finalize_reply", END)
    return g.compile()


def main() -> None:
    graph = build_graph()
    out = graph.invoke(
        {
            "logs": [],
            "messages": [HumanMessage(content="Return window for The Little Prince?")],
        }
    )
    print("logs (operator.add, duplicated line):", out["logs"])
    print("messages count (add_messages replaced same id):", len(out["messages"]))
    for m in out["messages"]:
        print(f"  {m.type}: {m.content}")


if __name__ == "__main__":
    print(__doc__)
    main()
