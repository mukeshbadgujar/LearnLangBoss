"""Q49. LangGraph vs. LangChain.

THE PROBLEM
People treat LangGraph and LangChain as rivals, or think create_agent is unrelated to graphs.

WHAT WE ARE GOING TO SOLVE
Contrast a one-shot LCEL chain with a tiny LangGraph cycle that retries until policy is clear.

WHAT THIS EXAMPLE IS ABOUT
Chain answers shipping in one shot; graph loops once if the draft lacks the number of days.

WHAT IT SOLVES
You see when a fixed chain is enough and when an explicit graph cycle earns its keep.

KEYWORDS
- LangGraph: Low-level runtime for stateful graphs with cycles, checkpoints, and interrupts.
- LangChain chain: A mostly linear composition of runnables.
- StateGraph: Graph builder where nodes update shared state and edges choose the next node.
- create_agent: High-level agent API built on LangGraph under the hood.

Run:
    python docs/interview/examples/langchain/q49_langgraph_vs_langchain.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import chat_model

from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

model = chat_model()


class DeskState(TypedDict):
    question: str
    draft: str
    tries: int


def draft_node(state: DeskState) -> dict:
    msg = model.invoke([
        SystemMessage("Bookstore desk. Mention the exact day count. Policy: standard shipping 5 days."),
        HumanMessage(state["question"]),
    ])
    return {"draft": msg.content, "tries": state["tries"] + 1}


def ok(state: DeskState) -> str:
    # Cycle until the draft includes "5" or we hit a try cap.
    if "5" in state["draft"] or state["tries"] >= 2:
        return "done"
    return "again"


def main() -> None:
    # One-shot LangChain-style call (no graph).
    oneshot = model.invoke([
        SystemMessage("One sentence. Standard shipping is 5 business days."),
        HumanMessage("How long is standard shipping?"),
    ]).content
    print("chain:", oneshot)

    graph = StateGraph(DeskState)
    graph.add_node("draft", draft_node)
    graph.add_edge(START, "draft")
    graph.add_conditional_edges("draft", ok, {"done": END, "again": "draft"})
    app = graph.compile()
    result = app.invoke({"question": "How long is standard shipping?", "draft": "", "tries": 0})
    print("graph:", result["draft"], f"(tries={result['tries']})")


if __name__ == "__main__":
    print(__doc__)
    main()
