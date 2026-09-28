"""Q49. LangGraph vs LangChain in 2026 - not a rivalry.

THE PROBLEM
Interviewers still ask "LangGraph or LangChain?" as if they were competing
products for the bookstore bot.

WHAT WE ARE GOING TO SOLVE
Show that a prebuilt agent idea and a hand-built StateGraph both sit on the
same runtime idea: LangGraph is the engine; LangChain is the batteries layer.

WHAT THIS EXAMPLE IS ABOUT
Order A100 policy answer built two ways: a tiny LCEL-style chain function, and
an explicit StateGraph with the same plain nodes - no fake models.

WHAT IT SOLVES
You answer: use LangChain prebuilts to move fast; drop to LangGraph when you
need custom control flow, multi-agent routing, or fine-grained persistence.

KEYWORDS
- LangGraph runtime: the state-machine engine under modern LangChain agents.
- Prebuilt agent: high-level helper (e.g. create_react_agent) on that runtime.
- Custom graph: StateGraph you wire yourself for HITL, branching, persistence.
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


def langchain_style_chain(question: str) -> str:
    """Batteries-included mental model: compose steps in a straight line."""
    order = "A100 The Little Prince shipped $18"
    policy = "Unread print books: 30-day return"
    return f"Chain answer: {order}. {policy}. Q={question}"


class GraphState(TypedDict):
    question: str
    order: str
    policy: str
    answer: str


def load_order(state: GraphState) -> dict:
    return {"order": "A100 The Little Prince shipped $18"}


def load_policy(state: GraphState) -> dict:
    return {"policy": "Unread print books: 30-day return"}


def compose(state: GraphState) -> dict:
    return {
        "answer": f"Graph answer: {state['order']}. {state['policy']}. Q={state['question']}"
    }


def build_graph():
    g = StateGraph(GraphState)
    g.add_node("order", load_order)
    g.add_node("policy", load_policy)
    g.add_node("compose", compose)
    g.add_edge(START, "order")
    g.add_edge("order", "policy")
    g.add_edge("policy", "compose")
    g.add_edge("compose", END)
    return g.compile()


def main() -> None:
    q = "Can I return The Little Prince?"
    print(langchain_style_chain(q))
    print(
        build_graph().invoke(
            {"question": q, "order": "", "policy": "", "answer": ""}
        )["answer"]
    )
    print("2026 framing: LangGraph = engine; LangChain = batteries on top.")


if __name__ == "__main__":
    print(__doc__)
    main()
