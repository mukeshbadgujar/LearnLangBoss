"""Q56. Chain vs agent vs graph.

THE PROBLEM
The team uses "chain", "agent", and "graph" interchangeably for the bookstore
support bot.

WHAT WE ARE GOING TO SOLVE
Run the same A100 question three ways: fixed chain of nodes, a tiny
tool-picking loop (agent-shaped), and an explicit StateGraph with a branch.

WHAT THIS EXAMPLE IS ABOUT
"Can I return The Little Prince from order A100?" answered by chain, agent-ish
loop, and graph routing - all plain Python (no fake LLM).

WHAT IT SOLVES
Chain = fixed path; agent = model chooses tools in a loop; graph = you own
state, branches, and persistence.

KEYWORDS
- Chain: predetermined sequence of steps.
- Agent: loop that chooses tools/actions at runtime (here a simple router).
- Graph: explicit state machine with edges you control.
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


def chain_answer(question: str) -> str:
    order = "A100 shipped $18"
    policy = "30-day unread print return"
    return f"Chain[{question}]: {order}; {policy}"


def agentish(question: str) -> str:
    # Stand-in for "model picks a tool" - still a runtime choice, not a fixed pipeline.
    tool = "policy_tool" if "return" in question.lower() else "shipping_tool"
    if tool == "policy_tool":
        return f"Agent[{tool}]: unread print books, 30-day return for A100"
    return f"Agent[{tool}]: A100 The Little Prince shipped"


class GraphState(TypedDict):
    question: str
    answer: str


def route(state: GraphState) -> str:
    return "policy" if "return" in state["question"].lower() else "shipping"


def policy_node(state: GraphState) -> dict:
    return {"answer": "Graph/policy: 30-day unread print return for A100 $18"}


def shipping_node(state: GraphState) -> dict:
    return {"answer": "Graph/shipping: A100 The Little Prince shipped"}


def build_graph():
    g = StateGraph(GraphState)
    g.add_node("policy", policy_node)
    g.add_node("shipping", shipping_node)
    g.add_conditional_edges(START, route, {"policy": "policy", "shipping": "shipping"})
    g.add_edge("policy", END)
    g.add_edge("shipping", END)
    return g.compile()


def main() -> None:
    q = "Can I return The Little Prince from order A100?"
    print(chain_answer(q))
    print(agentish(q))
    print(build_graph().invoke({"question": q, "answer": ""})["answer"])


if __name__ == "__main__":
    print(__doc__)
    main()
