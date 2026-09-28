"""Q51. Architectural difference: LangChain chains vs LangGraph state.

THE PROBLEM
A chain of bookstore helpers passes order_id and policy text as function
arguments; adding a loop or pause means rewriting the plumbing.

WHAT WE ARE GOING TO SOLVE
Contrast manual argument passing with a centralized StateGraph ledger that
every node reads and writes.

WHAT THIS EXAMPLE IS ABOUT
Order A100 lookup -> policy attach -> answer, once as plain functions and once
as a graph with shared state.

WHAT IT SOLVES
LangGraph's global state enables cycles, persistence, and multi-node
coordination without hand-plumbing return values.

KEYWORDS
- Chain: linear composition; you pass data between steps yourself.
- State machine: nodes update a shared state object under runtime control.
- Checkpointer: optional DB-backed persistence of that state per thread_id.
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


def chain_lookup(order_id: str) -> str:
    return f"{order_id} The Little Prince shipped $18"


def chain_policy(order_blob: str) -> str:
    return order_blob + " | unread print 30-day return"


def chain_answer(blob: str) -> str:
    return "Chain: " + blob


class DeskState(TypedDict):
    order_id: str
    blob: str
    answer: str


def g_lookup(state: DeskState) -> dict:
    return {"blob": f"{state['order_id']} The Little Prince shipped $18"}


def g_policy(state: DeskState) -> dict:
    return {"blob": state["blob"] + " | unread print 30-day return"}


def g_answer(state: DeskState) -> dict:
    return {"answer": "Graph: " + state["blob"]}


def build_graph():
    g = StateGraph(DeskState)
    g.add_node("lookup", g_lookup)
    g.add_node("policy", g_policy)
    g.add_node("answer", g_answer)
    g.add_edge(START, "lookup")
    g.add_edge("lookup", "policy")
    g.add_edge("policy", "answer")
    g.add_edge("answer", END)
    return g.compile()


def main() -> None:
    print(chain_answer(chain_policy(chain_lookup("A100"))))
    print(
        build_graph().invoke({"order_id": "A100", "blob": "", "answer": ""})["answer"]
    )
    print("LangGraph keeps one state ledger; chains pass values by hand.")


if __name__ == "__main__":
    print(__doc__)
    main()
