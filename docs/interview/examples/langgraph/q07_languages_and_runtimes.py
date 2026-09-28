"""Q7. Languages and runtimes

THE PROBLEM
Teams ask whether LangGraph is Python-only or also JS/TS, and what that means
for a bookstore support service.

WHAT WE ARE GOING TO SOLVE
Document the two official SDKs and show a tiny Python graph (this repo's runtime)
that could be mirrored in @langchain/langgraph.

WHAT THIS EXAMPLE IS ABOUT
Same refund fact for A100 in a one-node Python graph. Comments note the JS twin.

WHAT IT SOLVES
Python (langgraph) and JS/TS (@langchain/langgraph) share the core graph model;
this file runs the Python side.

KEYWORDS
- langgraph: Python package for the graph runtime.
- @langchain/langgraph: JavaScript/TypeScript package with core API parity.
- Runtime: the environment executing the compiled graph (CPython, Node, etc.).
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class InfoState(TypedDict):
    order_id: str
    blurb: str


def python_runtime_node(state: InfoState) -> dict:
    return {
        "blurb": (
            f"{state['order_id']}: Python langgraph runtime. "
            "JS/TS twin: npm package @langchain/langgraph (StateGraph, START, END)."
        )
    }


def build_graph():
    g = StateGraph(InfoState)
    g.add_node("python_runtime_node", python_runtime_node)
    g.add_edge(START, "python_runtime_node")
    g.add_edge("python_runtime_node", END)
    return g.compile()


def main() -> None:
    # Official SDKs: Python `langgraph` and JS/TS `@langchain/langgraph`.
    # Prebuilts (supervisor/swarm) matured first on Python; core graph API has parity.
    graph = build_graph()
    out = graph.invoke({"order_id": "A100", "blurb": ""})
    print(out["blurb"])
    print("Runtimes: CPython here; Node/Bun for the JS package.")


if __name__ == "__main__":
    print(__doc__)
    main()
