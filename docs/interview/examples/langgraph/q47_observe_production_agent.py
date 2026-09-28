"""Q47. Observe a production LangGraph agent.

THE PROBLEM
A refund decision on Order A100 looked wrong in production, and the on-call
engineer has no step-by-step record of which node wrote what.

WHAT WE ARE GOING TO SOLVE
Show the two tools interviewers expect: LangSmith traces for production, and
LangGraph Studio for interactive debugging - illustrated with a local span log
from a real graph run (no fake chat model).

WHAT THIS EXAMPLE IS ABOUT
A tiny A100 ticket graph that records per-node spans the way you would later
inspect them in LangSmith.

WHAT IT SOLVES
You can name production observability (LangSmith) and local step debugging
(Studio) and show what a span list looks like.

KEYWORDS
- LangSmith: hosted tracing/eval for LangChain and LangGraph runs.
- LangGraph Studio: visual debugger for stepping through graph state locally.
- Span: one recorded unit of work (a node, tool call, or model call).
- thread_id: correlates all checkpoints and traces for one customer ticket.
"""

from typing import TypedDict
from time import perf_counter

from langgraph.graph import END, START, StateGraph


class TicketState(TypedDict):
    order_id: str
    detail: str


# Local stand-in for what LangSmith would store per node.
SPANS: list[dict] = []


def traced(name: str):
    def wrap(fn):
        def inner(state: TicketState) -> dict:
            t0 = perf_counter()
            out = fn(state)
            SPANS.append(
                {
                    "name": name,
                    "input_order": state["order_id"],
                    "output": out,
                    "ms": round((perf_counter() - t0) * 1000, 3),
                }
            )
            return out

        return inner

    return wrap


@traced("lookup_order")
def lookup_order(state: TicketState) -> dict:
    return {"detail": f"{state['order_id']} The Little Prince shipped $18"}


@traced("policy_check")
def policy_check(state: TicketState) -> dict:
    return {"detail": state["detail"] + " | policy: 30-day unread print"}


def build_graph():
    g = StateGraph(TicketState)
    g.add_node("lookup_order", lookup_order)
    g.add_node("policy_check", policy_check)
    g.add_edge(START, "lookup_order")
    g.add_edge("lookup_order", "policy_check")
    g.add_edge("policy_check", END)
    return g.compile()


def main() -> None:
    SPANS.clear()
    out = build_graph().invoke({"order_id": "A100", "detail": ""})
    print("Final:", out["detail"])
    print("Production: open the LangSmith trace for thread_id=ticket-A100")
    print("Local debug: LangGraph Studio to step node-by-node")
    print("Recorded spans (what you would inspect):")
    for span in SPANS:
        print(span)


if __name__ == "__main__":
    print(__doc__)
    main()
