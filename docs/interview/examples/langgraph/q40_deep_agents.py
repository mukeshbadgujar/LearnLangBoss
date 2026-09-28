"""Q40. Deep agents beyond a bare ReAct loop.

THE PROBLEM
A single ReAct loop on a long bookstore ticket (plan return, check policy,
draft refund, notify warehouse) loses the plan and fills the context with
tool noise.

WHAT WE ARE GOING TO SOLVE
Model a deep-agent shape: plan up front, delegate sub-tasks, offload notes to
a scratchpad channel - with plain Python nodes (no fake LLM).

WHAT THIS EXAMPLE IS ABOUT
Order A100 return request. Planner writes steps; workers write to a scratchpad;
a final node summarizes without replaying every scratch line.

WHAT IT SOLVES
Long-horizon support work keeps a plan and an external scratchpad instead of
stuffing every intermediate tool result into one chat history.

KEYWORDS
- ReAct: reason, call a tool, observe, repeat in one loop.
- Deep agent: planning + delegation + scratchpad for long-horizon tasks.
- Scratchpad: external notes (here a state list) so the live context stays small.
- Delegation: handing a sub-task to a focused worker node or sub-agent.
"""

from typing import Annotated, TypedDict
import operator

from langgraph.graph import END, START, StateGraph


class DeepDesk(TypedDict):
    order_id: str
    plan: list[str]
    scratchpad: Annotated[list[str], operator.add]
    summary: str


def plan_return(state: DeepDesk) -> dict:
    # Explicit plan up front - what a deep agent adds before tool spam.
    return {
        "plan": [
            "confirm order A100 shipped",
            "check 30-day unread print policy",
            "draft refund if eligible",
        ]
    }


def worker_shipping(state: DeepDesk) -> dict:
    return {"scratchpad": [f"{state['order_id']}: carrier confirms shipped $18"]}


def worker_policy(state: DeepDesk) -> dict:
    return {"scratchpad": ["policy: unread print books, 30-day return"]}


def summarize(state: DeepDesk) -> dict:
    # Offload: summary uses plan + short scratch lines, not a full chat dump.
    lines = "; ".join(state["scratchpad"])
    return {
        "summary": (
            f"Plan ({len(state['plan'])} steps) done for {state['order_id']}. "
            f"Notes: {lines}. Eligible under 30-day unread print policy."
        )
    }


def build_graph():
    g = StateGraph(DeepDesk)
    g.add_node("plan", plan_return)
    g.add_node("shipping", worker_shipping)
    g.add_node("policy", worker_policy)
    g.add_node("summarize", summarize)
    g.add_edge(START, "plan")
    g.add_edge("plan", "shipping")
    g.add_edge("shipping", "policy")
    g.add_edge("policy", "summarize")
    g.add_edge("summarize", END)
    return g.compile()


def main() -> None:
    out = build_graph().invoke(
        {"order_id": "A100", "plan": [], "scratchpad": [], "summary": ""}
    )
    print("Plan:", out["plan"])
    print("Scratchpad entries:", out["scratchpad"])
    print(out["summary"])


if __name__ == "__main__":
    print(__doc__)
    main()
