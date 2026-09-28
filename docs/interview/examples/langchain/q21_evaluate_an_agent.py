"""Q21. How do you evaluate an agent?

THE PROBLEM
A correct final sentence can hide four wasted tool calls. The desk needs both
trajectory checks and outcome checks.

WHAT WE ARE GOING TO SOLVE
Run the agent once, then score: did it call lookup_order, and does the reply
mention A300 / Dune?

WHAT THIS EXAMPLE IS ABOUT
Question: status of order A300 (Dune, delivered, $16). We print pass/fail for
tool trajectory and answer content.

WHAT IT SOLVES
A tiny eval harness: trajectory + outcome, the two levels interviewers expect.

KEYWORDS
- Trajectory: Path of tools called (order and redundancy).
- Outcome: Whether the final answer is correct and usable.
- Eval set: Cases built from real failures over time.
- Steps per run: Operational metric for cost and latency.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import chat_model

from langchain.agents import create_agent
from langchain_core.tools import tool

model = chat_model()

ORDERS = {
    "A100": "The Little Prince  -  shipped  -  $18",
    "A200": "Clean Code  -  processing  -  $42",
    "A300": "Dune  -  delivered  -  $16",
}


@tool
def lookup_order(order_id: str) -> str:
    """Look up bookstore order by id (A100, A200, A300)."""
    return ORDERS.get(order_id.upper(), f"Unknown order {order_id}")


def main() -> None:
    agent = create_agent(model, tools=[lookup_order], system_prompt="Bookstore desk.")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What is the status of order A300?"}]}
    )

    tool_names = []
    for msg in result["messages"]:
        for call in getattr(msg, "tool_calls", None) or []:
            tool_names.append(call.get("name") or "")

    final = result["messages"][-1].content or ""
    trajectory_ok = "lookup_order" in tool_names
    outcome_ok = "A300" in final or "Dune" in final or "delivered" in final.lower()

    print("tools called:", tool_names)
    print("trajectory (called lookup_order):", trajectory_ok)
    print("outcome (mentions order facts):", outcome_ok)
    print("final:", final)


if __name__ == "__main__":
    print(__doc__)
    main()
