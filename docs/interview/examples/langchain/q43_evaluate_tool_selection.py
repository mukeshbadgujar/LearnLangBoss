"""Q43. How would you evaluate tool selection?

THE PROBLEM
An agent can reach a right final answer after calling the wrong tools three times.

WHAT WE ARE GOING TO SOLVE
Score the trajectory: which tool should have been chosen for a known ticket.

WHAT THIS EXAMPLE IS ABOUT
"Where is order A100?" should call lookup_order, not return_policy.

WHAT IT SOLVES
You catch bad tool descriptions by checking expected tool name, not only the final text.

KEYWORDS
- Trajectory: The sequence of tool calls the agent made.
- Tool selection: Choosing which function to call for a user goal.
- Expected tool: The ground-truth tool name for an eval case.
- Tool description: The text the model reads to decide when to call a tool.

Run:
    python docs/interview/examples/langchain/q43_evaluate_tool_selection.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import chat_model

from langchain.agents import create_agent

model = chat_model()

ORDERS = {"A100": "The Little Prince | shipped | $18"}


def lookup_order(order_id: str) -> str:
    """Look up order status by id (A100, A200, A300)."""
    return ORDERS.get(order_id.upper(), "not found")


def return_policy(_: str) -> str:
    """Return unread-print or digital refund policy text. Not for order status."""
    return "Unread print: 30 days. Digital after download: no refund."


def main() -> None:
    agent = create_agent(
        model,
        tools=[lookup_order, return_policy],
        system_prompt="Bookstore desk. Use tools. Prefer lookup_order for status questions.",
    )
    case = {"input": "Where is order A100?", "expected_tool": "lookup_order"}
    result = agent.invoke({"messages": [{"role": "user", "content": case["input"]}]})

    used = []
    for msg in result["messages"]:
        for call in getattr(msg, "tool_calls", None) or []:
            used.append(call["name"])
    print("tools_used:", used)
    print("pass:", case["expected_tool"] in used)


if __name__ == "__main__":
    print(__doc__)
    main()
