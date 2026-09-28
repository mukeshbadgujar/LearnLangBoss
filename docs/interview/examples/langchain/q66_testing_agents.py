"""Q66. How do you approach Testing and Evaluation for AI agents?

THE PROBLEM
Agent demos look fine until a tool path regresses and nobody has a regression set.

WHAT WE ARE GOING TO SOLVE
Define cases with expected tools and answer checks, then run the bookstore agent against them.

WHAT THIS EXAMPLE IS ABOUT
"Where is A100?" must call lookup_order; a shipping FAQ must mention 5 days.

WHAT IT SOLVES
You have a minimal agent eval harness: trajectory + content assertions.

KEYWORDS
- Agent eval: Scoring both tool trajectory and final answer quality.
- Expected tool: Ground-truth tool name for a case.
- Regression set: Cases you re-run whenever prompts or tools change.
- CI integration: Running the eval set automatically on each change.

Run:
    python docs/interview/examples/langchain/q66_testing_agents.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import chat_model

from langchain.agents import create_agent

model = chat_model()

ORDERS = {
    "A100": "The Little Prince | shipped | $18",
    "A200": "Clean Code | processing | $42",
    "A300": "Dune | delivered | $16",
}


def lookup_order(order_id: str) -> str:
    """Look up bookstore order status."""
    return ORDERS.get(order_id.upper(), "not found")


def shipping_policy(_: str = "") -> str:
    """Return standard and express shipping times."""
    return "Standard 5 business days. Express 2 days."


CASES = [
    {"q": "Where is order A100?", "expect_tool": "lookup_order", "must_include": "shipped"},
    {"q": "How long is standard shipping?", "expect_tool": "shipping_policy", "must_include": "5"},
]


def main() -> None:
    agent = create_agent(
        model,
        tools=[lookup_order, shipping_policy],
        system_prompt="Bookstore desk. Use tools for facts. Be brief.",
    )
    for case in CASES:
        result = agent.invoke({"messages": [{"role": "user", "content": case["q"]}]})
        tools = []
        for msg in result["messages"]:
            for call in getattr(msg, "tool_calls", None) or []:
                tools.append(call["name"])
        final = result["messages"][-1].content
        tool_ok = case["expect_tool"] in tools
        text_ok = case["must_include"].lower() in final.lower()
        print(case["q"], "tools=", tools, "tool_ok=", tool_ok, "text_ok=", text_ok)


if __name__ == "__main__":
    print(__doc__)
    main()
