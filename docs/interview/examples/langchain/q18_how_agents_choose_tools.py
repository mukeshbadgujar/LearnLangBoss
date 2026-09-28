"""Q18. How do LangChain agents choose tools?

THE PROBLEM
People assume LangChain has a special picker. Tool choice is the model's tool-
calling ability; LangChain only sends schemas and runs the chosen function.

WHAT WE ARE GOING TO SOLVE
Give two clearly named tools and ask a question that should pick one of them.

WHAT THIS EXAMPLE IS ABOUT
Tools: lookup_order and get_shipping_policy. Question about express shipping
should prefer get_shipping_policy, not invent order data.

WHAT IT SOLVES
Selection quality depends on names and docstrings you write, not a hidden
framework router.

KEYWORDS
- Tool calling: Provider feature where the model returns a structured call.
- Tool schema: Name, description, and argument types sent with the prompt.
- Description quality: Vague docstrings cause wrong tool picks.
- Tool count: Too many tools lowers selection accuracy.
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
    """Use only when the shopper gives an order id like A100, A200, or A300."""
    return ORDERS.get(order_id.upper(), f"Unknown order {order_id}")


@tool
def get_shipping_policy(speed: str) -> str:
    """Use for shipping time questions. speed is 'standard' or 'express'."""
    if speed.lower().startswith("exp"):
        return "Express shipping takes 2 business days."
    return "Standard shipping takes 5 business days."


def main() -> None:
    agent = create_agent(
        model,
        tools=[lookup_order, get_shipping_policy],
        system_prompt="Bookstore desk. Pick the tool that matches the question.",
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "How long is express shipping?"}]}
    )
    # Print tool names the model requested during the run.
    for msg in result["messages"]:
        calls = getattr(msg, "tool_calls", None) or []
        for call in calls:
            print("chose tool:", call.get("name") or call.get("function", {}).get("name"))
    print("final:", result["messages"][-1].content)


if __name__ == "__main__":
    print(__doc__)
    main()
