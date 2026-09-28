"""Q17. What is an AI agent?

THE PROBLEM
A normal program hard-codes branching. The desk wants the model to decide
whether to look up an order or just answer a policy question.

WHAT WE ARE GOING TO SOLVE
Run a tiny agent loop: goal + tools -> model picks tool or answers.

WHAT THIS EXAMPLE IS ABOUT
Shopper asks about order A100 (The Little Prince, shipped, $18). The agent
calls lookup_order, then replies with status and total.

WHAT IT SOLVES
You see the agent loop: model decides control flow at runtime.

KEYWORDS
- AI agent: System where the model chooses tools and when to stop.
- Control flow: Who decides the next step (you vs the model).
- Tool loop: Call tool -> feed result -> decide again.
- Goal: The user request the agent tries to finish.
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
    """Look up a bookstore order by id (A100, A200, A300)."""
    return ORDERS.get(order_id.upper(), f"Unknown order {order_id}")


def main() -> None:
    agent = create_agent(model, tools=[lookup_order], system_prompt="Bookstore desk agent.")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Tell me about order A100."}]}
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    print(__doc__)
    main()
