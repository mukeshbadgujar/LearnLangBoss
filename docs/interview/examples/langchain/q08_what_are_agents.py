"""Q8. What are agents?

THE PROBLEM
A fixed chain always runs the same steps. Looking up order A200 and answering
may need a tool only when the shopper mentions an order id.

WHAT WE ARE GOING TO SOLVE
Build a small agent with create_agent that can call lookup_order when needed.

WHAT THIS EXAMPLE IS ABOUT
The shopper asks for the status of order A200 (Clean Code, processing, $42).
The agent decides to call the tool, then replies.

WHAT IT SOLVES
A dynamic loop: model -> tool -> model -> final answer, powered by LangGraph
under create_agent.

KEYWORDS
- Agent: System where the model chooses the next tool or the final answer.
- create_agent: LangChain helper that builds an agent on the LangGraph runtime.
- Dynamic loop: Step count is not fixed; it depends on tool results.
- Tool call: Structured request naming a tool and its arguments.
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
    """Look up bookstore order status and total by id (A100, A200, A300)."""
    return ORDERS.get(order_id.upper(), f"Unknown order {order_id}")


def main() -> None:
    agent = create_agent(
        model,
        tools=[lookup_order],
        system_prompt="You are the bookstore desk. Use lookup_order for order facts.",
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What is the status of order A200?"}]}
    )
    # Last message is usually the final assistant reply.
    print(result["messages"][-1].content)


if __name__ == "__main__":
    print(__doc__)
    main()
