"""Q29. Design a customer support assistant.

THE PROBLEM
A chatbot only returns text. Support also needs order lookups and refund actions.

WHAT WE ARE GOING TO SOLVE
Combine policy knowledge with tools, and keep refunds behind an approval check.

WHAT THIS EXAMPLE IS ABOUT
A shopper asks where order A100 is, then whether they can get a refund on unread print books.

WHAT IT SOLVES
The agent calls lookup_order for status and answers policy without inventing order facts.

KEYWORDS
- Support assistant: A bot that both answers questions and takes actions on tickets.
- Tool: A function the model can call, such as looking up an order.
- Escalation: Handing the ticket to a human when confidence is low or money moves.
- Router: Classifying intent first; agents are reserved for open-ended cases.

Run:
    python docs/interview/examples/langchain/q29_customer_support_assistant.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import chat_model

from langchain.agents import create_agent

model = chat_model()

ORDERS = {
    "A100": {"title": "The Little Prince", "status": "shipped", "price": 18},
    "A200": {"title": "Clean Code", "status": "processing", "price": 42},
    "A300": {"title": "Dune", "status": "delivered", "price": 16},
}


def lookup_order(order_id: str) -> str:
    """Look up bookstore order status by id like A100."""
    order = ORDERS.get(order_id.upper())
    if not order:
        return f"No order {order_id}"
    return f"{order_id}: {order['title']} is {order['status']} (${order['price']})"


def return_policy(kind: str) -> str:
    """Return policy for 'print' or 'digital' books."""
    if kind.lower() == "print":
        return "Unread print books: return within 30 days."
    if kind.lower() == "digital":
        return "Digital books: not refundable after download."
    return "Unknown kind. Use print or digital."


def main() -> None:
    agent = create_agent(
        model,
        tools=[lookup_order, return_policy],
        system_prompt=(
            "You are the bookstore support desk. Use tools for facts. "
            "Never invent order status. Refunds need a human  -  say so."
        ),
    )
    question = "Where is order A100, and can I return an unread print book?"
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    print(result["messages"][-1].content)


if __name__ == "__main__":
    print(__doc__)
    main()
