"""Q7. What are tools?

THE PROBLEM
The model invents order totals. The desk already has a Python lookup for
orders A100, A200, and A300, but the model cannot call it by itself.

WHAT WE ARE GOING TO SOLVE
Wrap the lookup with @tool so an LLM (or agent) can request it by name.

WHAT THIS EXAMPLE IS ABOUT
We define lookup_order and call it directly for A100 (The Little Prince,
shipped, $18). The docstring is what the model would read at tool-choice time.

WHAT IT SOLVES
A real tool schema exists. The model never runs Python; your code does when
asked. Agents in later files bind this same pattern.

KEYWORDS
- Tool: A Python function plus metadata the model can request.
- @tool: Decorator that builds the schema from name, args, and docstring.
- Docstring-as-prompt: The description the model uses to decide when to call.
- Execution boundary: Model returns a call request; your app runs the function.
"""

from langchain_core.tools import tool

ORDERS = {
    "A100": {"title": "The Little Prince", "status": "shipped", "total": 18},
    "A200": {"title": "Clean Code", "status": "processing", "total": 42},
    "A300": {"title": "Dune", "status": "delivered", "total": 16},
}


@tool
def lookup_order(order_id: str) -> str:
    """Look up a bookstore order by id like A100, A200, or A300."""
    order = ORDERS.get(order_id.upper())
    if not order:
        return f"No order {order_id}"
    return f"{order_id}: {order['title']}, {order['status']}, ${order['total']}"


def main() -> None:
    # Direct invoke shows the tool works; agents later let the model choose it.
    print(lookup_order.invoke({"order_id": "A100"}))
    print("Name:", lookup_order.name)
    print("Description:", lookup_order.description)


if __name__ == "__main__":
    print(__doc__)
    main()
