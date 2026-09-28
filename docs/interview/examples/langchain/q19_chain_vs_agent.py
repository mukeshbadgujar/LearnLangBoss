"""Q19. What is the difference between a chain and an agent?

THE PROBLEM
Interviewers ask chain vs agent. The desk needs a side-by-side: fixed shipping
FAQ vs order lookup that may call a tool.

WHAT WE ARE GOING TO SOLVE
Run one chain (always prompt -> model) and one agent (may call lookup_order).

WHAT THIS EXAMPLE IS ABOUT
Chain answers standard shipping days. Agent answers status for order A200
(Clean Code, processing, $42) using a tool.

WHAT IT SOLVES
Chain = you decide the path at build time. Agent = model decides at runtime.

KEYWORDS
- Chain: Fixed steps every run; known model-call count.
- Agent: Model chooses tools; path and call count vary.
- Build time vs runtime: When control flow is decided.
- Determinism: Chains are more predictable; agents trade that for flexibility.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import chat_model

from langchain.agents import create_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

model = chat_model()

ORDERS = {
    "A100": "The Little Prince  -  shipped  -  $18",
    "A200": "Clean Code  -  processing  -  $42",
    "A300": "Dune  -  delivered  -  $16",
}


@tool
def lookup_order(order_id: str) -> str:
    """Look up bookstore order by id."""
    return ORDERS.get(order_id.upper(), f"Unknown order {order_id}")


def main() -> None:
    prompt = ChatPromptTemplate.from_template(
        "Policy: standard shipping is 5 business days. Answer briefly.\nQ: {q}"
    )
    chain = prompt | model | StrOutputParser()
    print("chain:", chain.invoke({"q": "How long is standard shipping?"}))

    agent = create_agent(model, tools=[lookup_order], system_prompt="Bookstore desk.")
    out = agent.invoke({"messages": [{"role": "user", "content": "Status of order A200?"}]})
    print("agent:", out["messages"][-1].content)


if __name__ == "__main__":
    print(__doc__)
    main()
