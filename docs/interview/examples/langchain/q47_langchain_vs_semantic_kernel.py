"""Q47. LangChain vs. Semantic Kernel.

THE PROBLEM
An interview asks which framework to pick without naming the team's language and cloud.

WHAT WE ARE GOING TO SOLVE
Show a real LangChain bookstore tool call, and note when Semantic Kernel fits better.

WHAT THIS EXAMPLE IS ABOUT
lookup_order for A200 via create_agent  -  Python orchestration with Groq.

WHAT IT SOLVES
You can say: Semantic Kernel fits .NET/Azure shops; LangChain fits Python ecosystems.

KEYWORDS
- Semantic Kernel: Microsoft's AI orchestration SDK, strong in C# and Azure.
- LangChain: Python-first framework with a large integration ecosystem.
- Integration: Ready-made connectors to models, stores, and tools.
- Ecosystem: The set of community packages and examples around a framework.

Run:
    python docs/interview/examples/langchain/q47_langchain_vs_semantic_kernel.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import chat_model

from langchain.agents import create_agent

model = chat_model()

# Semantic Kernel (not imported): would emphasize skills/plugins in a .NET or Azure host.
# Here we show the LangChain shape of the same bookstore lookup.


def lookup_order(order_id: str) -> str:
    """Return status for A100/A200/A300."""
    catalog = {
        "A100": "The Little Prince shipped $18",
        "A200": "Clean Code processing $42",
        "A300": "Dune delivered $16",
    }
    return catalog.get(order_id.upper(), "unknown")


def main() -> None:
    agent = create_agent(
        model,
        tools=[lookup_order],
        system_prompt="Bookstore desk. Use lookup_order for status questions.",
    )
    out = agent.invoke({"messages": [{"role": "user", "content": "Status of order A200?"}]})
    print(out["messages"][-1].content)
    print("Pick Semantic Kernel for C#/Azure-first orgs; LangChain for Python-first stacks.")


if __name__ == "__main__":
    print(__doc__)
    main()
