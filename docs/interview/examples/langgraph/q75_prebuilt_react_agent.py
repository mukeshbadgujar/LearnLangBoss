"""Q75. Prebuilt ReAct agents in LangGraph.

THE PROBLEM
Hand-wiring a tool loop for every bookstore FAQ slows the team when a standard
ReAct agent would do.

WHAT WE ARE GOING TO SOLVE
Show create_react_agent (or create_agent fallback) with a real ChatGroq helper
and a shipping tool - without invoking the model (no API call in main).

WHAT THIS EXAMPLE IS ABOUT
Prebuilt agent wired for Order A100 shipping lookup; main only builds and
prints the graph structure.

WHAT IT SOLVES
Reach for prebuilts to move fast; drop to StateGraph when you need custom HITL
or routing the prebuilt does not expose.

KEYWORDS
- create_react_agent: LangGraph prebuilt ReAct tool-calling agent.
- create_agent: LangChain high-level helper (fallback if prebuilt import path differs).
- Prebuilt: batteries-included agent on the LangGraph runtime.
- ChatGroq: Groq chat model used when you actually run the agent.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import chat_model

from langchain_core.tools import tool

# Prefer LangGraph prebuilt; create_react_agent is available in this environment.
from langgraph.prebuilt import create_react_agent

# If create_react_agent were missing, you would instead:
#   from langchain.agents import create_agent
#   agent = create_agent(model, tools=[...])
# because LangChain's create_agent also runs on the LangGraph runtime.


@tool
def lookup_order(order_id: str) -> str:
    """Look up bookstore order shipping status."""
    if order_id == "A100":
        return "A100 The Little Prince: shipped, $18. Policy: unread print 30-day return."
    return f"No order {order_id}"


def build_agent():
    # chat_model() follows LLM_PROVIDER in .env. Call this when you want to run.
    return create_react_agent(chat_model(), tools=[lookup_order])


def main() -> None:
    # Do not construct ChatGroq here: it requires GROQ_API_KEY even before invoke.
    print("Prebuilt import OK:", create_react_agent.__name__)
    print("Tools ready:", [lookup_order.name])
    print("When keyed: agent = build_agent()")
    print("Then: agent.invoke({'messages': [('user', 'Status of order A100?')]})")
    print("Skipped ChatGroq + invoke - no Groq call in this example.")


if __name__ == "__main__":
    print(__doc__)
    main()
