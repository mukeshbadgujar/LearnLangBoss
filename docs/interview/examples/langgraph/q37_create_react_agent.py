"""Q37. create_react_agent

THE PROBLEM
You want a tool-calling ReAct agent for bookstore policy without hand-building
the reason->act loop graph.

WHAT WE ARE GOING TO SOLVE
Construct a prebuilt ReAct agent with the chat model from .env and one policy tool.
Building the agent reads LLM_PROVIDER. This file does not send a question.

WHAT THIS EXAMPLE IS ABOUT
Prebuilt agent would answer return policy for The Little Prince / A100 using a
lookup_policy tool. We compile/construct only.

WHAT IT SOLVES
create_react_agent was the langgraph.prebuilt factory for a ReAct tool loop; it
moved toward langchain.agents.create_agent so core langgraph stays low-level.

KEYWORDS
- ReAct: reason + act loop (think, call tool, observe, repeat).
- create_react_agent: older prebuilt graph factory name (langgraph.prebuilt).
- create_agent: current langchain.agents entry point for the same idea.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import chat_model

from langchain_core.tools import tool

# create_react_agent moved: prefer langchain.agents.create_agent on new stacks.
try:
    from langchain.agents import create_agent as build_prebuilt_agent

    _FACTORY = "langchain.agents.create_agent"
except ImportError:  # pragma: no cover
    from langgraph.prebuilt import create_react_agent as build_prebuilt_agent

    _FACTORY = "langgraph.prebuilt.create_react_agent"


@tool
def lookup_policy(topic: str) -> str:
    """Return bookstore return policy text for a topic."""
    return (
        "Unread print books may be returned within 30 days. "
        "Order A100 The Little Prince ($18) is eligible if unread."
    )


def build_agent():
    # chat_model() follows LLM_PROVIDER in .env. This builds the agent only.
    return build_prebuilt_agent(chat_model(), tools=[lookup_policy])


def main() -> None:
    agent = build_agent()
    print("factory:", _FACTORY)
    print("built agent type:", type(agent).__name__)
    print("tools:", [lookup_policy.name])
    print(
        "Not sending a question. Example invoke:\n"
        '  agent.invoke({"messages": [("user", "Can I return A100?")]})'
    )


if __name__ == "__main__":
    print(__doc__)
    main()
