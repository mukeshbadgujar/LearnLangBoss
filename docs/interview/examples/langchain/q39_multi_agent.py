"""Q39. How would you build a multi-agent system?

THE PROBLEM
One agent with every bookstore tool mixes shipping facts with refund actions and wastes context.

WHAT WE ARE GOING TO SOLVE
Split into specialists and let a tiny supervisor route by intent.

WHAT THIS EXAMPLE IS ABOUT
Shipping questions go to the shipping specialist; refund questions go to the returns specialist.

WHAT IT SOLVES
Each specialist sees a narrow prompt, so tool choice and tokens stay focused.

KEYWORDS
- Multi-agent: Several agents cooperating, each with a smaller job.
- Supervisor: A coordinator that routes work to specialists.
- Handoff: Passing control directly from one agent to another.
- Scoped context: Giving each agent only the facts it needs.

Run:
    python docs/interview/examples/langchain/q39_multi_agent.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import chat_model

from langchain_core.messages import HumanMessage, SystemMessage

model = chat_model()


def shipping_agent(question: str) -> str:
    msgs = [
        SystemMessage("Shipping specialist. Policy: standard 5 days, express 2 days."),
        HumanMessage(question),
    ]
    return model.invoke(msgs).content


def returns_agent(question: str) -> str:
    msgs = [
        SystemMessage("Returns specialist. Unread print: 30 days. Digital after download: no refund."),
        HumanMessage(question),
    ]
    return model.invoke(msgs).content


def supervisor(question: str) -> str:
    # Supervisor only classifies; specialists answer. Prefer this over one giant agent.
    route = model.invoke([
        SystemMessage("Reply with exactly SHIPPING or RETURNS."),
        HumanMessage(question),
    ]).content.strip().upper()
    if "SHIP" in route:
        return shipping_agent(question)
    return returns_agent(question)


def main() -> None:
    print("Q1:", supervisor("How long is express shipping?"))
    print("Q2:", supervisor("Can I return unread print order A300?"))


if __name__ == "__main__":
    print(__doc__)
    main()
