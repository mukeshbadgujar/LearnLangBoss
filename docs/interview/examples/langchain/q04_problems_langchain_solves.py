"""Q4. What problems does LangChain solve?

THE PROBLEM
Raw LLM APIs differ by vendor, forget prior turns, and leave you to hand-write
multi-step desk logic for orders and refunds.

WHAT WE ARE GOING TO SOLVE
Show one normalized model call and a short message list so the desk is not
tied to vendor JSON shapes and is not forced to be single-turn forever.

WHAT THIS EXAMPLE IS ABOUT
A shopper first asks about order A200 (Clean Code, processing, $42), then
asks whether they can still cancel. We send both turns in one invoke.

WHAT IT SOLVES
Provider-shaped messages become LangChain messages; the second turn still
sees the first. Orchestration (tools/agents) builds on this same layer.

KEYWORDS
- Provider lock-in: App code stuck on one vendor's schemas and SDKs.
- Stateless API: Each raw request has no memory of earlier requests.
- Message list: Turns you resend so the model sees conversation context.
- Orchestration: Coordinating prompts, tools, and state around the model.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import chat_model

from langchain_core.messages import HumanMessage, SystemMessage

model = chat_model()


def main() -> None:
    # Same message objects work across providers LangChain supports.
    messages = [
        SystemMessage(
            content=(
                "You are the bookstore desk. Orders: A200 Clean Code is still "
                "processing at $42. Processing orders can be cancelled."
            )
        ),
        HumanMessage(content="What is the status of order A200?"),
        HumanMessage(content="Can I cancel it?"),
    ]
    print(model.invoke(messages).content)


if __name__ == "__main__":
    print(__doc__)
    main()
