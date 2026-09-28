"""Q22. What is memory in LangChain?

THE PROBLEM
LLM APIs are stateless. If the shopper said "order A100" earlier and then asks
"has it shipped?", a bare second call has no memory of A100.

WHAT WE ARE GOING TO SOLVE
Resend prior turns as a message list so the model sees short-term conversation
memory.

WHAT THIS EXAMPLE IS ABOUT
Turn 1 names order A100 (The Little Prince). Turn 2 asks if it has shipped.
We pass both Human/AI messages into one invoke.

WHAT IT SOLVES
Short-term memory as "what we put back in the prompt" between turns.

KEYWORDS
- Memory: Layer that stores state and chooses what to resend next turn.
- Stateless API: Raw calls forget everything unless you resend it.
- Short-term memory: History inside one conversation / thread.
- Long-term memory: Facts that persist across conversations (user-scoped).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import chat_model

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

model = chat_model()


def main() -> None:
    history = [
        SystemMessage(
            content="Bookstore desk. A100 The Little Prince is shipped at $18."
        ),
        HumanMessage(content="I'm asking about order A100."),
        AIMessage(content="Got it  -  order A100, The Little Prince."),
        HumanMessage(content="Has it shipped yet?"),
    ]
    # Memory here is literally the message list we resend.
    print(model.invoke(history).content)


if __name__ == "__main__":
    print(__doc__)
    main()
