"""Q53. What are "Deep Agents" and what specific problems do they solve?

THE PROBLEM
Long bookstore investigations bloat chat history, skip planning, and pollute working memory with raw notes.

WHAT WE ARE GOING TO SOLVE
Show the deep-agent idea: plan outside the chat, keep a scratchpad, return distilled results.

WHAT THIS EXAMPLE IS ABOUT
A refund investigation for A300 uses an explicit todo list and a short scratch note, not the full log.

WHAT IT SOLVES
You can name token bloat, missing macro-planning, and polluted memory  -  and the harness pattern that fixes them.

KEYWORDS
- Deep agent: Long-horizon agent with planning, files, and sub-agents outside raw chat.
- Token bloat: Chat history growing until prompts are huge and costly.
- Macro-planning: Maintaining an explicit checklist across many steps.
- Scratchpad: External notes so raw research does not flood the main messages.

Run:
    python docs/interview/examples/langchain/q53_deep_agents.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import chat_model

from langchain_core.messages import HumanMessage, SystemMessage

model = chat_model()


def main() -> None:
    # Stand-ins for deep-agent harness pieces (todo + virtual file), not a full framework.
    todo = [
        "1. Confirm A300 status",
        "2. Check unread print policy",
        "3. Draft refund note",
    ]
    scratchpad = "A300 Dune delivered $16; shopper says unread; policy=30-day print return."
    reply = model.invoke([
        SystemMessage("Bookstore deep desk. Follow the todo. Use only the scratchpad facts."),
        HumanMessage(f"TODO:\n" + "\n".join(todo) + f"\n\nSCRATCHPAD:\n{scratchpad}\n\nWrite the final refund note."),
    ]).content
    print("todo:")
    print("\n".join(todo))
    print("scratchpad:", scratchpad)
    print("final:", reply)


if __name__ == "__main__":
    print(__doc__)
    main()
