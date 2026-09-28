"""Q41. How would you manage context efficiently?

THE PROBLEM
History, policy dumps, and tool outputs expand until the desk prompt is slow and expensive.

WHAT WE ARE GOING TO SOLVE
Treat the context window as a budget: keep a short summary plus the latest question.

WHAT THIS EXAMPLE IS ABOUT
Six return-chat turns about A300 collapse into one summary line before the next model call.

WHAT IT SOLVES
The next prompt stays small without dropping the order id and unread fact.

KEYWORDS
- Context window: Maximum tokens a model can read and write in one call.
- Running summary: A short text updated as new turns arrive.
- Tool-output trim: Shortening large tool results before they enter history.
- Lost-in-the-middle: Models use mid-prompt facts worse than start/end facts.

Run:
    python docs/interview/examples/langchain/q41_manage_context.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import chat_model

from langchain_core.messages import HumanMessage, SystemMessage

model = chat_model()

OLD = [
    "Shopper: Order A300, the Dune paperback.",
    "Desk: Delivered 10 days ago, $16.",
    "Shopper: The book is unread.",
    "Desk: Inside the 30-day print return window.",
    "Shopper: I want a label.",
    "Desk: I can email a return label after approval.",
]


def main() -> None:
    summary = model.invoke([
        SystemMessage("Summarize in one sentence. Keep order id and key facts."),
        HumanMessage("\n".join(OLD)),
    ]).content
    latest = "Shopper: Has approval finished?"
    # Budget: summary + latest, not all six raw turns.
    reply = model.invoke([
        SystemMessage("Bookstore desk. Be brief. Use only the summary."),
        HumanMessage(f"Summary: {summary}\n{latest}"),
    ]).content
    print("summary:", summary)
    print("reply:", reply)
    print(f"raw_turns={len(OLD)} kept=summary+1")


if __name__ == "__main__":
    print(__doc__)
    main()
