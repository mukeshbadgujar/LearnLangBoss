"""Q24. How would you summarize conversation history?

THE PROBLEM
A long return chat burns tokens if you resend every line. Old turns only need
the gist; the latest question needs full detail.

WHAT WE ARE GOING TO SOLVE
Ask the model to summarize older turns, then build the next prompt from
summary + latest message.

WHAT THIS EXAMPLE IS ABOUT
Turns about A300 Dune (delivered, unread, 30-day window). Latest: ask for a
return label. We compress the old turns first.

WHAT IT SOLVES
A shorter next prompt: running summary plus the newest shopper line.

KEYWORDS
- Summarization: Model rewrites long text into a shorter fact-keeping digest.
- Running summary: Update the summary as new turns arrive.
- Token: Billing/context unit; long history multiplies cost.
- Lossy memory: Summaries drop detail; store exact ids elsewhere if needed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import chat_model

model = chat_model()


def main() -> None:
    old_turns = [
        "Shopper: Order A300, the Dune paperback.",
        "Desk: Delivered 10 days ago for $16.",
        "Shopper: The book is unread.",
        "Desk: Then it is inside the 30-day return window.",
    ]
    latest = "Shopper: Email me the return label."

    summary = model.invoke(
        "Summarize these bookstore support turns in one sentence:\n"
        + "\n".join(old_turns)
    ).content
    next_prompt = f"Summary so far: {summary}\n{latest}"
    print(next_prompt)
    print("---")
    print(f"Old turns: {len(old_turns)} lines -> summary + 1 new line.")


if __name__ == "__main__":
    print(__doc__)
    main()
