"""Q26. How would you reduce context growth?

THE PROBLEM
Full history for every return-desk turn is too expensive. You need a cheap
way to cap growth before you reach fancy retrieval-over-history.

WHAT WE ARE GOING TO SOLVE
Apply trimming (last N messages) and show size vs full buffer. Optionally
summarize the trimmed-away head with the model.

WHAT THIS EXAMPLE IS ABOUT
Eight lines about A300 Dune returns. We keep the last 4 (trim) and summarize
the dropped head so facts are not lost entirely.

WHAT IT SOLVES
Two practical levers: trim first, summarize second  -  context stays bounded.

KEYWORDS
- Trimming: Keep last N messages; drop the rest.
- Summarization: Compress old turns into a short digest.
- Retrieval over history: Vector-search past messages (heavier option).
- Structured facts: Store order ids outside the free-text prompt.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import chat_model

model = chat_model()

HISTORY = [
    "Shopper: Order A300, Dune paperback.",
    "Desk: Delivered 10 days ago for $16.",
    "Shopper: Unread  -  can I return it?",
    "Desk: Yes, unread print books have 30 days.",
    "Shopper: What about digital?",
    "Desk: Digital is not refundable after download.",
    "Shopper: Send the label.",
    "Desk: Label emailed.",
]


def main() -> None:
    window_n = 4
    trimmed = HISTORY[-window_n:]
    dropped = HISTORY[:-window_n]

    print("full chars:", sum(len(x) for x in HISTORY))
    print("trimmed chars:", sum(len(x) for x in trimmed))
    print("kept:\n" + "\n".join(trimmed))

    # Summarize what trimming removed so key facts survive.
    summary = model.invoke(
        "One-sentence summary of older bookstore turns:\n" + "\n".join(dropped)
    ).content
    next_context = f"Earlier summary: {summary}\n" + "\n".join(trimmed)
    print("---")
    print(next_context)
    print("next chars:", len(next_context))


if __name__ == "__main__":
    print(__doc__)
    main()
