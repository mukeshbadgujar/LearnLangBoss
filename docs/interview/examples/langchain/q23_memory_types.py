"""Q23. What memory types are available?

THE PROBLEM
Classic buffer / window memory classes are deprecated. The desk still needs to
understand buffer (all turns) vs window (last N) vs summary conceptually.

WHAT WE ARE GOING TO SOLVE
Show buffer vs window on the same return chat in plain Python, then optionally
ask the model with the windowed context.

WHAT THIS EXAMPLE IS ABOUT
A six-turn return chat about A300 Dune. Buffer keeps everything; window keeps
the last 4 lines before calling ChatGroq.

WHAT IT SOLVES
Clear contrast: full buffer grows forever; a window caps prompt size.

KEYWORDS
- Buffer memory: Keep every turn (grows without bound).
- Window memory: Keep only the last N turns.
- Summary memory: Compress old turns into a short digest (see Q24).
- Checkpointer: Modern LangGraph way to persist short-term thread state.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import chat_model

from langchain_core.messages import HumanMessage, SystemMessage

model = chat_model()

TURNS = [
    "Shopper: Order A300, the Dune paperback.",
    "Desk: Delivered 10 days ago for $16.",
    "Shopper: The book is unread.",
    "Desk: Inside the 30-day print return window.",
    "Shopper: Is digital different?",
    "Desk: Digital is not refundable after download.",
]


def main() -> None:
    buffer = TURNS[:]  # all turns
    window = TURNS[-4:]  # last N only
    print("buffer lines:", len(buffer))
    print("window lines:", len(window))
    print("window content:")
    print("\n".join(window))

    reply = model.invoke(
        [
            SystemMessage(content="Bookstore desk. Answer from the recent turns only."),
            HumanMessage(
                content="Recent chat:\n"
                + "\n".join(window)
                + "\n\nShopper: Email me the return label."
            ),
        ]
    ).content
    print("---")
    print(reply)


if __name__ == "__main__":
    print(__doc__)
    main()
