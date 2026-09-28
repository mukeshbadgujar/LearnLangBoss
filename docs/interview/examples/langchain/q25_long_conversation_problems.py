"""Q25. What problems arise with long conversations?

THE PROBLEM
Return chats grow. Resending the full history every turn raises cost and
latency, can overflow context, and can dilute attention to middle turns.

WHAT WE ARE GOING TO SOLVE
Measure prompt growth in characters across turns (a stand-in for tokens) and
name the four interview problems.

WHAT THIS EXAMPLE IS ABOUT
A growing A300 Dune return thread. We print cumulative prompt size after each
shopper/desk pair  -  no need for a model call to see the cost curve.

WHAT IT SOLVES
Concrete numbers for cost growth, plus the four risk labels interviewers want.

KEYWORDS
- Cost: Early messages are paid again on every later turn.
- Latency: Longer prompts take longer to process.
- Context overflow: History eventually exceeds the model window.
- Attention dilution: Middle-of-context details get ignored more often.
"""

TURNS = [
    ("Shopper", "Order A300, Dune paperback."),
    ("Desk", "Delivered 10 days ago for $16."),
    ("Shopper", "Unread  -  can I return it?"),
    ("Desk", "Yes, unread print books have 30 days."),
    ("Shopper", "What about digital?"),
    ("Desk", "Digital is not refundable after download."),
    ("Shopper", "Send the label."),
    ("Desk", "Label emailed."),
]


def main() -> None:
    history: list[str] = []
    print("turn | prompt_chars")
    for i in range(0, len(TURNS), 2):
        history.append(f"{TURNS[i][0]}: {TURNS[i][1]}")
        if i + 1 < len(TURNS):
            history.append(f"{TURNS[i + 1][0]}: {TURNS[i + 1][1]}")
        # Stand-in for tokens: character count of everything resent so far.
        size = sum(len(line) for line in history)
        print(f"{(i // 2) + 1:4d} | {size}")

    print("---")
    print("Problems: cost, latency, context overflow, attention dilution.")


if __name__ == "__main__":
    print(__doc__)
    main()
