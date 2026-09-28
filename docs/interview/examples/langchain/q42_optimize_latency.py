"""Q42. How would you optimize latency?

THE PROBLEM
Shoppers feel every second before the first token, even when total runtime is acceptable.

WHAT WE ARE GOING TO SOLVE
Stream tokens and batch independent questions so wall time drops where possible.

WHAT THIS EXAMPLE IS ABOUT
Two FAQ questions run via batch; one shipping answer is streamed token by token.

WHAT IT SOLVES
Time-to-first-token improves with streaming; independent tickets share wait time via batch.

KEYWORDS
- Latency: End-to-end time until the user has a useful response.
- Streaming: Yielding tokens as they arrive instead of waiting for the full string.
- batch: Running the same runnable on many inputs concurrently.
- Time to first token: How soon the UI can show the first piece of the answer.

Run:
    python docs/interview/examples/langchain/q42_optimize_latency.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import chat_model

from langchain_core.prompts import ChatPromptTemplate

model = chat_model()


def main() -> None:
    prompt = ChatPromptTemplate.from_template(
        "Bookstore desk. One short sentence.\n"
        "Policy: standard 5 days, express 2 days, unread print return 30 days.\nQ: {q}"
    )
    chain = prompt | model

    # Independent tickets: batch instead of serial invokes.
    answers = chain.batch([
        {"q": "How long is standard shipping?"},
        {"q": "How long is express shipping?"},
    ])
    for a in answers:
        print("batch:", a.content)

    print("stream:", end=" ", flush=True)
    for chunk in chain.stream({"q": "Can I return an unread print book?"}):
        # AIMessageChunk: print pieces as they arrive for faster perceived latency.
        print(chunk.content, end="", flush=True)
    print()


if __name__ == "__main__":
    print(__doc__)
    main()
