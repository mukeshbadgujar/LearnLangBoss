"""Q40. How would you implement retries and fallbacks?

THE PROBLEM
Teams conflate retries and fallbacks, then retry a dead provider forever or skip a useful backup.

WHAT WE ARE GOING TO SOLVE
Use with_retry for transient blips and with_fallbacks when a different path is required.

WHAT THIS EXAMPLE IS ABOUT
A flaky primary fails twice then a fallback Groq model answers the A100 status question.

WHAT IT SOLVES
Retries handle timeouts; fallbacks handle "this path will not work."

KEYWORDS
- with_retry: Re-run the same runnable with backoff after transient errors.
- with_fallbacks: Swap to another runnable when the primary cannot succeed.
- Idempotency: Safe to retry only if repeating the action does not double-charge.
- Exponential backoff: Waiting longer between each retry attempt.

Run:
    python docs/interview/examples/langchain/q40_retries_and_fallbacks.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import chat_model

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda

model = chat_model()

_attempts = {"n": 0}


def flaky_primary(messages):
    _attempts["n"] += 1
    if _attempts["n"] < 2:
        raise ConnectionError("transient desk API blip")
    # After one successful retry path we still demonstrate fallback separately below.
    raise ConnectionError("primary still down  -  need fallback")


def main() -> None:
    primary = RunnableLambda(flaky_primary)
    # Retry a few times, then fall back to a real ChatGroq call.
    resilient = primary.with_retry(stop_after_attempt=2).with_fallbacks([model])
    reply = resilient.invoke([
        HumanMessage("Order A100 The Little Prince  -  has it shipped? Answer briefly.")
    ])
    print("attempts:", _attempts["n"])
    print("answer:", reply.content)


if __name__ == "__main__":
    print(__doc__)
    main()
