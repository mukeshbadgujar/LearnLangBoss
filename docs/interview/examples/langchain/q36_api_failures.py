"""Q36. How do you handle API failures?

THE PROBLEM
Provider rate limits and outages are normal; a bare invoke crashes the desk ticket.

WHAT WE ARE GOING TO SOLVE
Attach with_fallbacks so a backup model answers when the primary path fails.

WHAT THIS EXAMPLE IS ABOUT
Primary path is wrapped to fail once; fallback ChatGroq still answers the shipping question.

WHAT IT SOLVES
Transient provider failure does not become a blank 500 for the shopper.

KEYWORDS
- Retry: Calling the same runnable again after a transient error.
- Fallback: Switching to a different runnable when the primary cannot succeed.
- Rate limit: Provider cap on how many calls you may make.
- Graceful degradation: Returning a useful backup answer instead of crashing.

Run:
    python docs/interview/examples/langchain/q36_api_failures.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import chat_model

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda

model = chat_model()
backup = chat_model()


def main() -> None:
    # Simulate a dead primary; real apps use primary.with_fallbacks([backup]).
    def boom(_messages):
        raise TimeoutError("primary Groq endpoint timed out")

    resilient = RunnableLambda(boom).with_fallbacks([backup])
    reply = resilient.invoke([HumanMessage("How many days is standard bookstore shipping? Policy: 5.")])
    print("fallback answer:", reply.content)


if __name__ == "__main__":
    print(__doc__)
    main()
