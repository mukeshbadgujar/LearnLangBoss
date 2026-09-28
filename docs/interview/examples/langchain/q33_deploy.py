"""Q33. How do you deploy LangChain applications?

THE PROBLEM
A notebook desk bot is not a service: no HTTP entry, no shared state, gateway timeouts kill long runs.

WHAT WE ARE GOING TO SOLVE
Show the deploy shape: FastAPI wrapper, shared checkpointer note, long-request awareness.

WHAT THIS EXAMPLE IS ABOUT
Expose one bookstore shipping answer as an HTTP-style handler function ready for a container.

WHAT IT SOLVES
The same LCEL chain becomes a callable service endpoint; state and timeouts are called out.

KEYWORDS
- Deploy: Packaging the app so other systems can call it over the network.
- FastAPI: A Python web framework commonly used to wrap LangChain chains.
- Checkpointer: Shared persistence so any instance can resume a conversation thread.
- Timeout: Gateway limit that can kill a 30-second agent run if left at defaults.

Run:
    python docs/interview/examples/langchain/q33_deploy.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import chat_model

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

model = chat_model()

# Production notes (not executed here):
# - Put API keys in a secrets manager, not the image.
# - Use Postgres/Redis checkpointer when you run more than one replica.
# - Raise gateway timeouts or queue long agent runs and stream results.


def answer_ticket(question: str) -> dict:
    """HTTP handler body: input JSON in, answer JSON out."""
    prompt = ChatPromptTemplate.from_template(
        "Bookstore desk. Be brief.\n"
        "Policy: standard shipping 5 business days; express 2 days.\nQ: {question}"
    )
    text = (prompt | model | StrOutputParser()).invoke({"question": question})
    return {"answer": text}


def main() -> None:
    # Stand-in for: POST /tickets {"question": "..."}
    payload = {"question": "How long is express shipping?"}
    print(answer_ticket(payload["question"]))


if __name__ == "__main__":
    print(__doc__)
    main()
