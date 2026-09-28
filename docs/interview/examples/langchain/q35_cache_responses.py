"""Q35. How do you cache responses?

THE PROBLEM
Shoppers ask the same shipping question dozens of times; each hit pays for another model call.

WHAT WE ARE GOING TO SOLVE
Turn on exact-match LLM caching so identical prompts reuse the prior reply.

WHAT THIS EXAMPLE IS ABOUT
The same "How long is standard shipping?" question is asked twice with InMemoryCache on.

WHAT IT SOLVES
The second call can hit the cache instead of paying the provider again.

KEYWORDS
- Cache: Store a prior result and reuse it when the same input appears again.
- Exact-match cache: Keys on the full prompt; only identical text hits.
- Semantic cache: Keys on meaning; similar questions can share an answer.
- Prompt caching: Provider-side reuse of long shared prefixes.

Run:
    python docs/interview/examples/langchain/q35_cache_responses.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import chat_model

import time

from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain_core.prompts import ChatPromptTemplate

model = chat_model()

# Exact-match cache sits in front of the model for the whole process.
set_llm_cache(InMemoryCache())


def main() -> None:
    prompt = ChatPromptTemplate.from_template(
        "Bookstore desk. One sentence. Policy: standard shipping is 5 business days.\nQ: {q}"
    )
    chain = prompt | model
    question = {"q": "How long is standard shipping?"}

    t0 = time.perf_counter()
    first = chain.invoke(question)
    t1 = time.perf_counter()
    second = chain.invoke(question)
    t2 = time.perf_counter()

    print("first:", first.content, f"({(t1 - t0) * 1000:.0f} ms)")
    print("second:", second.content, f"({(t2 - t1) * 1000:.0f} ms)")
    print("Second call should be faster when the exact prompt is cached.")


if __name__ == "__main__":
    print(__doc__)
    main()
