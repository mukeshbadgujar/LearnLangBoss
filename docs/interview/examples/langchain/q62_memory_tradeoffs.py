"""Q62. How does Memory work in LangChain, and what are the architectural trade-offs between Buffer, Summary, and VectorStore memory?

THE PROBLEM
Models are stateless; naively replaying every bookstore turn eventually blows the context window.

WHAT WE ARE GOING TO SOLVE
Compare buffer (all turns), summary (rolling), and vector recall (semantic) on a short ticket.

WHAT THIS EXAMPLE IS ABOUT
Maya's A100 return chat is stored three ways; each strategy builds a different next prompt.

WHAT IT SOLVES
You can state accuracy vs tokens vs detail-loss trade-offs clearly.

KEYWORDS
- Buffer memory: Replay the full message list every turn.
- Summary memory: Keep a rolling condensed summary of older turns.
- Vector memory: Embed past turns and retrieve only relevant ones.
- Stateless model: Each API call starts with only what you put in the prompt.

Run:
    python docs/interview/examples/langchain/q62_memory_tradeoffs.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import chat_model, embedding_model

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.vectorstores import InMemoryVectorStore

model = chat_model()
embeddings = embedding_model()

TURNS = [
    "Maya: I am Maya with order A100 The Little Prince.",
    "Desk: A100 has shipped ($18).",
    "Maya: Can I return it unread?",
    "Desk: Unread print books have a 30-day window.",
]


def main() -> None:
    latest = "Maya: What was my order id?"

    # Buffer: send everything (accurate early; costly later).
    buffer_prompt = "\n".join(TURNS + [latest])
    print("buffer_chars", len(buffer_prompt))

    # Summary: one condensed line (flat tokens; may drop ids if careless).
    summary = model.invoke([
        SystemMessage("Summarize in one sentence. Keep order id."),
        HumanMessage("\n".join(TURNS)),
    ]).content
    summary_prompt = f"Summary: {summary}\n{latest}"
    print("summary_chars", len(summary_prompt))

    # Vector: retrieve relevant past turns only.
    store = InMemoryVectorStore.from_documents(
        [Document(t) for t in TURNS], embeddings
    )
    recalled = "\n".join(d.page_content for d in store.similarity_search(latest, k=2))
    vector_prompt = f"Recalled:\n{recalled}\n{latest}"
    print("vector_chars", len(vector_prompt))

    print("answer:", model.invoke([
        SystemMessage("Bookstore desk. Answer from memory context only."),
        HumanMessage(summary_prompt),
    ]).content)


if __name__ == "__main__":
    print(__doc__)
    main()
