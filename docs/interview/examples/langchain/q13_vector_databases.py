"""Q13. How do vector databases fit into LangChain?

THE PROBLEM
Policy text must be searchable by meaning. A SQL LIKE on "return" is not enough
when shoppers say "get my money back."

WHAT WE ARE GOING TO SOLVE
Store policy embeddings in LangChain's VectorStore (here InMemoryVectorStore)
and run similarity_search.

WHAT THIS EXAMPLE IS ABOUT
We index return and shipping policies, then search for a refund-style query.
InMemoryVectorStore stands in for Chroma/Pinecone/pgvector with the same API.

WHAT IT SOLVES
Nearest-neighbor search over embeddings via a swap-friendly VectorStore interface.

KEYWORDS
- Vector database: Stores embedding vectors and finds nearest neighbors.
- VectorStore: LangChain interface shared by many backends.
- InMemoryVectorStore: In-process store for prototypes (real API, not fake data).
- Similarity search: Rank chunks by embedding distance to the query.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import embedding_model

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

embeddings = embedding_model()

DOCS = [
    Document(page_content="Unread print books can be returned within 30 days of delivery."),
    Document(page_content="Digital books are not refundable after download."),
    Document(page_content="Standard shipping takes 5 business days; express takes 2."),
]


def main() -> None:
    store = InMemoryVectorStore.from_documents(DOCS, embeddings)
    # Same methods exist on Chroma, Pinecone, pgvector wrappers.
    hits = store.similarity_search("I want a refund on a paperback", k=1)
    print(hits[0].page_content)


if __name__ == "__main__":
    print(__doc__)
    main()
