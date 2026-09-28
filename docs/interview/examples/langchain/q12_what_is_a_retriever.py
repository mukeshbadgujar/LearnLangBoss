"""Q12. What is a retriever?

THE PROBLEM
The desk needs a component that takes a shopper question and returns policy
documents  -  not a full RAG answer yet.

WHAT WE ARE GOING TO SOLVE
Build a vector-store retriever and call .invoke with a query string.

WHAT THIS EXAMPLE IS ABOUT
Query: "Can I return a digital book after I downloaded it?" The retriever
should surface the digital non-refundable policy chunk.

WHAT IT SOLVES
You see the retriever contract: query in -> list of Documents out.

KEYWORDS
- Retriever: Interface that maps a query string to a list of documents.
- as_retriever: Turns a vector store into that interface.
- Semantic search: Match by meaning via embeddings, not only keywords.
- k: How many top documents to return (here 1).
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
    retriever = store.as_retriever(search_kwargs={"k": 1})
    query = "Can I return a digital book after I downloaded it?"
    hits = retriever.invoke(query)
    for doc in hits:
        print(doc.page_content)


if __name__ == "__main__":
    print(__doc__)
    main()
