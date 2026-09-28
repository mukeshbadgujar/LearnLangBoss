"""Q15. How would you improve retrieval quality?

THE PROBLEM
A vague shopper question ("refund?") can miss the digital vs print policy
split. Bad answers often mean bad retrieval, not a weak model.

WHAT WE ARE GOING TO SOLVE
Show one cheap improvement: rewrite the query to be specific, then retrieve.
(Also name chunking, metadata, hybrid, and rerank as next levers.)

WHAT THIS EXAMPLE IS ABOUT
Original: "refund?" Rewritten: "Can I return a downloaded digital book?"
We retrieve with both and compare which policy chunk wins.

WHAT IT SOLVES
Query rewriting improves which document surfaces before you touch the LLM.

KEYWORDS
- Query rewriting: Expand or clarify the user question before search.
- Chunking: Split docs on meaning boundaries, not only character counts.
- Metadata filter: Narrow search by tags (e.g. policy_type=digital).
- Reranking: Rescore a wide candidate set with a stronger model.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import chat_model, embedding_model

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

model = chat_model()
embeddings = embedding_model()

DOCS = [
    Document(page_content="Unread print books can be returned within 30 days of delivery."),
    Document(page_content="Digital books are not refundable after download."),
    Document(page_content="Standard shipping takes 5 business days; express takes 2."),
]


def main() -> None:
    store = InMemoryVectorStore.from_documents(DOCS, embeddings)
    retriever = store.as_retriever(search_kwargs={"k": 1})

    vague = "refund?"
    # Ask the model to rewrite for search, not to answer the policy.
    rewritten = model.invoke(
        "Rewrite this bookstore support question into one clear search query "
        f"about digital vs print returns. Question: {vague}"
    ).content.strip()

    print("Vague hit:", retriever.invoke(vague)[0].page_content)
    print("Rewritten:", rewritten)
    print("Rewritten hit:", retriever.invoke(rewritten)[0].page_content)


if __name__ == "__main__":
    print(__doc__)
    main()
