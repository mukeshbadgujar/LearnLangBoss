"""Q44. How would you build an enterprise RAG pipeline?

THE PROBLEM
Enterprise desks need access control, citations, and multiple policy sources  -  not a toy index.

WHAT WE ARE GOING TO SOLVE
Store role metadata on chunks and filter before similarity search so private docs never leak.

WHAT THIS EXAMPLE IS ABOUT
A public shipping FAQ and an internal refund-playbook chunk; a shopper role only sees public text.

WHAT IT SOLVES
Retrieval is permission-aware and answers still cite sources.

KEYWORDS
- Access control: Restricting which documents a user is allowed to retrieve.
- Metadata filter: Applying permission fields before similarity ranking.
- Incremental indexing: Updating only changed documents instead of full rebuilds.
- Auditability: Logging query and sources for compliance review.

Run:
    python docs/interview/examples/langchain/q44_enterprise_rag.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import chat_model, embedding_model

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore

model = chat_model()
embeddings = embedding_model()

DOCS = [
    Document(
        "Standard shipping takes 5 business days. Express takes 2 days.",
        metadata={"source": "shipping.md", "acl": "public"},
    ),
    Document(
        "Internal: auto-approve unread print refunds under $20 for VIP accounts.",
        metadata={"source": "internal_refunds.md", "acl": "staff"},
    ),
]


def main() -> None:
    store = InMemoryVectorStore.from_documents(DOCS, embeddings)
    role = "public"  # shopper session  -  never see staff chunks
    question = "How long is standard shipping?"
    # Filter first; filtering after retrieval can leak that a staff doc exists.
    hits = store.similarity_search(question, k=3, filter={"acl": role})
    context = "\n".join(f"[{d.metadata['source']}] {d.page_content}" for d in hits)
    prompt = ChatPromptTemplate.from_template(
        "Answer only from context and cite sources.\n{context}\n\nQ: {question}"
    )
    print((prompt | model | StrOutputParser()).invoke({"context": context, "question": question}))
    print("sources:", [d.metadata["source"] for d in hits])


if __name__ == "__main__":
    print(__doc__)
    main()
