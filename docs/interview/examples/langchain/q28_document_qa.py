"""Q28. Design a document question-answering system.

THE PROBLEM
Shoppers ask policy questions, but the model invents rules unless the real pages are retrieved.

WHAT WE ARE GOING TO SOLVE
Ground answers in policy chunks that carry source metadata for citations.

WHAT THIS EXAMPLE IS ABOUT
A shopper asks about returning an unread print book versus a downloaded digital book.

WHAT IT SOLVES
The answer quotes retrieved policy text and prints the source ids used.

KEYWORDS
- Document QA: Answer questions only from stored documents, not from model memory.
- Metadata: Extra fields on a chunk (source, section) used for citations.
- Citation: Naming which document a claim came from.
- Empty retrieval: Returning no chunks when nothing matches is a valid result.

Run:
    python docs/interview/examples/langchain/q28_document_qa.py
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

POLICY = [
    Document(
        "Unread print books may be returned within 30 days of delivery.",
        metadata={"source": "returns.md", "section": "print"},
    ),
    Document(
        "Digital books are not refundable after download.",
        metadata={"source": "returns.md", "section": "digital"},
    ),
    Document(
        "Standard shipping takes 5 business days. Express shipping takes 2 days.",
        metadata={"source": "shipping.md", "section": "times"},
    ),
]


def main() -> None:
    store = InMemoryVectorStore.from_documents(POLICY, embeddings)
    question = "Can I return an unread print book I bought last week?"
    hits = store.similarity_search(question, k=2)
    if not hits:
        print("No policy found. Escalate to a human.")
        return

    context = "\n".join(f"[{d.metadata['source']}#{d.metadata['section']}] {d.page_content}" for d in hits)
    prompt = ChatPromptTemplate.from_template(
        "Answer only from the policy. Cite sources in brackets.\n"
        "Policy:\n{context}\n\nQuestion: {question}"
    )
    answer = (prompt | model | StrOutputParser()).invoke(
        {"context": context, "question": question}
    )
    print("Retrieved:", context)
    print("Answer:", answer)


if __name__ == "__main__":
    print(__doc__)
    main()
