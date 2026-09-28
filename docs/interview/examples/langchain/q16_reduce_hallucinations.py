"""Q16. How do you reduce hallucinations in a RAG application?

THE PROBLEM
RAG helps but does not erase hallucinations. If context is weak, the desk can
still invent a 90-day return window.

WHAT WE ARE GOING TO SOLVE
Strict prompt: answer only from context, or say you do not know. Also refuse
when retrieval is off-topic for the question.

WHAT THIS EXAMPLE IS ABOUT
We ask about gift-card balance (not in the policy store). The model should
refuse rather than invent bookstore rules.

WHAT IT SOLVES
Refusal behavior when context does not support an answer  -  safer than a guess.

KEYWORDS
- Hallucination: Fluent but ungrounded invented facts.
- Faithfulness: How tightly the answer sticks to provided context.
- Refusal: Saying "I don't know" when context is insufficient.
- Strict prompting: Instructions that forbid using outside knowledge.
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
    Document(page_content="Unread print books can be returned within 30 days of delivery."),
    Document(page_content="Digital books are not refundable after download."),
    Document(page_content="Standard shipping takes 5 business days; express takes 2."),
]


def main() -> None:
    store = InMemoryVectorStore.from_documents(DOCS, embeddings)
    retriever = store.as_retriever(search_kwargs={"k": 1})
    question = "What is my bookstore gift-card balance?"
    context = retriever.invoke(question)[0].page_content
    print("Retrieved (likely irrelevant):", context)

    prompt = ChatPromptTemplate.from_template(
        "Answer ONLY using the context. If the context does not contain the "
        "answer, reply exactly: I don't know from the policy.\n"
        "Context: {context}\nQuestion: {question}"
    )
    print((prompt | model | StrOutputParser()).invoke({"context": context, "question": question}))


if __name__ == "__main__":
    print(__doc__)
    main()
