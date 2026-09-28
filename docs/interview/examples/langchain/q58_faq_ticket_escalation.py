"""Q58. Outline the architecture for Project 1: Conversational FAQ & Ticket Escalation Agent.

THE PROBLEM
FAQ answers and out-of-scope tickets need different paths; dumping everything into one free-form agent is brittle.

WHAT WE ARE GOING TO SOLVE
Retrieve FAQ first; escalate to a ticket id when retrieval is empty or the model says out of scope.

WHAT THIS EXAMPLE IS ABOUT
Shipping FAQ hits the vector store; "change my payment method on A200" escalates to a mock ticket.

WHAT IT SOLVES
A clear ingest -> retrieve -> answer-or-escalate loop for a bookstore support desk.

KEYWORDS
- FAQ agent: Answers common questions from a knowledge base.
- Escalation: Creating a human ticket when automation cannot help.
- Conditional router: Branching on whether retrieval/confidence is good enough.
- Ticket id: Identifier handed back when the case goes to humans.

Run:
    python docs/interview/examples/langchain/q58_faq_ticket_escalation.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import chat_model, embedding_model

import uuid

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore

model = chat_model()
embeddings = embedding_model()

FAQ = [
    Document("Standard shipping takes 5 business days."),
    Document("Express shipping takes 2 business days."),
    Document("Unread print books may be returned within 30 days."),
]


def escalate(question: str) -> str:
    ticket = f"SN-{uuid.uuid4().hex[:8].upper()}"
    return f"Escalated to human queue as {ticket} for: {question}"


def main() -> None:
    store = InMemoryVectorStore.from_documents(FAQ, embeddings)
    for question in [
        "How long is express shipping?",
        "Please change the payment method on order A200",
    ]:
        hits = store.similarity_search_with_score(question, k=1)
        doc, score = hits[0]
        # Crude router: weak match => escalate (real systems also use an LLM scope check).
        if score > 0.4 and "payment" in question.lower():
            print(escalate(question))
            continue
        if "payment" in question.lower() or "change" in question.lower():
            print(escalate(question))
            continue
        prompt = ChatPromptTemplate.from_template(
            "Answer from FAQ only.\nFAQ: {faq}\nQ: {q}"
        )
        answer = (prompt | model | StrOutputParser()).invoke({"faq": doc.page_content, "q": question})
        print("FAQ:", answer)


if __name__ == "__main__":
    print(__doc__)
    main()
