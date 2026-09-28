"""Q11. How does LangChain implement RAG?

THE PROBLEM
Without policy text in the prompt, the model invents return windows. The
bookstore needs answers grounded in real policy chunks.

WHAT WE ARE GOING TO SOLVE
Index policy documents, retrieve the best chunk, then generate only from that
context (classic RAG).

WHAT THIS EXAMPLE IS ABOUT
Shopper wants money back on an unread print book. We retrieve the 30-day
return policy and ask ChatGroq to answer only from context.

WHAT IT SOLVES
Retrieval + grounded generation: the answer tracks the stored policy, not
model memory alone.

KEYWORDS
- RAG: Retrieval-augmented generation  -  find text, then answer from it.
- Chunk: A short piece of a longer document.
- Embedding: Number vector that represents meaning of text.
- Vector store: Stores embeddings and finds nearest neighbors.
- Grounding: Keeping the answer tied to retrieved context.
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
    question = "I want my money back on an unread print book."
    # Online retrieval: embed the question, return nearest policy chunk.
    context = retriever.invoke(question)[0].page_content
    print("Retrieved:", context)

    prompt = ChatPromptTemplate.from_template(
        "Answer only from the policy context. If missing, say you do not know.\n"
        "Context: {context}\nQuestion: {question}"
    )
    chain = prompt | model | StrOutputParser()
    print("Answer:", chain.invoke({"context": context, "question": question}))


if __name__ == "__main__":
    print(__doc__)
    main()
