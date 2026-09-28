"""Q60. Explain the 5 core components and data flow of a LangChain RAG pipeline.

THE PROBLEM
Candidates mix up loaders, splitters, embeddings, stores, and retrievers when describing RAG.

WHAT WE ARE GOING TO SOLVE
Walk one bookstore policy question through all five components in order.

WHAT THIS EXAMPLE IS ABOUT
Policy text becomes Documents, chunks, vectors, a store, then a retriever hit for a return question.

WHAT IT SOLVES
You can narrate loader -> splitter -> embedding -> vector store -> retriever with working code.

KEYWORDS
- Document loader: Turns raw files/APIs into Document objects.
- Text splitter: Breaks documents into smaller chunks.
- Embedding model: Maps text to vectors of meaning.
- Vector store: Indexes vectors for nearest-neighbor search.
- Retriever: Accepts a query string and returns top-k chunks.

Run:
    python docs/interview/examples/langchain/q60_rag_five_components.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import chat_model, embedding_model

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

model = chat_model()
embeddings = embedding_model()


def main() -> None:
    # 1) Loader stand-in: already-normalized Documents
    loaded = [
        Document(
            "Unread print books may be returned within 30 days of delivery. "
            "Digital books are not refundable after download. "
            "Standard shipping takes 5 business days. Express takes 2 days.",
            metadata={"source": "policy.md"},
        )
    ]
    # 2) Splitter
    chunks = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20).split_documents(loaded)
    # 3+4) Embeddings + vector store
    store = InMemoryVectorStore.from_documents(chunks, embeddings)
    # 5) Retriever
    retriever = store.as_retriever(search_kwargs={"k": 2})
    question = "How long do I have to return an unread print book?"
    hits = retriever.invoke(question)
    context = "\n".join(h.page_content for h in hits)
    prompt = ChatPromptTemplate.from_template(
        "Answer only from context.\n{context}\n\nQ: {question}"
    )
    print("chunks:", len(chunks))
    print("retrieved:", context)
    print("answer:", (prompt | model | StrOutputParser()).invoke({"context": context, "question": question}))


if __name__ == "__main__":
    print(__doc__)
    main()
