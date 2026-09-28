"""Q46. LangChain vs. LlamaIndex.

THE PROBLEM
Teams treat LangChain and LlamaIndex as interchangeable names for "talk to my PDFs."

WHAT WE ARE GOING TO SOLVE
Show the LangChain orchestration version of a bookstore policy lookup, and contrast centers of gravity.

WHAT THIS EXAMPLE IS ABOUT
Retrieve the print-return policy with LangChain Documents + vector store + Groq answer.

WHAT IT SOLVES
You can say: LlamaIndex is retrieval-first; LangChain is orchestration-first with retrieval as one part.

KEYWORDS
- LlamaIndex: A library centered on loaders, indexes, and query engines over documents.
- LangChain: A library centered on composing models, tools, and control flow.
- Index: The searchable structure built from your documents.
- Orchestration: Connecting retrieval, tools, and multi-step logic into one app.

Run:
    python docs/interview/examples/langchain/q46_langchain_vs_llamaindex.py
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

# LlamaIndex (not imported): would center on Index / QueryEngine over the same policy files.
# LangChain centers on piping retrieval into prompts, tools, and agents.


def main() -> None:
    docs = [
        Document("Unread print books may be returned within 30 days of delivery."),
        Document("Digital books are not refundable after download."),
    ]
    store = InMemoryVectorStore.from_documents(docs, embeddings)
    q = "How many days to return an unread print book?"
    chunk = store.similarity_search(q, k=1)[0].page_content
    prompt = ChatPromptTemplate.from_template(
        "Bookstore desk. Answer from context only.\nContext: {context}\nQ: {q}"
    )
    print((prompt | model | StrOutputParser()).invoke({"context": chunk, "q": q}))
    print("LangChain: retrieval is one runnable in a larger workflow.")


if __name__ == "__main__":
    print(__doc__)
    main()
