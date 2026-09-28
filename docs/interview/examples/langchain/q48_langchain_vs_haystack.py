"""Q48. LangChain vs. Haystack.

THE PROBLEM
Search-pipeline teams and agent teams both say "we need an LLM framework" and talk past each other.

WHAT WE ARE GOING TO SOLVE
Show LangChain RAG+answer for bookstore FAQ, and contrast Haystack's pipeline focus.

WHAT THIS EXAMPLE IS ABOUT
A linear retrieve-then-generate path for "are digital books refundable?"

WHAT IT SOLVES
You can say: Haystack shines at declarative search/QA services; LangChain when agents and branches appear.

KEYWORDS
- Haystack: A framework for production NLP and search pipelines.
- Pipeline: An explicit sequence of processing nodes for documents and queries.
- RAG: Retrieve relevant text, then generate an answer from it.
- Agent workflow: Branching tool use that a linear pipeline expresses poorly.

Run:
    python docs/interview/examples/langchain/q48_langchain_vs_haystack.py
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

# Haystack (not imported): would declare an explicit DocumentStore + Pipeline graph for QA.
# LangChain version below is the same retrieve-then-generate idea as LCEL.


def main() -> None:
    store = InMemoryVectorStore.from_documents(
        [
            Document("Digital books are not refundable after download."),
            Document("Unread print books may be returned within 30 days."),
        ],
        embeddings,
    )
    q = "Are digital books refundable after download?"
    ctx = store.similarity_search(q, k=1)[0].page_content
    prompt = ChatPromptTemplate.from_template(
        "Answer only from context.\nContext: {ctx}\nQ: {q}"
    )
    print((prompt | model | StrOutputParser()).invoke({"ctx": ctx, "q": q}))
    print("Haystack: search/QA pipelines. LangChain: add tools/agents when the path branches.")


if __name__ == "__main__":
    print(__doc__)
    main()
