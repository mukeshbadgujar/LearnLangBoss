"""Q5. What types of applications can be built with LangChain?

THE PROBLEM
"What can you build?" is vague. The bookstore needs four concrete categories
tied to real desk work: RAG, agents, chat memory, and structured extraction.

WHAT WE ARE GOING TO SOLVE
Run one structured-extraction mini-app (ticket -> JSON fields) and name the
other three types so the interview answer stays grounded.

WHAT THIS EXAMPLE IS ABOUT
A messy support email about order A300 (Dune, delivered, $16) is turned into
order_id, issue, and wants_return using a prompt and ChatGroq.

WHAT IT SOLVES
You demonstrate structured extraction live; RAG, agents, and memory are the
other three bookstore app types covered in later files.

KEYWORDS
- RAG: Retrieve private docs, then generate grounded answers.
- Autonomous agent: Model chooses tools to finish a multi-step goal.
- Context-aware chatbot: Support chat that keeps multi-turn history.
- Structured extraction: Unstructured text -> validated fields / JSON.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import chat_model

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

model = chat_model()


def main() -> None:
    email = (
        "Hi, order A300 Dune arrived yesterday for $16. "
        "The paperback is unread. I want a return label please."
    )
    prompt = ChatPromptTemplate.from_template(
        "Extract JSON with keys order_id, issue, wants_return (true/false).\n"
        "Email:\n{email}"
    )
    # Structured extraction pipeline: messy text in, fields out.
    chain = prompt | model | StrOutputParser()
    print(chain.invoke({"email": email}))
    print("---")
    print("Other bookstore apps: RAG policy Q&A, tool-using agents, memory chat.")


if __name__ == "__main__":
    print(__doc__)
    main()
