"""Q2. Why would you use LangChain instead of calling an LLM API directly?

THE PROBLEM
A one-off shipping FAQ could be a raw HTTP call. The desk now needs a prompt
template, plain-text output, and a path that can grow into tools later.

WHAT WE ARE GOING TO SOLVE
Show why an orchestration layer (prompt | model | parser) beats raw HTTP once
you need reusable prompts and structured steps.

WHAT THIS EXAMPLE IS ABOUT
The desk answers "How long is standard shipping?" with a small LCEL chain
instead of hand-building a Groq request body.

WHAT IT SOLVES
One chain.invoke returns a clean sentence. The same pattern later adds tools
or memory without rewriting HTTP glue.

KEYWORDS
- Orchestration layer: Control logic that wires prompts, models, and parsers.
- LCEL: LangChain Expression Language; you pipe runnables with |.
- StrOutputParser: Pulls plain text out of a chat message.
- Abstraction penalty: Frameworks speed building but can hide low-level errors.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import chat_model

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

model = chat_model()


def main() -> None:
    # Prompt templates inject variables without string-format bugs.
    prompt = ChatPromptTemplate.from_template(
        "You are the bookstore desk. Answer in one sentence.\n"
        "Policy: standard shipping is 5 business days.\n"
        "Question: {question}"
    )
    # Pipe = fixed pipeline: fill prompt, call model, parse to str.
    chain = prompt | model | StrOutputParser()
    print(chain.invoke({"question": "How long is standard shipping?"}))


if __name__ == "__main__":
    # print(__doc__)
    main()
