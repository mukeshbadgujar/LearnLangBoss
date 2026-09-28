"""Q3. What are the main components of LangChain?

THE PROBLEM
Interviewers list models, prompts, tools, agents, retrievers, and runnables.
The bookstore desk needs a tiny demo that touches several of them in one place.

WHAT WE ARE GOING TO SOLVE
Wire a prompt, a chat model, and a parser into one runnable chain, and name
the other components so the story stays concrete.

WHAT THIS EXAMPLE IS ABOUT
Order A100 (The Little Prince, shipped, $18) asks about express shipping.
A prompt + ChatGroq + StrOutputParser answers from the shipping policy line.

WHAT IT SOLVES
You see the six pillars: models and prompts in the code; tools, agents, and
retrievers named for later files; the chain itself is a Runnable.

KEYWORDS
- Models: Chat models for text; embedding models for vectors.
- Prompts: Templates that inject variables into messages.
- Tools: Python functions the model can request by name.
- Agents: Loops where the model chooses the next tool or final answer.
- Retrievers: Components that turn a query string into documents.
- Runnables: Anything with invoke / stream / batch that can be piped with |.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import chat_model

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

model = chat_model()


def main() -> None:
    prompt = ChatPromptTemplate.from_template(
        "Bookstore desk for order {order_id} ({title}).\n"
        "Policy: express shipping is 2 business days.\n"
        "Question: {question}"
    )
    # prompt | model | parser is itself a Runnable (a chain).
    chain = prompt | model | StrOutputParser()
    print(
        chain.invoke(
            {
                "order_id": "A100",
                "title": "The Little Prince",
                "question": "How fast is express shipping?",
            }
        )
    )


if __name__ == "__main__":
    print(__doc__)
    main()
