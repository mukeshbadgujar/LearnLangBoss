"""Q9. What are Runnables?

THE PROBLEM
Prompts, models, and parsers look like different APIs. The desk wants one way
to run a single shipping answer or a small batch of FAQ questions.

WHAT WE ARE GOING TO SOLVE
Show that a composed chain is a Runnable: invoke once, then batch two questions.

WHAT THIS EXAMPLE IS ABOUT
Two shoppers ask about standard vs express shipping. The same chain handles
one invoke and a batch list.

WHAT IT SOLVES
You see the Runnable contract: invoke and batch on the same piped object.

KEYWORDS
- Runnable: Interface with invoke, stream, and batch.
- invoke: Run once with one input.
- batch: Run the same runnable over a list of inputs.
- Composition: Piping runnables creates a larger runnable.
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
        "Bookstore desk. Policy: standard shipping 5 business days; express 2.\n"
        "Answer in one short sentence.\nQuestion: {question}"
    )
    chain = prompt | model | StrOutputParser()

    print("invoke:", chain.invoke({"question": "How long is standard shipping?"}))
    # batch uses the same Runnable contract on a list.
    answers = chain.batch(
        [
            {"question": "How long is standard shipping?"},
            {"question": "How long is express shipping?"},
        ]
    )
    print("batch:", answers)


if __name__ == "__main__":
    print(__doc__)
    main()
