"""Q61. What is the Runnable Interface, and how does it work under the hood?

THE PROBLEM
People use the pipe operator without knowing it builds a RunnableSequence via __or__.

WHAT WE ARE GOING TO SOLVE
Show invoke/batch/stream on one chain and print the composed type name.

WHAT THIS EXAMPLE IS ABOUT
A bookstore shipping FAQ chain is composed and executed three ways.

WHAT IT SOLVES
You can explain the contract and that a | b becomes a sequence object.

KEYWORDS
- Interface: A contract requiring specific methods with consistent names.
- RunnableSequence: Object created when you pipe two runnables with |.
- __or__: Python method that implements the | operator.
- Async variants: ainvoke, abatch, astream for non-blocking IO.

Run:
    python docs/interview/examples/langchain/q61_runnable_interface.py
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
        "Bookstore desk. One sentence. Policy: standard shipping is 5 business days.\nQ: {q}"
    )
    # prompt.__or__(model) builds RunnableSequence under the hood.
    chain = prompt | model | StrOutputParser()
    print("composed_type:", type(chain).__name__)
    print("invoke:", chain.invoke({"q": "How long is standard shipping?"}))
    print("batch:", chain.batch([{"q": "How long is standard shipping?"}]))
    streamed = "".join(chain.stream({"q": "How long is standard shipping?"}))
    print("stream:", streamed)


if __name__ == "__main__":
    print(__doc__)
    main()
