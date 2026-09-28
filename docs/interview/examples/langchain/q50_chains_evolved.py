"""Q50. How have "Chains" evolved in LangChain from legacy versions to modern architecture?

THE PROBLEM
Legacy LLMChain / SequentialChain hid steps inside opaque classes that were hard to debug or extend.

WHAT WE ARE GOING TO SOLVE
Build the same bookstore reply as a transparent LCEL pipe: prompt | model | parser.

WHAT THIS EXAMPLE IS ABOUT
A shopper asks how long express shipping takes; the pipe runs in plain sight.

WHAT IT SOLVES
You see modern chains as composed Runnables, not sealed black-box classes.

KEYWORDS
- Legacy chain: Older class-based chains like LLMChain that bundled steps internally.
- LCEL: LangChain Expression Language; declare pipelines with the pipe operator.
- Runnable: Shared interface with invoke, batch, stream, and async variants.
- Pipe (|): Operator that feeds the left runnable's output into the right one.

Run:
    python docs/interview/examples/langchain/q50_chains_evolved.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import chat_model

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

model = chat_model()


def main() -> None:
    # Old world: LLMChain(prompt=..., llm=...) hid the wiring.
    # New world: every piece is a Runnable you can test alone.
    prompt = ChatPromptTemplate.from_template(
        "Bookstore desk. One sentence.\nPolicy: express shipping is 2 business days.\nQ: {q}"
    )
    chain = prompt | model | StrOutputParser()
    print(chain.invoke({"q": "How long is express shipping?"}))
    print("type:", type(chain).__name__)


if __name__ == "__main__":
    print(__doc__)
    main()
