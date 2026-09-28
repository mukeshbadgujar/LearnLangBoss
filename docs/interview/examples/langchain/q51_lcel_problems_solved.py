"""Q51. What problems did LCEL and the Runnable interface solve?

THE PROBLEM
Prompts used .format(), models used .predict(), chains used .run()  -  every component had a different call style.

WHAT WE ARE GOING TO SOLVE
Show one bookstore chain using the unified invoke / batch / stream protocol.

WHAT THIS EXAMPLE IS ABOUT
The same shipping FAQ chain runs once, twice in batch, and once as a stream.

WHAT IT SOLVES
One interface covers sync, parallel inputs, and token streaming.

KEYWORDS
- Runnable interface: Shared methods invoke, batch, stream, ainvoke, abatch, astream.
- invoke: Run on one input and wait for one result.
- batch: Run on many inputs, often in parallel.
- stream: Yield output pieces as they are produced.

Run:
    python docs/interview/examples/langchain/q51_lcel_problems_solved.py
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
        "Bookstore desk. One short sentence. Policy: unread print return 30 days; digital no refund after download.\nQ: {q}"
    )
    chain = prompt | model | StrOutputParser()

    print("invoke:", chain.invoke({"q": "Print return window?"}))
    print("batch:", chain.batch([{"q": "Print return window?"}, {"q": "Digital refund after download?"}]))
    print("stream:", end=" ")
    for piece in chain.stream({"q": "Print return window?"}):
        print(piece, end="", flush=True)
    print()


if __name__ == "__main__":
    print(__doc__)
    main()
