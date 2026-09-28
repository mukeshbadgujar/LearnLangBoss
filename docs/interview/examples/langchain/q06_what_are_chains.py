"""Q6. What are chains?

THE PROBLEM
The desk fills a prompt, calls the model, and strips the message object in
three separate scripts. Nobody sees that they are one pipeline.

WHAT WE ARE GOING TO SOLVE
Join those steps with the pipe operator into one chain you can invoke once.

WHAT THIS EXAMPLE IS ABOUT
A shopper asks how long standard shipping takes. The chain answers from the
bookstore shipping policy in one invoke.

WHAT IT SOLVES
chain.invoke runs prompt -> model -> StrOutputParser in a fixed order.

KEYWORDS
- Chain: A fixed pipeline; output of one step feeds the next.
- LCEL: Build chains by piping runnables with |.
- Pipe (|): Connects runnables left-to-right.
- Fixed control flow: Steps known ahead of time (unlike agents).
- StrOutputParser: Turns a chat message into a plain string.
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
        "You are the bookstore desk. Answer in one sentence.\n"
        "Policy: standard shipping takes 5 business days; express takes 2.\n"
        "Question: {question}"
    )
    chain = prompt | model | StrOutputParser()
    print(chain.invoke({"question": "How long is standard shipping?"}))


if __name__ == "__main__":
    print(__doc__)
    main()
