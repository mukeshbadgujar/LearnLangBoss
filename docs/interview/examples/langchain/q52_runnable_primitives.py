"""Q52. What are the primary Runnable primitives used in LCEL pipelines?

THE PROBLEM
Interview answers name "the pipe" but miss Parallel, Passthrough, Lambda, and Branch.

WHAT WE ARE GOING TO SOLVE
Use each primitive once in a tiny bookstore ticket pipeline.

WHAT THIS EXAMPLE IS ABOUT
Lookup order A100 in parallel with policy text, then branch a short vs long reply style.

WHAT IT SOLVES
You can point to Sequence, Parallel, Passthrough, Lambda, and Branch in working code.

KEYWORDS
- RunnableSequence: The pipe A | B that feeds output forward.
- RunnableParallel: Run several runnables on the same input at once.
- RunnablePassthrough: Pass input through or assign extra fields.
- RunnableLambda: Wrap a plain Python function as a runnable.
- RunnableBranch: If/else routing inside a pipe.

Run:
    python docs/interview/examples/langchain/q52_runnable_primitives.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import chat_model

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableParallel, RunnablePassthrough

model = chat_model()

ORDERS = {"A100": "The Little Prince shipped $18"}


def main() -> None:
    order_fn = RunnableLambda(lambda x: ORDERS[x["order_id"]])
    policy_fn = RunnableLambda(lambda _: "Standard shipping: 5 business days.")

    # Parallel gathers both facts; Passthrough keeps the original question.
    gather = RunnableParallel(order=order_fn, policy=policy_fn, question=RunnablePassthrough())

    short = ChatPromptTemplate.from_template("One word status from: {order}") | model | StrOutputParser()
    long = (
        ChatPromptTemplate.from_template("Two sentences. Order: {order}. Policy: {policy}. Q: {question}")
        | model
        | StrOutputParser()
    )
    # Branch on a flag in the gathered dict.
    branch = RunnableBranch(
        (lambda d: d.get("style") == "short", short),
        long,
    )

    payload = {"order_id": "A100", "question": "Where is my book?", "style": "long"}
    gathered = gather.invoke(payload)
    gathered["style"] = payload["style"]
    print(branch.invoke(gathered))


if __name__ == "__main__":
    print(__doc__)
    main()
