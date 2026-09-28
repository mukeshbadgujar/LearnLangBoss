"""Q30. How would you build a multi-step workflow?

THE PROBLEM
Refund tickets need fixed steps: look up the order, check policy, draft a reply.

WHAT WE ARE GOING TO SOLVE
Compose a fixed LCEL chain when the sequence is known upfront.

WHAT THIS EXAMPLE IS ABOUT
Order A300 (Dune, delivered, $16) asks for a return on an unread print book.

WHAT IT SOLVES
One pipeline runs lookup -> policy check -> draft reply without an agent loop.

KEYWORDS
- Workflow: A sequence of steps that together finish a business task.
- LCEL: LangChain Expression Language; pipe runnables with |.
- Fixed sequence: Steps you know in advance; prefer a chain over an agent.
- State: Shared data each step reads and writes.

Run:
    python docs/interview/examples/langchain/q30_multi_step_workflow.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import chat_model

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

model = chat_model()

ORDERS = {
    "A100": "The Little Prince | shipped | $18",
    "A200": "Clean Code | processing | $42",
    "A300": "Dune | delivered | $16",
}


def lookup(inputs: dict) -> dict:
    order_id = inputs["order_id"]
    return {**inputs, "order": ORDERS.get(order_id, "unknown"), "policy": "Unread print: 30 days."}


def main() -> None:
    prompt = ChatPromptTemplate.from_template(
        "Draft one short refund reply.\nOrder: {order}\nPolicy: {policy}\nRequest: {request}"
    )
    # Fixed pipeline: enrich state, then generate. No branching, no agent.
    chain = (
        RunnablePassthrough()
        | RunnableLambda(lookup)
        | prompt
        | model
        | StrOutputParser()
    )
    print(chain.invoke({"order_id": "A300", "request": "Unread print return please"}))


if __name__ == "__main__":
    print(__doc__)
    main()
