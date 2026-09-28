"""Q37. How do you control costs?

THE PROBLEM
Sending twenty policy chunks and the full chat history on every ticket burns tokens.

WHAT WE ARE GOING TO SOLVE
Trim context before the model call: one order fact, one policy line, short instruction.

WHAT THIS EXAMPLE IS ABOUT
Order A200 status plus the digital refund rule, not the whole catalog.

WHAT IT SOLVES
The prompt stays small; cost per request is visible as character count before invoke.

KEYWORDS
- Cost control: Deliberately limiting tokens so the bill stays predictable.
- Model routing: Using a smaller model for easy steps and a larger one for hard ones.
- Context trim: Dropping or summarizing text before it reaches the prompt.
- Cost per request: Tracking spend per ticket, not only the monthly total.

Run:
    python docs/interview/examples/langchain/q37_control_costs.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import chat_model

from langchain_core.messages import HumanMessage, SystemMessage

model = chat_model()

ORDERS = {
    "A100": "The Little Prince shipped $18",
    "A200": "Clean Code processing $42",
    "A300": "Dune delivered $16",
}


def build_prompt(order_id: str, question: str) -> list:
    # Retrieve less: one order line + one policy line beats dumping the wiki.
    order = ORDERS[order_id]
    policy = "Digital books are not refundable after download."
    system = SystemMessage("Bookstore desk. Answer in one sentence from the facts.")
    human = HumanMessage(f"Order: {order}\nPolicy: {policy}\nQuestion: {question}")
    return [system, human]


def main() -> None:
    messages = build_prompt("A200", "Can I get a refund after I download Clean Code?")
    chars = sum(len(m.content) for m in messages)
    print(f"prompt_chars={chars}")
    print(model.invoke(messages).content)


if __name__ == "__main__":
    print(__doc__)
    main()
