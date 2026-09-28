"""Q31. How would you add human approval?

THE PROBLEM
An agent that issues refunds without a human gate can move money by mistake.

WHAT WE ARE GOING TO SOLVE
Pause before a refund action, require approve/reject, then continue.

WHAT THIS EXAMPLE IS ABOUT
Order A300 wants a $16 refund. The draft is shown; only approved refunds print as issued.

WHAT IT SOLVES
Reads stay automatic. Writes that move money wait for an explicit approval flag.

KEYWORDS
- Human-in-the-loop: Pausing automation until a person confirms a sensitive step.
- Approval gate: A check that blocks refunds until approved=True.
- Checkpointer: Saved state so a pause can last minutes or days across restarts.
- Side effect: An action that changes the outside world (refund, email, cancel).

Run:
    python docs/interview/examples/langchain/q31_human_approval.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import chat_model

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

model = chat_model()


def issue_refund(order_id: str, amount: float, approved: bool) -> str:
    """Only runs the money move when a human has set approved=True."""
    if not approved:
        return f"PAUSED: refund ${amount} for {order_id} waiting for human approval."
    return f"ISSUED: refund ${amount} for {order_id}."


def main() -> None:
    prompt = ChatPromptTemplate.from_template(
        "One sentence: draft a refund note for order {order_id} amount ${amount}. "
        "Policy: unread print books return within 30 days."
    )
    draft = (prompt | model | StrOutputParser()).invoke({"order_id": "A300", "amount": 16})
    print("Draft:", draft)

    # In production this flag comes from a UI after LangGraph interrupt + checkpointer.
    approved = True
    print(issue_refund("A300", 16.0, approved=approved))
    print(issue_refund("A300", 16.0, approved=False))


if __name__ == "__main__":
    print(__doc__)
    main()
