"""Q55. How do Claude Skills integrate with Deep Agents compared to static instructions?

THE PROBLEM
Stuffing every refund procedure into the system prompt wastes tokens on tickets that never need it.

WHAT WE ARE GOING TO SOLVE
Load a skill playbook only when the ticket is a refund; keep shipping tickets on a thin prompt.

WHAT THIS EXAMPLE IS ABOUT
A refund skill markdown is injected for A300 returns, but skipped for a shipping FAQ.

WHAT IT SOLVES
Skills are progressive disclosure: metadata always available, body loaded on demand.

KEYWORDS
- Skill: A modular playbook the agent loads when the task matches.
- Static instructions: Fixed system text present on every call.
- Progressive disclosure: Reveal detailed instructions only when needed.
- Playbook: Step-by-step operational procedure for one job.

Run:
    python docs/interview/examples/langchain/q55_skills_vs_static_instructions.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import chat_model

from langchain_core.messages import HumanMessage, SystemMessage

model = chat_model()

REFUND_SKILL = """
# refund-print
1. Confirm order id and unread status.
2. Check 30-day print window.
3. Draft refund; do not issue money without human approval.
""".strip()


def answer(ticket: str) -> str:
    base = "Bookstore desk. Available skills: refund-print (load only for refunds)."
    msgs = [SystemMessage(base)]
    if "refund" in ticket.lower() or "return" in ticket.lower():
        # Dynamic skill load  -  not a static always-on wall of rules.
        msgs.append(SystemMessage(f"Loaded skill:\n{REFUND_SKILL}"))
    msgs.append(HumanMessage(ticket))
    return model.invoke(msgs).content


def main() -> None:
    print("shipping:", answer("How long is standard shipping? (5 days policy)"))
    print("refund:", answer("I want a refund on unread print order A300."))


if __name__ == "__main__":
    print(__doc__)
    main()
