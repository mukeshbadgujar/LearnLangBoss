"""Q63. What is a SKILL.md file, and how is it used in agentic architectures?

THE PROBLEM
Putting every refund procedure in the system prompt bloats non-refund bookstore tickets.

WHAT WE ARE GOING TO SOLVE
Keep a SKILL.md playbook on disk and load it only when the task is a refund.

WHAT THIS EXAMPLE IS ABOUT
Shipping questions use a thin prompt; refund for A300 loads refund-print/SKILL.md.

WHAT IT SOLVES
Progressive disclosure: skill metadata is cheap; the body enters context on demand.

KEYWORDS
- SKILL.md: Markdown playbook with YAML frontmatter naming a skill.
- Frontmatter: YAML metadata block at the top of a markdown file.
- Progressive disclosure: Load detailed instructions only when needed.
- Playbook: Ordered steps the agent should follow for one job.

Run:
    python docs/interview/examples/langchain/q63_skill_md.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import chat_model

import tempfile

from langchain_core.messages import HumanMessage, SystemMessage

model = chat_model()

SKILL_MD = """---
name: refund-print
description: Handle unread print-book refund requests for the bookstore desk.
---
# Refund print books
1. Confirm order id and unread status.
2. Apply 30-day print return window.
3. Draft the note; do not move money without human approval.
"""


def load_skill_if_refund(task: str, skill_path: Path) -> str | None:
    if "refund" not in task.lower() and "return" not in task.lower():
        return None
    return skill_path.read_text(encoding="utf-8")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        skill_path = Path(tmp) / "SKILL.md"
        skill_path.write_text(SKILL_MD, encoding="utf-8")

        for task in [
            "How long is standard shipping?",
            "I need a refund for unread print order A300.",
        ]:
            skill = load_skill_if_refund(task, skill_path)
            system = "Bookstore desk. Skills available: refund-print."
            msgs = [SystemMessage(system)]
            if skill:
                msgs.append(SystemMessage(f"Loaded SKILL.md:\n{skill}"))
            msgs.append(HumanMessage(task + " Policy reminder: standard shipping 5 days."))
            print(task, "=>", model.invoke(msgs).content)
            print("loaded_skill:", bool(skill))


if __name__ == "__main__":
    print(__doc__)
    main()
