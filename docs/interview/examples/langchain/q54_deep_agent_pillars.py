"""Q54. What are the 4 core pillars of Deep Agent architecture?

THE PROBLEM
"Deep agent" answers stay vague unless you can name the four concrete pillars.

WHAT WE ARE GOING TO SOLVE
Demonstrate planning, scratchpad files, sub-agent quarantine, and context middleware in miniature.

WHAT THIS EXAMPLE IS ABOUT
A bookstore refund task updates a todo, writes a note file, and asks a sub-call for policy only.

WHAT IT SOLVES
Each pillar appears as a small, visible step you can describe in an interview.

KEYWORDS
- Explicit planning: An external checklist the agent updates (todo_write).
- Virtual filesystem: read_file/write_file style scratch space outside chat.
- Sub-agent quarantine: Delegate research; return only a distilled summary.
- Context middleware: Automatic trim/summary so history stays within budget.

Run:
    python docs/interview/examples/langchain/q54_deep_agent_pillars.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import chat_model

from langchain_core.messages import HumanMessage, SystemMessage

model = chat_model()


def policy_subagent() -> str:
    """Pillar 3: isolated call returns a short summary, not a long transcript."""
    return model.invoke([
        SystemMessage("Reply in one sentence only."),
        HumanMessage("Unread print return window for the bookstore?"),
    ]).content


def main() -> None:
    # Pillar 1  -  explicit plan
    todo = ["lookup A300", "fetch policy", "draft note"]
    todo[0] = "[x] lookup A300"

    # Pillar 2  -  scratch file outside chat
    note_path = Path(__file__).with_name("_q54_scratch.txt")
    note_path.write_text("A300 Dune delivered $16, unread claimed.\n", encoding="utf-8")
    scratch = note_path.read_text(encoding="utf-8").strip()

    # Pillar 3  -  quarantined sub-agent
    policy = policy_subagent()

    # Pillar 4  -  middleware stand-in: only scratch + policy enter the final prompt
    final = model.invoke([
        SystemMessage("Bookstore desk. One short refund draft."),
        HumanMessage(f"Notes: {scratch}\nPolicy: {policy}"),
    ]).content
    print("todo:", todo)
    print("scratch:", scratch)
    print("policy_summary:", policy)
    print("final:", final)
    note_path.unlink(missing_ok=True)


if __name__ == "__main__":
    print(__doc__)
    main()
