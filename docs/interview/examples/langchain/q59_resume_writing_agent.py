"""Q59. Outline the architecture for Project 2: Interactive Resume Document Writing Agent.

THE PROBLEM
Staff need a bookstore role resume drafted through interview turns, then filled into a template.

WHAT WE ARE GOING TO SOLVE
Collect fields conversationally, validate with a simple schema check, then render a text template.

WHAT THIS EXAMPLE IS ABOUT
A clerk named Priya supplies title and skills; the agent fills a tiny resume template for the desk.

WHAT IT SOLVES
Multi-turn gather -> validate -> render, the skeleton of a document-writing agent.

KEYWORDS
- Document writing agent: Conversationally gathers data then fills a template.
- Template: A document with placeholders like {{NAME}} replaced by collected fields.
- Validation: Checking required fields before rendering.
- Multi-turn interview: Asking follow-ups until the profile is complete.

Run:
    python docs/interview/examples/langchain/q59_resume_writing_agent.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import chat_model

from langchain_core.messages import HumanMessage, SystemMessage

model = chat_model()

TEMPLATE = """
# {{FULL_NAME}}
Role: {{TITLE}}
Bookstore desk skills: {{SKILLS}}
""".strip()


def main() -> None:
    # Simulated interview answers (UI would collect these across turns).
    profile = {
        "FULL_NAME": "Priya Shah",
        "TITLE": "Bookstore Support Specialist",
        "SKILLS": "order lookup, returns policy, ticket escalation",
    }
    missing = [k for k, v in profile.items() if not v]
    if missing:
        raise SystemExit(f"missing fields: {missing}")

    filled = TEMPLATE
    for key, value in profile.items():
        filled = filled.replace("{{" + key + "}}", value)

    polish = model.invoke([
        SystemMessage("Lightly polish the resume text. Keep placeholders already filled. Be brief."),
        HumanMessage(filled),
    ]).content
    print(polish)


if __name__ == "__main__":
    print(__doc__)
    main()
