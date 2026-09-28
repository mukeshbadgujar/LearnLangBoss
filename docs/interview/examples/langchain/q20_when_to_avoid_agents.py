"""Q20. When should you avoid agents?

THE PROBLEM
Agents are trendy, but a known shipping FAQ does not need a tool loop. Extra
loops raise cost and make errors harder to budget.

WHAT WE ARE GOING TO SOLVE
Prefer a fixed chain when you can draw the flowchart. Show that path in code.

WHAT THIS EXAMPLE IS ABOUT
Always: answer shipping from a fixed policy string via prompt | model. No tools.
We also print why an agent would be the wrong default here.

WHAT IT SOLVES
A predictable one-call chain for a known workflow  -  lower cost, clearer errors.

KEYWORDS
- Prefer chains: Use agents only when the path is unknown.
- Cost ceiling: Agents can loop and multiply token spend.
- Known workflow: If you can draw the flowchart, build the flowchart.
- Error cost: Sensitive actions need approval, not free agent loops.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import chat_model

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

model = chat_model()


def main() -> None:
    # Fixed path: retrieve policy text is already here; no tool choice needed.
    prompt = ChatPromptTemplate.from_template(
        "You are the bookstore desk. Answer in one sentence.\n"
        "Policy: standard shipping 5 business days; express 2 days.\n"
        "Question: {question}"
    )
    chain = prompt | model | StrOutputParser()
    print(chain.invoke({"question": "How long is express shipping?"}))
    print("---")
    print("Avoid an agent here: steps are known, one model call is enough.")


if __name__ == "__main__":
    print(__doc__)
    main()
