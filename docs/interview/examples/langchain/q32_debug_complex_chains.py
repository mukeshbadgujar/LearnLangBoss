"""Q32. How would you debug complex chains?

THE PROBLEM
A five-step desk chain returns a wrong answer and the final text alone does not show which step failed.

WHAT WE ARE GOING TO SOLVE
Attach a callback that prints each chain start and end so you see the trajectory.

WHAT THIS EXAMPLE IS ABOUT
A tiny policy chain answers a shipping question while a handler logs each runnable.

WHAT IT SOLVES
You can tell whether the bug was the prompt, the model, or the parser by reading the log.

KEYWORDS
- Tracing: Recording every step's inputs, outputs, and timing for a run.
- Callback: A hook LangChain calls around model and chain events.
- Trajectory: The sequence of steps an agent or chain actually took.
- Isolation: Testing one runnable alone with a known input.

Run:
    python docs/interview/examples/langchain/q32_debug_complex_chains.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import chat_model

from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

model = chat_model()


class DeskTrace(BaseCallbackHandler):
    """Prints chain boundaries  -  the same idea LangSmith hosts as a UI."""

    def on_chain_start(self, serialized: dict[str, Any], inputs: dict[str, Any], **kwargs: Any) -> None:
        name = (serialized or {}).get("name") or (serialized or {}).get("id", ["chain"])[-1]
        print(f"START {name} inputs={list(inputs) if isinstance(inputs, dict) else type(inputs).__name__}")

    def on_chain_end(self, outputs: Any, **kwargs: Any) -> None:
        preview = str(outputs)[:80]
        print(f"END   -> {preview}")


def main() -> None:
    prompt = ChatPromptTemplate.from_template(
        "Bookstore desk. One sentence.\nPolicy: standard shipping is 5 business days.\nQ: {q}"
    )
    chain = prompt | model | StrOutputParser()
    # Invoke with a callback instead of guessing from the final string alone.
    answer = chain.invoke({"q": "How long is standard shipping?"}, config={"callbacks": [DeskTrace()]})
    print("Answer:", answer)


if __name__ == "__main__":
    print(__doc__)
    main()
