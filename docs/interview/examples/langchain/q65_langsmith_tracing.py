"""Q65. How do you handle Observability and Tracing using LangSmith?

THE PROBLEM
Without a trace, a wrong bookstore answer is a black box across prompt, model, and parser.

WHAT WE ARE GOING TO SOLVE
Use a BaseCallbackHandler that prints chain start/end  -  the same idea LangSmith hosts as a UI.

WHAT THIS EXAMPLE IS ABOUT
A shipping FAQ chain runs with a local tracer; no LangSmith API key is required.

WHAT IT SOLVES
You see the execution tree locally and understand what LangSmith would store centrally.

KEYWORDS
- LangSmith: Hosted tracing and eval platform for LangChain runs.
- Trace: Recorded step-by-step path of one request.
- Callback handler: Local hook that receives chain/model events.
- Observability: Ability to inspect what the system did and why.

Run:
    python docs/interview/examples/langchain/q65_langsmith_tracing.py
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


class LocalTrace(BaseCallbackHandler):
    """Local stand-in for LangSmith: print the chain tree boundaries."""

    def on_chain_start(self, serialized: dict[str, Any], inputs: dict[str, Any], **kwargs: Any) -> None:
        name = (serialized or {}).get("name") or "chain"
        print(f"[trace] start {name}")

    def on_chain_end(self, outputs: Any, **kwargs: Any) -> None:
        print(f"[trace] end -> {str(outputs)[:60]}")


def main() -> None:
    # Production: LANGCHAIN_TRACING_V2=true and a LangSmith key send this tree to the hosted UI.
    prompt = ChatPromptTemplate.from_template(
        "Bookstore desk. One sentence. Policy: express shipping is 2 days.\nQ: {q}"
    )
    chain = prompt | model | StrOutputParser()
    answer = chain.invoke({"q": "How long is express shipping?"}, config={"callbacks": [LocalTrace()]})
    print("answer:", answer)


if __name__ == "__main__":
    print(__doc__)
    main()
