"""Q34. How do you monitor LLM applications?

THE PROBLEM
Uptime checks say the desk is healthy while answer quality and token cost silently degrade.

WHAT WE ARE GOING TO SOLVE
Add LLM-specific metrics: tokens and latency per request via a callback.

WHAT THIS EXAMPLE IS ABOUT
One shipping question runs while a handler records token usage and duration.

WHAT IT SOLVES
You see cost drivers per ticket, not only whether the process is up.

KEYWORDS
- Monitoring: Watching live metrics to detect problems early.
- Latency: How long a request takes from start to finish.
- Token usage: How many tokens the provider billed for the call.
- Quality drift: Answers get worse even when your code did not change.

Run:
    python docs/interview/examples/langchain/q34_monitor.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import chat_model

import time
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.prompts import ChatPromptTemplate

model = chat_model()


class MetricsHandler(BaseCallbackHandler):
    def __init__(self) -> None:
        self.t0 = 0.0
        self.tokens = 0

    def on_llm_start(self, *args: Any, **kwargs: Any) -> None:
        self.t0 = time.perf_counter()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        ms = (time.perf_counter() - self.t0) * 1000
        usage = (response.llm_output or {}).get("token_usage") or {}
        self.tokens = usage.get("total_tokens") or usage.get("total_tokens".upper(), 0) or 0
        # Groq may put usage on generations; still print wall time either way.
        print(f"latency_ms={ms:.0f} token_usage={usage or 'n/a'}")


def main() -> None:
    metrics = MetricsHandler()
    prompt = ChatPromptTemplate.from_template(
        "One word answer. Standard shipping days for the bookstore? Policy says 5."
    )
    msg = (prompt | model).invoke({}, config={"callbacks": [metrics]})
    print("answer:", msg.content)


if __name__ == "__main__":
    print(__doc__)
    main()
