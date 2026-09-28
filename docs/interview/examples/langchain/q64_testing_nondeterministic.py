"""Q64. How do you approach Testing and Evaluation for non-deterministic AI agents and graphs?

THE PROBLEM
assert reply == "exactly this string" fails on valid paraphrases from the bookstore desk model.

WHAT WE ARE GOING TO SOLVE
Use a tiny dataset, deterministic checks, and an LLM-as-judge instead of exact string match.

WHAT THIS EXAMPLE IS ABOUT
A shipping answer is graded for containing "5" and for groundedness against policy.

WHAT IT SOLVES
Non-deterministic outputs become testable with criteria and ground truth, not brittle equality.

KEYWORDS
- Non-deterministic: Same input can yield different valid phrasings.
- Dataset eval: Many cases with inputs and expected goals.
- LLM-as-judge: A model scores another model's output against rules.
- Ground truth: Trusted reference used for comparison.

Run:
    python docs/interview/examples/langchain/q64_testing_nondeterministic.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import chat_model

from langchain_core.messages import HumanMessage, SystemMessage

model = chat_model()

CASES = [
    {
        "q": "How long is standard shipping?",
        "must_include": "5",
        "ground_truth": "Standard shipping takes 5 business days.",
    }
]


def main() -> None:
    case = CASES[0]
    answer = model.invoke([
        SystemMessage("Bookstore desk. Be brief. Policy: standard shipping is 5 business days."),
        HumanMessage(case["q"]),
    ]).content
    deterministic = case["must_include"] in answer
    judge = model.invoke([
        SystemMessage("Reply PASS or FAIL only."),
        HumanMessage(
            f"Ground truth: {case['ground_truth']}\nAnswer: {answer}\n"
            "PASS if the answer states the same day count."
        ),
    ]).content.strip()
    print("answer:", answer)
    print("deterministic_pass:", deterministic)
    print("judge:", judge)


if __name__ == "__main__":
    print(__doc__)
    main()
