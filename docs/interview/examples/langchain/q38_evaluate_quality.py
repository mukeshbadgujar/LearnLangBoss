"""Q38. How do you evaluate application quality?

THE PROBLEM
Manual review of every desk reply does not scale past a handful of tickets.

WHAT WE ARE GOING TO SOLVE
Score one answer against a ground-truth policy with a second judge prompt.

WHAT THIS EXAMPLE IS ABOUT
A candidate reply about the 30-day print return window is graded pass/fail.

WHAT IT SOLVES
You get an automated groundedness check you can later put in CI.

KEYWORDS
- Evaluation: Scoring outputs against criteria or reference answers.
- Ground truth: The trusted correct answer used as a reference.
- LLM-as-judge: Using a model to grade another model's output.
- Dataset: A list of inputs and expected behaviors used for evals.

Run:
    python docs/interview/examples/langchain/q38_evaluate_quality.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import chat_model

from langchain_core.messages import HumanMessage

model = chat_model()

CASE = {
    "question": "How long to return an unread print book?",
    "ground_truth": "Unread print books may be returned within 30 days.",
    "candidate": "You have 30 days to return an unread print book.",
}


def main() -> None:
    judge = (
        "Score PASS or FAIL only.\n"
        f"Question: {CASE['question']}\n"
        f"Ground truth: {CASE['ground_truth']}\n"
        f"Candidate: {CASE['candidate']}\n"
        "PASS if the candidate matches the ground truth on the return window."
    )
    grade = model.invoke([HumanMessage(judge)]).content.strip()
    print("grade:", grade)
    # Deterministic check beside the judge: number must appear.
    print("contains_30:", "30" in CASE["candidate"])


if __name__ == "__main__":
    print(__doc__)
    main()
