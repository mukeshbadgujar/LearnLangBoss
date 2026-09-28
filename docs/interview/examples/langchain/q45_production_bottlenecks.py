"""Q45. What are common production bottlenecks?

THE PROBLEM
Teams blame the model when the desk is slow, but the delay is usually context, retrieval, or loops.

WHAT WE ARE GOING TO SOLVE
Measure four common bottlenecks on a tiny bookstore ticket path.

WHAT THIS EXAMPLE IS ABOUT
Compare a fat prompt versus a trimmed prompt, and show a hard agent step cap.

WHAT IT SOLVES
You name real bottlenecks: context growth, retrieval quality, unbounded loops, rate limits.

KEYWORDS
- Bottleneck: The step that limits overall speed or cost.
- Context growth: History and tool outputs expanding every turn.
- Step limit: Cap on how many agent iterations one ticket may use.
- Rate limit: Provider-wide call ceiling shared by all app instances.

Run:
    python docs/interview/examples/langchain/q45_production_bottlenecks.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import chat_model

from langchain_core.messages import HumanMessage, SystemMessage

model = chat_model()

FAT = "Policy dump: " + ("Unread print 30 days. Digital no refund after download. " * 40)
TRIM = "Unread print: 30 days. Digital after download: no refund."


def main() -> None:
    q = "Can I return an unread print book from order A300?"
    fat_msgs = [SystemMessage(FAT), HumanMessage(q)]
    trim_msgs = [SystemMessage(TRIM), HumanMessage(q)]
    print("fat_chars=", sum(len(m.content) for m in fat_msgs))
    print("trim_chars=", sum(len(m.content) for m in trim_msgs))
    print("trim_answer:", model.invoke(trim_msgs).content)

    # Unbounded agent loops: always set a hard step budget in production.
    max_agent_steps = 4
    print(f"agent_step_limit={max_agent_steps} (cap cost when tools loop)")
    print("other bottlenecks: bad retrieval chunks; provider rate limits across replicas")


if __name__ == "__main__":
    print(__doc__)
    main()
