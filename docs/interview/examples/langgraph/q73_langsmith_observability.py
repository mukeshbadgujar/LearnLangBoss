"""Q73. What to look at in LangSmith (local span list stand-in).

THE PROBLEM
Tracing is "on", but on-call does not know which LangSmith fields matter when
Order A100 refunds look slow or expensive.

WHAT WE ARE GOING TO SOLVE
Explain trace tree, latency, and token usage using a local list of spans from
a real graph run - not a fake chat model, and not duplicating q70's callback demo.

WHAT THIS EXAMPLE IS ABOUT
A100 desk graph records spans with latency_ms and token estimates you would
read in the LangSmith UI.

WHAT IT SOLVES
A checklist: open the trace, find the slow node, check token totals, confirm
tool inputs/outputs.

KEYWORDS
- Trace: nested tree of runs for one invoke.
- Latency: wall time per span / total.
- Tokens: prompt + completion usage (estimated here without calling a model).
"""

from typing import TypedDict
from time import perf_counter

from langgraph.graph import END, START, StateGraph


class DeskState(TypedDict):
    order_id: str
    body: str


# Stand-in for LangSmith run tree (no LLM - tokens are illustrative estimates).
SPANS: list[dict] = []


def span(name: str, tokens_in: int, tokens_out: int):
    def deco(fn):
        def inner(state: DeskState) -> dict:
            t0 = perf_counter()
            out = fn(state)
            SPANS.append(
                {
                    "name": name,
                    "latency_ms": round((perf_counter() - t0) * 1000, 3),
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                    "output_preview": str(out)[:80],
                }
            )
            return out

        return inner

    return deco


@span("shipping_lookup", tokens_in=40, tokens_out=0)
def shipping_lookup(state: DeskState) -> dict:
    return {"body": f"{state['order_id']} The Little Prince shipped $18"}


@span("policy_compose", tokens_in=120, tokens_out=60)
def policy_compose(state: DeskState) -> dict:
    return {"body": state["body"] + " | return window 30 days unread print"}


def build_graph():
    g = StateGraph(DeskState)
    g.add_node("shipping_lookup", shipping_lookup)
    g.add_node("policy_compose", policy_compose)
    g.add_edge(START, "shipping_lookup")
    g.add_edge("shipping_lookup", "policy_compose")
    g.add_edge("policy_compose", END)
    return g.compile()


def main() -> None:
    SPANS.clear()
    out = build_graph().invoke({"order_id": "A100", "body": ""})
    print("Final:", out["body"])
    print("In LangSmith you would open this trace and check:")
    total_lat = sum(s["latency_ms"] for s in SPANS)
    total_tok = sum(s["tokens_in"] + s["tokens_out"] for s in SPANS)
    for s in SPANS:
        print(" ", s)
    print(f"Total latency_ms~={total_lat} | total tokens~={total_tok}")
    slowest = max(SPANS, key=lambda s: s["latency_ms"])
    print("Investigate slowest span first:", slowest["name"])


if __name__ == "__main__":
    print(__doc__)
    main()
