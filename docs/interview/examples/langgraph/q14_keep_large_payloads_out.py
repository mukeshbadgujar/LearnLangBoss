"""Q14. Keep large payloads out of checkpointed state

THE PROBLEM
A customer attaches a long packing-slip OCR dump with the A100 refund. Putting
the full text in state balloons every checkpoint.

WHAT WE ARE GOING TO SOLVE
Store a reference (blob id / path) in state; fetch body only inside the node
that needs a snippet.

WHAT THIS EXAMPLE IS ABOUT
Fake object store holds the OCR text for The Little Prince shipment. State keeps
blob_ref only; policy node loads a short excerpt.

WHAT IT SOLVES
Checkpoints stay small; heavy content lives in external storage.

KEYWORDS
- Payload: bulk content (logs, OCR, PDFs).
- Reference: lightweight pointer (id/URL/path) stored in state.
- Checkpoint bloat: large state serialized after every superstep.
"""

from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

# Stand-in for S3 / blob store - not in graph state.
_BLOBS: dict[str, str] = {}


class LightState(TypedDict):
    order_id: str
    blob_ref: str
    excerpt: str
    decision: str


def ingest_ocr(state: LightState) -> dict:
    huge = (
        "PACKING SLIP\n" + ("line\n" * 200) + "Item: The Little Prince $18 Order A100\n"
    )
    ref = f"blob://returns/{state['order_id']}.txt"
    _BLOBS[ref] = huge
    return {"blob_ref": ref}  # not the huge string


def policy_from_excerpt(state: LightState) -> dict:
    body = _BLOBS[state["blob_ref"]]
    # Touch storage on demand; keep only a tiny excerpt in state.
    excerpt = body[-80:].strip()
    return {
        "excerpt": excerpt,
        "decision": "approve $18 - unread print within 30 days",
    }


def build_graph(checkpointer: MemorySaver):
    g = StateGraph(LightState)
    g.add_node("ingest_ocr", ingest_ocr)
    g.add_node("policy_from_excerpt", policy_from_excerpt)
    g.add_edge(START, "ingest_ocr")
    g.add_edge("ingest_ocr", "policy_from_excerpt")
    g.add_edge("policy_from_excerpt", END)
    return g.compile(checkpointer=checkpointer)


def main() -> None:
    saver = MemorySaver()
    graph = build_graph(saver)
    config = {"configurable": {"thread_id": "ocr-a100"}}
    out = graph.invoke(
        {"order_id": "A100", "blob_ref": "", "excerpt": "", "decision": ""},
        config,
    )
    snap = graph.get_state(config)
    print("blob_ref in checkpoint:", snap.values["blob_ref"])
    print("excerpt only:", out["excerpt"][:60], "...")
    print("decision:", out["decision"])
    print("full OCR lives in _BLOBS, not in checkpoint values.")


if __name__ == "__main__":
    print(__doc__)
    main()
