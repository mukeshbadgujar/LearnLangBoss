"""Q23. Encrypt sensitive checkpoint fields

THE PROBLEM
Checkpoints may hold shopper email / card-last-four for A100. At rest that must
not be plain serialization in the DB.

WHAT WE ARE GOING TO SOLVE
Wire EncryptedSerializer into MemorySaver (same serde hook PostgresSaver uses).

WHAT THIS EXAMPLE IS ABOUT
Store a refund state with email; serde encrypts before the checkpointer persists.

WHAT IT SOLVES
Custom serde (EncryptedSerializer) encrypts checkpoint payloads at rest.

KEYWORDS
- serde: serializer/deserializer for checkpoint blobs.
- EncryptedSerializer: AES wrapper shipped with LangGraph.
- At rest: data sitting in the database, not only in transit.
"""

import os
from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph.graph import END, START, StateGraph


class SensitiveState(TypedDict):
    order_id: str
    email: str
    decision: str


def stamp(state: SensitiveState) -> dict:
    return {"decision": "approve_unread_30d", "order_id": "A100"}


def build_graph(checkpointer: MemorySaver):
    g = StateGraph(SensitiveState)
    g.add_node("stamp", stamp)
    g.add_edge(START, "stamp")
    g.add_edge("stamp", END)
    return g.compile(checkpointer=checkpointer)


def main() -> None:
    # EncryptedSerializer reads LANGGRAPH_AES_KEY (32-byte key material).
    # Demo key only  -  use a secret manager in production.
    os.environ.setdefault(
        "LANGGRAPH_AES_KEY",
        "bookstore-demo-aes-key-32bytes!!",  # 32 chars
    )
    encrypted_serde = EncryptedSerializer.from_pycryptodome_aes()
    # Same pattern as: PostgresSaver(conn, serde=encrypted_serde)
    checkpointer = MemorySaver(serde=encrypted_serde)

    graph = build_graph(checkpointer)
    config = {"configurable": {"thread_id": "pii-a100"}}
    out = graph.invoke(
        {
            "order_id": "",
            "email": "reader@example.com",
            "decision": "",
        },
        config,
    )
    print("runtime values:", out)
    print("Checkpointer uses EncryptedSerializer  -  blobs encrypted at rest.")


if __name__ == "__main__":
    print(__doc__)
    main()
