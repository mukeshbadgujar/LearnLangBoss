"""Q20. MemorySaver vs SqliteSaver vs PostgresSaver

THE PROBLEM
Picking a checkpointer for local desk experiments vs a multi-pod bookstore API.

WHAT WE ARE GOING TO SOLVE
Construct MemorySaver and SqliteSaver (in-memory sqlite). Document PostgresSaver
in comments without requiring a Postgres server.

WHAT THIS EXAMPLE IS ABOUT
Persist A100 refund stage once in RAM and once in SqliteSaver(:memory:).

WHAT IT SOLVES
MemorySaver = dev/RAM. SqliteSaver = single-process disk/memory file.
PostgresSaver = production multi-instance (shown as comment).

KEYWORDS
- MemorySaver: in-process RAM checkpoints (lost on restart).
- SqliteSaver: SQLite-backed checkpoints.
- PostgresSaver: Postgres-backed checkpoints for production HA.
"""

import sqlite3
from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

# Production sketch (do not require Postgres here):
# from langgraph.checkpoint.postgres import PostgresSaver
# with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
#     checkpointer.setup()
#     graph = builder.compile(checkpointer=checkpointer)


class StageState(TypedDict):
    order_id: str
    stage: str


def advance(state: StageState) -> dict:
    return {"stage": "policy_ok_unread_30d", "order_id": "A100"}


def make_graph(checkpointer):
    g = StateGraph(StageState)
    g.add_node("advance", advance)
    g.add_edge(START, "advance")
    g.add_edge("advance", END)
    return g.compile(checkpointer=checkpointer)


def main() -> None:
    mem = MemorySaver()
    g_mem = make_graph(mem)
    cfg = {"configurable": {"thread_id": "mem-a100"}}
    print("MemorySaver:", g_mem.invoke({"order_id": "", "stage": ""}, cfg)["stage"])

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    sqlite_saver = SqliteSaver(conn)
    # SqliteSaver.from_conn_string(":memory:") also works as a context manager.
    g_sql = make_graph(sqlite_saver)
    cfg2 = {"configurable": {"thread_id": "sql-a100"}}
    print("SqliteSaver:", g_sql.invoke({"order_id": "", "stage": ""}, cfg2)["stage"])
    print("PostgresSaver: use from_conn_string + setup() in real deploys (see comment).")


if __name__ == "__main__":
    print(__doc__)
    main()
