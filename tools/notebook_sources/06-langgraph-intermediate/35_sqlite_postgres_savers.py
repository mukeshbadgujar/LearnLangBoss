# %% [markdown]
# # 35 - SQLite and Postgres Savers
#
# | | |
# |---|---|
# | **Level** | Intermediate (LangGraph) |
# | **Time** | 40 minutes |
# | **Prerequisites** | `34_checkpointing_thread_ids` |
# | **Checklist ID** | `35_sqlite_postgres_savers` |
#
# ## Why this matters
#
# `InMemorySaver` loses everything when the process exits. That is fine in a
# notebook and fatal in a product: a user's half-finished approval, a running
# multi-step job, an hour of conversation - all gone on the next deploy.
#
# This notebook swaps in durable storage. The API is identical; only the object
# you pass to `compile()` changes. That is the whole point of the checkpointer
# abstraction.

# %%
from pathlib import Path
import sys

_root = Path.cwd()
while not (_root / "shared").is_dir() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from shared.notebook_setup import setup, require  # noqa: E402

ctx = setup("35_sqlite_postgres_savers")

# %%
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from shared.llm import get_chat_model

model = get_chat_model()


class ChatState(MessagesState):
    turn_count: int


def chat_node(state: ChatState) -> dict:
    reply = model.invoke([("system", "You are a concise assistant. Two sentences maximum.")] + state["messages"])
    return {"messages": [reply], "turn_count": state.get("turn_count", 0) + 1}


builder = StateGraph(ChatState)
builder.add_node("chat", chat_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

# %% [markdown]
# ## 1. `SqliteSaver`
#
# One file, no server, real durability. It is the right choice for desktop
# tools, CLIs, single-process services and anything you want to be able to zip
# up and email.

# %%
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver

db_path = ctx.artifact("checkpoints.sqlite")
if db_path.exists():
    db_path.unlink()

# check_same_thread=False because the graph may run on a different thread than
# the one that opened the connection (a web server will).
conn = sqlite3.connect(db_path, check_same_thread=False)
saver = SqliteSaver(conn)

graph = builder.compile(checkpointer=saver)
config = {"configurable": {"thread_id": "durable-1"}}

graph.invoke({"messages": [HumanMessage("My name is Mukesh. Remember it.")], "turn_count": 0}, config)
graph.invoke({"messages": [HumanMessage("What is my name?")]}, config)

print(f"file: {db_path.name}  ({db_path.stat().st_size:,} bytes)")
print(f"turns: {graph.get_state(config).values['turn_count']}")
print(graph.get_state(config).values["messages"][-1].content.strip()[:100])

# %% [markdown]
# ### Proving it survives
#
# Close everything, reopen from the file, and ask a question that requires
# memory from before the "restart".

# %%
conn.close()
print("connection closed - simulating a process restart\n")

conn2 = sqlite3.connect(db_path, check_same_thread=False)
restarted = builder.compile(checkpointer=SqliteSaver(conn2))

state = restarted.get_state(config)
print(f"recovered {len(state.values['messages'])} messages, turn_count={state.values['turn_count']}")

answer = restarted.invoke({"messages": [HumanMessage("What did I tell you my name was?")]}, config)
print("\n", answer["messages"][-1].content.strip()[:130])

# %% [markdown]
# Nothing was re-sent and no history was rebuilt. A completely new process
# picked up a conversation from a file.
#
# ### `from_conn_string` - the context-manager form
#
# Convenient for scripts. Note it is a context manager, so the connection closes
# on exit - the graph is unusable afterwards.

# %%
with SqliteSaver.from_conn_string(str(db_path)) as scoped_saver:
    scoped = builder.compile(checkpointer=scoped_saver)
    print("inside the block:", len(scoped.get_state(config).values["messages"]), "messages")

try:
    scoped.get_state(config)
except Exception as exc:
    print(f"after the block:  {type(exc).__name__} - the connection is closed")

# %% [markdown]
# `SqliteSaver.from_conn_string(":memory:")` gives an in-memory SQLite database,
# which is useful in tests when you want real SQL semantics without a file.

# %%
with SqliteSaver.from_conn_string(":memory:") as memory_sqlite:
    test_graph = builder.compile(checkpointer=memory_sqlite)
    test_cfg = {"configurable": {"thread_id": "test-1"}}
    test_graph.invoke({"messages": [HumanMessage("Say OK.")], "turn_count": 0}, test_cfg)
    print("ephemeral SQLite turns:", test_graph.get_state(test_cfg).values["turn_count"])

# %% [markdown]
# ## 2. What is actually in the file
#
# Worth looking at once - it demystifies checkpointing completely.

# %%
inspect_conn = sqlite3.connect(db_path)
tables = [row[0] for row in inspect_conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
)]
print("tables:", tables)

for table in tables:
    count = inspect_conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    columns = [row[1] for row in inspect_conn.execute(f"PRAGMA table_info({table})")]
    print(f"\n{table}  ({count} rows)")
    print(f"  columns: {', '.join(columns)}")

# %%
rows = inspect_conn.execute(
    "SELECT thread_id, checkpoint_id, type, LENGTH(checkpoint) FROM checkpoints ORDER BY checkpoint_id LIMIT 6"
).fetchall()
print(f"{'thread':12} {'checkpoint_id':38} {'type':12} bytes")
for thread_id, checkpoint_id, kind, size in rows:
    print(f"{thread_id:12} {checkpoint_id:38} {str(kind):12} {size:,}")

inspect_conn.close()

# %% [markdown]
# The state blob is serialised per checkpoint, which is the concrete reason
# notebook 34 warned about keeping large payloads out of state: a 5 MB state
# writes 5 MB on **every** step.

# %% [markdown]
# ## 3. Async SQLite
#
# In an async server, the synchronous saver blocks the event loop on every step.
# Use `AsyncSqliteSaver` with the async graph methods.

# %%
async def async_sqlite_demo() -> None:
    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    except ImportError:
        print("[skipped] AsyncSqliteSaver needs aiosqlite: python -m pip install aiosqlite")
        return

    async with AsyncSqliteSaver.from_conn_string(str(ctx.artifact("async_checkpoints.sqlite"))) as async_saver:
        async_graph = builder.compile(checkpointer=async_saver)
        cfg = {"configurable": {"thread_id": "async-1"}}
        await async_graph.ainvoke({"messages": [HumanMessage("My favourite colour is teal.")], "turn_count": 0}, cfg)
        result = await async_graph.ainvoke({"messages": [HumanMessage("What is my favourite colour?")]}, cfg)
        print("async reply:", result["messages"][-1].content.strip()[:110])

        state = await async_graph.aget_state(cfg)
        print("async turns:", state.values["turn_count"])


await async_sqlite_demo()

# %% [markdown]
# **Pair the saver with the call style.** A sync saver under `ainvoke` will run,
# but it blocks the loop; an async saver under `invoke` raises. Matching them is
# not optional in a real server.

# %% [markdown]
# ## 4. SQLite's limits
#
# SQLite is a library, not a server. That has consequences.
#
# | Limit | Consequence |
# |---|---|
# | One writer at a time | Concurrent writes get `database is locked` |
# | File-based locking | Breaks over NFS and most container volume mounts |
# | Single host | Cannot be shared by several app instances |
# | No connection pooling | Every worker opens its own handle |
#
# Some of this is tunable - WAL mode allows one writer plus many readers, and a
# busy timeout makes contention wait rather than fail.

# %%
tuned_conn = sqlite3.connect(ctx.artifact("tuned.sqlite"), check_same_thread=False)
tuned_conn.execute("PRAGMA journal_mode=WAL")       # readers do not block the writer
tuned_conn.execute("PRAGMA busy_timeout=5000")      # wait 5s instead of failing instantly
tuned_conn.execute("PRAGMA synchronous=NORMAL")     # faster; safe with WAL

for pragma in ("journal_mode", "busy_timeout", "synchronous"):
    print(f"  {pragma:14} {tuned_conn.execute(f'PRAGMA {pragma}').fetchone()[0]}")

tuned = builder.compile(checkpointer=SqliteSaver(tuned_conn))
tuned.invoke({"messages": [HumanMessage("Say OK.")], "turn_count": 0},
             {"configurable": {"thread_id": "tuned-1"}})
print("\ntuned saver works:", tuned.get_state({"configurable": {"thread_id": "tuned-1"}}).values["turn_count"])
tuned_conn.close()

# %% [markdown]
# Even tuned, the rule stands: **one process writing.** The moment you run two
# app replicas, move to Postgres.

# %% [markdown]
# ## 5. `PostgresSaver` - the production answer
#
# Requires a running Postgres and `pip install langgraph-checkpoint-postgres`.
# The code below runs only if `POSTGRES_URI` is set in your `.env`.

# %%
import os

if require("POSTGRES_URI", feature="PostgresSaver"):
    try:
        from langgraph.checkpoint.postgres import PostgresSaver

        with PostgresSaver.from_conn_string(os.environ["POSTGRES_URI"]) as pg_saver:
            pg_saver.setup()                      # creates tables; idempotent, run once per deploy
            pg_graph = builder.compile(checkpointer=pg_saver)
            cfg = {"configurable": {"thread_id": "pg-1"}}
            pg_graph.invoke({"messages": [HumanMessage("Hello from Postgres.")], "turn_count": 0}, cfg)
            print("postgres turns:", pg_graph.get_state(cfg).values["turn_count"])
    except ImportError:
        print("[skipped] python -m pip install langgraph-checkpoint-postgres")
    except Exception as exc:
        print(f"[skipped] could not reach Postgres: {type(exc).__name__}: {str(exc)[:120]}")

# %% [markdown]
# ### The Postgres shape
#
# ```python
# from psycopg_pool import ConnectionPool
# from langgraph.checkpoint.postgres import PostgresSaver
#
# pool = ConnectionPool(
#     conninfo=os.environ["POSTGRES_URI"],
#     max_size=20,
#     kwargs={"autocommit": True, "prepare_threshold": 0},
# )
# saver = PostgresSaver(pool)
# saver.setup()                      # once, at deploy time - not per request
# graph = builder.compile(checkpointer=saver)
# ```
#
# Async version:
#
# ```python
# from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
#
# async with AsyncPostgresSaver.from_conn_string(uri) as saver:
#     await saver.setup()
#     graph = builder.compile(checkpointer=saver)
# ```
#
# Three things that bite people:
#
# 1. **`autocommit=True` and `prepare_threshold=0`** are required by the saver's
#    query pattern. Omit them and you will see prepared-statement errors.
# 2. **`setup()` is a migration.** Run it at deploy time, not on every request.
# 3. **Pool sizing.** Each concurrent graph run holds a connection for the length
#    of a step. Size the pool to peak concurrency, not average.

# %% [markdown]
# ## 6. Everything else is identical
#
# The strongest argument for this abstraction: swap the saver, change nothing
# else. Here is one graph and three backends.

# %%
def exercise(saver, label: str) -> None:
    graph = builder.compile(checkpointer=saver)
    cfg = {"configurable": {"thread_id": f"{label}-thread"}}
    graph.invoke({"messages": [HumanMessage("My project is codenamed Kestrel.")], "turn_count": 0}, cfg)
    graph.invoke({"messages": [HumanMessage("What is my project codename?")]}, cfg)

    state = graph.get_state(cfg)
    history = list(graph.get_state_history(cfg))
    print(f"{label:16} turns={state.values['turn_count']}  messages={len(state.values['messages'])}  "
          f"checkpoints={len(history)}")


from langgraph.checkpoint.memory import InMemorySaver

exercise(InMemorySaver(), "InMemorySaver")

file_conn = sqlite3.connect(ctx.artifact("compare.sqlite"), check_same_thread=False)
exercise(SqliteSaver(file_conn), "SqliteSaver")
file_conn.close()

with SqliteSaver.from_conn_string(":memory:") as ephemeral:
    exercise(ephemeral, "sqlite :memory:")

# %% [markdown]
# ## 7. Choosing
#
# | You are building | Use |
# |---|---|
# | A notebook or unit test | `InMemorySaver` |
# | A test needing real SQL semantics | `SqliteSaver.from_conn_string(":memory:")` |
# | A CLI, desktop app, or single-process service | `SqliteSaver` with a file |
# | Anything with more than one replica | `PostgresSaver` |
# | Anything an SRE will be paged about | `PostgresSaver` |
#
# Two operational habits worth forming now:
#
# - **Back up the checkpoint database.** It contains in-flight user work, not
#   just logs. Losing it loses pending approvals.
# - **Have a retention policy.** `delete_thread()` from notebook 34, run on a
#   schedule. Checkpoints grow without bound otherwise.

# %%
import datetime as dt

def prune_inactive_threads(saver, thread_ids: list[str], graph, days: int = 30) -> list[str]:
    """Sketch of a retention job: delete threads with no recent checkpoint."""
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)
    removed = []
    for thread_id in thread_ids:
        snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
        if not snapshot.created_at:
            continue
        if dt.datetime.fromisoformat(snapshot.created_at) < cutoff:
            saver.delete_thread(thread_id)
            removed.append(thread_id)
    return removed


demo_saver = InMemorySaver()
demo_graph = builder.compile(checkpointer=demo_saver)
for i in range(3):
    demo_graph.invoke({"messages": [HumanMessage(f"msg {i}")], "turn_count": 0},
                      {"configurable": {"thread_id": f"keep-{i}"}})

print("nothing is older than 30 days yet:", prune_inactive_threads(demo_saver, [f"keep-{i}" for i in range(3)], demo_graph))
print("same job with days=0 removes all:",
      prune_inactive_threads(demo_saver, [f"keep-{i}" for i in range(3)], demo_graph, days=0))

# %% [markdown]
# ## Try it yourself
#
# 1. **Genuine restart test.** Write a script that appends one message to a
#    SQLite-backed thread and exits. Run it three times and confirm the count grows.
# 2. **Trigger a lock.** Open two connections to the same SQLite file without
#    `busy_timeout` and write from both, then fix it with WAL and a timeout.
# 3. **Run Postgres locally** with `docker run -e POSTGRES_PASSWORD=x -p 5432:5432 postgres`,
#    set `POSTGRES_URI`, and re-run section 5.
# 4. **Measure the difference.** Time 50 graph steps against `InMemorySaver`,
#    `SqliteSaver` and (if available) Postgres.

# %% [markdown]
# ## Recap
#
# | Concept | Takeaway |
# |---|---|
# | Same API | Swap the saver; graph code is unchanged |
# | `SqliteSaver(conn)` | Pass `check_same_thread=False` for web servers |
# | `from_conn_string` | Context manager; connection closes on exit |
# | `":memory:"` | Real SQL semantics for tests, no file |
# | Inside the file | One serialised state blob per checkpoint per thread |
# | `AsyncSqliteSaver` | Match saver style to call style in async servers |
# | SQLite limits | One writer, single host, poor over network filesystems |
# | WAL + `busy_timeout` | Reduce contention; do not remove the single-writer limit |
# | `PostgresSaver` | Production; `setup()` at deploy, `autocommit=True`, size the pool |
# | Operations | Back up the checkpoint DB; prune old threads on a schedule |
#
# ## Next
#
# -> [36_human_in_the_loop.ipynb](36_human_in_the_loop.ipynb)
