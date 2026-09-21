# %% [markdown]
# # 46 - Deployment and Versioning
#
# | | |
# |---|---|
# | **Level** | Advanced (LangGraph) |
# | **Time** | 50 minutes |
# | **Prerequisites** | `45_langsmith_and_studio` |
# | **Checklist ID** | `46_deployment_and_versioning` |
#
# ## Why this matters
#
# A graph in a notebook and a graph serving traffic are different artefacts. The
# second one needs a durable checkpointer, an HTTP surface, streaming, health
# checks, configuration by environment, and a story for what happens to
# in-flight conversations when you deploy a new version.
#
# That last one is specific to LangGraph and catches people out: a user paused at
# an approval step yesterday resumes today against **your new code**. If the
# graph shape changed, that resume can fail.

# %%
from pathlib import Path
import sys

_root = Path.cwd()
while not (_root / "shared").is_dir() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from shared.notebook_setup import setup, require  # noqa: E402

ctx = setup("46_deployment_and_versioning")

# %%
import json
import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from shared.llm import get_chat_model

model = get_chat_model()
parser = StrOutputParser()

# %% [markdown]
# ## 1. Three ways to deploy
#
# | Option | You manage | Good for |
# |---|---|---|
# | **Self-hosted FastAPI** | Everything | Full control, existing infrastructure |
# | **LangGraph Platform** | Your graph code only | Fastest path to a production API |
# | **Serverless function** | Packaging and cold starts | Low, bursty traffic |
#
# This notebook builds the self-hosted version, because understanding it makes
# the managed options obvious. Notebook 53 turns it into a complete service.

# %% [markdown]
# ## 2. Structure the code for deployment
#
# A notebook defines everything inline. A deployable app separates the graph
# definition from the server.
#
# ```
# app/
#   __init__.py
#   config.py        # environment-driven settings
#   state.py         # state schemas
#   graphs.py        # builders + compiled graphs
#   dependencies.py  # checkpointer, store, model factories
#   server.py        # FastAPI
# langgraph.json     # for LangGraph Studio / Platform
# ```

# %%
APP_DIR = ctx.artifact("app")
APP_DIR.mkdir(parents=True, exist_ok=True)

(APP_DIR / "config.py").write_text('''"""Environment-driven settings. No secrets in code."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    environment: str = os.environ.get("APP_ENV", "dev")
    database_uri: str = os.environ.get("POSTGRES_URI", "")
    sqlite_path: str = os.environ.get("SQLITE_PATH", "./checkpoints.sqlite")
    model_name: str = os.environ.get("MODEL_NAME", "")
    max_concurrency: int = int(os.environ.get("MAX_CONCURRENCY", "8"))
    recursion_limit: int = int(os.environ.get("RECURSION_LIMIT", "25"))
    graph_version: str = os.environ.get("GRAPH_VERSION", "1.0.0")

    @property
    def is_production(self) -> bool:
        return self.environment == "prod"


settings = Settings()
''', encoding="utf-8")

print((APP_DIR / "config.py").read_text(encoding="utf-8")[:400])

# %%
(APP_DIR / "dependencies.py").write_text('''"""Process-wide singletons, created once at startup."""

import sqlite3
from contextlib import asynccontextmanager
from functools import lru_cache

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.memory import InMemoryStore

from .config import settings


@lru_cache(maxsize=1)
def get_checkpointer():
    """One checkpointer per process. Postgres in production."""
    if settings.database_uri:
        from langgraph.checkpoint.postgres import PostgresSaver

        saver = PostgresSaver.from_conn_string(settings.database_uri).__enter__()
        saver.setup()                      # idempotent; ideally a deploy-time migration
        return saver

    if settings.environment == "test":
        return InMemorySaver()

    conn = sqlite3.connect(settings.sqlite_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return SqliteSaver(conn)


@lru_cache(maxsize=1)
def get_store():
    return InMemoryStore()
''', encoding="utf-8")

print("wrote dependencies.py")

# %% [markdown]
# The `lru_cache(maxsize=1)` idiom matters: creating a checkpointer per request
# opens a new connection per request, and you will exhaust the pool under load.

# %% [markdown]
# ## 3. Version the graph
#
# The core deployment problem: a thread paused under v1 resumes under v2. Record
# the version in state so you can detect and handle that.

# %%
GRAPH_VERSION = "1.2.0"


class VersionedState(MessagesState):
    graph_version: str
    stage: str
    log: Annotated[list[str], operator.add]


def stamp_version(state: VersionedState) -> dict:
    """First node: record the version that started this thread."""
    existing = state.get("graph_version")
    if existing and existing != GRAPH_VERSION:
        return {"log": [f"MIGRATED from {existing} to {GRAPH_VERSION}"], "graph_version": GRAPH_VERSION}
    return {"graph_version": GRAPH_VERSION, "log": [f"started on {GRAPH_VERSION}"]}


def work(state: VersionedState) -> dict:
    return {"stage": "complete", "log": ["work done"]}


builder = StateGraph(VersionedState)
builder.add_sequence([("version", stamp_version), ("work", work)])
builder.add_edge(START, "version")
builder.add_edge("work", END)
versioned = builder.compile(checkpointer=InMemorySaver())

cfg = {"configurable": {"thread_id": "v-1"}}
print(versioned.invoke({"messages": [HumanMessage("hi")], "graph_version": "", "stage": "", "log": []}, cfg)["log"])

# simulate an older thread resuming under the new code
old_cfg = {"configurable": {"thread_id": "v-old"}}
versioned.update_state(old_cfg, {"graph_version": "1.0.0", "stage": "", "messages": [], "log": []})
print(versioned.invoke({"messages": [HumanMessage("resuming")]}, old_cfg)["log"])

# %% [markdown]
# ### Compatibility rules
#
# | Change | Safe for in-flight threads? |
# |---|---|
# | Add a new optional state key | Yes |
# | Add a new node not on the existing path | Yes |
# | Change a prompt | Yes (output may differ) |
# | **Rename a node** | **No** - a thread paused at the old name cannot resume |
# | **Remove a node** | **No** - same reason |
# | **Remove a state key** | **No** - downstream nodes may read it |
# | **Change a reducer** | **No** - existing values merge differently |
#
# For a breaking change, the safe pattern is: deploy the new version alongside,
# route new threads to it, and let old threads drain on the old version. If you
# cannot run both, drain first - finish or expire in-flight threads, then deploy.

# %%
def migrate_state(values: dict, from_version: str) -> dict:
    """Explicit migrations, the same discipline as a database schema."""
    migrated = dict(values)

    if from_version.startswith("1.0"):
        # 1.1 renamed `priority` to `urgency`
        if "priority" in migrated:
            migrated["urgency"] = migrated.pop("priority")
        migrated.setdefault("log", [])

    if from_version.startswith(("1.0", "1.1")):
        # 1.2 added a required field
        migrated.setdefault("tenant_id", "unknown")

    migrated["graph_version"] = GRAPH_VERSION
    return migrated


old_state = {"priority": "high", "messages": [], "graph_version": "1.0.0"}
print("before:", old_state)
print("after :", migrate_state(old_state, "1.0.0"))

# %% [markdown]
# ## 4. The FastAPI surface
#
# Four endpoints cover almost every real use: invoke, stream, read thread state,
# and resume from an interrupt.

# %%
SERVER = '''"""FastAPI service wrapping a LangGraph graph."""

import json
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from .config import settings
from .dependencies import get_checkpointer, get_store
from .graphs import build_graph

GRAPH = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Compile the graph once at startup, not per request."""
    global GRAPH
    GRAPH = build_graph(checkpointer=get_checkpointer(), store=get_store())
    yield
    GRAPH = None


app = FastAPI(title="Support Graph API", version=settings.graph_version, lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    thread_id: str = Field(min_length=1, max_length=128)
    user_id: str = ""
    tenant: str = "default"


def run_config(body: ChatRequest) -> dict[str, Any]:
    return {
        "configurable": {"thread_id": body.thread_id},
        "run_name": "support_graph",
        "tags": [f"env:{settings.environment}", f"tenant:{body.tenant}"],
        "metadata": {"user_id": body.user_id, "tenant": body.tenant,
                     "graph_version": settings.graph_version},
        "recursion_limit": settings.recursion_limit,
        "max_concurrency": settings.max_concurrency,
    }


@app.get("/health")
async def health() -> dict:
    """Liveness: is the process up?"""
    return {"status": "ok", "version": settings.graph_version}


@app.get("/ready")
async def ready() -> dict:
    """Readiness: can we actually serve? Checks the checkpointer."""
    try:
        GRAPH.get_state({"configurable": {"thread_id": "__healthcheck__"}})
    except Exception as exc:
        raise HTTPException(503, f"checkpointer unavailable: {type(exc).__name__}") from exc
    return {"status": "ready"}


@app.post("/chat")
async def chat(body: ChatRequest) -> dict:
    result = await GRAPH.ainvoke({"messages": [HumanMessage(body.message)]}, run_config(body))

    if "__interrupt__" in result:
        return {"status": "awaiting_approval",
                "thread_id": body.thread_id,
                "interrupt": result["__interrupt__"][0].value}

    return {"status": "complete",
            "thread_id": body.thread_id,
            "reply": result["messages"][-1].content}


@app.post("/chat/stream")
async def chat_stream(body: ChatRequest) -> StreamingResponse:
    async def events():
        try:
            async for mode, chunk in GRAPH.astream(
                {"messages": [HumanMessage(body.message)]},
                run_config(body),
                stream_mode=["updates", "messages", "custom"],
            ):
                if mode == "messages":
                    token, meta = chunk
                    if token.content:
                        payload = {"type": "token", "content": token.content}
                    else:
                        continue
                elif mode == "custom":
                    payload = {"type": "progress", **chunk}
                else:
                    payload = {"type": "node", "name": list(chunk)[0]}
                yield f"data: {json.dumps(payload)}\\n\\n"
            yield 'data: {"type": "done"}\\n\\n'
        except Exception as exc:
            yield f'data: {json.dumps({"type": "error", "message": str(exc)})}\\n\\n'

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class ResumeRequest(BaseModel):
    thread_id: str
    decision: Any


@app.post("/chat/resume")
async def resume(body: ResumeRequest) -> dict:
    config = {"configurable": {"thread_id": body.thread_id}}
    snapshot = await GRAPH.aget_state(config)
    if not snapshot.tasks or not any(t.interrupts for t in snapshot.tasks):
        raise HTTPException(409, "thread is not awaiting a decision")

    result = await GRAPH.ainvoke(Command(resume=body.decision), config)
    return {"status": "complete", "reply": result["messages"][-1].content}


@app.get("/threads/{thread_id}")
async def thread_state(thread_id: str) -> dict:
    snapshot = await GRAPH.aget_state({"configurable": {"thread_id": thread_id}})
    if not snapshot.values:
        raise HTTPException(404, "unknown thread")
    return {
        "thread_id": thread_id,
        "next": list(snapshot.next),
        "awaiting": [i.value for t in snapshot.tasks for i in t.interrupts],
        "message_count": len(snapshot.values.get("messages", [])),
        "updated_at": snapshot.created_at,
    }


@app.delete("/threads/{thread_id}")
async def delete_thread(thread_id: str) -> dict:
    get_checkpointer().delete_thread(thread_id)
    return {"status": "deleted", "thread_id": thread_id}
'''

(APP_DIR / "server.py").write_text(SERVER, encoding="utf-8")
print(SERVER[:1200])

# %% [markdown]
# Five things in there that separate this from a toy:
#
# 1. **`lifespan`** compiles the graph once. Compiling per request is a common
#    and expensive mistake.
# 2. **`/health` vs `/ready`** - liveness says the process is up; readiness says
#    it can serve. Kubernetes needs both, and conflating them causes restart loops.
# 3. **Async throughout** - `ainvoke` and `astream`, because a sync call blocks
#    the event loop (notebook 33).
# 4. **Interrupts are a normal response**, not an error. `status:
#    awaiting_approval` plus a `/chat/resume` endpoint is the HTTP shape of
#    notebook 36.
# 5. **`X-Accel-Buffering: no`** stops nginx buffering your SSE stream into
#    uselessness.

# %% [markdown]
# ## 5. Running it locally
#
# Let us actually exercise the graph the way the server would, without starting
# a server.

# %%
class SupportState(MessagesState):
    category: str


def classify(state: SupportState) -> dict:
    text = state["messages"][-1].content.lower()
    category = "billing" if any(w in text for w in ["charge", "invoice", "refund"]) else "general"
    return {"category": category}


def respond(state: SupportState) -> dict:
    reply = model.invoke([("system", f"You are a {state['category']} specialist. Two sentences.")]
                         + state["messages"])
    return {"messages": [reply]}


def build_graph(checkpointer=None, store=None):
    b = StateGraph(SupportState)
    b.add_sequence([("classify", classify), ("respond", respond)])
    b.add_edge(START, "classify")
    b.add_edge("respond", END)
    return b.compile(checkpointer=checkpointer, store=store, name="support_graph")


service_graph = build_graph(checkpointer=InMemorySaver())


async def simulate_request(message: str, thread_id: str) -> dict:
    """Exactly what the /chat handler does."""
    config = {
        "configurable": {"thread_id": thread_id},
        "run_name": "support_graph",
        "tags": ["env:dev"],
        "metadata": {"graph_version": GRAPH_VERSION},
        "recursion_limit": 25,
    }
    result = await service_graph.ainvoke({"messages": [HumanMessage(message)], "category": ""}, config)
    return {"status": "complete", "thread_id": thread_id,
            "category": result["category"], "reply": result["messages"][-1].content}


response = await simulate_request("We were charged twice for January.", "api-1")
print(json.dumps({**response, "reply": response["reply"][:120]}, indent=2))

# %%
async def simulate_stream(message: str, thread_id: str) -> None:
    config = {"configurable": {"thread_id": thread_id}}
    async for mode, chunk in service_graph.astream(
        {"messages": [HumanMessage(message)], "category": ""}, config,
        stream_mode=["updates", "messages"],
    ):
        if mode == "updates":
            print(f"\ndata: {json.dumps({'type': 'node', 'name': list(chunk)[0]})}")
        else:
            token, meta = chunk
            if token.content:
                print(token.content, end="", flush=True)
    print('\ndata: {"type": "done"}')


await simulate_stream("What is your data retention policy?", "api-2")

# %% [markdown]
# ## 6. `langgraph.json` and the CLI
#
# The same file Studio uses (notebook 45) also drives the LangGraph CLI and
# Platform deployment.

# %%
langgraph_json = {
    "dependencies": ["."],
    "graphs": {"support": "./app/graphs.py:support_graph"},
    "env": ".env",
    "python_version": "3.11",
}
(ctx.artifact("langgraph.json")).write_text(json.dumps(langgraph_json, indent=2), encoding="utf-8")
print(json.dumps(langgraph_json, indent=2))

# %% [markdown]
# ```bash
# python -m pip install --upgrade "langgraph-cli[inmem]"
#
# langgraph dev      # local dev server + Studio, in memory
# langgraph build    # build a Docker image
# langgraph up       # run that image with Postgres and Redis via docker compose
# ```
#
# `langgraph up` gives you the platform's runtime locally: persistence, task
# queue, streaming and a thread API, without writing the FastAPI layer yourself.
#
# The trade-off is the usual one:
#
# | | Self-hosted FastAPI | LangGraph Platform |
# |---|---|---|
# | Control over the HTTP surface | Total | Fixed API shape |
# | Infrastructure to run | Yours | Managed (or `langgraph up`) |
# | Background runs, cron, queueing | Build it | Included |
# | Cost | Your servers | Per-node pricing |
# | Fits existing auth/middleware | Easily | Needs adapting |

# %% [markdown]
# ## 7. Containerising the self-hosted version

# %%
DOCKERFILE = '''FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Never bake secrets into the image - inject them at runtime.
ENV APP_ENV=prod

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \\
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
'''
(ctx.artifact("Dockerfile")).write_text(DOCKERFILE, encoding="utf-8")
print(DOCKERFILE)

# %% [markdown]
# **Worker count and the checkpointer must match.** Four uvicorn workers are four
# processes. SQLite cannot take concurrent writers from four processes - use one
# worker with SQLite, or Postgres with several.

# %%
def validate_deployment(*, workers: int, checkpointer: str, replicas: int = 1) -> list[str]:
    """Catch the misconfigurations that only appear under load."""
    problems = []
    processes = workers * replicas

    if checkpointer == "memory":
        problems.append("InMemorySaver loses every conversation on restart")
        if processes > 1:
            problems.append("InMemorySaver is not shared between processes - users will see random threads")
    if checkpointer == "sqlite" and processes > 1:
        problems.append(f"SQLite with {processes} processes will hit 'database is locked'")
    if checkpointer == "postgres" and workers > 8:
        problems.append("check the connection pool is sized for this worker count")
    return problems


for setup_name, kwargs in [
    ("dev laptop", {"workers": 1, "checkpointer": "sqlite"}),
    ("naive prod", {"workers": 4, "checkpointer": "sqlite", "replicas": 3}),
    ("worst case", {"workers": 4, "checkpointer": "memory", "replicas": 3}),
    ("correct prod", {"workers": 4, "checkpointer": "postgres", "replicas": 3}),
]:
    issues = validate_deployment(**kwargs)
    print(f"{setup_name:13} {'OK' if not issues else 'PROBLEMS'}")
    for issue in issues:
        print(f"                - {issue}")

# %% [markdown]
# ## 8. Deploying safely
#
# ### Draining in-flight threads

# %%
import datetime as dt


def drain_report(graph, thread_ids: list[str]) -> dict:
    """Before a breaking deploy: what is still paused?"""
    paused, active, done = [], [], []
    for thread_id in thread_ids:
        snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
        if any(task.interrupts for task in snapshot.tasks):
            paused.append(thread_id)
        elif snapshot.next:
            active.append(thread_id)
        else:
            done.append(thread_id)
    return {"paused_awaiting_human": paused, "mid_execution": active, "complete": done}


for thread in ["d-1", "d-2"]:
    service_graph.invoke({"messages": [HumanMessage("hello")], "category": ""},
                         {"configurable": {"thread_id": thread}})

print(drain_report(service_graph, ["d-1", "d-2", "never-seen"]))

# %% [markdown]
# ### A rollout checklist
#
# | Step | Detail |
# |---|---|
# | 1. Classify the change | Additive or breaking? Use the table in section 3 |
# | 2. Run the regression suite | Notebook 45's `regression_gate` in CI |
# | 3. Migrate the checkpoint schema | `saver.setup()` as a deploy step, not at request time |
# | 4. Deploy to staging with production-like data | Especially long threads |
# | 5. Canary | 5% of traffic, watch error rate and p95 |
# | 6. Check paused threads | `drain_report`; a breaking change needs them drained |
# | 7. Full rollout | Keep the previous image ready |
# | 8. Watch for an hour | Error rate, latency, cost per request, approval queue depth |
#
# ### Rollback
#
# Rolling code back is easy. Rolling **state** back is not - threads created
# under v2 may contain keys v1 does not understand. Make new state keys optional
# and have v1 tolerate unknown keys, or accept that rollback abandons those
# threads.

# %% [markdown]
# ## 9. Configuration by environment

# %%
ENVIRONMENTS = {
    "dev":     {"checkpointer": "sqlite",   "workers": 1, "tracing": True,  "sample": 1.0,  "recursion_limit": 50},
    "staging": {"checkpointer": "postgres", "workers": 2, "tracing": True,  "sample": 1.0,  "recursion_limit": 25},
    "prod":    {"checkpointer": "postgres", "workers": 4, "tracing": True,  "sample": 0.1,  "recursion_limit": 25},
    "test":    {"checkpointer": "memory",   "workers": 1, "tracing": False, "sample": 0.0,  "recursion_limit": 10},
}

print(f"{'env':9} {'checkpointer':13} {'workers':>8} {'trace %':>8} {'recursion':>10}")
for name, config in ENVIRONMENTS.items():
    print(f"{name:9} {config['checkpointer']:13} {config['workers']:>8} "
          f"{config['sample'] * 100:>7.0f}% {config['recursion_limit']:>10}")

# %% [markdown]
# ## 10. What to monitor
#
# | Metric | Alert when |
# |---|---|
# | Request error rate | > 1% over 5 minutes |
# | p95 latency | > 2x the 7-day baseline |
# | Tokens per request | > 1.5x baseline (a loop is running away) |
# | Checkpoint write latency | > 200 ms (your database is struggling) |
# | Approval queue depth | Growing for more than an hour |
# | Threads paused > 24h | Any (your timeout policy is not working) |
# | `GraphRecursionError` count | Any in production |
# | Checkpoint table size | Growth without a retention job |
#
# The LangGraph-specific ones are the last four. Standard APM will not tell you
# that 40 customers are waiting for an approval nobody is looking at.

# %%
def operational_snapshot(graph, thread_ids: list[str]) -> dict:
    """The numbers to export to your metrics system."""
    now = dt.datetime.now(dt.timezone.utc)
    stale_cutoff = now - dt.timedelta(hours=24)

    paused = stale = total_messages = 0
    for thread_id in thread_ids:
        snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
        if not snapshot.values:
            continue
        total_messages += len(snapshot.values.get("messages", []))
        if any(task.interrupts for task in snapshot.tasks):
            paused += 1
            if snapshot.created_at and dt.datetime.fromisoformat(snapshot.created_at) < stale_cutoff:
                stale += 1

    return {"threads": len(thread_ids), "awaiting_approval": paused,
            "stale_over_24h": stale, "total_messages": total_messages}


print(operational_snapshot(service_graph, ["api-1", "api-2", "d-1", "d-2"]))

# %% [markdown]
# ## Try it yourself
#
# 1. **Run the server for real.** Write the files to a proper `app/` directory,
#    `uvicorn app.server:app --reload`, and drive it with `curl`.
# 2. **Break a deploy on purpose.** Pause a thread at an interrupt, rename the
#    node, restart, and try to resume. Read the error.
# 3. **Write the migration.** Add a required state key, write `migrate_state` for
#    it, and prove an old thread resumes correctly.
# 4. **Load test.** Run the SQLite configuration with 4 uvicorn workers and 20
#    concurrent requests, observe the lock errors, then switch to Postgres.

# %% [markdown]
# ## Recap
#
# | Concept | Takeaway |
# |---|---|
# | Project layout | Separate state, graphs, dependencies and server |
# | `lifespan` | Compile the graph once at startup, never per request |
# | `lru_cache` singletons | One checkpointer per process, not per request |
# | Graph versioning | Stamp the version in state; detect old threads on resume |
# | Breaking changes | Renaming or removing a node breaks paused threads |
# | `migrate_state` | Explicit migrations, like a database schema |
# | `/health` vs `/ready` | Liveness and readiness are different questions |
# | Interrupts over HTTP | `awaiting_approval` + a `/chat/resume` endpoint |
# | SSE | `X-Accel-Buffering: no` or your proxy buffers the stream |
# | Workers vs checkpointer | SQLite needs one process; Postgres for replicas |
# | `langgraph dev` / `build` / `up` | Local Studio, Docker image, full platform runtime |
# | Drain before breaking deploys | Paused threads are user work in progress |
# | Monitor | Approval queue depth and stale threads, not just latency |
#
# ### LangServe (historical note)
#
# Older courses (including the LLG `LCEL.zip` material) deploy chains with
# LangServe's `add_routes(app, chain, path="/...")`. That path still works for
# simple LCEL chains, but for graphs the supported surface is what this notebook
# builds: a FastAPI app calling `graph.ainvoke` / `astream`, or LangGraph
# Platform via `langgraph.json`. Prefer those for anything with checkpoints,
# interrupts, or streaming modes.
#
# ## Next
#
# -> [47_token_limits_and_summarization_nodes.ipynb](47_token_limits_and_summarization_nodes.ipynb)
