# %% [markdown]
# # 53 - Capstone: FastAPI + LangGraph Service
#
# | | |
# |---|---|
# | **Level** | Capstone |
# | **Time** | 110 minutes |
# | **Prerequisites** | `46_deployment_and_versioning`, `51_policy_rag_chatbot` |
# | **Checklist ID** | `53_fastapi_langgraph_service` |
#
# ## Why this matters
#
# Everything so far ran in a notebook, single-user, one request at a time. A
# service is different in ways that break naive code: concurrent requests,
# multiple workers sharing state, streaming to a browser that may disconnect,
# authentication deciding which thread a caller may touch, and a health check
# that has to tell the truth.
#
# This capstone turns the policy assistant into a deployable HTTP service. We
# build the app **in this notebook**, test it in-process with `httpx` against
# the ASGI app - no separate server, no port conflicts - then write it out as
# a real project you can run with `uvicorn`.

# %%
from pathlib import Path
import sys

_root = Path.cwd()
while not (_root / "shared").is_dir() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from shared.notebook_setup import setup, require  # noqa: E402

ctx = setup("53_fastapi_langgraph_service")

# %%
import asyncio
import json
import operator
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, AsyncIterator, Literal, TypedDict

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from shared.llm import get_chat_model, get_embeddings

# %% [markdown]
# ## 1. The graph, extracted from the notebook
#
# The first real change: the graph becomes a **factory function** with no
# module-level side effects. Building an index at import time means every
# uvicorn worker rebuilds it, and a failure during import gives you an
# unreadable stack trace instead of a health check that reports "degraded".

# %%
GRAPH_VERSION = "1.2.0"


class ServiceState(TypedDict):
    messages: Annotated[list, add_messages]
    question: str
    documents: list[Document]
    answer: str
    citations: list[str]
    escalated: bool
    trace: Annotated[list[str], operator.add]


def build_retriever():
    """Build the policy index. Called once at startup, not at import."""
    documents: list[Document] = []
    for filename in ("company_handbook.md", "product_faq.md"):
        text = ctx.data(filename).read_text(encoding="utf-8")
        for section in MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "title"), ("##", "section")], strip_headers=False
        ).split_text(text):
            section.metadata["source"] = filename
            documents.append(section)
    policy = ctx.data("leave_policy.txt").read_text(encoding="utf-8")
    documents.append(Document(policy, metadata={"source": "leave_policy.txt"}))

    chunks = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100) \
        .split_documents(documents)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"{chunk.metadata['source']}#{index}"

    return FAISS.from_documents(chunks, get_embeddings()).as_retriever(search_kwargs={"k": 4})


def build_graph(retriever, model, checkpointer):
    """Compile the service graph. Dependencies are injected, never imported globally."""
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are Northwind's policy assistant. Answer only from the context and cite the "
         "chunk id in brackets. If the context does not answer the question, say so and "
         "point the employee at hr@northwind.example. Be concise.\n\nContext:\n{context}"),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ])
    contextualiser = ChatPromptTemplate.from_messages([
        ("system", "Rewrite the user's message as a standalone question using the conversation. "
                   "Return only the question."),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]) | model | StrOutputParser()

    def contextualise(state: ServiceState) -> dict:
        history = state["messages"][:-1][-6:]
        latest = state["messages"][-1].content
        question = contextualiser.invoke({"history": history, "question": latest}) if history else latest
        return {"question": question.strip(), "trace": ["contextualise"]}

    def retrieve(state: ServiceState) -> dict:
        found = retriever.invoke(state["question"])
        return {"documents": found, "trace": [f"retrieve:{len(found)}"]}

    def generate(state: ServiceState) -> dict:
        context = "\n\n".join(f"[{d.metadata['chunk_id']}] {d.page_content}"
                              for d in state["documents"])
        reply = (answer_prompt | model).invoke({
            "context": context, "history": state["messages"][:-1][-6:],
            "question": state["question"],
        })
        cited = [d.metadata["chunk_id"] for d in state["documents"]
                 if d.metadata["chunk_id"] in reply.content]
        escalated = "hr@northwind.example" in reply.content
        return {"messages": [reply], "answer": reply.content, "citations": cited,
                "escalated": escalated, "trace": ["generate"]}

    builder = StateGraph(ServiceState)
    builder.add_sequence([("contextualise", contextualise), ("retrieve", retrieve),
                          ("generate", generate)])
    builder.add_edge(START, "contextualise")
    builder.add_edge("generate", END)
    return builder.compile(checkpointer=checkpointer)


print(f"graph factory ready (version {GRAPH_VERSION})")

# %% [markdown]
# ## 2. Request and response contracts
#
# Pydantic models are your API contract. Validate at the edge and the rest of
# the service can assume its inputs are sane.

# %%
class ChatRequest(BaseModel):
    """A question from a client."""

    message: str = Field(min_length=1, max_length=2000)
    conversation_id: str | None = Field(
        default=None,
        description="Omit to start a new conversation; pass the returned id to continue one.",
    )


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    citations: list[str]
    escalated: bool
    latency_ms: int
    graph_version: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "starting"]
    graph_version: str
    checks: dict[str, bool]
    uptime_seconds: int


class ConversationResponse(BaseModel):
    conversation_id: str
    turns: list[dict]
    message_count: int


print("contracts:", [m.__name__ for m in (ChatRequest, ChatResponse, HealthResponse,
                                          ConversationResponse)])

# %% [markdown]
# **Note `conversation_id` is optional, not a `thread_id`.** The client never
# names a thread directly - if it could, one user could read another's
# conversation by guessing an id. We derive the thread from the authenticated
# caller plus the conversation, in section 4.

# %% [markdown]
# ## 3. Lifespan - build once, share across requests
#
# `lifespan` runs on startup and shutdown. Build the index and the checkpointer
# here, hang them on `app.state`, and every request reuses them.

# %%
STARTED_AT = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build expensive resources once per process."""
    app.state.ready = False
    app.state.errors = {}

    try:
        app.state.model = get_chat_model()
        app.state.retriever = build_retriever()
        # Production: PostgresSaver, so every replica shares threads.
        app.state.checkpointer = InMemorySaver()
        app.state.graph = build_graph(app.state.retriever, app.state.model, app.state.checkpointer)
        app.state.ready = True
    except Exception as exc:                     # noqa: BLE001 - startup must not crash silently
        app.state.errors["startup"] = f"{type(exc).__name__}: {exc}"

    yield                                        # <- the app serves requests here

    # Shutdown: close pools, flush traces.
    app.state.ready = False


print("lifespan defined: index and checkpointer built once, not per request")

# %% [markdown]
# ## 4. Authentication and thread ownership
#
# This is the security-critical part of the whole service. A conversation id
# from the client is **untrusted input**. The thread key must include the
# authenticated user, so a caller physically cannot address someone else's
# thread.

# %%
API_KEYS = {                       # a real service looks this up in a database
    "demo-key-alice": "user-alice",
    "demo-key-bob": "user-bob",
}


async def current_user(x_api_key: Annotated[str | None, Header()] = None) -> str:
    if not x_api_key or x_api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")
    return API_KEYS[x_api_key]


def thread_id_for(user_id: str, conversation_id: str) -> str:
    """The user id is part of the key, so cross-user access is impossible by construction."""
    return f"{user_id}::{conversation_id}"


print(thread_id_for("user-alice", "conv-1"))
print(thread_id_for("user-bob", "conv-1"), "<- same conversation id, different thread")

# %% [markdown]
# ## 5. The endpoints

# %%
app = FastAPI(title="Northwind Policy Assistant", version=GRAPH_VERSION, lifespan=lifespan)


def initial_state(message: str) -> dict:
    return {"messages": [HumanMessage(message)], "question": "", "documents": [],
            "answer": "", "citations": [], "escalated": False, "trace": []}


@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Liveness and readiness in one. Must never call the model."""
    state = request.app.state
    checks = {
        "graph_compiled": getattr(state, "graph", None) is not None,
        "retriever_ready": getattr(state, "retriever", None) is not None,
        "checkpointer_ready": getattr(state, "checkpointer", None) is not None,
        "model_configured": getattr(state, "model", None) is not None,
    }
    status = "ok" if all(checks.values()) else ("starting" if not state.errors else "degraded")
    return HealthResponse(status=status, graph_version=GRAPH_VERSION, checks=checks,
                          uptime_seconds=int(time.time() - STARTED_AT))


@app.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request,
               user_id: Annotated[str, Depends(current_user)]) -> ChatResponse:
    """Ask a question. Returns the whole answer once it is ready."""
    if not request.app.state.ready:
        raise HTTPException(status_code=503, detail="Service is starting")

    conversation_id = payload.conversation_id or f"conv-{uuid.uuid4().hex[:12]}"
    config = {
        "configurable": {"thread_id": thread_id_for(user_id, conversation_id)},
        "run_name": "policy-chat",
        "tags": [f"user:{user_id}", f"version:{GRAPH_VERSION}"],
        "metadata": {"conversation_id": conversation_id},
    }

    started = time.perf_counter()
    try:
        # ainvoke, not invoke: a sync call would block the event loop for every other request.
        result = await request.app.state.graph.ainvoke(initial_state(payload.message), config)
    except Exception as exc:                     # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Model call failed: {type(exc).__name__}") from exc

    return ChatResponse(
        conversation_id=conversation_id,
        answer=result["answer"],
        citations=result["citations"],
        escalated=result["escalated"],
        latency_ms=int((time.perf_counter() - started) * 1000),
        graph_version=GRAPH_VERSION,
    )


@app.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(conversation_id: str, request: Request,
                           user_id: Annotated[str, Depends(current_user)]) -> ConversationResponse:
    """Read a conversation back. Only ever your own - the thread key includes your user id."""
    config = {"configurable": {"thread_id": thread_id_for(user_id, conversation_id)}}
    snapshot = await request.app.state.graph.aget_state(config)
    if not snapshot.values:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = snapshot.values.get("messages", [])
    return ConversationResponse(
        conversation_id=conversation_id,
        turns=[{"role": m.type, "content": m.content} for m in messages],
        message_count=len(messages),
    )


@app.delete("/conversations/{conversation_id}", status_code=204)
async def delete_conversation(conversation_id: str, request: Request,
                              user_id: Annotated[str, Depends(current_user)]) -> None:
    """Right to erasure. Deleting a thread must actually delete its checkpoints."""
    request.app.state.checkpointer.delete_thread(thread_id_for(user_id, conversation_id))


print("routes:", sorted(r.path for r in app.routes if hasattr(r, "path")))

# %% [markdown]
# ## 6. Streaming with Server-Sent Events
#
# A four-second wait with no output feels broken. SSE is the right transport
# here: one-directional, plain HTTP, reconnects for free, and every browser
# supports it via `EventSource`.
#
# Three things people get wrong:
#
# - **Every event needs `data: ` and a blank line.** Miss the blank line and
#   the browser buffers forever.
# - **Send a terminal event.** The client cannot tell "finished" from "died".
# - **Handle disconnects.** A user who closes the tab should not keep billing
#   you.

# %%
@app.post("/chat/stream")
async def chat_stream(payload: ChatRequest, request: Request,
                      user_id: Annotated[str, Depends(current_user)]) -> StreamingResponse:
    """Stream status events and answer tokens as Server-Sent Events."""
    if not request.app.state.ready:
        raise HTTPException(status_code=503, detail="Service is starting")

    conversation_id = payload.conversation_id or f"conv-{uuid.uuid4().hex[:12]}"
    config = {"configurable": {"thread_id": thread_id_for(user_id, conversation_id)},
              "run_name": "policy-chat-stream",
              "tags": [f"user:{user_id}"]}
    graph = request.app.state.graph

    STATUS = {"contextualise": "Understanding your question",
              "retrieve": "Searching policy documents",
              "generate": "Writing the answer"}

    async def event_stream() -> AsyncIterator[str]:
        def sse(event: str, data: dict) -> str:
            return f"event: {event}\ndata: {json.dumps(data)}\n\n"

        yield sse("start", {"conversation_id": conversation_id})
        try:
            async for mode, chunk in graph.astream(
                initial_state(payload.message), config, stream_mode=["updates", "messages"]
            ):
                if await request.is_disconnected():
                    break                       # client left - stop paying for tokens

                if mode == "updates":
                    for node in chunk:
                        if node in STATUS:
                            yield sse("status", {"step": node, "label": STATUS[node]})
                elif mode == "messages":
                    message, metadata = chunk
                    if metadata.get("langgraph_node") == "generate" and message.content:
                        yield sse("token", {"text": message.content})

            snapshot = await graph.aget_state(config)
            yield sse("done", {"conversation_id": conversation_id,
                               "citations": snapshot.values.get("citations", []),
                               "escalated": snapshot.values.get("escalated", False)})
        except Exception as exc:                 # noqa: BLE001
            yield sse("error", {"detail": f"{type(exc).__name__}"})

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})   # disable nginx buffering


print("streaming endpoint registered: POST /chat/stream")

# %% [markdown]
# ## 7. Testing the service in-process
#
# `httpx.ASGITransport` calls the app directly - no port, no server process, no
# flakiness. This is exactly how you should write the service's test suite.

# %%
@asynccontextmanager
async def test_client():
    """An httpx client wired straight into the ASGI app, with lifespan run."""
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


ALICE = {"X-API-Key": "demo-key-alice"}
BOB = {"X-API-Key": "demo-key-bob"}


async def smoke() -> dict:
    async with test_client() as client:
        results = {}

        health_response = await client.get("/health")
        results["health"] = health_response.json()

        unauthorised = await client.post("/chat", json={"message": "hi"})
        results["no_key_status"] = unauthorised.status_code

        bad_request = await client.post("/chat", json={"message": ""}, headers=ALICE)
        results["empty_message_status"] = bad_request.status_code

        first = await client.post(
            "/chat", json={"message": "How many days of annual leave do employees get?"},
            headers=ALICE,
        )
        results["first_turn"] = first.json()
        return results


smoke_results = asyncio.run(smoke())
print(json.dumps(smoke_results["health"], indent=2))
print(f"\nno API key      -> {smoke_results['no_key_status']} (expect 401)")
print(f"empty message   -> {smoke_results['empty_message_status']} (expect 422)")
answer = smoke_results["first_turn"]
print(f"\nconversation_id: {answer['conversation_id']}")
print(f"latency:         {answer['latency_ms']}ms")
print(f"citations:       {answer['citations']}")
print(f"\n{answer['answer'][:260]}")

# %% [markdown]
# ### Conversation continuity and isolation
#
# Two properties worth an explicit test, because both are easy to break and
# neither shows up in manual clicking.

# %%
async def continuity_and_isolation() -> dict:
    async with test_client() as client:
        first = (await client.post("/chat", json={"message": "How many sick leave days are there?"},
                                   headers=ALICE)).json()
        conversation = first["conversation_id"]

        follow_up = (await client.post(
            "/chat", json={"message": "And what about casual leave?", "conversation_id": conversation},
            headers=ALICE)).json()

        mine = await client.get(f"/conversations/{conversation}", headers=ALICE)
        theirs = await client.get(f"/conversations/{conversation}", headers=BOB)

        deleted = await client.delete(f"/conversations/{conversation}", headers=ALICE)
        after_delete = await client.get(f"/conversations/{conversation}", headers=ALICE)

        return {
            "first": first["answer"][:90],
            "follow_up": follow_up["answer"][:90],
            "same_conversation": follow_up["conversation_id"] == conversation,
            "owner_can_read": mine.status_code,
            "owner_turns": mine.json()["message_count"],
            "other_user_status": theirs.status_code,
            "delete_status": deleted.status_code,
            "after_delete_status": after_delete.status_code,
        }


outcome = asyncio.run(continuity_and_isolation())
print(f"turn 1: {outcome['first']}")
print(f"turn 2: {outcome['follow_up']}")
print(f"\nsame conversation reused:  {outcome['same_conversation']}")
print(f"owner reads history:       {outcome['owner_can_read']} ({outcome['owner_turns']} messages)")
print(f"other user reads it:       {outcome['other_user_status']} (expect 404 - not 403, "
      f"we do not confirm it exists)")
print(f"delete:                    {outcome['delete_status']} (expect 204)")
print(f"read after delete:         {outcome['after_delete_status']} (expect 404)")

# %% [markdown]
# **404, not 403.** Returning "forbidden" tells an attacker the conversation
# exists. Since the thread key embeds the user id, Bob's lookup simply finds
# nothing - the safe answer falls out of the design rather than needing a check.

# %% [markdown]
# ### The streaming endpoint

# %%
async def stream_demo() -> list[tuple[str, str]]:
    events: list[tuple[str, str]] = []
    async with test_client() as client:
        async with client.stream(
            "POST", "/chat/stream",
            json={"message": "What is the notice period during probation?"},
            headers=ALICE, timeout=60.0,
        ) as response:
            event_name = ""
            async for line in response.aiter_lines():
                if line.startswith("event: "):
                    event_name = line[7:]
                elif line.startswith("data: "):
                    events.append((event_name, line[6:]))
    return events


events = asyncio.run(stream_demo())
kinds = [name for name, _ in events]
print(f"{len(events)} events: {kinds.count('status')} status, {kinds.count('token')} token, "
      f"{kinds.count('done')} done")

for name, data in events:
    if name in ("start", "status", "done"):
        print(f"   {name:7} {data[:110]}")

tokens = "".join(json.loads(data)["text"] for name, data in events if name == "token")
print(f"\nreassembled answer ({len(tokens)} chars):\n{tokens[:260]}")

# %% [markdown]
# ## 8. Concurrency
#
# The reason every handler is `async` and every graph call is `ainvoke`: a
# synchronous call blocks the event loop, so ten concurrent users queue behind
# each other instead of overlapping.

# %%
async def concurrency_check(n: int = 5) -> dict:
    questions = [
        "How many days of annual leave do employees get?",
        "What is the notice period during probation?",
        "How many sick days are there?",
        "What is the API rate limit?",
        "How long does a refund take?",
    ][:n]

    async with test_client() as client:
        started = time.perf_counter()
        responses = await asyncio.gather(*[
            client.post("/chat", json={"message": q}, headers=ALICE, timeout=120.0)
            for q in questions
        ])
        wall_clock = time.perf_counter() - started

    latencies = [r.json()["latency_ms"] for r in responses]
    conversations = {r.json()["conversation_id"] for r in responses}
    return {
        "requests": n,
        "wall_clock_s": round(wall_clock, 1),
        "sum_of_latencies_s": round(sum(latencies) / 1000, 1),
        "slowest_single_s": round(max(latencies) / 1000, 1),
        "distinct_conversations": len(conversations),
        "all_succeeded": all(r.status_code == 200 for r in responses),
    }


concurrency = asyncio.run(concurrency_check())
print(json.dumps(concurrency, indent=2))
print(f"\nSpeedup vs sequential: "
      f"{concurrency['sum_of_latencies_s'] / max(concurrency['wall_clock_s'], 0.1):.1f}x")

# %% [markdown]
# Wall clock close to the slowest single request means the requests genuinely
# overlapped. If it were close to the *sum*, something synchronous is blocking
# the event loop - usually an `invoke` that should be `ainvoke`, or a CPU-bound
# embedding call that belongs in a thread pool.
#
# Note each request got its own `conversation_id`: requests are independent
# unless the client says otherwise.

# %% [markdown]
# ## 9. Production middleware
#
# Four things every service needs, that a notebook never does.

# %%
from collections import defaultdict, deque

import logging

logger = logging.getLogger("policy-service")

# 1. Request id + structured access log.
@app.middleware("http")
async def observability(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])
    started = time.perf_counter()
    response = await call_next(request)
    duration_ms = int((time.perf_counter() - started) * 1000)

    logger.info(json.dumps({
        "request_id": request_id,
        "method": request.method,
        "path": request.url.path,
        "status": response.status_code,
        "duration_ms": duration_ms,
        "graph_version": GRAPH_VERSION,
    }))
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Graph-Version"] = GRAPH_VERSION
    return response


# 2. Per-user rate limiting. One user must not be able to drain the budget.
_REQUEST_LOG: dict[str, deque] = defaultdict(deque)
RATE_LIMIT, RATE_WINDOW_S = 20, 60


def check_rate_limit(user_id: str) -> None:
    now = time.time()
    bucket = _REQUEST_LOG[user_id]
    while bucket and now - bucket[0] > RATE_WINDOW_S:
        bucket.popleft()
    if len(bucket) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Rate limit exceeded",
                            headers={"Retry-After": str(RATE_WINDOW_S)})
    bucket.append(now)


# 3. A cost ceiling per user per day, enforced before the call, not after the bill.
_DAILY_TOKENS: dict[str, int] = defaultdict(int)
DAILY_TOKEN_BUDGET = 200_000


def check_budget(user_id: str) -> None:
    if _DAILY_TOKENS[user_id] >= DAILY_TOKEN_BUDGET:
        raise HTTPException(status_code=429, detail="Daily token budget exhausted")


# 4. Graceful degradation rather than a 500 when the provider is down.
def fallback_answer() -> str:
    return ("Our assistant is temporarily unavailable. Policy documents are on the intranet, "
            "and HR can be reached at hr@northwind.example.")


for user, expected in [("user-alice", "allowed"), ("user-alice", "allowed")]:
    check_rate_limit(user)
print(f"rate limiter: {len(_REQUEST_LOG['user-alice'])}/{RATE_LIMIT} used in this window")
print(f"budget guard: {_DAILY_TOKENS['user-alice']:,}/{DAILY_TOKEN_BUDGET:,} tokens today")

# %% [markdown]
# ## 10. Writing out a real project
#
# Notebooks are for learning. Here is the same service as files you can run.

# %%
service_dir = ctx.artifact("policy_service")
service_dir.mkdir(parents=True, exist_ok=True)

(service_dir / "config.py").write_text('''"""Configuration from the environment. No secrets in code."""
from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    graph_version: str = "1.2.0"
    checkpoint_url: str = "sqlite:///./checkpoints.sqlite"
    rate_limit_per_minute: int = 20
    daily_token_budget: int = 200_000
    langsmith_project: str = "policy-assistant"
    environment: str = "development"

    class Config:
        env_file = ".env"
        env_prefix = "POLICY_"


@lru_cache
def get_settings() -> Settings:
    return Settings()
''', encoding="utf-8")

(service_dir / "graph.py").write_text('''"""The graph, with no import-time side effects."""
from langgraph.graph import StateGraph


def build_graph(retriever, model, checkpointer):
    """Import this from main.py and call it once inside the lifespan handler."""
    # The full implementation is in notebook 53, section 1.
    raise NotImplementedError("Copy build_graph from the notebook")
''', encoding="utf-8")

(service_dir / "requirements.txt").write_text(
    "fastapi>=0.115\nuvicorn[standard]>=0.32\nlanggraph>=1.2\nlangchain>=1.4\n"
    "langgraph-checkpoint-postgres>=2.0\npsycopg[binary,pool]>=3.2\n"
    "pydantic-settings>=2.6\nlangsmith>=0.3\n",
    encoding="utf-8",
)

(service_dir / "Dockerfile").write_text('''FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run as a non-root user.
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \\
  CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://localhost:8000/health').json()['status']=='ok' else 1)"

# One worker per container; scale with replicas so each has its own event loop.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
''', encoding="utf-8")

(service_dir / "docker-compose.yml").write_text('''services:
  api:
    build: .
    ports: ["8000:8000"]
    environment:
      POLICY_ENVIRONMENT: production
      POLICY_CHECKPOINT_URL: postgresql://postgres:postgres@db:5432/checkpoints
      GROQ_API_KEY: ${GROQ_API_KEY}
      LANGSMITH_API_KEY: ${LANGSMITH_API_KEY}
    depends_on:
      db: {condition: service_healthy}
    deploy:
      replicas: 3

  db:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: checkpoints
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      retries: 10
    volumes: ["pgdata:/var/lib/postgresql/data"]

volumes:
  pgdata:
''', encoding="utf-8")

(service_dir / "test_api.py").write_text('''"""Tests that run the ASGI app in-process - no server, no ports, no flakes."""
import httpx
import pytest

from main import app

ALICE = {"X-API-Key": "demo-key-alice"}
BOB = {"X-API-Key": "demo-key-bob"}


@pytest.fixture
async def client():
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


async def test_health_reports_ok(client):
    body = (await client.get("/health")).json()
    assert body["status"] == "ok"
    assert all(body["checks"].values())


async def test_requires_api_key(client):
    assert (await client.post("/chat", json={"message": "hi"})).status_code == 401


async def test_rejects_empty_message(client):
    assert (await client.post("/chat", json={"message": ""}, headers=ALICE)).status_code == 422


async def test_conversation_continues(client):
    first = (await client.post("/chat", json={"message": "How much annual leave?"},
                               headers=ALICE)).json()
    second = (await client.post("/chat", json={"message": "And sick leave?",
                                               "conversation_id": first["conversation_id"]},
                                headers=ALICE)).json()
    assert second["conversation_id"] == first["conversation_id"]


async def test_users_cannot_read_each_others_threads(client):
    mine = (await client.post("/chat", json={"message": "How much annual leave?"},
                              headers=ALICE)).json()
    theirs = await client.get(f"/conversations/{mine['conversation_id']}", headers=BOB)
    assert theirs.status_code == 404


async def test_stream_terminates_with_done(client):
    async with client.stream("POST", "/chat/stream", json={"message": "How much annual leave?"},
                             headers=ALICE, timeout=60.0) as response:
        events = [line[7:] async for line in response.aiter_lines() if line.startswith("event: ")]
    assert events[0] == "start"
    assert events[-1] == "done"
''', encoding="utf-8")

(service_dir / "langgraph.json").write_text(json.dumps({
    "dependencies": ["."],
    "graphs": {"policy_assistant": "./graph.py:build_graph"},
    "env": ".env",
}, indent=2), encoding="utf-8")

(service_dir / "README.md").write_text('''# Northwind Policy Assistant

## Run locally

```bash
python -m pip install -r requirements.txt
cp ../../.env.example .env        # add your GROQ_API_KEY
uvicorn main:app --reload
```

Open http://localhost:8000/docs for the generated API reference.

## Test

```bash
pytest test_api.py
```

## Deploy

```bash
docker compose up --build
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Readiness; never calls the model |
| POST | `/chat` | Ask a question, get the whole answer |
| POST | `/chat/stream` | Same, as Server-Sent Events |
| GET | `/conversations/{id}` | Read your own history |
| DELETE | `/conversations/{id}` | Erase a conversation |

Authenticate with `X-API-Key`. Thread keys embed the caller's user id, so one
user cannot address another's conversation.
''', encoding="utf-8")

for path in sorted(service_dir.iterdir()):
    print(f"   {path.name:22} {path.stat().st_size:>6} bytes")

# %% [markdown]
# `main.py` is the one file you assemble yourself - copy sections 1 to 9 of
# this notebook into it. Everything else is written above.

# %% [markdown]
# ## 11. Going to production
#
# The checklist, in the order these things bite.

# %%
CHECKLIST = {
    "Swap InMemorySaver for PostgresSaver": "In-memory state dies on deploy and is not shared "
                                            "between replicas. This is the first thing to break.",
    "One uvicorn worker per container": "Multiple workers mean multiple event loops and "
                                        "multiple in-process indexes. Scale with replicas.",
    "Move the vector index out of process": "FAISS in memory is rebuilt per replica and cannot "
                                            "be updated without a deploy.",
    "Set a request timeout": "A hung provider call holds a connection forever. "
                             "Use asyncio.timeout around the graph call.",
    "Enable LangSmith with the version tag": "You cannot debug a production answer you "
                                             "cannot see. Tag traces with graph_version.",
    "Alert on escalation rate": "A sudden rise means retrieval broke, not that users changed.",
    "Cap tokens per user per day": "The first abusive user arrives sooner than you expect.",
    "Log every answer with its chunk ids": "The only way to answer 'why did it say that?'",
    "Handle thread migration": "Notebook 46: old threads must still load after a state "
                               "schema change, or every conversation breaks on deploy.",
    "Load test streaming specifically": "SSE holds a connection open per user. "
                                        "Concurrency limits bite far earlier than with /chat.",
}

for index, (item, why) in enumerate(CHECKLIST.items(), 1):
    print(f"{index:2}. {item}\n    {why}\n")

# %% [markdown]
# ## Try it yourself
#
# 1. **Wire in the rate limiter.** Add `Depends` wrappers calling
#    `check_rate_limit` and `check_budget`, then prove a 21st request gets 429.
# 2. **Add a timeout.** Wrap the graph call in `asyncio.timeout(30)` and return
#    `fallback_answer()` instead of a 500.
# 3. **Add feedback.** `POST /conversations/{id}/feedback` storing a rating
#    against the chunk ids, then find your worst-performing chunks.
# 4. **Swap the checkpointer.** Use `SqliteSaver` in the lifespan and confirm a
#    conversation survives a restart of the test client.
# 5. **Break isolation deliberately.** Change `thread_id_for` to ignore
#    `user_id` and watch `test_users_cannot_read_each_others_threads` fail. That
#    test is the one protecting your users.
# 6. **Add a second graph version.** Serve `v1` and `v2` behind a header and
#    compare escalation rates between them.
# 7. **Optional Streamlit client.** Point a Streamlit chat UI at `/chat/stream`
#    (SSE). The LLG `BAsicChatbot` / `AINEWSAgentic` zips and Kris Naik's
#    `streamlit_app.py` are reference front ends; FastAPI stays the source of
#    truth for auth and isolation.
#
# ### Alternative project layouts
#
# Two layouts show up constantly in the courses we crosswalked. Both are fine;
# pick one and stay consistent:
#
# ```
# # Eden Marco agentic-RAG style
# graph/
#   chains/          # graders, routers, generation
#   nodes/           # retrieve, grade, generate, web_search
#   state.py
#   consts.py
#   graph.py
#
# # Kris Naik / LLG Streamlit style
# src/<package>/
#   graph/ nodes/ state/ tools/ LLMS/ ui/
# ```
#
# Our notebook keeps everything inline for teaching. Capstone 53's written-out
# `artifacts/policy_service/` is the middle ground: flat files, FastAPI entrypoint.

# %% [markdown]
# ## Recap
#
# | Decision | Why |
# |---|---|
# | Graph factory, no import-time work | Every worker would otherwise rebuild the index |
# | `lifespan` | Build expensive resources once per process |
# | Pydantic contracts | Validate at the edge; 422 instead of a 500 later |
# | Thread key embeds the user id | Cross-user access becomes impossible, not merely checked |
# | 404, not 403 | Do not confirm that someone else's conversation exists |
# | `ainvoke` / `astream` everywhere | A sync call serialises every concurrent user |
# | SSE with a terminal `done` event | The client can tell "finished" from "died" |
# | `request.is_disconnected()` | Stop paying for tokens nobody will read |
# | `/health` never calls the model | A health check that costs money is not a health check |
# | `httpx.ASGITransport` | Real end-to-end tests with no server and no flakes |
# | Rate limit and token budget | The first abusive user arrives sooner than you expect |
# | Version tag on every trace | You cannot debug what you cannot attribute |
#
# ## Next
#
# You have finished the curriculum. Go back to
# [CHECKLIST.md](../CHECKLIST.md) and mark off what you have verified - and
# revisit `references/official-sources.md` whenever the libraries move.
