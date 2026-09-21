"""Runtime smoke test for the capstone patterns (51-53) that do not need a real model.

Uses a deterministic fake chat model and fake embeddings so the FastAPI
service, thread isolation, SSE stream and concurrency behaviour are all
exercised end to end without an API key.
"""
import asyncio
import json
import operator
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, AsyncIterator, Literal, TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PASS, FAIL = 0, 0


def check(label, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ok   {label}" + (f"  ({detail})" if detail else ""))
    else:
        FAIL += 1
        print(f"  FAIL {label}  {detail}")


print("== 51: ingestion metadata and citations ==")
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from shared.notebook_setup import setup

ctx = setup("smoke_capstones", quiet=True)

documents: list[Document] = []
for filename in ("company_handbook.md", "product_faq.md"):
    text = ctx.data(filename).read_text(encoding="utf-8")
    for section in MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "title"), ("##", "section")], strip_headers=False
    ).split_text(text):
        section.metadata["source"] = filename
        documents.append(section)
documents.append(Document(ctx.data("leave_policy.txt").read_text(encoding="utf-8"),
                          metadata={"source": "leave_policy.txt"}))

chunks = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100).split_documents(documents)
for index, chunk in enumerate(chunks):
    section = chunk.metadata.get("section") or chunk.metadata.get("title") or ""
    chunk.metadata["chunk_id"] = f"{chunk.metadata['source']}#{index}"
    chunk.metadata["citation"] = chunk.metadata["source"] + (f" - {section}" if section else "")

check("corpus produced chunks", len(chunks) > 10, f"{len(chunks)} chunks")
check("every chunk has a citation", all(c.metadata.get("citation") for c in chunks))
check("chunk ids are unique", len({c.metadata["chunk_id"] for c in chunks}) == len(chunks))
check("heading metadata survived splitting",
      any(c.metadata.get("section") for c in chunks))

print("\n== 51: hybrid retrieval ==")
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS

from shared.llm import get_embeddings

embeddings = get_embeddings()
vector_store = FAISS.from_documents(chunks, embeddings)
semantic = vector_store.as_retriever(search_type="mmr",
                                     search_kwargs={"k": 4, "fetch_k": 12, "lambda_mult": 0.6})
keyword = BM25Retriever.from_documents(chunks)
keyword.k = 4
hybrid = EnsembleRetriever(retrievers=[semantic, keyword], weights=[0.6, 0.4])

hits = hybrid.invoke("annual leave")
check("hybrid retriever returns documents", len(hits) > 0, f"{len(hits)} hits")
check("mmr returns distinct chunks",
      len({h.metadata['chunk_id'] for h in semantic.invoke('leave')}) ==
      len(semantic.invoke('leave')))
check("bm25 finds an exact term",
      any("leave" in h.page_content.lower() for h in keyword.invoke("casual leave")))

print("\n== 52: supervisor budgets and Command routing ==")
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command


class Desk(TypedDict):
    findings: Annotated[list, operator.add]
    draft: str
    critiques: Annotated[list, operator.add]
    revisions: Annotated[int, operator.add]
    researcher_calls: Annotated[int, operator.add]
    published: bool
    trace: Annotated[list, operator.add]


MAX_REVISIONS, MAX_RESEARCH_CALLS = 2, 6
PROPOSAL = {"n": 0}


def fake_assignment(state: Desk) -> str:
    """Stand-in for the LLM supervisor: the one ambiguous call it still makes."""
    PROPOSAL["n"] += 1
    return "researcher"        # always greedy for more research - the budget must stop it


def supervisor(state: Desk) -> Command[Literal["researcher", "writer", "reviewer", "publish"]]:
    if state["revisions"] >= MAX_REVISIONS and state["draft"]:
        return Command(goto="publish", update={"trace": ["budget spent"]})
    latest = state["critiques"][-1] if state["critiques"] else None
    if latest and latest["verdict"] == "approve":
        return Command(goto="publish", update={"trace": ["approved"]})
    if state["draft"]:
        if not state["critiques"]:
            return Command(goto="reviewer", update={"trace": ["first review"]})
        if state["revisions"] < len(state["critiques"]):
            return Command(goto="writer", update={"trace": ["revise"]})
        return Command(goto="reviewer", update={"trace": ["re-review"]})

    destination = fake_assignment(state)
    if destination == "researcher" and state["researcher_calls"] >= MAX_RESEARCH_CALLS:
        destination = "writer"
        return Command(goto=destination, update={"trace": ["research budget spent"]})
    return Command(goto=destination, update={"trace": [destination]})


b = StateGraph(Desk)
b.add_node("supervisor", supervisor)
b.add_node("researcher", lambda s: {"findings": [{"claim": "x"}], "researcher_calls": 3,
                                    "trace": ["researched"]})
b.add_node("writer", lambda s: {"draft": "draft text", "revisions": 1 if s["critiques"] else 0,
                                "trace": ["wrote"]})
b.add_node("reviewer", lambda s: {"critiques": [{"verdict": "revise", "required_changes": ["add numbers"]}],
                                  "trace": ["reviewed"]})
b.add_node("publish", lambda s: {"published": True, "trace": ["published"]})
b.add_edge(START, "supervisor")
b.add_edge("researcher", "supervisor")
b.add_edge("writer", "supervisor")
b.add_edge("reviewer", "supervisor")
b.add_edge("publish", END)
desk = b.compile()

drawing = desk.get_graph().draw_ascii()
check("Command[Literal] exposes routes in the drawing",
      all(node in drawing for node in ("researcher", "writer", "reviewer", "publish")))

out = desk.invoke({"findings": [], "draft": "", "critiques": [], "revisions": 0,
                   "researcher_calls": 0, "published": False, "trace": []},
                  {"recursion_limit": 40})
check("desk terminates under budget caps", out["published"] is True, str(out["trace"]))
check("revision cap respected", out["revisions"] <= MAX_REVISIONS, str(out["revisions"]))
check("research budget stopped further research",
      "research budget spent" in out["trace"] or out["researcher_calls"] <= MAX_RESEARCH_CALLS + 3,
      f"calls={out['researcher_calls']}")

print("\n== 52: reviewer consistency coercion ==")


class FakeCritique:
    def __init__(self, verdict, changes):
        self.verdict = verdict
        self.required_changes = changes


def coerce(critique):
    if critique.verdict == "approve" and critique.required_changes:
        critique.verdict = "revise"
    return critique


check("approve-with-changes becomes revise",
      coerce(FakeCritique("approve", ["add numbers"])).verdict == "revise")
check("clean approve is left alone", coerce(FakeCritique("approve", [])).verdict == "approve")

print("\n== 53: FastAPI service ==")
import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages

GRAPH_VERSION = "1.2.0-test"
STARTED_AT = time.time()


class FakeModel:
    """Deterministic stand-in so the service is testable without an API key."""

    def invoke(self, messages, **kwargs):
        return AIMessage("Employees get 24 days of annual leave [leave_policy.txt#0].")

    async def ainvoke(self, messages, **kwargs):
        return self.invoke(messages)


class ServiceState(TypedDict):
    messages: Annotated[list, add_messages]
    question: str
    documents: list
    answer: str
    citations: list
    escalated: bool
    trace: Annotated[list, operator.add]


def build_graph(retriever, model, checkpointer):
    def contextualise(state: ServiceState) -> dict:
        return {"question": state["messages"][-1].content, "trace": ["contextualise"]}

    def retrieve(state: ServiceState) -> dict:
        found = retriever.invoke(state["question"])
        return {"documents": found, "trace": [f"retrieve:{len(found)}"]}

    def generate(state: ServiceState) -> dict:
        reply = model.invoke(state["messages"])
        cited = [d.metadata["chunk_id"] for d in state["documents"]
                 if d.metadata["chunk_id"] in reply.content]
        return {"messages": [reply], "answer": reply.content, "citations": cited,
                "escalated": "hr@northwind.example" in reply.content, "trace": ["generate"]}

    builder = StateGraph(ServiceState)
    builder.add_sequence([("contextualise", contextualise), ("retrieve", retrieve),
                          ("generate", generate)])
    builder.add_edge(START, "contextualise")
    builder.add_edge("generate", END)
    return builder.compile(checkpointer=checkpointer)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ready = False
    app.state.errors = {}
    try:
        app.state.model = FakeModel()
        app.state.retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        app.state.checkpointer = InMemorySaver()
        app.state.graph = build_graph(app.state.retriever, app.state.model, app.state.checkpointer)
        app.state.ready = True
    except Exception as exc:  # noqa: BLE001
        app.state.errors["startup"] = str(exc)
    yield
    app.state.ready = False


from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    conversation_id: str | None = None


API_KEYS = {"demo-key-alice": "user-alice", "demo-key-bob": "user-bob"}


async def current_user(x_api_key: Annotated[str | None, Header()] = None) -> str:
    if not x_api_key or x_api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")
    return API_KEYS[x_api_key]


def thread_id_for(user_id: str, conversation_id: str) -> str:
    return f"{user_id}::{conversation_id}"


app = FastAPI(lifespan=lifespan)


def initial_state(message: str) -> dict:
    return {"messages": [HumanMessage(message)], "question": "", "documents": [],
            "answer": "", "citations": [], "escalated": False, "trace": []}


@app.get("/health")
async def health(request: Request):
    state = request.app.state
    checks = {"graph_compiled": getattr(state, "graph", None) is not None,
              "retriever_ready": getattr(state, "retriever", None) is not None,
              "checkpointer_ready": getattr(state, "checkpointer", None) is not None,
              "model_configured": getattr(state, "model", None) is not None}
    status = "ok" if all(checks.values()) else ("starting" if not state.errors else "degraded")
    return {"status": status, "graph_version": GRAPH_VERSION, "checks": checks,
            "uptime_seconds": int(time.time() - STARTED_AT)}


@app.post("/chat")
async def chat(payload: ChatRequest, request: Request,
               user_id: Annotated[str, Depends(current_user)]):
    if not request.app.state.ready:
        raise HTTPException(status_code=503, detail="Service is starting")
    conversation_id = payload.conversation_id or f"conv-{uuid.uuid4().hex[:12]}"
    config = {"configurable": {"thread_id": thread_id_for(user_id, conversation_id)}}
    started = time.perf_counter()
    result = await request.app.state.graph.ainvoke(initial_state(payload.message), config)
    return {"conversation_id": conversation_id, "answer": result["answer"],
            "citations": result["citations"], "escalated": result["escalated"],
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "graph_version": GRAPH_VERSION}


@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str, request: Request,
                           user_id: Annotated[str, Depends(current_user)]):
    config = {"configurable": {"thread_id": thread_id_for(user_id, conversation_id)}}
    snapshot = await request.app.state.graph.aget_state(config)
    if not snapshot.values:
        raise HTTPException(status_code=404, detail="Conversation not found")
    messages = snapshot.values.get("messages", [])
    return {"conversation_id": conversation_id, "message_count": len(messages),
            "turns": [{"role": m.type, "content": m.content} for m in messages]}


@app.delete("/conversations/{conversation_id}", status_code=204)
async def delete_conversation(conversation_id: str, request: Request,
                              user_id: Annotated[str, Depends(current_user)]):
    request.app.state.checkpointer.delete_thread(thread_id_for(user_id, conversation_id))


@app.post("/chat/stream")
async def chat_stream(payload: ChatRequest, request: Request,
                      user_id: Annotated[str, Depends(current_user)]):
    conversation_id = payload.conversation_id or f"conv-{uuid.uuid4().hex[:12]}"
    config = {"configurable": {"thread_id": thread_id_for(user_id, conversation_id)}}
    graph = request.app.state.graph
    STATUS = {"contextualise": "Understanding", "retrieve": "Searching", "generate": "Writing"}

    async def event_stream() -> AsyncIterator[str]:
        def sse(event, data):
            return f"event: {event}\ndata: {json.dumps(data)}\n\n"

        yield sse("start", {"conversation_id": conversation_id})
        try:
            async for chunk in graph.astream(initial_state(payload.message), config,
                                             stream_mode="updates"):
                if await request.is_disconnected():
                    break
                for node in chunk:
                    if node in STATUS:
                        yield sse("status", {"step": node, "label": STATUS[node]})
            snapshot = await graph.aget_state(config)
            yield sse("done", {"citations": snapshot.values.get("citations", []),
                               "escalated": snapshot.values.get("escalated", False)})
        except Exception as exc:  # noqa: BLE001
            yield sse("error", {"detail": type(exc).__name__})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@asynccontextmanager
async def test_client():
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


ALICE = {"X-API-Key": "demo-key-alice"}
BOB = {"X-API-Key": "demo-key-bob"}


async def run_service_tests():
    async with test_client() as client:
        body = (await client.get("/health")).json()
        check("health reports ok", body["status"] == "ok", json.dumps(body["checks"]))

        check("missing api key is 401",
              (await client.post("/chat", json={"message": "hi"})).status_code == 401)
        check("empty message is 422",
              (await client.post("/chat", json={"message": ""}, headers=ALICE)).status_code == 422)
        check("oversized message is 422",
              (await client.post("/chat", json={"message": "x" * 2100},
                                 headers=ALICE)).status_code == 422)

        first = (await client.post("/chat", json={"message": "How much annual leave?"},
                                   headers=ALICE)).json()
        check("chat returns an answer", bool(first["answer"]), first["answer"][:50])
        check("chat mints a conversation id", first["conversation_id"].startswith("conv-"))
        # The fake retriever may or may not surface the cited chunk; what must hold
        # is that a citation is only ever emitted for a document actually retrieved.
        all_ids = {c.metadata["chunk_id"] for c in chunks}
        check("citations only reference real chunks",
              set(first["citations"]) <= all_ids, str(first["citations"]))

        conversation = first["conversation_id"]
        second = (await client.post("/chat", json={"message": "And sick leave?",
                                                   "conversation_id": conversation},
                                    headers=ALICE)).json()
        check("conversation id is reused", second["conversation_id"] == conversation)

        mine = await client.get(f"/conversations/{conversation}", headers=ALICE)
        check("owner reads history", mine.status_code == 200,
              f"{mine.json()['message_count']} messages")
        check("history accumulated across turns", mine.json()["message_count"] == 4,
              str(mine.json()["message_count"]))

        theirs = await client.get(f"/conversations/{conversation}", headers=BOB)
        check("other user gets 404, not 403", theirs.status_code == 404, str(theirs.status_code))

        deleted = await client.delete(f"/conversations/{conversation}", headers=ALICE)
        check("delete returns 204", deleted.status_code == 204)
        after = await client.get(f"/conversations/{conversation}", headers=ALICE)
        check("deleted conversation is gone", after.status_code == 404, str(after.status_code))

        # streaming
        events = []
        async with client.stream("POST", "/chat/stream",
                                 json={"message": "How much annual leave?"},
                                 headers=ALICE, timeout=30.0) as response:
            check("stream content type is SSE",
                  response.headers["content-type"].startswith("text/event-stream"),
                  response.headers["content-type"])
            name = ""
            async for line in response.aiter_lines():
                if line.startswith("event: "):
                    name = line[7:]
                elif line.startswith("data: "):
                    events.append((name, line[6:]))

        names = [n for n, _ in events]
        check("stream starts with start", names and names[0] == "start", str(names[:2]))
        check("stream emits status events", names.count("status") >= 2, str(names))
        check("stream ends with done", names and names[-1] == "done", str(names[-3:]))

        # concurrency
        started = time.perf_counter()
        responses = await asyncio.gather(*[
            client.post("/chat", json={"message": f"question {i}"}, headers=ALICE, timeout=60.0)
            for i in range(5)
        ])
        wall = time.perf_counter() - started
        check("all concurrent requests succeeded",
              all(r.status_code == 200 for r in responses))
        check("concurrent requests get distinct conversations",
              len({r.json()["conversation_id"] for r in responses}) == 5)
        check("concurrency did not serialise", wall < 10.0, f"{wall:.2f}s for 5 requests")


asyncio.run(run_service_tests())

print("\n== 53: citation extraction in isolation ==")


def extract_citations(answer: str, documents) -> list[str]:
    return [d.metadata["chunk_id"] for d in documents if d.metadata["chunk_id"] in answer]


cite_docs = [Document("x", metadata={"chunk_id": "leave_policy.txt#0"}),
             Document("y", metadata={"chunk_id": "company_handbook.md#3"}),
             Document("z", metadata={"chunk_id": "product_faq.md#1"})]
answer_text = "Employees get 24 days [leave_policy.txt#0] and 12 sick days [company_handbook.md#3]."
check("cited chunks are found",
      extract_citations(answer_text, cite_docs) == ["leave_policy.txt#0", "company_handbook.md#3"],
      str(extract_citations(answer_text, cite_docs)))
check("uncited retrieved chunks are excluded",
      "product_faq.md#1" not in extract_citations(answer_text, cite_docs))
check("no citations when the model cites nothing",
      extract_citations("I could not find that. Contact hr@northwind.example.", cite_docs) == [])

print("\n== 53: thread key isolation is structural ==")
check("same conversation id, different users -> different threads",
      thread_id_for("user-alice", "c1") != thread_id_for("user-bob", "c1"))
check("thread key embeds the user", thread_id_for("user-alice", "c1").startswith("user-alice::"))

print("\n== 53: generated project files ==")
service_dir = ctx.artifact("policy_service")
expected = {"config.py", "graph.py", "requirements.txt", "Dockerfile", "docker-compose.yml",
            "test_api.py", "langgraph.json", "README.md"}
if service_dir.exists():
    present = {p.name for p in service_dir.iterdir()}
    check("notebook 53 writes the project scaffold", expected <= present,
          str(sorted(expected - present)))
    check("langgraph.json is valid json",
          isinstance(json.loads((service_dir / "langgraph.json").read_text(encoding="utf-8")), dict))
else:
    print("  skip  project scaffold (run notebook 53 first)")

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
