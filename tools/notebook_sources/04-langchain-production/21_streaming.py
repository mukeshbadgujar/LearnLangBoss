# %% [markdown]
# # 21 - Streaming
#
# | | |
# |---|---|
# | **Level** | Intermediate |
# | **Time** | 40 minutes |
# | **Prerequisites** | `20_callbacks` |
# | **Checklist ID** | `21_streaming` |
#
# ## Why this matters
#
# A RAG answer takes 4 seconds. An agent takes 20. Without streaming your user
# stares at a spinner and concludes the product is broken; with streaming they see
# progress within 300 ms and perceive the same system as fast.
#
# Perceived latency is the whole game, and it is the cheapest UX win in an LLM
# application. This notebook covers the four streaming modes and which to use
# where.

# %%
from pathlib import Path
import sys

_root = Path.cwd()
while not (_root / "shared").is_dir() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from shared.notebook_setup import setup, require  # noqa: E402

ctx = setup("21_streaming")

# %%
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from shared.llm import get_chat_model

model = get_chat_model()

# %% [markdown]
# ## 1. Streaming a model directly

# %%
import time

start = time.perf_counter()
first_token_at = None
pieces = 0

for chunk in model.stream("Explain why streaming improves perceived latency. Three sentences."):
    if first_token_at is None:
        first_token_at = time.perf_counter() - start
    pieces += 1
    print(chunk.content, end="", flush=True)

total = time.perf_counter() - start
print(f"\n\nfirst token: {first_token_at:.2f}s | total: {total:.2f}s | {pieces} chunks")

# %%
start = time.perf_counter()
model.invoke("Explain why streaming improves perceived latency. Three sentences.")
print(f"invoke (nothing visible until done): {time.perf_counter() - start:.2f}s")

# %% [markdown]
# Same total time, completely different experience. Time-to-first-token is the
# metric users actually feel.

# %% [markdown]
# ### Chunks are `AIMessageChunk` and they add up

# %%
collected = None
for chunk in model.stream("Count: one two three"):
    collected = chunk if collected is None else collected + chunk

print("type          :", type(collected).__name__)
print("accumulated   :", collected.content)
print("usage on final:", collected.usage_metadata)

# %% [markdown]
# Adding chunks together rebuilds the full message, including tool calls and
# usage metadata. That is how you get both streaming *and* a complete record.

# %% [markdown]
# ## 2. Streaming a chain
#
# LCEL streams end to end as long as every step supports it. `StrOutputParser` and
# `JsonOutputParser` do; a step that must see the whole input does not.

# %%
chain = ChatPromptTemplate.from_template("Write a short paragraph about {topic}.") | model | StrOutputParser()

for piece in chain.stream({"topic": "idempotency in webhook delivery"}):
    print(piece, end="", flush=True)
print()

# %%
# A blocking step kills streaming - know how to spot it.
from langchain_core.runnables import RunnableLambda

blocking_chain = chain | RunnableLambda(lambda text: text.upper())

start = time.perf_counter()
chunks = list(blocking_chain.stream({"topic": "connection pooling"}))
print(f"{len(chunks)} chunk(s) in {time.perf_counter() - start:.2f}s  <- one chunk means no streaming")
print("Reason: the lambda needs the whole string before it can uppercase it.")

# %%
# A generator function preserves streaming.
def upper_stream(tokens):
    for token in tokens:
        yield token.upper()


streaming_chain = chain | RunnableLambda(upper_stream)
chunks = list(streaming_chain.stream({"topic": "connection pooling"}))
print(f"{len(chunks)} chunks - streaming preserved by using a generator")

# %% [markdown]
# ## 3. Streaming structured output
#
# `JsonOutputParser` emits progressively completed objects. This is how you build
# a form that fills itself in field by field.

# %%
from langchain_core.output_parsers import JsonOutputParser

json_chain = (
    ChatPromptTemplate.from_template(
        'Reply with JSON only: {{"customer": string, "category": string, '
        '"priority": string, "summary": string, "next_step": string}}\n\nTicket: {ticket}'
    )
    | model
    | JsonOutputParser()
)

for partial in json_chain.stream({"ticket": "Everline Bank: API p99 latency 4s during EU hours, Enterprise plan"}):
    print(partial)

# %% [markdown]
# ## 4. `astream_events`: the fine-grained API
#
# `stream` gives you the final output incrementally. `astream_events` gives you
# *every* internal event - which chain started, what the retriever returned, which
# tool ran - as a typed async stream.

# %%
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from shared.llm import get_embeddings

policy = Path(ctx.data("leave_policy.txt")).read_text(encoding="utf-8")
store = FAISS.from_documents(
    [Document(c, metadata={"source": "leave_policy.txt"})
     for c in RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60).split_text(policy)],
    get_embeddings(),
)
retriever = store.as_retriever(search_kwargs={"k": 2}).with_config(run_name="policy_retriever")

rag_chain = (
    {"context": retriever, "question": lambda x: x["question"]}
    | ChatPromptTemplate.from_template("Answer from the context only.\n\n{context}\n\nQ: {question}")
    | model
    | StrOutputParser()
).with_config(run_name="policy_rag")

# %%
async def show_events(question: str) -> None:
    async for event in rag_chain.astream_events({"question": question}, version="v2"):
        kind = event["event"]
        name = event.get("name", "")

        if kind == "on_retriever_end":
            docs = event["data"]["output"]
            print(f"[retrieved] {len(docs)} documents from {name}")
            for doc in docs:
                print("            -", " ".join(doc.page_content.split())[:70])

        elif kind == "on_chat_model_stream":
            token = event["data"]["chunk"].content
            if token:
                print(token, end="", flush=True)

        elif kind == "on_chain_end" and name == "policy_rag":
            print("\n[done]")


await show_events("How many days of bereavement leave for an immediate family member?")

# %% [markdown]
# That is a real UI in miniature: show the sources the instant retrieval finishes,
# then stream the answer on top of them. The user sees the system working rather
# than waiting.

# %%
# The event catalogue - useful reference.
async def catalogue(question: str) -> None:
    seen: dict[str, int] = {}
    async for event in rag_chain.astream_events({"question": question}, version="v2"):
        key = f"{event['event']:26} {event.get('name', '')[:28]}"
        seen[key] = seen.get(key, 0) + 1
    for key, count in seen.items():
        print(f"  {key:56} x{count}")


await catalogue("How many casual leave days are there?")

# %% [markdown]
# | Event | `data` contains |
# |---|---|
# | `on_chain_start` / `on_chain_end` | `input` / `output` |
# | `on_chat_model_start` | `input` (the messages) |
# | `on_chat_model_stream` | `chunk` (an `AIMessageChunk`) |
# | `on_chat_model_end` | `output` (the full message with usage) |
# | `on_retriever_end` | `output` (list of `Document`) |
# | `on_tool_start` / `on_tool_end` | `input` / `output` |
# | `on_parser_stream` | `chunk` |
#
# Filter by `name` and `tags` to isolate one component in a large graph.

# %%
async def only_named(question: str, wanted: str) -> None:
    async for event in rag_chain.astream_events(
        {"question": question}, version="v2", include_names=[wanted]
    ):
        print(f"  {event['event']:26} {event['name']}")


await only_named("How many sick days?", "policy_retriever")

# %% [markdown]
# ## 5. Streaming agents and graphs
#
# LangGraph adds `stream_mode`, which is the control you want for agents.

# %%
import sqlite3
from contextlib import closing

from langchain.agents import create_agent
from langchain.tools import tool

from shared.sample_data import ensure_all

DB_PATH = str(ensure_all()["sqlite"])


@tool
def get_employee(employee_id: str) -> str:
    """Look up an employee's department, level and leave balances by id (e.g. E-104)."""
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM employees WHERE employee_id = ?", (employee_id.upper(),)).fetchone()
    if row is None:
        return f"No employee {employee_id}."
    return f"{row['name']} - {row['department']} {row['level']}, {row['annual_leave_balance']} leave days left."


agent = create_agent(
    model=model,
    tools=[get_employee],
    system_prompt="You are the Ops Assistant. Always use tools for employee facts.",
)

question = {"messages": [{"role": "user", "content": "Compare the leave balances of E-101 and E-104."}]}

# %%
print("stream_mode='updates' - what each node produced:\n")
for chunk in agent.stream(question, stream_mode="updates"):
    for node, update in chunk.items():
        for message in update.get("messages", []):
            if message.type == "ai" and message.tool_calls:
                print(f"  [{node}] wants {[c['name'] + str(c['args']) for c in message.tool_calls]}")
            elif message.type == "tool":
                print(f"  [tools] {message.name} -> {str(message.content)[:70]}")
            elif message.type == "ai":
                print(f"  [{node}] {str(message.content)[:200]}")

# %%
print("stream_mode='messages' - token by token, only the answer:\n")
for token, metadata in agent.stream(question, stream_mode="messages"):
    if token.content and metadata.get("langgraph_node") == "model":
        print(token.content, end="", flush=True)
print()

# %%
print("stream_mode='values' - the full state after each step:\n")
for state in agent.stream(question, stream_mode="values"):
    print(f"  {len(state['messages'])} messages, last is {state['messages'][-1].type}")

# %%
print("multiple modes at once:\n")
for mode, payload in agent.stream(question, stream_mode=["updates", "messages"]):
    if mode == "messages":
        token, metadata = payload
        if token.content and metadata.get("langgraph_node") == "model":
            print(token.content, end="", flush=True)
    else:
        print(f"\n[{mode}] {list(payload)}")
print()

# %% [markdown]
# | `stream_mode` | Yields | Use for |
# |---|---|---|
# | `"values"` | full state after each step | debugging, progress bars |
# | `"updates"` | only what each node changed | showing "calling tool X..." |
# | `"messages"` | `(token, metadata)` tuples | token-by-token UI output |
# | `"custom"` | whatever you emit | progress from inside a node (notebook 42) |
# | `"debug"` | everything | deep debugging |

# %% [markdown]
# ## 6. Async streaming for a web server

# %%
async def astream_demo() -> None:
    print("async model stream: ", end="")
    async for chunk in model.astream("Name three benefits of async IO. One line."):
        print(chunk.content, end="", flush=True)
    print()

    print("async agent stream: ", end="")
    async for token, metadata in agent.astream(question, stream_mode="messages"):
        if token.content and metadata.get("langgraph_node") == "model":
            print(token.content, end="", flush=True)
    print()


await astream_demo()

# %% [markdown]
# ### Server-Sent Events shape for FastAPI
#
# This is the pattern notebook 53 uses for real.
#
# ```python
# from fastapi import FastAPI
# from fastapi.responses import StreamingResponse
# import json
#
# app = FastAPI()
#
# @app.post("/chat/stream")
# async def chat_stream(body: dict):
#     async def event_source():
#         async for event in rag_chain.astream_events({"question": body["question"]}, version="v2"):
#             if event["event"] == "on_retriever_end":
#                 sources = [d.metadata.get("source") for d in event["data"]["output"]]
#                 yield f"event: sources\ndata: {json.dumps(sources)}\n\n"
#             elif event["event"] == "on_chat_model_stream":
#                 token = event["data"]["chunk"].content
#                 if token:
#                     yield f"event: token\ndata: {json.dumps(token)}\n\n"
#         yield "event: done\ndata: {}\n\n"
#
#     return StreamingResponse(event_source(), media_type="text/event-stream")
# ```
#
# Three production details: send a heartbeat comment every ~15 s so proxies do not
# time the connection out, set `X-Accel-Buffering: no` if you are behind nginx,
# and handle client disconnects so you stop paying for an abandoned generation.

# %% [markdown]
# ## 7. Streaming and error handling
#
# A failure halfway through a stream leaves the user with half an answer. Decide
# in advance whether you show it or discard it.

# %%
def guarded_stream(runnable, payload, on_error="partial"):
    """Yield tokens, and deal with a mid-stream failure deliberately."""
    buffer: list[str] = []
    try:
        for piece in runnable.stream(payload):
            buffer.append(piece)
            yield piece
    except Exception as exc:
        if on_error == "partial":
            yield f"\n\n[interrupted: {type(exc).__name__}. Partial answer above.]"
        else:
            raise


for piece in guarded_stream(chain, {"topic": "graceful degradation during a provider outage"}):
    print(piece, end="", flush=True)
print()

# %% [markdown]
# ## 8. Measuring what matters

# %%
def measure(runnable, payload, label: str) -> None:
    start = time.perf_counter()
    first = None
    chunks = 0
    characters = 0
    for piece in runnable.stream(payload):
        text = piece if isinstance(piece, str) else getattr(piece, "content", "")
        if first is None and text:
            first = time.perf_counter() - start
        chunks += 1
        characters += len(text)
    total = time.perf_counter() - start
    print(f"{label:22} TTFT={first:5.2f}s  total={total:5.2f}s  chunks={chunks:4}  "
          f"~{characters / max(total, 0.01):5.0f} chars/s")


measure(chain, {"topic": "vector indexes"}, "simple chain")
measure(rag_chain, {"question": "How many sick days?"}, "rag chain")

# %% [markdown]
# **TTFT (time to first token)** is the number to optimise. Retrieval and query
# rewriting both happen *before* the first token, so a history-aware RAG chain has
# noticeably worse TTFT than a plain one - which is a real argument for skipping
# the rewrite when history is empty (notebook 14 does this automatically).

# %% [markdown]
# ## Try it yourself
#
# 1. **Build a sources-then-answer UI** in the notebook: print the retrieved
#    sources as a numbered list the moment retrieval ends, then stream the answer
#    below it.
# 2. **Find the blocking step.** Add a reranking step to `rag_chain` and measure
#    how much TTFT degrades. Decide whether it is worth it.
# 3. **Stream a structured form.** Use `JsonOutputParser` streaming to print a
#    field only once it is complete, rather than reprinting the whole dict.
# 4. **Cancellation.** Wrap an async stream in `asyncio.wait_for` with a 2-second
#    timeout and confirm the generation stops.

# %% [markdown]
# ## Recap
#
# | Concept | Takeaway |
# |---|---|
# | `.stream()` | Incremental final output; same total time, far better UX |
# | `AIMessageChunk` | Chunks add with `+` to rebuild the full message |
# | Blocking steps | Any step needing the whole input kills streaming - use generators |
# | `JsonOutputParser` | Streams progressively completed objects |
# | `astream_events(version="v2")` | Every internal event; filter with `include_names` |
# | `stream_mode` | `values` / `updates` / `messages` / `custom` / `debug` |
# | TTFT | The metric users feel; everything before the first token counts against it |
# | SSE | The standard transport; heartbeats and disconnect handling required |
#
# ## Next
#
# -> [22_multimodal.ipynb](22_multimodal.ipynb)
