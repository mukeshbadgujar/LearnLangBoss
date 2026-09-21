# %% [markdown]
# # 33 - Compile, Invoke and Stream
#
# | | |
# |---|---|
# | **Level** | Intermediate (LangGraph) |
# | **Time** | 45 minutes |
# | **Prerequisites** | `32_conditional_routing` |
# | **Checklist ID** | `33_compile_invoke_stream` |
#
# ## Why this matters
#
# `invoke` returns the final state and nothing else. For a graph that takes 15
# seconds and makes four model calls, that is a blank screen followed by a wall
# of text - and no way to see where the time went.
#
# `stream` is the difference between a demo and a product. LangGraph offers seven
# stream modes, and knowing which one answers which question is the whole skill.

# %%
from pathlib import Path
import sys

_root = Path.cwd()
while not (_root / "shared").is_dir() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from shared.notebook_setup import setup, require  # noqa: E402

ctx = setup("33_compile_invoke_stream")

# %%
import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

from shared.llm import get_chat_model

model = get_chat_model()
parser = StrOutputParser()

# %% [markdown]
# ## 1. A graph worth streaming

# %%
class ResearchState(TypedDict):
    topic: str
    outline: str
    draft: str
    summary: str
    trace: Annotated[list[str], operator.add]


def plan(state: ResearchState) -> dict:
    outline = (ChatPromptTemplate.from_template(
        "Write a 3-bullet outline for a short internal brief on: {topic}"
    ) | model | parser).invoke(state)
    return {"outline": outline, "trace": ["plan"]}


def write(state: ResearchState) -> dict:
    draft = (ChatPromptTemplate.from_template(
        "Write a 150-word internal brief following this outline.\n\nTopic: {topic}\nOutline: {outline}"
    ) | model | parser).invoke(state)
    return {"draft": draft, "trace": ["write"]}


def condense(state: ResearchState) -> dict:
    summary = (ChatPromptTemplate.from_template(
        "Condense to a single sentence a busy executive would read:\n\n{draft}"
    ) | model | parser).invoke(state)
    return {"summary": summary, "trace": ["condense"]}


builder = StateGraph(ResearchState)
builder.add_sequence([("plan", plan), ("write", write), ("condense", condense)])
builder.add_edge(START, "plan")
builder.add_edge("condense", END)
graph = builder.compile()

print(graph.get_graph().draw_ascii())

# %% [markdown]
# ## 2. `compile()` options
#
# Compilation turns a builder into a runnable and is where you attach
# infrastructure. The builder itself is reusable - compile it several ways.

# %%
from langgraph.checkpoint.memory import InMemorySaver

plain = builder.compile()
persistent = builder.compile(checkpointer=InMemorySaver())
approval = builder.compile(checkpointer=InMemorySaver(), interrupt_before=["write"])
named = builder.compile(name="research_brief")

print(f"{'variant':14} {'checkpointer':14} {'interrupts':12} name")
for label, compiled, ckpt, interrupts in [
    ("plain", plain, "none", "-"),
    ("persistent", persistent, "InMemorySaver", "-"),
    ("approval", approval, "InMemorySaver", "before write"),
    ("named", named, "none", "-"),
]:
    print(f"{label:14} {ckpt:14} {interrupts:12} {compiled.name}")

# %% [markdown]
# | Option | What it does | Covered in |
# |---|---|---|
# | `checkpointer` | Persist state per `thread_id` | 34, 35 |
# | `store` | Cross-thread long-term memory | 49 |
# | `interrupt_before` / `interrupt_after` | Static pause points | 36 |
# | `cache` | Skip re-running unchanged nodes | 44 |
# | `name` | Label in traces and when used as a subgraph | 40 |
# | `debug` | Verbose step logging to stdout | here |

# %% [markdown]
# ## 3. `invoke`, `batch`, and their async twins

# %%
import time

TOPIC = "why our support ticket volume spiked after the January release"

start = time.perf_counter()
final = graph.invoke({"topic": TOPIC, "outline": "", "draft": "", "summary": "", "trace": []})
print(f"invoke: {time.perf_counter() - start:.1f}s, returned keys {sorted(final)}")
print(f"\n{final['summary'].strip()[:200]}")

# %%
topics = [
    "our API rate limits and who is hitting them",
    "the cost of the new embedding model",
]

start = time.perf_counter()
results = graph.batch(
    [{"topic": t, "outline": "", "draft": "", "summary": "", "trace": []} for t in topics],
    config={"max_concurrency": 2},
)
print(f"batch of {len(topics)}: {time.perf_counter() - start:.1f}s")
for topic, result in zip(topics, results):
    print(f"  {topic[:44]:46} -> {result['summary'].strip()[:70]}")

# %% [markdown]
# `batch` runs whole graphs concurrently, which is different from the *within*-graph
# parallelism of notebook 38. Both are useful; they compose.

# %% [markdown]
# ## 4. The seven stream modes
#
# | Mode | Yields | Use it for |
# |---|---|---|
# | `updates` | The partial update from each node | **Progress indicators** - the default choice |
# | `values` | The complete state after each step | Watching state accumulate; debugging |
# | `messages` | LLM tokens as they are generated | **Typewriter output in a chat UI** |
# | `custom` | Whatever your nodes write | Progress inside a long node |
# | `debug` | Verbose per-task events | Deep debugging |
# | `tasks` | Task start/finish events | Timing and scheduling questions |
# | `checkpoints` | Each checkpoint written | Auditing persistence |

# %% [markdown]
# ### `updates` - what just happened

# %%
for chunk in graph.stream({"topic": TOPIC, "outline": "", "draft": "", "summary": "", "trace": []},
                          stream_mode="updates"):
    for node_name, update in chunk.items():
        changed = {k: (str(v)[:48] + "...") if isinstance(v, str) and len(str(v)) > 48 else v
                   for k, v in update.items()}
        print(f"[{node_name:9}] {changed}")

# %% [markdown]
# One chunk per node, containing only what that node changed. This is what you
# render as "Planning... Writing... Condensing...".

# %% [markdown]
# ### `values` - the whole state each time

# %%
for step, snapshot in enumerate(graph.stream(
    {"topic": TOPIC, "outline": "", "draft": "", "summary": "", "trace": []}, stream_mode="values"
)):
    filled = [k for k, v in snapshot.items() if v and k != "topic"]
    print(f"step {step}: populated -> {filled}")

# %% [markdown]
# Note there is one more `values` chunk than `updates` chunks: the first is the
# initial state before any node ran.
#
# ### `messages` - tokens

# %%
print("streaming tokens from the condense node only:\n")
for token, metadata in graph.stream(
    {"topic": TOPIC, "outline": "", "draft": "", "summary": "", "trace": []}, stream_mode="messages"
):
    if metadata.get("langgraph_node") == "condense" and token.content:
        print(token.content, end="", flush=True)
print()

# %% [markdown]
# `messages` yields `(token, metadata)` from **every** LLM call in the graph.
# Filtering on `metadata["langgraph_node"]` is essential - otherwise your chat UI
# shows the planning and drafting calls too, which the user should never see.
#
# Other useful metadata keys: `langgraph_step`, `langgraph_triggers`, `tags`, and
# `ls_model_name`.

# %%
seen = {}
for token, metadata in graph.stream(
    {"topic": "one-line test", "outline": "", "draft": "", "summary": "", "trace": []}, stream_mode="messages"
):
    node = metadata.get("langgraph_node")
    seen[node] = seen.get(node, 0) + 1
print("tokens per node:", seen)
print("metadata keys  :", sorted(metadata)[:10])

# %% [markdown]
# ### Tagging model calls for precise filtering
#
# Filtering by node breaks down when one node makes two model calls. Tag them.

# %%
def dual_call_node(state: ResearchState) -> dict:
    tagged = model.with_config(tags=["user_visible"])
    internal = model.with_config(tags=["internal"])
    _ = internal.invoke("Reply with one word: ok")
    visible = tagged.invoke(f"In one sentence, why does {state['topic']} matter?")
    return {"summary": visible.content, "trace": ["dual"]}


builder2 = StateGraph(ResearchState)
builder2.add_node("dual", dual_call_node)
builder2.add_edge(START, "dual")
builder2.add_edge("dual", END)

print("only 'user_visible' tokens:\n")
for token, metadata in builder2.compile().stream(
    {"topic": TOPIC, "outline": "", "draft": "", "summary": "", "trace": []}, stream_mode="messages"
):
    if "user_visible" in (metadata.get("tags") or []) and token.content:
        print(token.content, end="", flush=True)
print()

# %% [markdown]
# ### `custom` - progress from inside a node
#
# A node that loops over 200 documents produces no `updates` chunks until it
# finishes. `get_stream_writer()` lets it report progress itself.

# %%
from langgraph.config import get_stream_writer


class IndexState(TypedDict):
    documents: list[str]
    indexed: int


def index_documents(state: IndexState) -> dict:
    writer = get_stream_writer()
    total = len(state["documents"])
    for i, doc in enumerate(state["documents"], 1):
        writer({"phase": "indexing", "done": i, "total": total, "current": doc})
    return {"indexed": total}


builder3 = StateGraph(IndexState)
builder3.add_node("index", index_documents)
builder3.add_edge(START, "index")
builder3.add_edge("index", END)

docs = ["handbook.md", "leave_policy.txt", "product_faq.md", "tickets.csv"]
for chunk in builder3.compile().stream({"documents": docs, "indexed": 0}, stream_mode="custom"):
    bar = "#" * chunk["done"] + "." * (chunk["total"] - chunk["done"])
    print(f"  [{bar}] {chunk['done']}/{chunk['total']}  {chunk['current']}")

# %% [markdown]
# `get_stream_writer()` is a no-op outside a streaming run, so nodes using it
# still work under plain `invoke`.

# %%
print(builder3.compile().invoke({"documents": docs, "indexed": 0}))

# %% [markdown]
# ### Combining modes
#
# Pass a list and each chunk arrives as `(mode, payload)`.

# %%
for mode, payload in builder3.compile().stream(
    {"documents": docs[:2], "indexed": 0}, stream_mode=["updates", "custom"]
):
    print(f"{mode:8} {payload}")

# %% [markdown]
# ### `debug` and `tasks` - when you need the machinery

# %%
for chunk in graph.stream({"topic": "one-line test", "outline": "", "draft": "", "summary": "", "trace": []},
                          stream_mode="tasks"):
    name = chunk.get("name")
    kind = "start" if "result" not in chunk else "finish"
    print(f"  {kind:7} {name}")

# %% [markdown]
# ## 5. Async
#
# Every method has an `a`-prefixed twin. Use them in FastAPI or any async server
# - a synchronous `invoke` inside an async handler blocks the event loop and
# destroys your throughput under concurrency.

# %%
async def run_async() -> None:
    result = await graph.ainvoke({"topic": "async test", "outline": "", "draft": "", "summary": "", "trace": []})
    print("ainvoke ->", result["summary"].strip()[:90])

    print("\nastream updates:")
    async for chunk in graph.astream(
        {"topic": "async test", "outline": "", "draft": "", "summary": "", "trace": []}, stream_mode="updates"
    ):
        print("  ", list(chunk))


await run_async()

# %% [markdown]
# ### `astream_events` - the finest granularity
#
# Every start/end/stream event for every model, node and chain in the graph.
# Heavier than `astream`, but it is what you need when you want to show tool
# calls, retrieval hits and tokens in the same UI.

# %%
async def show_events() -> None:
    counts: dict[str, int] = {}
    async for event in graph.astream_events(
        {"topic": "events test", "outline": "", "draft": "", "summary": "", "trace": []}
    ):
        counts[event["event"]] = counts.get(event["event"], 0) + 1
    for name, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {name:28} {count}")


await show_events()

# %%
async def stream_one_node() -> None:
    """Typical UI wiring: node-level progress plus tokens from the final node."""
    async for event in graph.astream_events(
        {"topic": "events test", "outline": "", "draft": "", "summary": "", "trace": []}
    ):
        kind = event["event"]
        if kind == "on_chain_start" and event["name"] in {"plan", "write", "condense"}:
            print(f"\n[{event['name']} started]", end=" ")
        elif kind == "on_chat_model_stream" and event["metadata"].get("langgraph_node") == "condense":
            print(event["data"]["chunk"].content, end="", flush=True)
    print()


await stream_one_node()

# %% [markdown]
# ## 6. Runtime configuration
#
# `config` carries per-run settings. `configurable` is the part your own code can
# read via `get_config()`.

# %%
from langgraph.config import get_config


class ConfigState(TypedDict):
    question: str
    answer: str


def configurable_node(state: ConfigState) -> dict:
    config = get_config()
    settings = config.get("configurable", {})
    tone = settings.get("tone", "neutral")
    max_words = settings.get("max_words", 50)
    answer = (ChatPromptTemplate.from_template(
        "Answer in a {tone} tone, at most {max_words} words: {question}"
    ) | model | parser).invoke({"question": state["question"], "tone": tone, "max_words": max_words})
    return {"answer": answer}


builder4 = StateGraph(ConfigState)
builder4.add_node("answer", configurable_node)
builder4.add_edge(START, "answer")
builder4.add_edge("answer", END)
configurable_graph = builder4.compile()

for tone in ["formal", "casual"]:
    outcome = configurable_graph.invoke(
        {"question": "Why did our ticket volume spike?", "answer": ""},
        config={"configurable": {"tone": tone, "max_words": 25}},
    )
    print(f"[{tone:7}] {outcome['answer'].strip()[:130]}")

# %% [markdown]
# Other config keys you will use constantly:

# %%
outcome = graph.invoke(
    {"topic": "config demo", "outline": "", "draft": "", "summary": "", "trace": []},
    config={
        "run_name": "research_brief",            # readable name in LangSmith
        "tags": ["tenant:acme", "v2"],           # filterable
        "metadata": {"user_id": "u-4821"},       # searchable
        "recursion_limit": 15,                   # superstep ceiling
        "max_concurrency": 4,                    # parallel branch limit
        "configurable": {"thread_id": "t-1"},    # required when checkpointing
    },
)
print("ran with config; summary:", outcome["summary"].strip()[:90])

# %% [markdown]
# ## 7. Choosing a mode
#
# ```
# Building a chat UI?                      -> messages (+ updates for step labels)
# Showing "step 2 of 4"?                   -> updates
# Long-running node needs a progress bar?  -> custom
# Debugging what state looks like?         -> values
# Debugging why a node ran?                -> debug or tasks
# Need tool calls AND tokens in one feed?  -> astream_events
# Server-side, no UI?                      -> invoke
# ```
#
# The common production combination is `stream_mode=["updates", "messages"]`:
# step labels for the progress area, tokens for the answer area.

# %%
print("labels + tokens in one pass:\n")
for mode, payload in graph.stream(
    {"topic": "combined demo", "outline": "", "draft": "", "summary": "", "trace": []},
    stream_mode=["updates", "messages"],
):
    if mode == "updates":
        print(f"\n\n[{list(payload)[0]}] ", end="")
    elif mode == "messages":
        token, metadata = payload
        if metadata.get("langgraph_node") == "condense" and token.content:
            print(token.content, end="", flush=True)
print()

# %% [markdown]
# ## Try it yourself
#
# 1. **Add a retrieval node** that emits `custom` progress per document scored,
#    and render it as a progress bar alongside `updates`.
# 2. **Measure the overhead.** Time the same run under `invoke`, `stream`, and
#    `astream_events` and quantify what the finest granularity costs.
# 3. **Stream only the final answer** in a graph where three nodes call the
#    model, using tags rather than node names.
# 4. **Async batch.** Run 10 topics with `abatch` at `max_concurrency` 2 and 10,
#    and compare total wall time.

# %% [markdown]
# ## Recap
#
# | Concept | Takeaway |
# |---|---|
# | `compile()` | Where checkpointer, store, interrupts, cache and name are attached |
# | One builder, many compiles | Same graph with and without persistence |
# | `updates` | Per-node partial updates - the default progress feed |
# | `values` | Full state per step; one extra chunk for the initial state |
# | `messages` | `(token, metadata)`; **filter by `langgraph_node` or tags** |
# | `custom` + `get_stream_writer()` | Progress from inside a long node; no-op under `invoke` |
# | `tasks` / `debug` | Scheduling and deep debugging |
# | Multiple modes | Pass a list; chunks arrive as `(mode, payload)` |
# | `astream_events` | Finest granularity; tool calls and tokens in one feed |
# | Async twins | Mandatory in FastAPI - sync `invoke` blocks the event loop |
# | `configurable` | Per-run settings readable with `get_config()` |
#
# ## Next
#
# -> [34_checkpointing_thread_ids.ipynb](34_checkpointing_thread_ids.ipynb)
