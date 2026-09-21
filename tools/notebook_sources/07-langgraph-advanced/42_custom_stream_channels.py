# %% [markdown]
# # 42 - Custom Stream Channels
#
# | | |
# |---|---|
# | **Level** | Advanced (LangGraph) |
# | **Time** | 40 minutes |
# | **Prerequisites** | `41_multi_agent_handoffs` |
# | **Checklist ID** | `42_custom_stream_channels` |
#
# ## Why this matters
#
# Notebook 33 covered the built-in stream modes. They tell the user *which node*
# is running. They cannot tell the user "searching 47 documents", "found 3 of 5
# required facts", or "this is taking longer than usual because the CRM is slow".
#
# Custom channels let a node emit whatever its own progress means. Done well,
# this is the difference between a spinner and an interface that explains itself.

# %%
from pathlib import Path
import sys

_root = Path.cwd()
while not (_root / "shared").is_dir() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from shared.notebook_setup import setup, require  # noqa: E402

ctx = setup("42_custom_stream_channels")

# %%
import operator
import time
from typing import Annotated, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from shared.llm import get_chat_model

model = get_chat_model()
parser = StrOutputParser()

# %% [markdown]
# ## 1. `get_stream_writer()`
#
# Call it inside a node to get a function that pushes anything into the `custom`
# stream. Outside a streaming run it is a no-op, so nodes stay usable under
# `invoke`.

# %%
class SearchState(TypedDict):
    query: str
    hits: list[str]


CORPUS = [f"document-{i:03d}" for i in range(1, 25)]


def search(state: SearchState) -> dict:
    writer = get_stream_writer()
    writer({"event": "start", "total": len(CORPUS)})

    hits = []
    for index, document in enumerate(CORPUS, start=1):
        time.sleep(0.01)
        if index % 7 == 0:
            hits.append(document)
            writer({"event": "hit", "document": document})
        if index % 6 == 0:
            writer({"event": "progress", "scanned": index, "total": len(CORPUS)})

    writer({"event": "done", "hits": len(hits)})
    return {"hits": hits}


builder = StateGraph(SearchState)
builder.add_node("search", search)
builder.add_edge(START, "search")
builder.add_edge("search", END)
search_graph = builder.compile()

for chunk in search_graph.stream({"query": "leave policy", "hits": []}, stream_mode="custom"):
    if chunk["event"] == "progress":
        filled = int(20 * chunk["scanned"] / chunk["total"])
        print(f"  [{'#' * filled}{'.' * (20 - filled)}] {chunk['scanned']}/{chunk['total']}")
    elif chunk["event"] == "hit":
        print(f"      + {chunk['document']}")
    else:
        print(f"  {chunk}")

# %%
print("under invoke, the writer is a no-op:", search_graph.invoke({"query": "x", "hits": []}))

# %% [markdown]
# ## 2. Designing an event protocol
#
# Untyped dicts get messy fast. Give your custom stream a schema - your frontend
# is going to switch on it.

# %%
from dataclasses import asdict, dataclass, field
from typing import Literal


@dataclass
class ProgressEvent:
    """One unit of user-visible progress."""

    kind: Literal["stage", "progress", "finding", "warning", "metric"]
    stage: str
    message: str = ""
    current: int = 0
    total: int = 0
    data: dict = field(default_factory=dict)

    def emit(self) -> None:
        writer = get_stream_writer()
        writer(asdict(self))


def render(event: dict) -> str:
    """One renderer, driven by `kind`. Your React component looks like this too."""
    kind = event["kind"]
    if kind == "stage":
        return f"\n>> {event['stage']}: {event['message']}"
    if kind == "progress":
        filled = int(18 * event["current"] / max(event["total"], 1))
        return f"   [{'#' * filled}{'.' * (18 - filled)}] {event['current']}/{event['total']} {event['message']}"
    if kind == "finding":
        return f"   + {event['message']}"
    if kind == "warning":
        return f"   ! {event['message']}"
    return f"   . {event['stage']}: {event['data']}"


# %% [markdown]
# ## 3. A research pipeline that narrates itself

# %%
class ResearchState(TypedDict):
    question: str
    sources: Annotated[list[str], operator.add]
    facts: Annotated[list[str], operator.add]
    answer: str


DOCUMENTS = {
    "leave_policy.txt": "24 days annual leave; 12 may carry forward until 30 June; 12 sick days.",
    "company_handbook.md": "Notice: L1-L3 30 days, L4 60 days, L5+ 90 days. Learning budget INR 60,000.",
    "product_faq.md": "Growth plan 600 req/min. Data in ap-south-1. Enterprise SLA 1 hour.",
    "support_tickets.csv": "14 open billing tickets, 37 platform, 5 onboarding.",
    "archive_2024.md": "",           # empty - will produce a warning
}


def gather(state: ResearchState) -> dict:
    ProgressEvent("stage", "gather", "reading source documents").emit()

    sources, facts = [], []
    for index, (name, content) in enumerate(DOCUMENTS.items(), start=1):
        time.sleep(0.05)
        ProgressEvent("progress", "gather", name, index, len(DOCUMENTS)).emit()

        if not content:
            ProgressEvent("warning", "gather", f"{name} is empty - skipped").emit()
            continue

        sources.append(name)
        facts.append(f"[{name}] {content}")
        ProgressEvent("finding", "gather", f"{name}: {content[:56]}").emit()

    ProgressEvent("metric", "gather", data={"sources": len(sources), "skipped": len(DOCUMENTS) - len(sources)}).emit()
    return {"sources": sources, "facts": facts}


def answer(state: ResearchState) -> dict:
    ProgressEvent("stage", "answer", f"composing from {len(state['facts'])} sources").emit()
    start = time.perf_counter()
    text = (ChatPromptTemplate.from_template(
        "Answer using only these facts:\n{facts}\n\nQuestion: {question}\nTwo sentences."
    ) | model | parser).invoke({"facts": "\n".join(state["facts"]), "question": state["question"]})
    ProgressEvent("metric", "answer", data={"seconds": round(time.perf_counter() - start, 2)}).emit()
    return {"answer": text}


builder2 = StateGraph(ResearchState)
builder2.add_sequence([("gather", gather), ("answer", answer)])
builder2.add_edge(START, "gather")
builder2.add_edge("answer", END)
research = builder2.compile()

for event in research.stream(
    {"question": "How much leave do I get and what is the L5 notice period?",
     "sources": [], "facts": [], "answer": ""},
    stream_mode="custom",
):
    print(render(event))

# %% [markdown]
# A user watching that knows what the system read, what it skipped and why, and
# how long the model call took. None of that is available from `updates`.

# %% [markdown]
# ## 4. Combining custom events with tokens
#
# The full production feed: stage labels, progress, and the answer typing out.

# %%
print("=" * 62)
for mode, payload in research.stream(
    {"question": "What is the Enterprise SLA?", "sources": [], "facts": [], "answer": ""},
    stream_mode=["custom", "messages"],
):
    if mode == "custom":
        print(render(payload))
    else:
        token, metadata = payload
        if metadata.get("langgraph_node") == "answer" and token.content:
            print(token.content, end="", flush=True)
print("\n" + "=" * 62)

# %% [markdown]
# ## 5. Streaming from inside tools
#
# `get_stream_writer()` works in a tool function too, so a slow tool can report
# progress rather than appearing frozen.

# %%
from langchain_core.tools import tool
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition


@tool
def audit_accounts(region: str) -> str:
    """Audit every account in a region for billing anomalies. Slow."""
    writer = get_stream_writer()
    accounts = [f"ACC-{n:04d}" for n in range(1, 13)]
    anomalies = []

    for index, account in enumerate(accounts, start=1):
        time.sleep(0.02)
        if index % 5 == 0:
            anomalies.append(account)
            writer({"kind": "finding", "stage": "audit", "message": f"anomaly in {account}"})
        if index % 4 == 0:
            writer({"kind": "progress", "stage": "audit", "message": region,
                    "current": index, "total": len(accounts)})

    return f"Audited {len(accounts)} accounts in {region}; {len(anomalies)} anomalies: {anomalies}"


tools = [audit_accounts]
model_with_tools = model.bind_tools(tools)


def agent_node(state: MessagesState) -> dict:
    return {"messages": [model_with_tools.invoke(state["messages"])]}


builder3 = StateGraph(MessagesState)
builder3.add_node("agent", agent_node)
builder3.add_node("tools", ToolNode(tools))
builder3.add_edge(START, "agent")
builder3.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
builder3.add_edge("tools", "agent")
tool_graph = builder3.compile()

for mode, payload in tool_graph.stream(
    {"messages": [("user", "Audit the ap-south-1 region for billing anomalies.")]},
    stream_mode=["custom", "updates"],
):
    if mode == "custom":
        print(render(payload))
    else:
        print(f"[{list(payload)[0]}]")

# %% [markdown]
# ## 6. Custom events across subgraphs
#
# Events from a child surface in the parent's stream. With `subgraphs=True` you
# also learn which child sent them.

# %%
class InnerState(TypedDict):
    items: list[str]
    processed: int


def inner_work(state: InnerState) -> dict:
    for index, item in enumerate(state["items"], start=1):
        time.sleep(0.02)
        ProgressEvent("progress", "inner", item, index, len(state["items"])).emit()
    return {"processed": len(state["items"])}


inner = StateGraph(InnerState)
inner.add_node("work", inner_work)
inner.add_edge(START, "work")
inner.add_edge("work", END)
inner_graph = inner.compile(name="worker")

outer = StateGraph(InnerState)
outer.add_node("stage_one", inner_graph)
outer.add_node("stage_two", lambda s: (ProgressEvent("stage", "outer", "finalising").emit(), {"processed": s["processed"]})[1])
outer.add_edge(START, "stage_one")
outer.add_edge("stage_one", "stage_two")
outer.add_edge("stage_two", END)
outer_graph = outer.compile()

for namespace, event in outer_graph.stream(
    {"items": ["a", "b", "c"], "processed": 0}, stream_mode="custom", subgraphs=True
):
    where = ".".join(part.split(":")[0] for part in namespace) or "parent"
    print(f"  {where:12} {render(event).strip()}")

# %% [markdown]
# ## 7. Server-sent events
#
# What this looks like behind a FastAPI endpoint. Notebook 53 builds the full
# service; this is the streaming core.

# %%
import json


async def sse_stream(graph, payload: dict):
    """Yield Server-Sent Events a browser's EventSource can consume."""
    async for mode, chunk in graph.astream(payload, stream_mode=["custom", "messages", "updates"]):
        if mode == "custom":
            event = {"type": "progress", **chunk}
        elif mode == "messages":
            token, metadata = chunk
            if metadata.get("langgraph_node") != "answer" or not token.content:
                continue
            event = {"type": "token", "content": token.content}
        else:
            event = {"type": "node", "name": list(chunk)[0]}
        yield f"data: {json.dumps(event)}\n\n"


async def preview() -> None:
    shown = 0
    async for line in sse_stream(research, {"question": "What is the data region?",
                                            "sources": [], "facts": [], "answer": ""}):
        if shown < 8:
            print("  " + line.strip())
        shown += 1
    print(f"  ... {shown} events total")


await preview()

# %% [markdown]
# ```javascript
# // In the browser
# const events = new EventSource("/chat/stream?q=...");
# events.onmessage = (e) => {
#   const event = JSON.parse(e.data);
#   if (event.type === "progress") updateProgressPanel(event);
#   if (event.type === "token")    appendToAnswer(event.content);
#   if (event.type === "node")     setCurrentStage(event.name);
# };
# ```

# %% [markdown]
# ## 8. Design guidance
#
# **Emit what the user cares about, not what the code is doing.** "Checking the
# billing system" is useful; "entering node `_validate_3`" is not.
#
# **Do not emit per item at high volume.** 10,000 events will overwhelm the
# channel and the browser. Throttle.

# %%
def throttled_progress(total: int, target_updates: int = 20):
    """Emit at most `target_updates` progress events regardless of `total`."""
    every = max(1, total // target_updates)

    def maybe(index: int, message: str = "") -> None:
        if index % every == 0 or index == total:
            ProgressEvent("progress", "scan", message, index, total).emit()

    return maybe


def scan_many(state: SearchState) -> dict:
    total = 5000
    maybe = throttled_progress(total)
    for index in range(1, total + 1):
        maybe(index, "scanning")
    return {"hits": []}


builder4 = StateGraph(SearchState)
builder4.add_node("scan", scan_many)
builder4.add_edge(START, "scan")
builder4.add_edge("scan", END)

events = list(builder4.compile().stream({"query": "x", "hits": []}, stream_mode="custom"))
print(f"5,000 items -> {len(events)} events (last: {events[-1]['current']}/{events[-1]['total']})")

# %% [markdown]
# **Keep payloads small and serialisable.** They may cross a network and be
# JSON-encoded. Send an id, not the document.
#
# **Never put secrets in a custom event.** These go straight to the browser -
# they are the least protected surface in your system.

# %%
def safe_event(record: dict) -> dict:
    """Whitelist fields before emitting anything derived from internal data."""
    allowed = {"id", "title", "score", "source"}
    return {k: v for k, v in record.items() if k in allowed}


internal = {"id": "doc-1", "title": "Leave Policy", "score": 0.82,
            "source": "leave_policy.txt", "api_key": "sk-secret", "raw_text": "..." * 500}
print("emitted:", safe_event(internal))

# %% [markdown]
# ## Try it yourself
#
# 1. **Add an ETA.** Track elapsed time per item and include a projected
#    completion time in progress events.
# 2. **Cancellable work.** Emit a `checkpoint` event every 100 items with enough
#    information to resume, and test resuming after an interruption.
# 3. **Render it properly.** Build a small terminal renderer that keeps a live
#    progress bar on one line using `\r`.
# 4. **Multi-agent narration.** Add custom events to notebook 41's supervisor so
#    the user sees which specialist is working and why.

# %% [markdown]
# ## Recap
#
# | Concept | Takeaway |
# |---|---|
# | `get_stream_writer()` | Emit anything to the `custom` channel; no-op under `invoke` |
# | Event protocol | Give events a `kind` and a dataclass - your UI switches on it |
# | Combined modes | `["custom", "messages"]` gives progress plus tokens |
# | Inside tools | A slow tool can narrate instead of appearing frozen |
# | Subgraphs | Child events reach the parent; `subgraphs=True` names the source |
# | SSE | Map custom/messages/updates onto one JSON event stream |
# | Throttle | Cap events regardless of item count |
# | Small payloads | Ids, not documents - these cross the wire |
# | **No secrets** | Custom events go straight to the browser; whitelist fields |
#
# ## Next
#
# -> [43_hybrid_deterministic_llm.ipynb](43_hybrid_deterministic_llm.ipynb)
