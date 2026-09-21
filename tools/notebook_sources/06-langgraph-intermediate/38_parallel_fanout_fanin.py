# %% [markdown]
# # 38 - Parallel Execution: Fan-out and Fan-in
#
# | | |
# |---|---|
# | **Level** | Intermediate (LangGraph) |
# | **Time** | 45 minutes |
# | **Prerequisites** | `37_message_history_and_deletion` |
# | **Checklist ID** | `38_parallel_fanout_fanin` |
#
# ## Why this matters
#
# Three independent model calls run one after another take three times as long
# as they need to. A research agent checking five sources, a document pipeline
# running four extractors, a RAG system querying three indexes - all of these are
# embarrassingly parallel and most implementations run them in sequence.
#
# LangGraph parallelises automatically when the graph shape allows it. The skills
# are: drawing the right shape, choosing reducers that merge concurrent writes,
# and handling the branch that fails.

# %%
from pathlib import Path
import sys

_root = Path.cwd()
while not (_root / "shared").is_dir() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from shared.notebook_setup import setup, require  # noqa: E402

ctx = setup("38_parallel_fanout_fanin")

# %%
import operator
import time
from typing import Annotated, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

from shared.llm import get_chat_model

model = get_chat_model()
parser = StrOutputParser()

# %% [markdown]
# ## 1. The superstep model
#
# LangGraph does not execute node by node. It executes in **supersteps**: every
# node whose predecessors have finished runs together, their writes are merged,
# then the next superstep begins.
#
# Two consequences follow, and they explain most parallelism surprises:
#
# 1. Nodes in the same superstep **cannot see each other's writes**.
# 2. A node with several incoming edges waits for **all** of them.

# %%
class TraceState(TypedDict):
    log: Annotated[list[str], operator.add]


def make_node(name: str, seconds: float = 0.3):
    def node(state: TraceState) -> dict:
        start = time.perf_counter()
        time.sleep(seconds)
        return {"log": [f"{name} ({time.perf_counter() - start:.2f}s)"]}
    return node


# Sequential: A -> B -> C
sequential = StateGraph(TraceState)
sequential.add_sequence([("a", make_node("a")), ("b", make_node("b")), ("c", make_node("c"))])
sequential.add_edge(START, "a")
sequential.add_edge("c", END)

# Parallel: START -> {A, B, C} -> END
parallel = StateGraph(TraceState)
for name in ("a", "b", "c"):
    parallel.add_node(name, make_node(name))
    parallel.add_edge(START, name)
    parallel.add_edge(name, END)

for label, builder in [("sequential", sequential), ("parallel", parallel)]:
    start = time.perf_counter()
    result = builder.compile().invoke({"log": []})
    print(f"{label:11} {time.perf_counter() - start:.2f}s total  {result['log']}")

# %% [markdown]
# Same work, one third of the time. No threads, no `asyncio` - just graph shape.

# %% [markdown]
# ## 2. Fan-out, fan-in with real work
#
# A research assistant that gathers three perspectives at once.

# %%
class ResearchState(TypedDict):
    question: str
    findings: Annotated[list[str], operator.add]
    report: str


def perspective(name: str, instruction: str):
    def node(state: ResearchState) -> dict:
        answer = (ChatPromptTemplate.from_template(
            f"{instruction}\n\nQuestion: {{question}}\nAnswer in 2 sentences."
        ) | model | parser).invoke(state)
        return {"findings": [f"[{name}] {answer.strip()}"]}
    return node


def synthesise(state: ResearchState) -> dict:
    joined = "\n\n".join(state["findings"])
    report = (ChatPromptTemplate.from_template(
        "Combine these three perspectives into one 3-sentence recommendation.\n\n{findings}"
    ) | model | parser).invoke({"findings": joined})
    return {"report": report}


builder = StateGraph(ResearchState)
builder.add_node("technical", perspective("technical", "You are a staff engineer. Focus on implementation risk."))
builder.add_node("financial", perspective("financial", "You are a finance partner. Focus on cost and ROI."))
builder.add_node("customer", perspective("customer", "You are head of support. Focus on customer impact."))
builder.add_node("synthesise", synthesise)

for branch in ("technical", "financial", "customer"):
    builder.add_edge(START, branch)
    builder.add_edge(branch, "synthesise")
builder.add_edge("synthesise", END)

research = builder.compile()
print(research.get_graph().draw_ascii())

# %%
QUESTION = "Should we migrate our vector store from FAISS to a managed service?"

start = time.perf_counter()
outcome = research.invoke({"question": QUESTION, "findings": [], "report": ""})
print(f"parallel: {time.perf_counter() - start:.1f}s\n")

for finding in outcome["findings"]:
    print(f"  {finding[:120]}")
print(f"\nreport: {outcome['report'].strip()[:260]}")

# %% [markdown]
# ### The reducer is doing the work
#
# `findings: Annotated[list[str], operator.add]` is what makes three concurrent
# writes to the same key legal. Remove it and the graph raises rather than
# silently losing two thirds of the research - LangGraph refuses to guess.

# %%
class Unsafe(TypedDict):
    findings: list[str]      # no reducer


bad = StateGraph(Unsafe)
bad.add_node("x", lambda s: {"findings": ["x"]})
bad.add_node("y", lambda s: {"findings": ["y"]})
bad.add_edge(START, "x")
bad.add_edge(START, "y")
bad.add_edge("x", END)
bad.add_edge("y", END)
try:
    bad.compile().invoke({"findings": []})
except Exception as exc:
    print(f"{type(exc).__name__}: {str(exc)[:160]}")

# %% [markdown]
# ## 3. `Send` - dynamic fan-out (map-reduce)
#
# The pattern above has a fixed three branches. When the number of branches
# depends on the data - one per document, one per subtopic - use `Send`.

# %%
from langgraph.types import Send


class MapReduceState(TypedDict):
    topic: str
    subtopics: list[str]
    notes: Annotated[list[str], operator.add]
    summary: str


def plan(state: MapReduceState) -> dict:
    raw = (ChatPromptTemplate.from_template(
        "List exactly 4 subtopics to research for: {topic}. One per line, no numbering."
    ) | model | parser).invoke(state)
    subtopics = [line.strip("-* ").strip() for line in raw.splitlines() if line.strip()][:4]
    return {"subtopics": subtopics}


def fan_out(state: MapReduceState) -> list[Send]:
    """One `research` invocation per subtopic, each with its own private input."""
    return [Send("research", {"subtopic": topic}) for topic in state["subtopics"]]


def research_one(state: dict) -> dict:
    """Note the input: what `Send` passed, not the full graph state."""
    note = (ChatPromptTemplate.from_template(
        "Write 2 sentences of substance about: {subtopic}"
    ) | model | parser).invoke(state)
    return {"notes": [f"[{state['subtopic']}] {note.strip()}"]}


def reduce_notes(state: MapReduceState) -> dict:
    summary = (ChatPromptTemplate.from_template(
        "Summarise these research notes into 3 sentences:\n\n{notes}"
    ) | model | parser).invoke({"notes": "\n\n".join(state["notes"])})
    return {"summary": summary}


builder2 = StateGraph(MapReduceState)
builder2.add_node("plan", plan)
builder2.add_node("research", research_one)
builder2.add_node("reduce", reduce_notes)
builder2.add_edge(START, "plan")
builder2.add_conditional_edges("plan", fan_out, ["research"])
builder2.add_edge("research", "reduce")
builder2.add_edge("reduce", END)

map_reduce = builder2.compile()
print(map_reduce.get_graph().draw_ascii())

# %%
start = time.perf_counter()
outcome = map_reduce.invoke({"topic": "reducing our LLM inference costs", "subtopics": [],
                             "notes": [], "summary": ""})
print(f"{len(outcome['subtopics'])} subtopics researched in parallel: {time.perf_counter() - start:.1f}s\n")
for subtopic in outcome["subtopics"]:
    print(f"  - {subtopic}")
print(f"\n{outcome['summary'].strip()[:280]}")

# %% [markdown]
# ### `Send` semantics worth knowing
#
# - The node receives **exactly the dict you send**, not the graph state. That is
#   why `research_one` reads `state["subtopic"]` - a key that is not in
#   `MapReduceState` at all.
# - Its **return value** is merged into the real state through the normal
#   reducers, which is how `notes` accumulates.
# - The count is decided at runtime, so it can be 1 or 100.
# - `add_conditional_edges(source, fn, ["research"])` - the list tells LangGraph
#   which nodes can be targeted, for the diagram.

# %%
print("what the fan-out function returns:")
for send in fan_out({"topic": "x", "subtopics": ["a", "b"], "notes": [], "summary": ""}):
    print(f"   Send(node={send.node!r}, arg={send.arg})")

# %% [markdown]
# ## 4. Parallel branches of different lengths
#
# Fan-in waits for whole **paths**, not just immediate predecessors.

# %%
class UnevenState(TypedDict):
    log: Annotated[list[str], operator.add]


builder3 = StateGraph(UnevenState)
builder3.add_node("quick", make_node("quick", 0.1))
builder3.add_node("slow_1", make_node("slow_1", 0.2))
builder3.add_node("slow_2", make_node("slow_2", 0.2))
builder3.add_node("slow_3", make_node("slow_3", 0.2))
builder3.add_node("merge", lambda s: {"log": [f"merge saw {len(s['log'])} results"]})

builder3.add_edge(START, "quick")
builder3.add_edge(START, "slow_1")
builder3.add_edge("slow_1", "slow_2")
builder3.add_edge("slow_2", "slow_3")
builder3.add_edge("quick", "merge")
builder3.add_edge("slow_3", "merge")
builder3.add_edge("merge", END)

uneven = builder3.compile()
print(uneven.get_graph().draw_ascii())
print(uneven.invoke({"log": []})["log"])

# %% [markdown]
# `merge` ran once, after the three-node chain finished, and saw all four
# results. Watch the supersteps:

# %%
for step, chunk in enumerate(uneven.stream({"log": []}, stream_mode="updates")):
    print(f"  superstep {step}: {list(chunk)}")

# %% [markdown]
# ### `defer=True` - wait even when an edge is ready early
#
# Occasionally a node has an edge that becomes ready before the other branches
# finish, and you want it to wait anyway. `defer=True` postpones the node until
# nothing else is pending.

# %%
class DeferState(TypedDict):
    seen: Annotated[list[str], operator.add]


for defer in (False, True):
    builder4 = StateGraph(DeferState)
    builder4.add_node("fast", lambda s: {"seen": ["fast"]})
    builder4.add_node("slow_a", lambda s: {"seen": ["slow_a"]})
    builder4.add_node("slow_b", lambda s: {"seen": ["slow_b"]})
    builder4.add_node("collect", lambda s: {"seen": [f"collect saw {len(s['seen'])}"]}, defer=defer)
    builder4.add_edge(START, "fast")
    builder4.add_edge(START, "slow_a")
    builder4.add_edge("slow_a", "slow_b")
    builder4.add_edge("fast", "collect")
    builder4.add_edge("slow_b", "collect")
    builder4.add_edge("collect", END)
    print(f"defer={defer!s:5} -> {builder4.compile().invoke({'seen': []})['seen']}")

# %% [markdown]
# ## 5. When one branch fails
#
# By default, an exception in any branch fails the whole superstep. For research
# and enrichment work that is usually wrong - three good results beat zero.

# %%
class ResilientState(TypedDict):
    results: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]


def flaky_source(name: str, fail: bool):
    def node(state: ResilientState) -> dict:
        if fail:
            raise ConnectionError(f"{name} is unreachable")
        return {"results": [f"{name}: ok"]}
    return node


strict = StateGraph(ResilientState)
for name, fail in [("crm", False), ("warehouse", True), ("billing", False)]:
    strict.add_node(name, flaky_source(name, fail))
    strict.add_edge(START, name)
    strict.add_edge(name, END)

try:
    strict.compile().invoke({"results": [], "errors": []})
except Exception as exc:
    print(f"strict graph: {type(exc).__name__}: {exc}")

# %% [markdown]
# ### Isolate failures inside the node

# %%
def resilient_source(name: str, fail: bool):
    def node(state: ResilientState) -> dict:
        try:
            if fail:
                raise ConnectionError(f"{name} is unreachable")
            return {"results": [f"{name}: ok"]}
        except Exception as exc:
            return {"errors": [f"{name}: {type(exc).__name__}: {exc}"]}
    return node


resilient = StateGraph(ResilientState)
resilient.add_node("gather", lambda s: {"results": [f"gathered {len(s['results'])} of 3 sources"]})
for name, fail in [("crm", False), ("warehouse", True), ("billing", False)]:
    resilient.add_node(name, resilient_source(name, fail))
    resilient.add_edge(START, name)
    resilient.add_edge(name, "gather")
resilient.add_edge("gather", END)

outcome = resilient.compile().invoke({"results": [], "errors": []})
print("results:", outcome["results"])
print("errors :", outcome["errors"])

# %% [markdown]
# ### `retry_policy` for genuinely transient failures

# %%
from langgraph.types import RetryPolicy


class RetryState(TypedDict):
    attempts: Annotated[int, operator.add]
    value: str


calls = {"n": 0}


def transient(state: RetryState) -> dict:
    calls["n"] += 1
    if calls["n"] < 3:
        raise ConnectionError("temporary network blip")
    return {"attempts": 1, "value": f"succeeded on attempt {calls['n']}"}


builder5 = StateGraph(RetryState)
builder5.add_node(
    "transient",
    transient,
    retry_policy=RetryPolicy(max_attempts=5, initial_interval=0.05, backoff_factor=2.0,
                             retry_on=ConnectionError),
)
builder5.add_edge(START, "transient")
builder5.add_edge("transient", END)

print(builder5.compile().invoke({"attempts": 0, "value": ""}))

# %% [markdown]
# `retry_on` matters. Retrying a `ValueError` from bad input just wastes time and
# money - retry only what can succeed on a second attempt (timeouts, rate limits,
# connection errors).

# %% [markdown]
# ## 6. Controlling concurrency
#
# Unbounded fan-out will hit provider rate limits. `max_concurrency` caps how
# many branches run at once.

# %%
class WideState(TypedDict):
    done: Annotated[list[int], operator.add]


wide = StateGraph(WideState)
for i in range(8):
    wide.add_node(f"task_{i}", (lambda s, n=i: (time.sleep(0.15), {"done": [n]})[1]))
    wide.add_edge(START, f"task_{i}")
    wide.add_edge(f"task_{i}", END)
wide_graph = wide.compile()

for limit in (2, 4, None):
    config = {"max_concurrency": limit} if limit else {}
    start = time.perf_counter()
    wide_graph.invoke({"done": []}, config=config)
    print(f"max_concurrency={str(limit):4} -> {time.perf_counter() - start:.2f}s")

# %% [markdown]
# Set it deliberately. Your provider's rate limit, not your CPU, is almost
# always the binding constraint.

# %% [markdown]
# ## 7. Parallel RAG - a realistic use
#
# Query three retrievers at once and fuse the results.

# %%
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from shared.llm import get_embeddings

docs: list[Document] = []
for file_name in ("company_handbook.md", "leave_policy.txt", "product_faq.md"):
    text = Path(ctx.data(file_name)).read_text(encoding="utf-8")
    for chunk in RecursiveCharacterTextSplitter(chunk_size=420, chunk_overlap=60).split_text(text):
        docs.append(Document(chunk, metadata={"source": file_name}))

vector_store = FAISS.from_documents(docs, get_embeddings())
dense = vector_store.as_retriever(search_kwargs={"k": 3})
sparse = BM25Retriever.from_documents(docs, k=3)
mmr = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 10})


class RagState(TypedDict):
    question: str
    candidates: Annotated[list[tuple[str, str]], operator.add]
    answer: str


def retriever_node(name: str, retriever):
    def node(state: RagState) -> dict:
        found = retriever.invoke(state["question"])
        return {"candidates": [(name, d.page_content) for d in found]}
    return node


def fuse_and_answer(state: RagState) -> dict:
    seen, unique = set(), []
    for source, content in state["candidates"]:
        key = content[:80]
        if key not in seen:
            seen.add(key)
            unique.append(content)
    context = "\n\n".join(unique[:6])
    answer = (ChatPromptTemplate.from_template(
        "Answer using only this context. If it is not there, say so.\n\n{context}\n\nQ: {question}"
    ) | model | parser).invoke({"context": context, "question": state["question"]})
    return {"answer": answer}


builder6 = StateGraph(RagState)
builder6.add_node("dense", retriever_node("dense", dense))
builder6.add_node("sparse", retriever_node("sparse", sparse))
builder6.add_node("mmr", retriever_node("mmr", mmr))
builder6.add_node("fuse", fuse_and_answer)
for name in ("dense", "sparse", "mmr"):
    builder6.add_edge(START, name)
    builder6.add_edge(name, "fuse")
builder6.add_edge("fuse", END)
parallel_rag = builder6.compile()

start = time.perf_counter()
outcome = parallel_rag.invoke({"question": "How many days of casual leave and what is the notice period for L5?",
                               "candidates": [], "answer": ""})
print(f"3 retrievers in {time.perf_counter() - start:.2f}s, "
      f"{len(outcome['candidates'])} candidates -> "
      f"{len({c[1][:80] for c in outcome['candidates']})} unique\n")
print(outcome["answer"].strip()[:280])

# %% [markdown]
# ## 8. Parallelism pitfalls
#
# | Pitfall | Symptom | Fix |
# |---|---|---|
# | No reducer on a shared key | `InvalidUpdateError` at runtime | Add `Annotated[T, reducer]` |
# | Branch reads another branch's write | Sees stale or missing data | They are in the same superstep - sequence them |
# | One branch fails, all fail | Whole run raises | `try/except` inside the node, or `retry_policy` |
# | Rate limit errors under load | 429s from the provider | `max_concurrency` |
# | Parallel is slower than sequential | Overhead exceeds the work | Only parallelise real I/O-bound work |
#
# The second one deserves a demonstration, because it looks like a bug in
# LangGraph until you understand supersteps.

# %%
class VisibilityState(TypedDict):
    a: str
    b: str
    log: Annotated[list[str], operator.add]


def writes_a(state: VisibilityState) -> dict:
    return {"a": "value-from-a"}


def reads_a(state: VisibilityState) -> dict:
    return {"b": "read", "log": [f"reads_a saw a={state['a']!r}"]}


same_step = StateGraph(VisibilityState)
same_step.add_node("writes_a", writes_a)
same_step.add_node("reads_a", reads_a)
same_step.add_edge(START, "writes_a")
same_step.add_edge(START, "reads_a")          # same superstep
same_step.add_edge("writes_a", END)
same_step.add_edge("reads_a", END)
print("parallel  ->", same_step.compile().invoke({"a": "", "b": "", "log": []})["log"])

sequenced = StateGraph(VisibilityState)
sequenced.add_node("writes_a", writes_a)
sequenced.add_node("reads_a", reads_a)
sequenced.add_edge(START, "writes_a")
sequenced.add_edge("writes_a", "reads_a")     # next superstep
sequenced.add_edge("reads_a", END)
print("sequenced ->", sequenced.compile().invoke({"a": "", "b": "", "log": []})["log"])

# %% [markdown]
# ## Try it yourself
#
# 1. **Parallelise something real.** Take the RAG chain from notebook 14 and run
#    retrieval and query-expansion concurrently. Measure the saving.
# 2. **Map-reduce over files.** Use `Send` to summarise each file in
#    `shared/sample_data`, then produce one combined summary.
# 3. **Partial-failure budget.** Extend the resilient example to proceed only if
#    at least two of three sources succeed, and fail loudly otherwise.
# 4. **Find the crossover.** With nodes that sleep 10 ms, at what branch count
#    does graph overhead outweigh the parallelism benefit?

# %% [markdown]
# ## Recap
#
# | Concept | Takeaway |
# |---|---|
# | Supersteps | All ready nodes run together; writes merge; then the next step |
# | Fan-out | Several edges from one node run its targets in parallel |
# | Fan-in | Several edges into a node make it wait for all paths |
# | Reducers | Mandatory for keys written concurrently; no reducer means an error |
# | `Send(node, arg)` | Runtime-sized fan-out; the node receives **only** `arg` |
# | `defer=True` | Postpone a node until nothing else is pending |
# | Failure isolation | `try/except` in the node keeps sibling branches alive |
# | `RetryPolicy(retry_on=...)` | Retry only what can succeed on a second attempt |
# | `max_concurrency` | Your provider's rate limit is the real constraint |
# | Same-superstep visibility | Parallel nodes cannot see each other's writes |
#
# ## Next
#
# -> [39_time_travel.ipynb](../07-langgraph-advanced/39_time_travel.ipynb)
