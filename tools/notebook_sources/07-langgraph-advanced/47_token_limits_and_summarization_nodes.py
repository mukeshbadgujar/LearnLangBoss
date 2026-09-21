# %% [markdown]
# # 47 - Token Limits and Summarisation Nodes
#
# | | |
# |---|---|
# | **Level** | Advanced (LangGraph) |
# | **Time** | 45 minutes |
# | **Prerequisites** | `46_deployment_and_versioning`, `37_message_history_and_deletion` |
# | **Checklist ID** | `47_token_limits_and_summarization_nodes` |
#
# ## Why this matters
#
# Notebook 37 managed conversation history. This notebook is about the harder
# case: **content that is larger than the context window no matter how you trim
# it.** A 200-page contract, a 90-minute meeting transcript, 500 support tickets.
#
# You cannot fit it, so you have to choose a compression strategy - and the
# strategies differ enormously in cost, latency and what they lose. Picking the
# wrong one is why some document-summarisation features cost $4 per document and
# others cost four cents.

# %%
from pathlib import Path
import sys

_root = Path.cwd()
while not (_root / "shared").is_dir() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from shared.notebook_setup import setup, require  # noqa: E402

ctx = setup("47_token_limits_and_summarization_nodes")

# %%
import operator
import time
from typing import Annotated, Literal, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from shared.llm import get_chat_model

model = get_chat_model()
parser = StrOutputParser()

# %% [markdown]
# ## 1. Know your budget
#
# Context window is not the same as usable budget. Subtract the system prompt,
# tool schemas, and room for the answer.

# %%
def token_budget(context_window: int, system_prompt: str, tool_schema_tokens: int = 0,
                 reserve_output: int = 1500, safety_margin: int = 200) -> dict:
    system_tokens = model.get_num_tokens(system_prompt)
    usable = context_window - system_tokens - tool_schema_tokens - reserve_output - safety_margin
    return {
        "context_window": context_window,
        "system_prompt": system_tokens,
        "tool_schemas": tool_schema_tokens,
        "reserved_for_output": reserve_output,
        "safety_margin": safety_margin,
        "usable_for_content": usable,
    }


SYSTEM = ("You are Northwind's contract analyst. Answer only from the provided text, "
          "cite the clause number, and never speculate about terms that are not present.")

for window in (8_192, 32_768, 128_000):
    budget = token_budget(window, SYSTEM, tool_schema_tokens=600)
    print(f"{window:>7,} window -> {budget['usable_for_content']:>7,} usable "
          f"({budget['usable_for_content'] / window:.0%})")

# %% [markdown]
# On an 8k model with a few tools, you have roughly 5,900 tokens for content -
# about 12 pages. That is the real constraint, not "8k".

# %% [markdown]
# ## 2. A document that does not fit

# %%
def build_long_document() -> str:
    """A synthetic contract long enough to exceed a small context window."""
    clauses = []
    topics = [
        ("Term and Renewal", "This Agreement commences on the Effective Date and continues for 24 months, "
         "renewing automatically for successive 12-month terms unless either party gives 60 days notice."),
        ("Fees", "Customer shall pay INR 18,40,000 annually, invoiced quarterly in advance. "
         "Late payments accrue interest at 1.5% per month."),
        ("Service Levels", "Provider guarantees 99.9% monthly uptime. Credits of 10% of monthly fees "
         "apply for each full hour of downtime beyond the threshold, capped at 50% of monthly fees."),
        ("Data Protection", "Customer data is stored in ap-south-1 by default. Provider shall not transfer "
         "personal data outside the agreed region without prior written consent."),
        ("Liability", "Aggregate liability is limited to fees paid in the preceding 12 months, except for "
         "breaches of confidentiality or data protection obligations."),
        ("Termination", "Either party may terminate for material breach on 30 days written notice if the "
         "breach remains uncured. Customer data is retained for 30 days post-termination."),
        ("Support", "Enterprise customers receive 1-hour response for P1 incidents, 4 hours for P2, "
         "and next business day for P3."),
        ("Intellectual Property", "Each party retains ownership of its pre-existing intellectual property. "
         "Customer grants Provider a licence to process Customer Data solely to deliver the Services."),
    ]
    for index in range(1, 41):
        title, body = topics[(index - 1) % len(topics)]
        clauses.append(
            f"Clause {index}: {title} (Schedule {chr(65 + index % 5)})\n{body} "
            f"For the purposes of this clause, references to 'Business Day' mean any day other than a "
            f"Saturday, Sunday or public holiday in Maharashtra, India. This clause {index} shall survive "
            f"termination to the extent necessary to give effect to its provisions."
        )
    return "\n\n".join(clauses)


CONTRACT = build_long_document()
contract_tokens = model.get_num_tokens(CONTRACT)
print(f"document: {len(CONTRACT):,} characters, ~{contract_tokens:,} tokens")
print(f"fits in an 8k window? {contract_tokens < token_budget(8192, SYSTEM)['usable_for_content']}")

# %% [markdown]
# ## 3. Strategy A - stuff (when it fits)
#
# One call, full context, best quality. Always try this first; only reach for the
# others when it genuinely does not fit.

# %%
def stuff_summarise(text: str, question: str) -> str:
    return (ChatPromptTemplate.from_messages([
        ("system", SYSTEM),
        ("human", "Document:\n{text}\n\nQuestion: {question}"),
    ]) | model | parser).invoke({"text": text, "question": question})


SHORT = "\n\n".join(CONTRACT.split("\n\n")[:6])
print(f"short extract: ~{model.get_num_tokens(SHORT)} tokens")
print(stuff_summarise(SHORT, "What is the notice period for non-renewal?").strip()[:200])

# %% [markdown]
# ## 4. Strategy B - map-reduce
#
# Summarise each chunk independently (parallel), then summarise the summaries.
# Fast, because the map step fans out. Loses cross-chunk connections.

# %%
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.types import Send


class MapReduceState(TypedDict):
    document: str
    question: str
    chunks: list[str]
    chunk_summaries: Annotated[list[str], operator.add]
    answer: str


def split_document(state: MapReduceState) -> dict:
    chunks = RecursiveCharacterTextSplitter(chunk_size=2200, chunk_overlap=200).split_text(state["document"])
    return {"chunks": chunks}


def fan_out_chunks(state: MapReduceState) -> list[Send]:
    return [
        Send("summarise_chunk", {"chunk": chunk, "index": i, "question": state["question"]})
        for i, chunk in enumerate(state["chunks"])
    ]


def summarise_chunk(state: dict) -> dict:
    summary = (ChatPromptTemplate.from_template(
        "Extract only the facts relevant to: {question}\n\nText:\n{chunk}\n\n"
        "If nothing is relevant, reply exactly: NOTHING RELEVANT."
    ) | model | parser).invoke(state)
    if "NOTHING RELEVANT" in summary.upper():
        return {"chunk_summaries": []}
    return {"chunk_summaries": [f"[chunk {state['index']}] {summary.strip()}"]}


def reduce_summaries(state: MapReduceState) -> dict:
    answer = (ChatPromptTemplate.from_messages([
        ("system", SYSTEM),
        ("human", "Extracted facts:\n{facts}\n\nQuestion: {question}"),
    ]) | model | parser).invoke({"facts": "\n\n".join(state["chunk_summaries"]), "question": state["question"]})
    return {"answer": answer}


builder = StateGraph(MapReduceState)
builder.add_node("split", split_document)
builder.add_node("summarise_chunk", summarise_chunk)
builder.add_node("reduce", reduce_summaries)
builder.add_edge(START, "split")
builder.add_conditional_edges("split", fan_out_chunks, ["summarise_chunk"])
builder.add_edge("summarise_chunk", "reduce")
builder.add_edge("reduce", END)
map_reduce = builder.compile()

print(map_reduce.get_graph().draw_ascii())

# %%
QUESTION = "What are the service level commitments and what credits apply if they are missed?"

start = time.perf_counter()
outcome = map_reduce.invoke({"document": CONTRACT, "question": QUESTION, "chunks": [],
                             "chunk_summaries": [], "answer": ""})
map_reduce_seconds = time.perf_counter() - start

print(f"{len(outcome['chunks'])} chunks -> {len(outcome['chunk_summaries'])} relevant, "
      f"{map_reduce_seconds:.1f}s")
print(f"\n{outcome['answer'].strip()[:320]}")

# %% [markdown]
# Note the `NOTHING RELEVANT` filter. Without it, irrelevant chunks contribute
# noise to the reduce step and the answer gets worse as the document gets longer.

# %% [markdown]
# ## 5. Strategy C - refine
#
# Carry a running answer through the chunks in order. Preserves narrative and
# cross-chunk reasoning, but it is strictly sequential and therefore slow.

# %%
class RefineState(TypedDict):
    chunks: list[str]
    question: str
    index: int
    running: str
    steps: Annotated[int, operator.add]


def prepare(state: RefineState) -> dict:
    chunks = RecursiveCharacterTextSplitter(chunk_size=2200, chunk_overlap=200).split_text(CONTRACT)
    return {"chunks": chunks, "index": 0, "running": ""}


def refine_step(state: RefineState) -> dict:
    chunk = state["chunks"][state["index"]]
    template = (
        "Answer this question from the text.\n\nQuestion: {question}\n\nText:\n{chunk}"
        if not state["running"] else
        "Refine the existing answer using new text. Keep it accurate and do not lose earlier facts.\n\n"
        "Question: {question}\n\nExisting answer:\n{running}\n\nNew text:\n{chunk}"
    )
    running = (ChatPromptTemplate.from_template(template) | model | parser).invoke(
        {"question": state["question"], "running": state["running"], "chunk": chunk}
    )
    return {"running": running, "index": state["index"] + 1, "steps": 1}


def more_chunks(state: RefineState) -> Literal["refine", "__end__"]:
    return "refine" if state["index"] < len(state["chunks"]) else END


builder2 = StateGraph(RefineState)
builder2.add_node("prepare", prepare)
builder2.add_node("refine", refine_step)
builder2.add_edge(START, "prepare")
builder2.add_edge("prepare", "refine")
builder2.add_conditional_edges("refine", more_chunks, {"refine": "refine", END: END})
refine_graph = builder2.compile()

start = time.perf_counter()
refined = refine_graph.invoke({"chunks": [], "question": QUESTION, "index": 0, "running": "", "steps": 0},
                              {"recursion_limit": 60})
refine_seconds = time.perf_counter() - start
print(f"{refined['steps']} sequential steps, {refine_seconds:.1f}s")
print(f"\n{refined['running'].strip()[:320]}")

# %% [markdown]
# ## 6. Strategy D - retrieve then stuff
#
# Do not summarise the whole document at all. Index it, retrieve only the
# relevant parts, and stuff those. Usually the cheapest and the best, provided
# the question is specific.

# %%
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from shared.llm import get_embeddings


class RetrieveState(TypedDict):
    document: str
    question: str
    retrieved: list[str]
    answer: str


_index_cache: dict[str, FAISS] = {}


def index_document(state: RetrieveState) -> dict:
    key = str(hash(state["document"]))
    if key not in _index_cache:
        chunks = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120).split_text(state["document"])
        _index_cache[key] = FAISS.from_documents(
            [Document(c, metadata={"chunk": i}) for i, c in enumerate(chunks)], get_embeddings()
        )
    return {}


def retrieve(state: RetrieveState) -> dict:
    store = _index_cache[str(hash(state["document"]))]
    found = store.similarity_search(state["question"], k=4)
    return {"retrieved": [d.page_content for d in found]}


def answer_from_retrieved(state: RetrieveState) -> dict:
    answer = (ChatPromptTemplate.from_messages([
        ("system", SYSTEM),
        ("human", "Relevant clauses:\n{context}\n\nQuestion: {question}"),
    ]) | model | parser).invoke({"context": "\n\n".join(state["retrieved"]), "question": state["question"]})
    return {"answer": answer}


builder3 = StateGraph(RetrieveState)
builder3.add_sequence([("index", index_document), ("retrieve", retrieve), ("answer", answer_from_retrieved)])
builder3.add_edge(START, "index")
builder3.add_edge("answer", END)
retrieve_graph = builder3.compile()

start = time.perf_counter()
retrieved = retrieve_graph.invoke({"document": CONTRACT, "question": QUESTION, "retrieved": [], "answer": ""})
retrieve_seconds = time.perf_counter() - start
print(f"{len(retrieved['retrieved'])} clauses retrieved, {retrieve_seconds:.1f}s")
print(f"\n{retrieved['answer'].strip()[:320]}")

# %% [markdown]
# ## 7. Comparing the strategies

# %%
from langchain_core.callbacks import BaseCallbackHandler


class Meter(BaseCallbackHandler):
    def __init__(self):
        self.calls = self.input_tokens = self.output_tokens = 0

    def on_llm_end(self, response, **kwargs):
        self.calls += 1
        try:
            usage = response.generations[0][0].message.usage_metadata or {}
        except (AttributeError, IndexError):
            return
        self.input_tokens += usage.get("input_tokens", 0)
        self.output_tokens += usage.get("output_tokens", 0)


RUNS = {
    "map-reduce": (map_reduce, {"document": CONTRACT, "question": QUESTION, "chunks": [],
                                "chunk_summaries": [], "answer": ""}, {}),
    "refine": (refine_graph, {"chunks": [], "question": QUESTION, "index": 0, "running": "", "steps": 0},
               {"recursion_limit": 60}),
    "retrieve+stuff": (retrieve_graph, {"document": CONTRACT, "question": QUESTION,
                                        "retrieved": [], "answer": ""}, {}),
}

print(f"{'strategy':16} {'calls':>6} {'in tokens':>10} {'out':>7} {'seconds':>8}")
for label, (graph, payload, extra) in RUNS.items():
    meter = Meter()
    start = time.perf_counter()
    graph.invoke(payload, {**extra, "callbacks": [meter]})
    print(f"{label:16} {meter.calls:>6} {meter.input_tokens:>10,} {meter.output_tokens:>7,} "
          f"{time.perf_counter() - start:>8.1f}")

# %% [markdown]
# | Strategy | Calls | Parallel | Cross-chunk reasoning | Use when |
# |---|---|---|---|---|
# | **Stuff** | 1 | n/a | Perfect | It fits - always try first |
# | **Map-reduce** | n+1 | Yes | Poor | Broad summary of a large corpus |
# | **Refine** | n | No | Good | Narrative order matters (transcripts) |
# | **Retrieve + stuff** | 1 (+ embedding) | n/a | Only within retrieved chunks | **Specific questions - usually the answer** |
#
# The instinct to reach for map-reduce is usually wrong. If the question is
# specific, retrieval beats it on cost, latency and quality simultaneously.
# Map-reduce earns its keep for "summarise this whole thing".

# %% [markdown]
# ## 8. Summarisation nodes in a conversation
#
# Back to chat. The robust pattern is a summarisation node that fires on a
# **token** threshold, not a message count.

# %%
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.graph.message import add_messages


class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
    summary: str
    compressions: Annotated[int, operator.add]


TRIGGER_TOKENS = 500
KEEP_RECENT = 4


def count_tokens(messages) -> int:
    text = "\n".join(m.content for m in messages if isinstance(m.content, str))
    return model.get_num_tokens(text)


def chat(state: ChatState) -> dict:
    system = "You are a concise assistant. Two sentences maximum."
    if state.get("summary"):
        system += f"\n\nContext from earlier in this conversation:\n{state['summary']}"
    return {"messages": [model.invoke([("system", system)] + state["messages"])]}


def maybe_compress(state: ChatState) -> dict:
    messages = state["messages"]
    if count_tokens(messages) < TRIGGER_TOKENS or len(messages) <= KEEP_RECENT:
        return {}

    older, _recent = messages[:-KEEP_RECENT], messages[-KEEP_RECENT:]
    transcript = "\n".join(f"{m.__class__.__name__}: {m.content}" for m in older if isinstance(m.content, str))
    instruction = (
        f"Existing summary:\n{state['summary']}\n\nIncorporate this newer transcript, "
        f"keeping every fact about the user and their situation:\n{transcript}"
        if state.get("summary") else
        f"Summarise in 3 sentences, keeping every fact about the user:\n{transcript}"
    )
    summary = model.invoke(instruction).content

    return {"summary": summary,
            "messages": [RemoveMessage(id=m.id) for m in older],
            "compressions": 1}


def needs_compression(state: ChatState) -> Literal["compress", "__end__"]:
    return "compress" if count_tokens(state["messages"]) >= TRIGGER_TOKENS else END


builder4 = StateGraph(ChatState)
builder4.add_node("chat", chat)
builder4.add_node("compress", maybe_compress)
builder4.add_edge(START, "chat")
builder4.add_conditional_edges("chat", needs_compression, {"compress": "compress", END: END})
builder4.add_edge("compress", END)
chat_graph = builder4.compile(checkpointer=InMemorySaver())

print(chat_graph.get_graph().draw_ascii())

# %%
config = {"configurable": {"thread_id": "long-chat"}}
TURNS = [
    "My name is Mukesh and I lead the Kestrel platform team.",
    "We are evaluating whether to move from FAISS to a managed vector store.",
    "Our corpus is about 2 million chunks and grows 5% a month.",
    "Latency matters more than cost for us.",
    "We are on AWS in ap-south-1.",
    "What would you recommend?",
    "How long would a migration take?",
    "What is the main risk?",
]

print(f"{'turn':>4} {'msgs':>5} {'tokens':>7} {'compressions':>13}")
for turn, text in enumerate(TURNS, start=1):
    chat_graph.invoke({"messages": [HumanMessage(text)], "summary": "", "compressions": 0}, config)
    state = chat_graph.get_state(config).values
    print(f"{turn:>4} {len(state['messages']):>5} {count_tokens(state['messages']):>7} "
          f"{state['compressions']:>13}")

# %%
answer = chat_graph.invoke({"messages": [HumanMessage("What is my team called and which region are we in?")]},
                           config)
print("state holds only recent messages plus a summary, yet:")
print(" ", answer["messages"][-1].content.strip()[:180])
print(f"\nsummary in state:\n  {chat_graph.get_state(config).values['summary'][:260]}")

# %% [markdown]
# ## 9. What summarisation loses
#
# Compression is lossy by definition. Know what you are throwing away.
#
# | Reliably survives | Reliably lost |
# |---|---|
# | Names, entities, stated preferences | Exact wording and tone |
# | Decisions and their stated reasons | Which turn something was said in |
# | Numbers, if you ask for them explicitly | Numbers, if you do not |
# | The overall trajectory | Ideas that were raised and dropped |
#
# **Ask for what you need.** A generic "summarise this" loses numbers; a prompt
# that says "keep every figure, date and name verbatim" mostly does not.

# %%
TRANSCRIPT = """Mukesh: Our p95 is 840ms and the SLO is 500ms.
Priya: The index rebuild takes 4 hours and we do it nightly at 02:00 IST.
Mukesh: Budget is INR 12 lakh for the year. Deadline is 31 March.
Priya: We could shard into 4 partitions, which would cut rebuild time to about 70 minutes."""

for label, instruction in [
    ("generic", "Summarise this in two sentences:\n\n{t}"),
    ("fact-preserving", "Summarise in two sentences. Preserve every number, date and name exactly:\n\n{t}"),
]:
    summary = (ChatPromptTemplate.from_template(instruction) | model | parser).invoke({"t": TRANSCRIPT})
    facts = ["840", "500", "4 hour", "02:00", "12 lakh", "31 March", "70"]
    kept = [f for f in facts if f.lower() in summary.lower()]
    print(f"\n{label}: kept {len(kept)}/{len(facts)} facts {kept}")
    print(f"  {summary.strip()[:200]}")

# %% [markdown]
# ## 10. Choosing
#
# ```
# Does the content fit the budget?
#   yes -> STUFF
#   no  -> Is the question specific?
#            yes -> RETRIEVE + STUFF
#            no  -> Does order/narrative matter?
#                     yes -> REFINE
#                     no  -> MAP-REDUCE
#
# Conversation growing?
#   -> trim for the model (37), add a token-triggered summarisation node when
#      users notice forgetting
# ```

# %%
def choose_strategy(content_tokens: int, budget: int, question_is_specific: bool,
                    order_matters: bool) -> str:
    if content_tokens <= budget:
        return "stuff"
    if question_is_specific:
        return "retrieve + stuff"
    return "refine" if order_matters else "map-reduce"


budget = token_budget(8192, SYSTEM)["usable_for_content"]
for description, tokens, specific, ordered in [
    ("short policy, specific question", 900, True, False),
    ("long contract, specific question", contract_tokens, True, False),
    ("long contract, 'summarise it all'", contract_tokens, False, False),
    ("90-minute transcript, 'what was decided'", 40_000, False, True),
]:
    print(f"{description:42} -> {choose_strategy(tokens, budget, specific, ordered)}")

# %% [markdown]
# ## Try it yourself
#
# 1. **Measure quality, not just cost.** Write 5 questions with known answers in
#    the contract and score each strategy with notebook 26's harness.
# 2. **Hierarchical map-reduce.** For a corpus too large even for the reduce
#    step, summarise summaries in a second tier.
# 3. **Adaptive chunking.** Split the contract on `Clause N:` boundaries instead
#    of by character count, and see whether map-reduce improves.
# 4. **Fact-retention test.** Compress a conversation 5 times in a row and track
#    how many of 10 seeded facts survive each round.

# %% [markdown]
# ## Recap
#
# | Concept | Takeaway |
# |---|---|
# | Usable budget | Window minus system prompt, tool schemas and output reserve |
# | Stuff | One call, best quality - always try first |
# | Map-reduce | `Send` fan-out; fast; poor cross-chunk reasoning |
# | `NOTHING RELEVANT` filter | Stops irrelevant chunks polluting the reduce step |
# | Refine | Sequential, preserves narrative, slowest |
# | **Retrieve + stuff** | Usually the best answer for a specific question |
# | Token-triggered compression | Trigger on tokens, not message count |
# | Rolling summary | Extend the existing summary rather than re-summarising |
# | Lossy by design | Numbers and dates disappear unless you demand them |
# | Choosing | Fits? Specific question? Order matters? - in that order |
#
# ## Next
#
# -> [48_self_reflective_rag.ipynb](48_self_reflective_rag.ipynb)
