# %% [markdown]
# # 07 - LangChain Expression Language (LCEL)
#
# | | |
# |---|---|
# | **Level** | Intermediate |
# | **Time** | 60 minutes |
# | **Prerequisites** | `05_output_parsers` |
# | **Checklist ID** | `07_lcel` |
#
# ## Why this matters
#
# This is the most important notebook in Track 01.
#
# LCEL is the composition layer underneath everything else in LangChain. Once you
# understand that *every* component - prompts, models, parsers, retrievers, your
# own functions, even whole agents - implements the same `Runnable` interface, the
# rest of the framework stops looking like a hundred unrelated classes and starts
# looking like Lego.
#
# You get four things for free when you compose with `|`:
#
# 1. **`invoke` / `batch` / `stream`** on every composite, with no extra code
# 2. **Async variants** (`ainvoke`, `abatch`, `astream`) automatically
# 3. **Parallelism** where the graph allows it
# 4. **Tracing** - LangSmith sees each step as a named span

# %%
from pathlib import Path
import sys

_root = Path.cwd()
while not (_root / "shared").is_dir() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from shared.notebook_setup import setup, require  # noqa: E402

ctx = setup("07_lcel")

# %%
from shared.llm import get_chat_model, get_embeddings

model = get_chat_model()

# %% [markdown]
# ## 1. The `Runnable` contract
#
# A Runnable is anything with these methods:
#
# | Method | Input -> Output | Notes |
# |---|---|---|
# | `invoke(x)` | one -> one | The core method |
# | `batch([x, y])` | many -> many | Runs concurrently |
# | `stream(x)` | one -> iterator | Incremental output |
# | `ainvoke` / `abatch` / `astream` | async versions | Free if you implement `invoke` |
# | `astream_events(x)` | async iterator of typed events | Step-level streaming (notebook 21) |
#
# Check it yourself:

# %%
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

prompt = ChatPromptTemplate.from_template("Explain {topic} in one sentence.")
parser = StrOutputParser()

for component in (prompt, model, parser):
    print(f"{type(component).__name__:22} is Runnable: {isinstance(component, Runnable)}")

# %% [markdown]
# ## 2. The pipe operator
#
# `a | b` means "run `a`, feed its output into `b`". The result is itself a
# Runnable, so chains compose endlessly.

# %%
chain = prompt | model | parser

print(type(chain).__name__)
print(chain.invoke({"topic": "vector embeddings"}))

# %% [markdown]
# ### Data flowing through the pipe
#
# Build it up one stage at a time and print the intermediate type. This single
# exercise clears up most LCEL confusion.

# %%
step_input = {"topic": "idempotency keys"}

after_prompt = prompt.invoke(step_input)
print("1. dict          ->", type(after_prompt).__name__)

after_model = model.invoke(after_prompt)
print("2. ChatPromptValue ->", type(after_model).__name__)

after_parser = parser.invoke(after_model)
print("3. AIMessage     ->", type(after_parser).__name__)
print("\nfinal:", after_parser)

# %% [markdown]
# **Type mismatches are the #1 LCEL bug.** A model expects messages or a string,
# not a dict. A parser expects a message, not a dict. When a chain fails, print
# the intermediate types exactly like this.

# %% [markdown]
# ## 3. `invoke`, `batch`, `stream` on the composite

# %%
import time

topics = [{"topic": t} for t in ["RAG", "tool calling", "checkpointing", "reranking"]]

start = time.perf_counter()
results = chain.batch(topics)
print(f"batch of {len(topics)} in {time.perf_counter() - start:.2f}s\n")
for topic, text in zip(topics, results):
    print(f"  {topic['topic']:14} {text.strip()[:90]}")

# %%
print("streaming:\n")
for piece in chain.stream({"topic": "why idempotency matters in webhooks"}):
    print(piece, end="", flush=True)
print()

# %%
# Control concurrency so you do not trip provider rate limits.
many = [{"topic": f"database index type {i}"} for i in range(6)]
limited = chain.batch(many, config={"max_concurrency": 2})
print(f"{len(limited)} results with at most 2 in flight at a time")

# %% [markdown]
# ## 4. `RunnablePassthrough`: keep the input around
#
# Chains lose the original input by default - each stage only sees the previous
# output. `RunnablePassthrough` forwards it unchanged.

# %%
from langchain_core.runnables import RunnablePassthrough

print(RunnablePassthrough().invoke({"question": "anything"}))

# %% [markdown]
# The real use is inside a dict, building the input a prompt needs:

# %%
policy_text = Path(ctx.data("leave_policy.txt")).read_text(encoding="utf-8")


def fetch_context(question: str) -> str:
    """Stand-in for a retriever - notebook 13 replaces this with the real thing."""
    sections = policy_text.split("\n\n")
    words = {w.lower().strip("?.,") for w in question.split()}
    scored = sorted(sections, key=lambda s: -len(words & {w.lower() for w in s.split()}))
    return "\n\n".join(scored[:2])


rag_prompt = ChatPromptTemplate.from_template(
    "Answer using only this policy context.\n\nContext:\n{context}\n\nQuestion: {question}"
)

rag_chain = (
    {"context": RunnablePassthrough() | fetch_context, "question": RunnablePassthrough()}
    | rag_prompt
    | model
    | parser
)

print(rag_chain.invoke("How many sick leave days do I get and when is a certificate needed?"))

# %% [markdown]
# That dict literal is the idiom to internalise. `{"a": runnable_x, "b": runnable_y}`
# is automatically converted into a `RunnableParallel`: both branches receive the
# same input, run concurrently, and the results are merged into one dict.

# %% [markdown]
# ### `assign`: add a key without losing the others

# %%
enrich = RunnablePassthrough.assign(
    context=lambda x: fetch_context(x["question"]),
    word_count=lambda x: len(x["question"].split()),
)

print(enrich.invoke({"question": "Can I carry forward unused leave?", "user": "E-102"}).keys())

pipeline = enrich | rag_prompt | model | parser
print("\n", pipeline.invoke({"question": "Can I carry forward unused leave?", "user": "E-102"}))

# %% [markdown]
# ## 5. `RunnableParallel`: fan out, then merge
#
# Independent work should run at the same time. `RunnableParallel` (or a plain
# dict) does that.

# %%
from langchain_core.runnables import RunnableParallel

ticket = "Everline Bank reports API p99 latency of 4 seconds during EU business hours on Enterprise."

summary_chain = ChatPromptTemplate.from_template("One-line summary: {text}") | model | parser
category_chain = ChatPromptTemplate.from_template(
    "One word category (billing/integration/performance/bug/security/howto): {text}"
) | model | parser
priority_chain = ChatPromptTemplate.from_template(
    "One word priority (low/medium/high/critical): {text}"
) | model | parser

analyse = RunnableParallel(
    summary=summary_chain,
    category=category_chain,
    priority=priority_chain,
)

start = time.perf_counter()
parallel_result = analyse.invoke({"text": ticket})
parallel_time = time.perf_counter() - start

start = time.perf_counter()
_ = {
    "summary": summary_chain.invoke({"text": ticket}),
    "category": category_chain.invoke({"text": ticket}),
    "priority": priority_chain.invoke({"text": ticket}),
}
serial_time = time.perf_counter() - start

for key, value in parallel_result.items():
    print(f"{key:9}: {value.strip()}")
print(f"\nparallel {parallel_time:.2f}s vs serial {serial_time:.2f}s "
      f"({serial_time / max(parallel_time, 0.01):.1f}x)")

# %% [markdown]
# ## 6. `RunnableLambda`: your own functions in a chain
#
# Any callable can join the pipeline. Plain functions are auto-wrapped when they
# appear in a pipe; wrap explicitly when you want a name in traces or need to pass
# config.

# %%
from langchain_core.runnables import RunnableLambda


def redact_pii(text: str) -> str:
    """Strip email addresses before anything is sent to a model."""
    import re

    return re.sub(r"[\w.+-]+@[\w-]+\.[\w.]+", "[EMAIL REDACTED]", text)


def enforce_length(text: str) -> str:
    return text if len(text) <= 300 else text[:297] + "..."


safe_chain = (
    RunnableLambda(redact_pii)
    | ChatPromptTemplate.from_template("Summarise this support message in one line:\n{text}")
    | model
    | parser
    | RunnableLambda(enforce_length)
)

print(safe_chain.invoke(
    "Please contact me at priya.nair@cedarlogistics.example about the CSV export bug; "
    "my colleague arjun@cedarlogistics.example is also affected."
))

# %% [markdown]
# Note the redaction happened **before** the model call - the email never left
# the process. That is a real compliance pattern, not a toy example.

# %% [markdown]
# ## 7. `RunnableBranch`: conditional routing
#
# Full treatment in notebook 25; here is the primitive.

# %%
from langchain_core.runnables import RunnableBranch

billing_chain = ChatPromptTemplate.from_template(
    "You are a billing specialist. Answer briefly: {question}"
) | model | parser
security_chain = ChatPromptTemplate.from_template(
    "You are a security officer. Answer briefly and cautiously: {question}"
) | model | parser
general_chain = ChatPromptTemplate.from_template(
    "You are a general support agent. Answer briefly: {question}"
) | model | parser

router = RunnableBranch(
    (lambda x: any(w in x["question"].lower() for w in ["invoice", "charge", "refund", "billing"]), billing_chain),
    (lambda x: any(w in x["question"].lower() for w in ["soc2", "gdpr", "security", "residency"]), security_chain),
    general_chain,  # default - always last
)

for question in [
    "We were charged twice for January.",
    "Where is our data stored for GDPR purposes?",
    "How do I rename a dashboard?",
]:
    print(f"Q: {question}\nA: {router.invoke({'question': question}).strip()}\n")

# %% [markdown]
# ## 8. Fallbacks: chains that survive failure
#
# Providers rate-limit, time out and have outages. `with_fallbacks` gives you a
# second chance without try/except scattered through your code.

# %%
from shared.llm import available_providers

primary = model
fallback_models = [get_chat_model(provider=p) for p in available_providers()[1:]]

resilient_model = primary.with_fallbacks(fallback_models) if fallback_models else primary
print(f"Primary plus {len(fallback_models)} fallback provider(s) configured")

resilient_chain = prompt | resilient_model | parser
print(resilient_chain.invoke({"topic": "graceful degradation"}))

# %%
# Prove the mechanism with a deliberately broken first option.
from langchain_core.runnables import RunnableLambda


def always_fails(_):
    raise RuntimeError("simulated provider outage")


guarded = RunnableLambda(always_fails).with_fallbacks(
    [RunnableLambda(lambda x: f"fallback handled: {x}")]
)
print(guarded.invoke("customer question"))

# %%
# Retries handle transient errors; fallbacks handle persistent ones. Use both.
flaky_calls = {"n": 0}


def flaky(_):
    flaky_calls["n"] += 1
    if flaky_calls["n"] < 3:
        raise ConnectionError("transient network blip")
    return f"succeeded on attempt {flaky_calls['n']}"


print(RunnableLambda(flaky).with_retry(stop_after_attempt=4).invoke("x"))

# %% [markdown]
# ## 9. Configuration, naming and inspection
#
# Small things that pay off hugely when debugging a chain in LangSmith.

# %%
named_chain = (
    prompt.with_config(run_name="explain_prompt")
    | model.with_config(run_name="main_model")
    | parser.with_config(run_name="to_string")
).with_config(run_name="explainer_chain", tags=["track-01", "lcel-demo"])

print(named_chain.invoke({"topic": "idempotent retries"}))

# %%
# Make a parameter swappable at call time instead of build time.
from langchain_core.runnables import ConfigurableField

tunable = get_chat_model().configurable_fields(
    temperature=ConfigurableField(id="temperature", name="Sampling temperature")
)
tunable_chain = ChatPromptTemplate.from_template("Give one product name idea for: {thing}") | tunable | parser

print("t=0.0 ->", tunable_chain.invoke({"thing": "a stale feature flag finder"},
                                       config={"configurable": {"temperature": 0.0}}).strip())
print("t=1.3 ->", tunable_chain.invoke({"thing": "a stale feature flag finder"},
                                       config={"configurable": {"temperature": 1.3}}).strip())

# %%
# Inspect the structure of a composed chain.
print("Chain steps:")
for i, step in enumerate(rag_chain.steps if hasattr(rag_chain, "steps") else [], start=1):
    print(f"  {i}. {type(step).__name__}")

print("\nInput schema keys:", list(chain.input_schema.model_json_schema().get("properties", {})))

# %%
# ASCII picture of the graph - handy when a chain gets complex.
try:
    rag_chain.get_graph().print_ascii()
except Exception as exc:
    print(f"(install grandalf for ASCII graphs: {type(exc).__name__})")

# %% [markdown]
# ## 10. Async, because your API server needs it
#
# Every Runnable has async twins. FastAPI endpoints (notebook 53) should use them
# so one slow model call does not block the event loop.

# %%
import asyncio


async def async_demo():
    single = await chain.ainvoke({"topic": "connection pooling"})

    many = await chain.abatch([{"topic": t} for t in ["sharding", "read replicas"]])

    streamed = []
    async for piece in chain.astream({"topic": "backpressure"}):
        streamed.append(piece)

    return single, many, "".join(streamed)


single, many, streamed = await async_demo()
print("ainvoke :", single.strip()[:90])
print("abatch  :", len(many), "results")
print("astream :", streamed.strip()[:90])

# %% [markdown]
# ## Try it yourself
#
# 1. **Build a 4-way parallel analyser** for a ticket: summary, category,
#    priority, and suggested reply. Then add `RunnablePassthrough.assign` so the
#    original ticket text is still in the output dict.
# 2. **Type-error hunt.** Write `model | prompt | parser` (wrong order) on purpose,
#    read the error, and explain precisely which stage received the wrong type.
# 3. **Cheap-then-strong routing.** Build a chain that uses a small fast model
#    first and falls back to a bigger one when the answer is shorter than 20
#    characters. (Hint: `RunnableLambda` + `RunnableBranch`.)
# 4. **Trace it.** With LangSmith on, run the named chain and confirm each
#    `run_name` appears as its own span.

# %% [markdown]
# ## Recap
#
# | Primitive | What it does |
# |---|---|
# | `\|` | Compose: output of left becomes input of right |
# | `RunnablePassthrough` | Forward the input unchanged |
# | `RunnablePassthrough.assign(k=fn)` | Add keys, keep existing ones |
# | `RunnableParallel` / `{...}` | Fan out concurrently, merge into a dict |
# | `RunnableLambda` / bare function | Your Python inside the chain |
# | `RunnableBranch` | Conditional routing |
# | `.with_fallbacks([...])` | Try alternatives on failure |
# | `.with_retry(...)` | Retry transient failures |
# | `.with_config(run_name=..., tags=[...])` | Readable traces |
# | `.configurable_fields(...)` | Change parameters per call |
# | `a*` methods | Async everywhere, for free |
#
# **When to stop using LCEL:** when your flow needs cycles (agent loops), pausing
# for a human, or persistence between steps. LCEL is a DAG. The moment you need a
# loop, move to LangGraph - which is Track 05 onward.
#
# ## Next
#
# -> [08_chains_sequential_and_custom.ipynb](08_chains_sequential_and_custom.ipynb)
