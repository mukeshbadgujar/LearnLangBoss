# %% [markdown]
# # 19 - LangSmith: Tracing, Debugging and Datasets
#
# | | |
# |---|---|
# | **Level** | Intermediate |
# | **Time** | 50 minutes |
# | **Prerequisites** | `18_from_agentexecutor_to_graphs` |
# | **Checklist ID** | `19_langsmith` |
#
# ## Why this matters
#
# An LLM application fails differently from normal software. There is no stack
# trace when the model picks the wrong tool, retrieves the wrong chunk, or answers
# from a truncated context. You get a plausible-sounding wrong answer and no clue
# why.
#
# LangSmith records every prompt, every intermediate step, every token count and
# every latency figure. Debugging an agent without it is guesswork.
#
# > Everything here works with a free personal LangSmith account. If you have no
# > key, the notebook still runs — the tracing cells skip and the local
# > alternatives still demonstrate the concepts.
#
# ---
#
# ## Knowledge base (read this before the code)
#
# ### What LangSmith is
#
# LangSmith is the **observability + evaluation** platform for LangChain (and
# later LangGraph). It sits beside your process. It does not replace your app,
# your models, or your vector store.
#
# Docs: https://docs.langchain.com/langsmith/home
#
# ### What LangSmith can do
#
# | Capability | In one sentence | You will practice it in |
# |---|---|---|
# | **Tracing** | Record nested spans for every LLM / tool / chain call | §§1–4 |
# | **Debugging** | Open a run and inspect the exact rendered prompt and I/O | §10 + UI |
# | **Datasets** | Store inputs + expected outputs as a regression set | §7 |
# | **Evaluations** | Re-score a dataset after a prompt change (`evaluate`) | §7 |
# | **Feedback** | Attach thumbs / scores / comments to a specific run | §9 |
# | **Monitoring** | Filter by tag/metadata; watch errors, latency, cost | §§3, 6 |
#
# Notebook **45** extends the same platform to **graphs** and adds Studio.
#
# ### Use cases (when you actually open LangSmith)
#
# | Situation | What you look for |
# |---|---|
# | User reports a bad answer | Filter by `user_id` / `thread_id` → open that run |
# | “The model is broken” | Almost always the **rendered prompt** or truncated context |
# | Latency complaint | Waterfall: which span dominates time? |
# | Cost spike | Token counts per step; a hidden second LLM call |
# | Prompt change debate | Dataset experiment: v1 vs v2 score |
# | Production bug | Add the failing example to the dataset so it cannot silently return |
#
# ### Debugging order (memorise this)
#
# 1. **Rendered prompt** — variables `None`, truncated context, missing system msg.
# 2. **`finish_reason`** — `length` means truncation.
# 3. **Retrieved docs / tool I/O** — wrong chunk or bad tool args.
# 4. **Token counts** — one step usually owns the bill.
# 5. **Latency waterfall** — one step usually owns the wait.
#
# ### Keywords you will see in every later notebook
#
# | Keyword | Meaning |
# |---|---|
# | Trace / run | One top-level invocation + all nested work |
# | Span | One timed step inside a trace |
# | Project | Named bucket of runs (`LANGSMITH_PROJECT`) |
# | `run_name` | Human title in the UI |
# | `tags` / `metadata` | Filters and searchable key/values |
# | `@traceable` | Make *your* Python function a span |
# | `run_type` | How the UI renders a span (`llm`, `chain`, `tool`, `retriever`, …) |
# | Dataset / experiment | Fixed examples + scored re-runs |
#
# ### How it fits with LangChain and LangGraph
#
# ```
# LangChain  → build chains, tools, RAG
# LangGraph  → orchestrate stateful multi-step agents (Tracks 05–07)
# LangSmith  → see and evaluate what either of them did
# ```
#
# ---

# %%
from pathlib import Path
import sys

_root = Path.cwd()
while not (_root / "shared").is_dir() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from shared.notebook_setup import setup, require  # noqa: E402

ctx = setup("19_langsmith")

# %%
import os

from shared.llm import get_chat_model, has_key

model = get_chat_model()
tracing_on = has_key("LANGSMITH_API_KEY")
print("LangSmith key present:", tracing_on)
print("Project             :", os.environ.get("LANGSMITH_PROJECT", "(unset)"))

# %% [markdown]
# ## 1. Turning tracing on
#
# **Knowledge.** Tracing is configured entirely by environment variables. There
# is no per-call wrapper to add — any LangChain component in the process is
# traced automatically once the switch is on.
#
# ```ini
# # .env
# LANGSMITH_TRACING=true
# LANGSMITH_API_KEY=lsv2_pt_...
# LANGSMITH_PROJECT=genai-mastery
# LANGSMITH_ENDPOINT=https://api.smith.langchain.com
# ```
#
# | Variable | Role |
# |---|---|
# | `LANGSMITH_TRACING` | Master switch (`true` / `false`) |
# | `LANGSMITH_API_KEY` | Auth for https://smith.langchain.com |
# | `LANGSMITH_PROJECT` | Which project receives runs |
# | `LANGSMITH_ENDPOINT` | Cloud default; override only for self-hosted |
#
# Older docs used `LANGCHAIN_*` names; both work. Prefer `LANGSMITH_*` in new code.

# %%
from shared.llm import enable_tracing

if enable_tracing(project="genai-mastery-nb19"):
    print("tracing enabled, project =", os.environ["LANGSMITH_PROJECT"])
else:
    print("[skipped] add LANGSMITH_API_KEY to .env to send traces")

# %% [markdown]
# ## 2. Your first trace
#
# **Knowledge.** An LCEL pipe (`prompt | model | parser`) becomes a small trace
# tree: parent chain → prompt → LLM → parser. `.with_config(run_name=...)`
# labels the parent so you can find it in the UI.

# %%
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

triage_chain = (
    ChatPromptTemplate.from_template(
        "Classify this support ticket into exactly one of: "
        "billing, integration, performance, bug, security, howto.\n\nTicket: {ticket}"
    )
    | model
    | StrOutputParser()
).with_config(run_name="ticket_triage")

print(triage_chain.invoke({"ticket": "webhook deliveries failing with 401 after key rotation"}).strip())

if tracing_on:
    print("\nOpen https://smith.langchain.com -> project 'genai-mastery-nb19' to see the run.")
    print("In the run, open the LLM span and read the *rendered* prompt — that is the debugging habit.")

# %% [markdown]
# ## 3. Making traces readable
#
# **Knowledge.** A tree full of `RunnableSequence` / `RunnableParallel` is
# technically correct and useless for search. Name, tag, and attach metadata.

# %%
from langchain_core.runnables import RunnableParallel

summary = (ChatPromptTemplate.from_template("One-line summary: {ticket}") | model | StrOutputParser()).with_config(
    run_name="summarise"
)
category = (ChatPromptTemplate.from_template("One word category: {ticket}") | model | StrOutputParser()).with_config(
    run_name="categorise"
)
priority = (ChatPromptTemplate.from_template("One word priority: {ticket}") | model | StrOutputParser()).with_config(
    run_name="prioritise"
)

analyser = RunnableParallel(summary=summary, category=category, priority=priority).with_config(
    run_name="ticket_analyser",
    tags=["track-04", "triage", "demo"],
    metadata={"pipeline_version": "2.1", "owner": "support-platform"},
)

result = analyser.invoke({"ticket": "dashboards take over 30 seconds to load on large datasets"})
for key, value in result.items():
    print(f"{key:9}: {value.strip()[:90]}")

# %% [markdown]
# | Config field | Purpose |
# |---|---|
# | `run_name` | The label in the trace tree. Use verbs |
# | `tags` | Filterable labels. Use for environment, version, feature |
# | `metadata` | Arbitrary key/values. Use for `user_id`, `tenant`, `prompt_version` |
#
# **Always tag with user and tenant identifiers in production.** When a customer
# reports a bad answer, you want to find *their* run in seconds.

# %%
user_scoped = analyser.with_config(
    metadata={"user_id": "E-104", "tenant": "northwind", "environment": "dev"},
    tags=["env:dev"],
)
user_scoped.invoke({"ticket": "salesforce sync stopped after sandbox refresh"})
print("run tagged with user/tenant metadata")

# %% [markdown]
# ## 4. Tracing your own functions with `@traceable`
#
# **Knowledge.** LangChain components trace themselves. Your business logic does
# not — unless you decorate it. The interesting bugs are usually in *your* code
# (retrieval ranking, post-filters), not the raw model call.

# %%
from langsmith import traceable


@traceable(run_type="retriever", name="policy_lookup")
def lookup_policy(question: str) -> list[str]:
    """Pretend retriever - traced so you can see what it returned."""
    policy = Path(ctx.data("leave_policy.txt")).read_text(encoding="utf-8")
    sections = policy.split("\n\n")
    words = {w.lower().strip("?.,") for w in question.split()}
    ranked = sorted(sections, key=lambda s: -len(words & {w.lower() for w in s.split()}))
    return ranked[:2]


@traceable(run_type="chain", name="answer_policy_question")
def answer(question: str) -> str:
    context = "\n\n".join(lookup_policy(question))
    return (
        ChatPromptTemplate.from_template("Answer from the context only.\n\n{context}\n\nQ: {question}")
        | model
        | StrOutputParser()
    ).invoke({"context": context, "question": question})


print(answer("How many casual leave days are there?").strip())

# %% [markdown]
# `run_type` controls how LangSmith displays the span: `llm`, `chain`, `tool`,
# `retriever`, `prompt` or `parser`. Marking your retriever as `retriever` gets
# you the document viewer in the UI.

# %% [markdown]
# ## 5. Debugging without a LangSmith account
#
# **Knowledge.** Same questions LangSmith answers — “what was sent?”, “which
# steps ran?”, “show everything” — using local tools. Useful offline and as a
# complement when tracing is on.

# %%
# (a) Inspect exactly what was sent to the model.
rendered = ChatPromptTemplate.from_template(
    "Classify: {ticket}"
).invoke({"ticket": "duplicate charge in january"})

for message in rendered.to_messages():
    print(f"[{message.type}] {message.content}")

# %%
# (b) Capture intermediate values with astream_events.
import asyncio


async def watch_events(chain, payload):
    seen = []
    async for event in chain.astream_events(payload, version="v2"):
        kind = event["event"]
        if kind in {"on_chain_start", "on_llm_start", "on_chain_end"}:
            seen.append((kind, event.get("name")))
    return seen


events = await watch_events(triage_chain, {"ticket": "SOC2 report requested"})
for kind, name in events[:12]:
    print(f"  {kind:18} {name}")

# %%
# (c) Global debug mode - extremely verbose, but it shows everything.
from langchain_core.globals import set_debug, set_verbose

set_verbose(True)
triage_chain.invoke({"ticket": "csv export drops the last row"})
set_verbose(False)
print("\n(verbose off again)")

# %% [markdown]
# ## 6. Finding the slow step
#
# Latency complaints are almost always one step, not the whole pipeline. Measure
# per stage before optimising anything.

# %%
import time

from langchain_core.runnables import RunnableLambda


def timed(name: str, runnable):
    """Wrap a runnable so it prints its own duration."""

    def run(payload):
        start = time.perf_counter()
        output = runnable.invoke(payload)
        print(f"  {name:14} {time.perf_counter() - start:6.2f}s")
        return output

    return RunnableLambda(run).with_config(run_name=name)


pipeline = (
    timed("summarise", summary)
    | RunnableLambda(lambda text: {"ticket": text})
    | timed("categorise", category)
)

start = time.perf_counter()
pipeline.invoke({"ticket": "API p99 latency spiked to 4 seconds during EU business hours"})
print(f"  {'TOTAL':14} {time.perf_counter() - start:6.2f}s")

# %% [markdown]
# In LangSmith the same information is a waterfall chart you do not have to build.
# The usual findings: retrieval is fast, the model call dominates, and a
# "cheap" reranking step you added is quietly costing 2 seconds.

# %% [markdown]
# ## 7. Datasets: turning bugs into regression tests
#
# This is the part teams skip and later regret. A dataset is a set of inputs plus
# expected outputs. Every prompt change gets scored against it, so you find out
# *before* shipping that your improvement broke three other cases.

# %%
EVAL_EXAMPLES = [
    {"ticket": "we were charged twice for january seats", "category": "billing"},
    {"ticket": "webhook deliveries failing with 401 after key rotation", "category": "integration"},
    {"ticket": "dashboards take 30+ seconds on large datasets", "category": "performance"},
    {"ticket": "csv export drops the last row when filtered", "category": "bug"},
    {"ticket": "we need your SOC2 report for procurement", "category": "security"},
    {"ticket": "how do I export a dashboard to PDF", "category": "howto"},
    {"ticket": "invoice needs reissuing with a corrected GST number", "category": "billing"},
    {"ticket": "google sheets connector asks to reauthorize daily", "category": "integration"},
]

print(f"{len(EVAL_EXAMPLES)} examples ready")

# %%
if require("LANGSMITH_API_KEY", feature="LangSmith dataset upload"):
    from langsmith import Client

    client = Client()
    dataset_name = "northwind-ticket-triage"

    if not client.has_dataset(dataset_name=dataset_name):
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="Support ticket text -> expected category. Regression set for triage prompts.",
        )
        client.create_examples(
            inputs=[{"ticket": e["ticket"]} for e in EVAL_EXAMPLES],
            outputs=[{"category": e["category"]} for e in EVAL_EXAMPLES],
            dataset_id=dataset.id,
        )
        print(f"created dataset '{dataset_name}' with {len(EVAL_EXAMPLES)} examples")
    else:
        print(f"dataset '{dataset_name}' already exists")

# %% [markdown]
# ### Running an evaluation

# %%
if require("LANGSMITH_API_KEY", feature="LangSmith evaluation run"):
    from langsmith import evaluate

    def triage_target(inputs: dict) -> dict:
        return {"category": triage_chain.invoke({"ticket": inputs["ticket"]}).strip().lower()}

    def exact_category(outputs: dict, reference_outputs: dict) -> dict:
        predicted = outputs.get("category", "").strip().lower()
        expected = reference_outputs.get("category", "").strip().lower()
        return {"key": "category_match", "score": float(predicted.startswith(expected))}

    experiment = evaluate(
        triage_target,
        data="northwind-ticket-triage",
        evaluators=[exact_category],
        experiment_prefix="triage-v1",
        max_concurrency=4,
    )
    print("experiment complete - compare runs in the LangSmith UI")

# %% [markdown]
# ### The same evaluation, locally
#
# You do not need LangSmith to get the discipline. Run it in-process and keep the
# numbers in a file.

# %%
def local_evaluate(chain, examples: list[dict]) -> dict:
    predictions = chain.batch(
        [{"ticket": e["ticket"]} for e in examples],
        config={"max_concurrency": 4},
    )
    rows = []
    correct = 0
    for example, raw in zip(examples, predictions):
        predicted = raw.strip().lower().strip(".")
        hit = predicted.startswith(example["category"])
        correct += hit
        rows.append((hit, example["category"], predicted, example["ticket"][:48]))

    print(f"{'':3} {'expected':13} {'predicted':16} ticket")
    for hit, expected, predicted, ticket in rows:
        print(f"{'OK ' if hit else 'X  '} {expected:13} {predicted[:16]:16} {ticket}")

    accuracy = correct / len(examples)
    print(f"\naccuracy: {correct}/{len(examples)} = {accuracy:.0%}")
    return {"accuracy": accuracy, "correct": correct, "total": len(examples)}


baseline = local_evaluate(triage_chain, EVAL_EXAMPLES)

# %%
# Now change the prompt and see whether it actually improved.
improved_chain = (
    ChatPromptTemplate.from_messages([
        ("system",
         "You classify support tickets. Reply with exactly one lowercase word from this list "
         "and nothing else: billing, integration, performance, bug, security, howto.\n"
         "- billing: invoices, charges, refunds, plan changes\n"
         "- integration: connectors, webhooks, API auth, third-party sync\n"
         "- performance: slowness, latency, timeouts\n"
         "- bug: incorrect behaviour or wrong output\n"
         "- security: compliance, certifications, data residency, access\n"
         "- howto: usage questions with no defect"),
        ("human", "{ticket}"),
    ])
    | model
    | StrOutputParser()
).with_config(run_name="ticket_triage_v2")

improved = local_evaluate(improved_chain, EVAL_EXAMPLES)

print(f"\nv1 {baseline['accuracy']:.0%}  ->  v2 {improved['accuracy']:.0%}"
      f"   ({'improved' if improved['accuracy'] > baseline['accuracy'] else 'no better'})")

# %% [markdown]
# That comparison is the entire point of a dataset. Without it, "I improved the
# prompt" is an opinion.

# %% [markdown]
# ## 8. Capturing production failures as examples
#
# The best dataset is built from real failures. When a user reports a bad answer,
# take the run and add it to the dataset with the correct output.

# %%
if require("LANGSMITH_API_KEY", feature="adding a failure to the dataset"):
    from langsmith import Client

    client = Client()
    client.create_examples(
        inputs=[{"ticket": "our SSO metadata certificate expires next week"}],
        outputs=[{"category": "security"}],
        dataset_name="northwind-ticket-triage",
    )
    print("added a real-world failure to the regression set")

# %%
# Locally, this is just appending to a JSON file - do that from day one.
import json

failures_path = ctx.artifact("triage_failures.jsonl")
with open(failures_path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps({"ticket": "our SSO metadata certificate expires next week", "category": "security"}) + "\n")

print("appended to", failures_path)
print(failures_path.read_text(encoding="utf-8").strip().splitlines()[-1])

# %% [markdown]
# ## 9. Feedback and annotation
#
# Thumbs up/down from users, attached to the run that produced the answer, is the
# cheapest quality signal you will ever get.

# %%
if require("LANGSMITH_API_KEY", feature="attaching user feedback to a run"):
    from langchain_core.tracers.context import collect_runs
    from langsmith import Client

    client = Client()
    with collect_runs() as runs:
        triage_chain.invoke({"ticket": "we need a DPA signed before renewal"})

    run_id = runs.traced_runs[0].id
    client.create_feedback(run_id, key="user_thumbs", score=0, comment="Should have been 'security', not 'howto'")
    print("feedback attached to run", run_id)

# %% [markdown]
# ## 10. What to look at in a trace (debugging playbook)
#
# When something is wrong, open the run in LangSmith and check **in this order**:
#
# 1. **The rendered prompt.** ~90% of bugs are here — a variable that rendered as
#    `None`, truncated context, a template that lost its system message.
# 2. **`finish_reason`.** `length` means truncation; that alone explains many
#    “the model is broken” reports.
# 3. **Retrieved documents.** Was the right chunk even in the context?
# 4. **Tool inputs and outputs.** Sensible arguments? Readable return value?
# 5. **Token counts per step.** One step usually dominates cost.
# 6. **Latency waterfall.** One step usually dominates time — often not the one
#    you assumed.
#
# **Practice once with a real key:** deliberately pass a missing template
# variable, find it in the UI, then fix it. That single loop is the skill.

# %% [markdown]
# ## Try it yourself
#
# 1. **Trace an agent.** Run the Ops Assistant from notebook 17 with tracing on
#    and read the tool-call spans. Find a case where it chose the wrong tool.
# 2. **Grow the dataset to 20 examples**, including three genuinely ambiguous
#    tickets. See whether v2's advantage survives.
# 3. **Add a second evaluator** that checks the output is *only* one word (no
#    explanation). Measure how often each version violates it.
# 4. **Metadata discipline.** Add `user_id` and `prompt_version` metadata to every
#    chain in this notebook, then filter by them in the UI.

# %% [markdown]
# ## Recap
#
# | Concept | Takeaway |
# |---|---|
# | `LANGSMITH_TRACING=true` | Env vars only; no code changes needed |
# | `.with_config(run_name=..., tags=..., metadata=...)` | Readable, filterable traces |
# | `@traceable` | Traces *your* functions, where the real bugs live |
# | `run_type` | `llm`/`chain`/`tool`/`retriever` changes how the span renders |
# | `astream_events` / `set_verbose` | Offline debugging that still works |
# | Datasets | Turn every bug into a regression test |
# | `evaluate(...)` | Score a change before shipping it |
# | Local eval | Same discipline without an account - do it from day one |
# | Feedback | Attach user thumbs to the exact run |
# | Debugging order | Prompt → finish_reason → retrieval/tools → tokens → latency |
#
# Graph-level tracing, Studio, and production sampling land in notebook **45**.
#
# ## Next
#
# -> [20_callbacks.ipynb](20_callbacks.ipynb)
