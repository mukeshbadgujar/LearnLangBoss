# %% [markdown]
# # 45 - LangSmith and LangGraph Studio
#
# | | |
# |---|---|
# | **Level** | Advanced (LangGraph) |
# | **Time** | 55 minutes |
# | **Prerequisites** | `44_error_retries_tool_failures`, `19_langsmith` |
# | **Checklist ID** | `45_langsmith_and_studio` |
#
# ## Why this matters
#
# `print()` works until your graph has twelve nodes, three of which loop. Then
# you need the actual execution tree: which nodes ran, in what order, what each
# received, how long each took, and what it cost.
#
# Notebook 19 taught LangSmith on **chains**. This lesson applies the same
# platform to **graphs**, then adds **LangGraph Studio** — a visual debugger
# where you step through nodes and edit state live.
#
# ---
#
# ## Knowledge base (read this before the code)
#
# ### What LangSmith is
#
# LangSmith is the observability and evaluation platform for LangChain and
# LangGraph. It is **not** a model and **not** a replacement for your app
# server. It sits beside your process and records what happened.
#
# | Layer | Job |
# |---|---|
# | **Tracing** | Record every LLM call, tool call, node, and custom function as nested *spans* |
# | **Debugging** | Open a failed run and see the exact prompt, tool args, and state |
# | **Datasets** | Store example inputs + expected outputs as a regression set |
# | **Evaluations / experiments** | Re-run a dataset after a prompt or graph change and compare scores |
# | **Feedback** | Attach thumbs, scores, or comments to a specific run |
# | **Monitoring** | Filter by tag/metadata; watch latency, errors, and cost over time |
#
# Official docs: https://docs.langchain.com/langsmith/home
#
# ### What LangSmith can do (use cases)
#
# | Use case | What you do | What you get |
# |---|---|---|
# | **"Why did this user get a bad answer?"** | Filter by `user_id` / `thread_id` in metadata | The exact prompt, retrieval, and tool calls for that turn |
# | **"Which node is slow / expensive?"** | Open the waterfall in a graph trace | Per-node latency and token counts |
# | **"Did my prompt change help?"** | Run an experiment on a dataset | Side-by-side scores for v1 vs v2 |
# | **"Did quality drop after deploy?"** | Tag runs with `prompt_version` / `env:prod` | Slice metrics by version |
# | **"Turn this bug into a test"** | Add the failing input + correct output to a dataset | It fails CI next time the bug returns |
# | **"Make my Python visible"** | `@traceable` on helpers | Custom logic shows as spans, not a black box |
# | **"Debug a looping graph"** | Trace a graph with conditional edges | See how many times `draft` / `review` actually ran |
#
# ### Debugging workflow with LangSmith (the habit)
#
# When something is wrong, open the trace and check **in this order**:
#
# 1. **Rendered prompt** — missing variables, truncated context, wrong system message.
# 2. **`finish_reason`** — `length` means the model was cut off.
# 3. **Node path** — did the graph take the branch you expected?
# 4. **Tool / retriever I/O** — bad args in → garbage out.
# 5. **Token and latency waterfall** — one step usually dominates.
# 6. **Loop count** — graphs with revise loops often run more iterations than you think.
#
# That workflow is the same for chains (notebook 19) and graphs (this notebook).
# Graphs just add nesting: one parent run → child runs per node → grandchildren
# for each LLM / tool inside the node.
#
# ### Keyword glossary for this lesson
#
# | Keyword | Meaning |
# |---|---|
# | **Trace / run** | One top-level invocation of a chain or graph, plus all nested work |
# | **Span** | One timed step inside a trace (a node, an LLM call, a `@traceable` function) |
# | **Project** | A named bucket of runs in LangSmith (`LANGSMITH_PROJECT`) |
# | **`run_name`** | Human title for the run in the UI (instead of generic `LangGraph`) |
# | **`tags`** | Filter chips (`env:prod`, `v1`, `research`) |
# | **`metadata`** | Searchable key/value (`user_id`, `tenant`, `prompt_version`) |
# | **`thread_id`** | Checkpointer key that groups turns of one conversation |
# | **`@traceable`** | Decorator that turns a plain Python function into a span |
# | **`run_type`** | How LangSmith renders a span: `llm`, `chain`, `tool`, `retriever`, … |
# | **Dataset** | Collection of `{inputs, outputs}` examples for regression |
# | **Experiment / evaluate** | Running a target function over a dataset with scorers |
# | **Studio** | Local visual IDE for graphs (`langgraph dev`) |
# | **`langgraph.json`** | Config that tells Studio which compiled graphs to load |
# | **Checkpointer** | Persistence for graph state between steps / turns (notebook 34) |
#
# ### LangChain vs LangGraph vs LangSmith (how they fit)
#
# ```
# LangChain  → building blocks (models, prompts, tools, LCEL)
# LangGraph  → orchestration (state, nodes, edges, loops, HITL)
# LangSmith  → observe and evaluate what those two actually did
# Studio     → visual debugger for LangGraph (uses LangSmith traces)
# ```
#
# You can run graphs without LangSmith. You should not ship them without *some*
# form of the same discipline (this notebook ends with an offline `debug_run`).
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

ctx = setup("45_langsmith_and_studio")

# %%
import operator
import os
from typing import Annotated, Literal, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from shared.llm import get_chat_model

model = get_chat_model()
parser = StrOutputParser()

print("tracing enabled:", ctx.tracing)

# %% [markdown]
# ## 1. Enabling tracing
#
# **Knowledge.** Tracing is opt-in via environment variables. Once on, every
# LangChain / LangGraph call in the process emits spans — you do not wrap each
# call by hand.
#
# | Variable | Role |
# |---|---|
# | `LANGSMITH_TRACING=true` | Master switch |
# | `LANGSMITH_API_KEY` | Auth to https://smith.langchain.com |
# | `LANGSMITH_PROJECT` | Which project receives the runs (group by app or lesson) |
# | `LANGSMITH_ENDPOINT` | Cloud default is fine; set this only for self-hosted |
#
# Older docs used `LANGCHAIN_*` names; both work. Prefer `LANGSMITH_*` in new code.
#
# `shared/notebook_setup.py` loads `.env` and calls `enable_tracing()` for you.

# %%
from shared.llm import describe_environment, has_key

print(describe_environment())
if not has_key("LANGSMITH_API_KEY"):
    print("\nGet a free key at https://smith.langchain.com - the rest of this notebook still runs.")
else:
    print("\nWith tracing on, every graph.invoke below will appear in your LangSmith project.")

# %% [markdown]
# ## 2. A graph worth tracing
#
# **Knowledge.** A flat single-node graph produces a boring trace. To learn how
# to *read* a graph trace you need:
#
# - more than one **node** (`plan` → `draft` → `review`)
# - a **conditional edge** that can loop (`keep_going`)
# - **structured output** inside a node (`Review` via `with_structured_output`)
# - a **reducer** on some fields (`Annotated[int, operator.add]` for `revisions`)
#
# | Piece in the code | What it teaches in the trace |
# |---|---|
# | `ResearchState` | The state schema — what each node may read/write |
# | `Annotated[..., operator.add]` | Reducers: updates *accumulate* instead of overwrite |
# | `plan_node` / `draft_node` / `review_node` | Become child spans under the parent graph run |
# | `keep_going` | Conditional routing — you will see 1–N draft/review pairs |
# | `compile(name=..., checkpointer=...)` | Named parent run + resumable `thread_id` |

# %%
class ResearchState(TypedDict):
    question: str
    plan: str
    draft: str
    score: int
    revisions: Annotated[int, operator.add]
    trace: Annotated[list[str], operator.add]


class Review(BaseModel):
    """A reviewer's verdict on a draft — becomes structured LLM output in the trace."""

    score: int = Field(description="Quality out of 10", ge=1, le=10)
    issue: str = Field(description="The single most important thing to fix")


reviewer = model.with_structured_output(Review)


def plan_node(state: ResearchState) -> dict:
    """Node: produce a short plan. Returns a partial state update."""
    plan = (ChatPromptTemplate.from_template(
        "Write a 3-bullet plan for answering: {question}"
    ) | model | parser).invoke(state)
    return {"plan": plan, "trace": ["plan"]}


def draft_node(state: ResearchState) -> dict:
    """Node: write or revise the answer. `revisions: 1` accumulates via the reducer."""
    template = ("Answer in 80 words following this plan.\n\nQuestion: {question}\nPlan: {plan}"
                if not state["draft"] else
                "Improve this answer.\n\nQuestion: {question}\nDraft: {draft}\nFix: {trace}")
    draft = (ChatPromptTemplate.from_template(template) | model | parser).invoke(
        {**state, "trace": state["trace"][-1]}
    )
    return {"draft": draft, "revisions": 1, "trace": ["draft"]}


def review_node(state: ResearchState) -> dict:
    """Node: score the draft. Look for the structured `Review` span in LangSmith."""
    verdict = reviewer.invoke([("system", "Review this answer strictly."),
                               ("human", f"Q: {state['question']}\n\nA: {state['draft']}")])
    return {"score": verdict.score, "trace": [f"review:{verdict.score} - {verdict.issue}"]}


def keep_going(state: ResearchState) -> Literal["draft", "__end__"]:
    """Router: stop when score is good enough or we hit a revision cap."""
    return END if state["score"] >= 8 or state["revisions"] >= 3 else "draft"


builder = StateGraph(ResearchState)
builder.add_node("plan", plan_node)
builder.add_node("draft", draft_node)
builder.add_node("review", review_node)
builder.add_edge(START, "plan")
builder.add_edge("plan", "draft")
builder.add_edge("draft", "review")
builder.add_conditional_edges("review", keep_going, {"draft": "draft", END: END})
graph = builder.compile(name="research_loop", checkpointer=InMemorySaver())

print(graph.get_graph().draw_ascii())

# %% [markdown]
# **After this cell.** The ASCII diagram is the *static* topology. The LangSmith
# trace of a live run is the *dynamic* path (how many times you looped). Keep
# both ideas separate when debugging.

# %% [markdown]
# ## 3. Making traces readable
#
# **Knowledge.** An unnamed graph run shows up as `LangGraph` with anonymous
# children. That is almost useless in a project with hundreds of runs. You pass
# observability fields through the **`config`** argument of `invoke` / `stream`.
#
# | Config key | What it buys you in the UI |
# |---|---|
# | `run_name` | A readable title instead of `LangGraph` |
# | `tags` | Filter chips — `track-07`, `v1`, `tenant:acme` |
# | `metadata` | Searchable key/value; group cost by tenant or prompt version |
# | `configurable.thread_id` | Groups every turn of one conversation (also required by the checkpointer) |
#
# **Always include `prompt_version` in metadata.** Three weeks later, "did quality
# drop when we shipped 1.2.0?" is answerable only if that string is on the run.

# %%
QUESTION = "Should we cap the number of tool calls an agent can make, and why?"

outcome = graph.invoke(
    {"question": QUESTION, "plan": "", "draft": "", "score": 0, "revisions": 0, "trace": []},
    config={
        "configurable": {"thread_id": "trace-1"},
        "run_name": "research_loop:tool_call_caps",
        "tags": ["track-07", "research", "v1"],
        "metadata": {"user_id": "u-4821", "tenant": "acme", "prompt_version": "1.2.0"},
    },
)
print(f"revisions={outcome['revisions']} final score={outcome['score']}")
for step in outcome["trace"]:
    print(f"  {step[:96]}")

# %% [markdown]
# **After this cell — debugging checklist in the UI**
#
# 1. Open https://smith.langchain.com → your project.
# 2. Find the run named `research_loop:tool_call_caps`.
# 3. Expand `plan` → `draft` → `review`. Count how many draft/review pairs ran.
# 4. Click an LLM span: confirm the *rendered* prompt matches what you expected.
# 5. Filter the project by tag `v1` or metadata `prompt_version=1.2.0`.

# %% [markdown]
# ## 4. What a graph trace looks like
#
# **Knowledge.** Mentally map every live run to a tree like this:
#
# ```
# research_loop                                     12.4s   4,812 tokens
#  |- plan                                           2.1s     612
#  |   `- ChatGroq                                   2.0s     612
#  |- draft                          (iteration 1)   3.4s   1,204
#  |   `- ChatGroq                                   3.3s   1,204
#  |- review                         (iteration 1)   1.8s     890
#  |   `- ChatGroq (structured)                      1.7s     890
#  |- draft                          (iteration 2)   3.3s   1,310
#  |- review                         (iteration 2)   1.8s     796
#  `- __end__
# ```
#
# Read three facts off it immediately:
#
# 1. **Which node dominates** latency (here: `draft`, twice).
# 2. **How many loop iterations** actually ran.
# 3. **Where tokens went** — often an unnoticed second call.
#
# If the UI and your mental model disagree, trust the UI: the router ran what it ran.

# %% [markdown]
# ## 5. Tracing custom functions with `@traceable`
#
# **Knowledge.** Nodes that only call LangChain runnables are traced for free.
# Plain Python inside a node (scoring, ranking, I/O) is **invisible** unless you
# mark it. `@traceable` creates a span; nested calls nest in the UI.
#
# | Argument | Meaning |
# |---|---|
# | `name=` | Label in the tree |
# | `run_type=` | UI treatment: `tool`, `chain`, `retriever`, `llm`, … |
#
# Use `@traceable` on anything you might later ask: "what did this helper return?"

# %%
from langsmith import traceable


@traceable(name="score_relevance", run_type="tool")
def score_relevance(question: str, text: str) -> float:
    """Custom scoring logic worth seeing as its own span."""
    question_words = set(question.lower().split())
    text_words = set(text.lower().split())
    overlap = question_words & text_words
    return round(len(overlap) / max(len(question_words), 1), 3)


@traceable(name="rank_candidates", run_type="chain")
def rank_candidates(question: str, candidates: list[str]) -> list[tuple[str, float]]:
    """Parent span: will show three `score_relevance` children when tracing is on."""
    scored = [(c, score_relevance(question, c)) for c in candidates]
    return sorted(scored, key=lambda pair: pair[1], reverse=True)


ranked = rank_candidates(QUESTION, [
    "Capping tool calls bounds the worst-case cost of an agent run.",
    "Vector databases store embeddings for similarity search.",
    "An agent that loops forever will exhaust its recursion limit.",
])
for text, score in ranked:
    print(f"  {score:.3f}  {text[:66]}")

# %% [markdown]
# **After this cell.** In LangSmith, open the `rank_candidates` run. You should
# see three nested `score_relevance` spans. If you only see one flat span, the
# child was not decorated (or tracing is off).

# %% [markdown]
# ## 6. Finding the slow step programmatically
#
# **Knowledge.** The browser UI is best for one-off debugging. For scripts and
# alerts, use the **LangSmith Client SDK** (`Client.list_runs`). Same data, no
# clicking.
#
# Typical questions this answers:
#
# - What are the last N runs' latencies?
# - Which run error'd?
# - What is the slowest run in this project right now?

# %%
if require("LANGSMITH_API_KEY", feature="querying traces from the SDK"):
    from langsmith import Client

    client = Client()
    project = os.environ.get("LANGSMITH_PROJECT", "genai-mastery")

    try:
        runs = list(client.list_runs(project_name=project, limit=25))
        if runs:
            print(f"{'name':28} {'type':10} {'sec':>7} {'tokens':>8} status")
            for run in runs[:12]:
                seconds = (run.end_time - run.start_time).total_seconds() if run.end_time else 0
                tokens = (run.total_tokens or 0)
                print(f"{(run.name or '')[:27]:28} {str(run.run_type):10} {seconds:>7.2f} "
                      f"{tokens:>8} {'error' if run.error else 'ok'}")

            slowest = max((r for r in runs if r.end_time), key=lambda r: (r.end_time - r.start_time))
            print(f"\nslowest: {slowest.name} "
                  f"({(slowest.end_time - slowest.start_time).total_seconds():.2f}s)")
        else:
            print(f"no runs yet in project {project!r} — invoke the graph above with tracing on first")
    except Exception as exc:
        print(f"[skipped] {type(exc).__name__}: {str(exc)[:120]}")

# %% [markdown]
# ## 7. Datasets and regression testing for graphs
#
# **Knowledge.** A **dataset** is a list of examples: `inputs` (what you send the
# graph) and `outputs` (what “good” looks like — exact text, keywords, grades).
# An **experiment** runs your graph over that dataset and scores each example.
#
# Without datasets, “I improved the prompt” is an opinion. With them, it is a
# number you can put in CI.
#
# This section: upload (or reuse) a tiny dataset in LangSmith, then run the same
# checks **locally** so the lesson works without a key.

# %%
EVAL_CASES = [
    {"question": "Why cap agent tool calls?", "must_mention": ["cost"]},
    {"question": "What is a checkpointer for?", "must_mention": ["state"]},
    {"question": "When should you use a subgraph?", "must_mention": ["reus"]},
]

if require("LANGSMITH_API_KEY", feature="LangSmith datasets"):
    from langsmith import Client

    client = Client()
    dataset_name = "langgraph-research-loop"

    try:
        if client.has_dataset(dataset_name=dataset_name):
            dataset = client.read_dataset(dataset_name=dataset_name)
        else:
            dataset = client.create_dataset(dataset_name=dataset_name,
                                            description="Research loop regression cases")
            client.create_examples(
                inputs=[{"question": c["question"]} for c in EVAL_CASES],
                outputs=[{"must_mention": c["must_mention"]} for c in EVAL_CASES],
                dataset_id=dataset.id,
            )
        print(f"dataset ready: {dataset_name} ({client.count_examples(dataset_id=dataset.id)} examples)")
    except Exception as exc:
        print(f"[skipped] {type(exc).__name__}: {str(exc)[:120]}")

# %% [markdown]
# ### Evaluating locally
#
# **Knowledge.** `must_mention` here is a deliberately dumb evaluator (substring
# check). Production evaluators are often LLM-as-judge (notebook 26) or exact
# match on structured fields. The *discipline* is the same: fixed cases, scored
# outputs, a gate that fails the build.

# %%
def run_case(case: dict) -> dict:
    """Invoke the graph once and score with a simple keyword check."""
    outcome = graph.invoke(
        {"question": case["question"], "plan": "", "draft": "", "score": 0, "revisions": 0, "trace": []},
        config={"configurable": {"thread_id": f"eval-{hash(case['question']) & 0xffff}"},
                "run_name": "eval:research_loop",
                "tags": ["eval"]},
    )
    draft = outcome["draft"].lower()
    return {
        "question": case["question"],
        "passed": all(term in draft for term in case["must_mention"]),
        "score": outcome["score"],
        "revisions": outcome["revisions"],
    }


results = [run_case(case) for case in EVAL_CASES]
print(f"{'ok':4} {'score':>6} {'rev':>4}  question")
for result in results:
    print(f"{'PASS' if result['passed'] else 'FAIL':4} {result['score']:>6} "
          f"{result['revisions']:>4}  {result['question'][:48]}")

passed = sum(r["passed"] for r in results)
print(f"\n{passed}/{len(results)} passed, mean score "
      f"{sum(r['score'] for r in results) / len(results):.1f}/10")


def regression_gate(results: list[dict], min_pass_rate: float = 0.8, min_score: float = 7.0) -> None:
    """Raise in CI if quality regressed — this is the production pattern."""
    rate = sum(r["passed"] for r in results) / len(results)
    mean = sum(r["score"] for r in results) / len(results)
    if rate < min_pass_rate or mean < min_score:
        raise AssertionError(f"regression: pass rate {rate:.0%}, mean score {mean:.1f}")
    print(f"gate ok: pass rate {rate:.0%}, mean score {mean:.1f}")


try:
    regression_gate(results)
except AssertionError as exc:
    print(f"gate would fail the build: {exc}")

# %% [markdown]
# ## 8. LangGraph Studio
#
# **Knowledge.** Studio is a **visual IDE for LangGraph**, not a separate product
# from LangSmith. It runs your graph locally, animates node execution, lets you
# inspect and **edit** state mid-run, and deep-links into LangSmith traces.
#
# | Capability | Why it matters |
# |---|---|
# | Live node highlighting | See the actual path, including loops |
# | Inspect state at any step | No `print` archaeology |
# | **Edit state and continue** | Test “what if classifier said billing?” without code changes |
# | Fork from any checkpoint | Compare two futures (pairs with notebook 39 time travel) |
# | Interrupt UI | Click approve/reject for human-in-the-loop nodes |
# | Trace link | Jump into the LangSmith waterfall for the same run |
#
# ### Setup: `langgraph.json`
#
# Studio discovers graphs from a JSON file at the project root. Each entry points
# at a **module:attribute** that is a *compiled* graph (usually without your own
# checkpointer — the platform supplies one).

# %%
import json

studio_config = {
    "dependencies": ["."],
    "graphs": {
        "research_loop": "./app/graphs.py:research_graph",
        "support_router": "./app/graphs.py:support_graph",
    },
    "env": ".env",
}

config_path = ctx.artifact("langgraph.json")
config_path.write_text(json.dumps(studio_config, indent=2), encoding="utf-8")
print(config_path.read_text(encoding="utf-8"))

# %% [markdown]
# **What each field means**
#
# | Field | Meaning |
# |---|---|
# | `dependencies` | Packages / paths installed into the Studio process (`.` = this repo) |
# | `graphs` | Map of UI name → `path/to/module.py:exported_compiled_graph` |
# | `env` | File of environment variables loaded before import |
#
# Example export module:
#
# ```python
# # app/graphs.py
# from langgraph.graph import StateGraph, START, END
#
# builder = StateGraph(ResearchState)
# # ... add nodes and edges ...
# research_graph = builder.compile()   # no checkpointer — Studio/Platform adds one
# ```
#
# Then locally:
#
# ```bash
# python -m pip install --upgrade "langgraph-cli[inmem]"
# langgraph dev            # starts the local API and opens Studio in the browser
# ```
#
# `langgraph dev` keeps everything in memory — no Docker required for learning.

# %% [markdown]
# ## 9. Debugging without LangSmith or Studio
#
# **Knowledge.** Keys expire, networks fail, and interviews happen offline. The
# same information a waterfall gives you can be approximated with
# `graph.stream(..., stream_mode="updates")`: each chunk is `{node_name: update}`.
#
# Use this in CI logs and when you cannot open a browser. Prefer LangSmith when
# you have it — richer prompts, tokens, and nested LLM spans.

# %%
import time


def debug_run(graph, payload: dict, config: dict | None = None) -> dict:
    """Print a node-by-node execution report with timings and state deltas."""
    config = config or {}
    print(f"{'step':>4} {'node':14} {'sec':>6}  update")
    print("-" * 78)

    started = time.perf_counter()
    last = started
    final: dict = {}

    for step, chunk in enumerate(graph.stream(payload, config, stream_mode="updates"), start=1):
        now = time.perf_counter()
        for node_name, update in chunk.items():
            summary = ", ".join(
                f"{k}={(str(v)[:30] + '...') if len(str(v)) > 30 else v}"
                for k, v in (update or {}).items()
            )
            print(f"{step:>4} {node_name:14} {now - last:>6.2f}  {summary[:56]}")
            final.update(update or {})
        last = now

    print("-" * 78)
    print(f"total {time.perf_counter() - started:.2f}s over {step} superstep(s)")
    return final


debug_run(
    graph,
    {"question": "What is a reducer in LangGraph?", "plan": "", "draft": "", "score": 0, "revisions": 0, "trace": []},
    {"configurable": {"thread_id": "debug-1"}},
)

# %% [markdown]
# **After this cell.** Compare the printed node sequence with a LangSmith trace
# of the same question. Same story; different UI.

# %% [markdown]
# ## 10. Production observability checklist
#
# **Knowledge.** Tracing 100% of production traffic is often too expensive and
# too noisy. Standardise a `config` factory: always set identity fields; sample
# in prod; never put secrets in tags or metadata (traces are widely readable).

# %%
def production_config(*, thread_id: str, user_id: str, tenant: str,
                      environment: str = "prod", prompt_version: str = "1.2.0",
                      sample_rate: float = 0.1) -> dict:
    """The config shape to standardise on across your services."""
    import random

    traced = environment != "prod" or random.random() < sample_rate
    return {
        "configurable": {"thread_id": thread_id},
        "run_name": "research_loop",
        "tags": [f"env:{environment}", f"tenant:{tenant}", f"prompt:{prompt_version}"],
        "metadata": {"user_id": user_id, "tenant": tenant, "prompt_version": prompt_version,
                     "environment": environment},
        **({} if traced else {"callbacks": []}),
    }


example = production_config(thread_id="t-1", user_id="u-4821", tenant="acme")
print(json.dumps({k: v for k, v in example.items() if k != "callbacks"}, indent=2))

# %% [markdown]
# | Practice | Why |
# |---|---|
# | `run_name` on every entry point | Traces are searchable |
# | `thread_id` on every conversation | Turns group together |
# | `metadata` with tenant, user, prompt version | Slice cost and quality |
# | Tag the environment (`dev` / `staging` / `prod`) | Do not mix test traffic |
# | Separate LangSmith projects per environment | Same reason |
# | Sample traces at high volume | Full tracing at 10M req/day is costly |
# | Alert on error rate and p95 latency | Not just on process crashes |
# | A regression suite in CI | Catch quality drops before users do |
# | **Never put secrets in metadata or tags** | Traces are widely readable |

# %% [markdown]
# ## Try it yourself
#
# 1. **Debug a bad answer in LangSmith.** Invoke the graph with a wrong-ish
#    question, open the trace, and write down which check from the debugging
#    workflow (prompt → finish_reason → path → tools → cost) found the issue.
# 2. **Create `langgraph.json` for real.** Move one graph into `app/graphs.py`,
#    run `langgraph dev`, edit state mid-run in Studio.
# 3. **Find the expensive node.** Use `debug_run` and (if keyed) `list_runs` —
#    do they agree on which node dominates?
# 4. **Wire the gate into CI.** Make `regression_gate` exit non-zero on failure.
# 5. **Tag and slice.** Run 10 times with two `prompt_version` values; compare
#    mean score per version in the UI.

# %% [markdown]
# ## Recap
#
# | Concept | Takeaway |
# |---|---|
# | LangSmith | Trace, debug, dataset, evaluate, feedback, monitor |
# | Debugging order | Prompt → finish_reason → path → tools → tokens/latency → loops |
# | Graph traces | Parent run + per-node children + nested LLM/tool spans |
# | `run_name` / `tags` / `metadata` | Make runs searchable; always store `prompt_version` |
# | `@traceable` | Your Python becomes a span |
# | `Client.list_runs` | Latency/error queries without a browser |
# | Datasets + gate | Opinion → measurable regression check |
# | Studio + `langgraph.json` | Visual step-through; edit state; fork checkpoints |
# | `debug_run` | Offline stand-in for the waterfall |
# | Sampling | 100% in dev; a fraction in prod |
#
# ## Next
#
# -> [46_deployment_and_versioning.ipynb](46_deployment_and_versioning.ipynb)
