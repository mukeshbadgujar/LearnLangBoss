# %% [markdown]
# # 45 - LangSmith and LangGraph Studio
#
# | | |
# |---|---|
# | **Level** | Advanced (LangGraph) |
# | **Time** | 40 minutes |
# | **Prerequisites** | `44_error_retries_tool_failures`, `19_langsmith` |
# | **Checklist ID** | `45_langsmith_and_studio` |
#
# ## Why this matters
#
# `print()` works until your graph has twelve nodes, three of which loop. Then
# you need to see the actual execution tree: which nodes ran, in what order, what
# each one received, how long it took and what it cost.
#
# Notebook 19 covered LangSmith for chains. Graphs add structure - nested spans
# per node, subgraph boundaries, loop iterations - and LangGraph Studio adds a
# visual debugger where you can step through a graph and edit its state live.

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
# Two environment variables and every graph run is traced. No code changes.
#
# ```bash
# LANGSMITH_TRACING=true
# LANGSMITH_API_KEY=lsv2_...
# LANGSMITH_PROJECT=genai-mastery      # optional, groups runs
# ```
#
# `shared/notebook_setup.py` reads these from `.env` and calls `enable_tracing()`
# for you.

# %%
from shared.llm import describe_environment, has_key

print(describe_environment())
if not has_key("LANGSMITH_API_KEY"):
    print("\nGet a free key at https://smith.langchain.com - the rest of this notebook still runs.")

# %% [markdown]
# ## 2. A graph worth tracing
#
# Something with a loop, a branch and a tool call - so the trace has structure.

# %%
class ResearchState(TypedDict):
    question: str
    plan: str
    draft: str
    score: int
    revisions: Annotated[int, operator.add]
    trace: Annotated[list[str], operator.add]


class Review(BaseModel):
    """A reviewer's verdict on a draft."""

    score: int = Field(description="Quality out of 10", ge=1, le=10)
    issue: str = Field(description="The single most important thing to fix")


reviewer = model.with_structured_output(Review)


def plan_node(state: ResearchState) -> dict:
    plan = (ChatPromptTemplate.from_template(
        "Write a 3-bullet plan for answering: {question}"
    ) | model | parser).invoke(state)
    return {"plan": plan, "trace": ["plan"]}


def draft_node(state: ResearchState) -> dict:
    template = ("Answer in 80 words following this plan.\n\nQuestion: {question}\nPlan: {plan}"
                if not state["draft"] else
                "Improve this answer.\n\nQuestion: {question}\nDraft: {draft}\nFix: {trace}")
    draft = (ChatPromptTemplate.from_template(template) | model | parser).invoke(
        {**state, "trace": state["trace"][-1]}
    )
    return {"draft": draft, "revisions": 1, "trace": ["draft"]}


def review_node(state: ResearchState) -> dict:
    verdict = reviewer.invoke([("system", "Review this answer strictly."),
                               ("human", f"Q: {state['question']}\n\nA: {state['draft']}")])
    return {"score": verdict.score, "trace": [f"review:{verdict.score} - {verdict.issue}"]}


def keep_going(state: ResearchState) -> Literal["draft", "__end__"]:
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
# ## 3. Making traces readable
#
# A trace full of `RunnableSequence` entries is barely better than no trace. Name
# and tag your runs.

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
# | Config key | What it buys you in the UI |
# |---|---|
# | `run_name` | A readable title instead of `LangGraph` |
# | `tags` | Filter chips - `track-07`, `v1`, `tenant:acme` |
# | `metadata` | Searchable key/value; group cost by tenant or prompt version |
# | `thread_id` | Groups every turn of one conversation |
#
# **Always include `prompt_version`.** Three weeks later, "did quality drop when
# we shipped 1.2.0?" is answerable only if it is in the metadata.

# %% [markdown]
# ## 4. What a graph trace looks like
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
# Three things to read off it immediately:
#
# 1. **Which node dominates** the latency (here: `draft`, twice).
# 2. **How many loop iterations** actually ran.
# 3. **Where tokens went** - often the surprise is an unnoticed second call.

# %% [markdown]
# ## 5. Tracing custom functions with `@traceable`
#
# Plain Python inside a node is invisible unless you mark it.

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
# Nested `@traceable` functions nest in the trace too, so `rank_candidates`
# shows three `score_relevance` children.

# %% [markdown]
# ## 6. Finding the slow step programmatically
#
# You do not always want to open a browser. The client API answers latency
# questions directly.

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
            print(f"no runs yet in project {project!r}")
    except Exception as exc:
        print(f"[skipped] {type(exc).__name__}: {str(exc)[:120]}")

# %% [markdown]
# ## 7. Datasets and regression testing for graphs
#
# The same evaluation discipline as notebook 26, applied to a whole graph.

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
# You do not need LangSmith to run a regression suite. This version works
# offline and is what you would put in CI.

# %%
def run_case(case: dict) -> dict:
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
    """Raise in CI if quality regressed."""
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
# Studio is a visual IDE for graphs: run them, watch execution animate node by
# node, inspect and **edit** state at any step, and fork from any checkpoint -
# the time travel of notebook 39, with a UI.
#
# ### Setting it up
#
# Studio needs a `langgraph.json` at the project root describing your graphs.

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
# The graph module exports a **compiled** graph at module level:
#
# ```python
# # app/graphs.py
# from langgraph.graph import StateGraph, START, END
#
# builder = StateGraph(ResearchState)
# ...
# research_graph = builder.compile()      # no checkpointer - the platform supplies one
# ```
#
# Then:
#
# ```bash
# python -m pip install --upgrade "langgraph-cli[inmem]"
# langgraph dev            # starts the API and opens Studio in the browser
# ```
#
# `langgraph dev` runs everything locally in memory - no Docker, no account.
#
# ### What Studio gives you
#
# | Capability | Why it matters |
# |---|---|
# | Visual graph with live node highlighting | See the actual path, including loops |
# | Inspect state at any step | No `print` statements |
# | **Edit state and continue** | Test a branch without changing code |
# | Fork from any checkpoint | Compare two futures side by side |
# | Interrupt UI | Approve/reject human-in-the-loop steps by hand |
# | Trace link | Jump straight into the LangSmith trace |
#
# The state-editing feature is the one that changes how you work. Reproducing
# "what if the classifier had said billing?" becomes a two-second edit instead of
# a code change and a rerun.

# %% [markdown]
# ## 9. Debugging without any of it
#
# Studio and LangSmith are conveniences. When you have neither, this function
# gets you most of the way.

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
# ## 10. Production observability checklist
#
# | Practice | Why |
# |---|---|
# | `run_name` on every entry point | Traces are searchable |
# | `thread_id` on every conversation | Turns group together |
# | `metadata` with tenant, user, prompt version | Slice cost and quality |
# | Tag the environment (`dev` / `staging` / `prod`) | Do not mix test traffic |
# | Separate LangSmith projects per environment | Same reason |
# | Sample traces at high volume | Tracing 100% of 10M requests is expensive |
# | Alert on error rate and p95 latency | Not just on crashes |
# | A regression suite in CI | Catch quality drops before users do |
# | **Never put secrets in metadata or tags** | Traces are widely readable |

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
# ## Try it yourself
#
# 1. **Create `langgraph.json` for real.** Move one of your graphs into
#    `app/graphs.py`, run `langgraph dev`, and step through it in Studio.
# 2. **Find the expensive node.** Use `debug_run` on the research loop and work
#    out how much each revision iteration costs.
# 3. **Wire the gate into CI.** Make `regression_gate` a script that exits
#    non-zero, and run it on every change to a prompt.
# 4. **Tag and slice.** Run the graph 10 times with two different
#    `prompt_version` tags and compare mean score per version.

# %% [markdown]
# ## Recap
#
# | Concept | Takeaway |
# |---|---|
# | Enable tracing | Two env vars; no code change |
# | `run_name`, `tags`, `metadata` | Turn an unreadable trace into a searchable one |
# | `prompt_version` in metadata | The only way to answer "did the change hurt?" later |
# | Graph traces | Show node nesting, loop iterations and per-node cost |
# | `@traceable` | Make custom Python visible as its own span |
# | `client.list_runs` | Query latency and errors without a browser |
# | Local regression suite | Works offline; belongs in CI |
# | `langgraph.json` + `langgraph dev` | Runs Studio locally, in memory, no Docker |
# | Studio's killer feature | Edit state mid-run and fork from any checkpoint |
# | `debug_run` | Node-by-node timings when you have no tooling at all |
# | Sampling | Trace 100% in dev, a fraction in production |
#
# ## Next
#
# -> [46_deployment_and_versioning.ipynb](46_deployment_and_versioning.ipynb)
