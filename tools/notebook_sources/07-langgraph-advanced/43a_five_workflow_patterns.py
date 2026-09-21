# %% [markdown]
# # 43a - The Five Workflow Patterns
#
# | | |
# |---|---|
# | **Level** | Advanced |
# | **Time** | 55 minutes |
# | **Prerequisites** | `32_conditional_routing`, `38_parallel_fanout_fanin`, `43_hybrid_deterministic_llm` |
# | **Checklist ID** | `43a_five_workflow_patterns` |
# | **Sourced from** | `LLG/4-Workflows` (chaining, parallelization, routing, orchestrator-worker, evaluator-optimizer) |
#
# ## Why this matters
#
# Notebooks 32, 38 and 43 teach the pieces. The LLG workflows course names the
# catalogue. Once you can name the pattern, you can pick it in a design review
# in ten seconds instead of reinventing it.
#
# | # | Pattern | One-line |
# |---|---|---|
# | 1 | **Prompt chaining** | Fixed sequence of LLM steps |
# | 2 | **Parallelization** | Fan-out, then fan-in |
# | 3 | **Routing** | Classify, then send to one specialist |
# | 4 | **Orchestrator-worker** | Planner spawns N workers dynamically (`Send`) |
# | 5 | **Evaluator-optimizer** | Generate → critique → revise until good enough |
#
# Patterns 1-3 you have seen. Patterns 4 and 5 are the gaps this lesson fills.

# %%
from pathlib import Path
import sys

_root = Path.cwd()
while not (_root / "shared").is_dir() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from shared.notebook_setup import setup  # noqa: E402

ctx = setup("43a_five_workflow_patterns")

# %%
import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from shared.llm import get_chat_model

model = get_chat_model()
parser = StrOutputParser()

# %% [markdown]
# ## Pattern 1 - Prompt chaining
#
# Fixed pipeline. No branching. Use when every input takes the same steps.

# %%
class ChainState(TypedDict):
    topic: str
    outline: str
    draft: str


def outline(state: ChainState) -> dict:
    text = (ChatPromptTemplate.from_template(
        "Write a 3-bullet outline for a blog post about: {topic}"
    ) | model | parser).invoke(state)
    return {"outline": text}


def draft(state: ChainState) -> dict:
    text = (ChatPromptTemplate.from_template(
        "Write a 120-word draft from this outline:\n{outline}"
    ) | model | parser).invoke(state)
    return {"draft": text}


b = StateGraph(ChainState)
b.add_sequence([("outline", outline), ("draft", draft)])
b.add_edge(START, "outline")
b.add_edge("draft", END)
chain = b.compile()

print(chain.invoke({"topic": "why RAG needs hybrid search", "outline": "", "draft": ""})["draft"][:300])

# %% [markdown]
# ## Pattern 2 - Parallelization
#
# Fan-out from one node, fan-in with a reducer. Notebook 38 covered this.

# %%
class ParallelState(TypedDict):
    topic: str
    takes: Annotated[list[str], operator.add]
    summary: str


def fan_out(state: ParallelState) -> list[Send]:
    return [Send("one_take", {"topic": state["topic"], "angle": angle})
            for angle in ("cost", "latency", "accuracy")]


def one_take(state: dict) -> dict:
    text = (ChatPromptTemplate.from_template(
        "In two sentences, the {angle} angle on: {topic}"
    ) | model | parser).invoke(state)
    return {"takes": [f"[{state['angle']}] {text.strip()}"]}


def summarise(state: ParallelState) -> dict:
    text = (ChatPromptTemplate.from_template(
        "Synthesize these angles into one paragraph:\n{takes}"
    ) | model | parser).invoke({"takes": "\n".join(state["takes"])})
    return {"summary": text}


b2 = StateGraph(ParallelState)
b2.add_node("one_take", one_take)
b2.add_node("summarise", summarise)
b2.add_conditional_edges(START, fan_out, ["one_take"])
b2.add_edge("one_take", "summarise")
b2.add_edge("summarise", END)
parallel = b2.compile()

print(parallel.invoke({"topic": "moving from FAISS to a managed vector DB",
                       "takes": [], "summary": ""})["summary"][:300])

# %% [markdown]
# ## Pattern 3 - Routing
#
# Classify, then send to exactly one specialist. Notebooks 25 and 32.

# %%
class Route(BaseModel):
    destination: Literal["billing", "technical", "general"]


class RouteState(TypedDict):
    question: str
    destination: str
    answer: str


def classify(state: RouteState) -> dict:
    route = (ChatPromptTemplate.from_template(
        "Classify as billing, technical, or general:\n{question}"
    ) | model.with_structured_output(Route)).invoke(state)
    return {"destination": route.destination}


def answer_billing(state: RouteState) -> dict:
    return {"answer": f"[billing] {state['question']}"}


def answer_technical(state: RouteState) -> dict:
    return {"answer": f"[technical] {state['question']}"}


def answer_general(state: RouteState) -> dict:
    return {"answer": f"[general] {state['question']}"}


b3 = StateGraph(RouteState)
b3.add_node("classify", classify)
b3.add_node("billing", answer_billing)
b3.add_node("technical", answer_technical)
b3.add_node("general", answer_general)
b3.add_edge(START, "classify")
b3.add_conditional_edges("classify", lambda s: s["destination"],
                         {"billing": "billing", "technical": "technical", "general": "general"})
b3.add_edge("billing", END)
b3.add_edge("technical", END)
b3.add_edge("general", END)
router = b3.compile()

print(router.invoke({"question": "Why was I charged twice this month?",
                     "destination": "", "answer": ""})["answer"])

# %% [markdown]
# ## Pattern 4 - Orchestrator-worker (the gap)
#
# The orchestrator **decides at runtime** how many workers to spawn and what
# each one does. That is `Send` with a dynamic list - not a fixed fan-out.
#
# Use when: the number of subtasks depends on the input (a report with N
# sections, a codebase with N files, a brief covering N competitors).

# %%
class Section(BaseModel):
    name: str = Field(description="Section title")
    brief: str = Field(description="What this section must cover")


class Plan(BaseModel):
    sections: list[Section]


class OrchState(TypedDict):
    topic: str
    sections: list[dict]
    completed: Annotated[list[str], operator.add]
    report: str


def orchestrate(state: OrchState) -> dict:
    plan = (ChatPromptTemplate.from_template(
        "Plan 3 short report sections for: {topic}. "
        "Each section needs a name and a one-sentence brief."
    ) | model.with_structured_output(Plan)).invoke(state)
    return {"sections": [s.model_dump() for s in plan.sections]}


def assign_workers(state: OrchState) -> list[Send]:
    return [Send("worker", {"section": section, "topic": state["topic"]})
            for section in state["sections"]]


def worker(state: dict) -> dict:
    section = state["section"]
    text = (ChatPromptTemplate.from_template(
        "Write a 60-word section called '{name}' for a report on {topic}.\n"
        "Brief: {brief}"
    ) | model | parser).invoke({"topic": state["topic"], **section})
    return {"completed": [f"## {section['name']}\n{text.strip()}"]}


def compile_report(state: OrchState) -> dict:
    return {"report": f"# {state['topic']}\n\n" + "\n\n".join(state["completed"])}


b4 = StateGraph(OrchState)
b4.add_node("orchestrate", orchestrate)
b4.add_node("worker", worker)
b4.add_node("compile", compile_report)
b4.add_edge(START, "orchestrate")
b4.add_conditional_edges("orchestrate", assign_workers, ["worker"])
b4.add_edge("worker", "compile")
b4.add_edge("compile", END)
orchestrator = b4.compile()

report = orchestrator.invoke({
    "topic": "Should Northwind adopt hybrid search?",
    "sections": [], "completed": [], "report": "",
})
print(report["report"][:500])
print(f"\nsections planned: {[s['name'] for s in report['sections']]}")

# %% [markdown]
# **Orchestrator-worker vs plain parallelization:** parallelization fans out a
# *fixed* list you wrote in code. Orchestrator-worker fans out a list the
# *model produced*. The second is how you handle inputs of unknown shape.

# %% [markdown]
# ## Pattern 5 - Evaluator-optimizer (the other gap)
#
# Generate → evaluate → revise until a structured critic approves, or you hit
# a cap. This is the writing-desk half of what notebook 48 does for retrieval,
# and what notebook 52's reviewer does for research briefs.

# %%
class Critique(BaseModel):
    passable: bool
    required_changes: list[str] = Field(default_factory=list)


class EOState(TypedDict):
    brief: str
    draft: str
    revisions: Annotated[int, operator.add]
    critiques: Annotated[list[dict], operator.add]


MAX_REVISIONS = 2


def generate(state: EOState) -> dict:
    if state["critiques"]:
        changes = "\n".join(f"- {c}" for c in state["critiques"][-1]["required_changes"])
        prompt = (f"Revise this draft.\n\nBrief: {state['brief']}\n\nDraft:\n{state['draft']}\n\n"
                  f"Required changes:\n{changes}")
    else:
        prompt = f"Write a 80-word answer for: {state['brief']}"
    draft = model.invoke(prompt).content
    return {"draft": draft, "revisions": 1 if state["critiques"] else 0}


def evaluate(state: EOState) -> dict:
    critique = (ChatPromptTemplate.from_template(
        "Critique this draft against the brief. Be severe.\n"
        "Approve only if it is specific, complete, and under 100 words.\n\n"
        "Brief: {brief}\n\nDraft:\n{draft}"
    ) | model.with_structured_output(Critique)).invoke(state)
    return {"critiques": [critique.model_dump()]}


def should_continue(state: EOState) -> Literal["generate", "__end__"]:
    latest = state["critiques"][-1]
    if latest["passable"]:
        return END
    if state["revisions"] >= MAX_REVISIONS:
        return END
    return "generate"


b5 = StateGraph(EOState)
b5.add_node("generate", generate)
b5.add_node("evaluate", evaluate)
b5.add_edge(START, "generate")
b5.add_edge("generate", "evaluate")
b5.add_conditional_edges("evaluate", should_continue, {"generate": "generate", END: END})
eo = b5.compile()

result = eo.invoke({"brief": "Explain hybrid search to an engineering manager in under 100 words.",
                    "draft": "", "revisions": 0, "critiques": []},
                   {"recursion_limit": 20})
print(f"revisions: {result['revisions']}")
print(f"verdicts:  {[c['passable'] for c in result['critiques']]}")
print(f"\n{result['draft']}")

# %% [markdown]
# ## Choosing a pattern
#
# ```
# Is the step sequence fixed?
#   yes -> chaining
#   no  -> Does one input need many independent subtasks?
#            yes -> Is the subtask list known in code?
#                     yes -> parallelization
#                     no  -> orchestrator-worker
#            no  -> Does quality need a generate/critique loop?
#                     yes -> evaluator-optimizer
#                     no  -> routing (one of N specialists)
# ```

# %%
def pick(fixed: bool, many_subtasks: bool, subtasks_known: bool, needs_critique: bool) -> str:
    if fixed:
        return "chaining"
    if many_subtasks:
        return "parallelization" if subtasks_known else "orchestrator-worker"
    return "evaluator-optimizer" if needs_critique else "routing"


for label, args in [
    ("summarise a ticket", (True, False, True, False)),
    ("score a resume on 4 fixed rubrics", (False, True, True, False)),
    ("write a report with N sections from a brief", (False, True, False, False)),
    ("draft a tweet until a critic approves", (False, False, False, True)),
    ("billing vs technical support", (False, False, False, False)),
]:
    print(f"{label:48} -> {pick(*args)}")

# %% [markdown]
# ## Try it yourself
#
# 1. Extend the orchestrator to plan *between 2 and 5* sections based on topic
#    complexity, and assert the worker count matches.
# 2. Add a hard word-count check to the evaluator so "passable" also requires
#    `len(draft.split()) <= 100` in code - not just in the prompt.
# 3. Combine patterns: route a request to either a chain or an
#    evaluator-optimizer depending on whether the user asked for "quick" or
#    "polished".
#
# ## Recap
#
# | Pattern | Mechanism | Use when |
# |---|---|---|
# | Chaining | Fixed edges | Every input takes the same steps |
# | Parallelization | Fixed `Send` list | Known independent subtasks |
# | Routing | Conditional edge | One of N specialists |
# | **Orchestrator-worker** | Dynamic `Send` from a plan | Subtask list depends on the input |
# | **Evaluator-optimizer** | Generate ↔ critique loop with a cap | Quality needs iteration |
#
# ## Next
#
# -> [44_error_retries_tool_failures.ipynb](44_error_retries_tool_failures.ipynb)
