# %% [markdown]
# # 40 - Subgraphs
#
# | | |
# |---|---|
# | **Level** | Advanced (LangGraph) |
# | **Time** | 45 minutes |
# | **Prerequisites** | `39_time_travel` |
# | **Checklist ID** | `40_subgraphs` |
#
# ## Why this matters
#
# A graph with thirty nodes is unmaintainable for the same reason a thousand-line
# function is. Subgraphs are the refactoring tool: a compiled graph is itself a
# valid node, so you can build small, independently testable graphs and compose
# them.
#
# The one thing to get right is **state**. The parent and the child may share a
# schema, or they may not, and the two cases are wired differently.

# %%
from pathlib import Path
import sys

_root = Path.cwd()
while not (_root / "shared").is_dir() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from shared.notebook_setup import setup, require  # noqa: E402

ctx = setup("40_subgraphs")

# %%
import operator
from typing import Annotated, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from shared.llm import get_chat_model

model = get_chat_model()
parser = StrOutputParser()

# %% [markdown]
# ## 1. Shared state - the simple case
#
# When the child's schema overlaps the parent's, add the compiled graph directly
# as a node. State flows through as if the subgraph's nodes were inlined.

# %%
class DocState(TypedDict):
    text: str
    cleaned: str
    findings: Annotated[list[str], operator.add]


# --- child: a reusable quality-check pipeline ---------------------------- #
def strip_boilerplate(state: DocState) -> dict:
    cleaned = "\n".join(
        line for line in state["text"].splitlines()
        if not line.strip().startswith(("Confidential", "Page ", "---"))
    )
    return {"cleaned": cleaned.strip(), "findings": ["stripped boilerplate"]}


def check_length(state: DocState) -> dict:
    words = len(state["cleaned"].split())
    verdict = "too short" if words < 20 else "acceptable length"
    return {"findings": [f"{words} words - {verdict}"]}


quality = StateGraph(DocState)
quality.add_sequence([("strip", strip_boilerplate), ("length", check_length)])
quality.add_edge(START, "strip")
quality.add_edge("length", END)
quality_graph = quality.compile(name="quality_check")

# --- parent -------------------------------------------------------------- #
def summarise(state: DocState) -> dict:
    summary = (ChatPromptTemplate.from_template(
        "Summarise in one sentence:\n\n{cleaned}"
    ) | model | parser).invoke(state)
    return {"findings": [f"summary: {summary.strip()[:90]}"]}


parent = StateGraph(DocState)
parent.add_node("quality", quality_graph)         # a compiled graph used as a node
parent.add_node("summarise", summarise)
parent.add_edge(START, "quality")
parent.add_edge("quality", "summarise")
parent.add_edge("summarise", END)
pipeline = parent.compile()

print(pipeline.get_graph().draw_ascii())

# %%
RAW = """Confidential - internal only
Page 1

Employees accrue 24 days of annual leave per year. Up to 12 unused days may be
carried into the next year and must be used before 30 June.
---
Page 2"""

outcome = pipeline.invoke({"text": RAW, "cleaned": "", "findings": []})
for finding in outcome["findings"]:
    print(f"  {finding}")

# %% [markdown]
# The reducer on `findings` merged writes from both the child and the parent.
# Shared keys just work.
#
# ### Seeing inside

# %%
print(pipeline.get_graph(xray=True).draw_ascii())

# %% [markdown]
# `xray=True` expands subgraphs in the diagram. Essential when a graph is three
# levels deep and you have lost track of what runs where.

# %% [markdown]
# ## 2. Different schemas - the wrapper function
#
# More often the child has its own vocabulary. Then you do **not** add the
# compiled graph directly; you add a function that translates in and out.

# %%
class ReviewInput(TypedDict):
    content: str
    criteria: str


class ReviewOutput(TypedDict):
    score: int
    issues: list[str]


class ReviewInternal(TypedDict):
    content: str
    criteria: str
    raw_assessment: str
    score: int
    issues: list[str]


def assess(state: ReviewInternal) -> dict:
    assessment = (ChatPromptTemplate.from_template(
        "Assess this content against: {criteria}\n\nContent:\n{content}\n\n"
        "Reply with a score out of 10 on the first line, then one issue per line."
    ) | model | parser).invoke(state)
    return {"raw_assessment": assessment}


def parse_assessment(state: ReviewInternal) -> dict:
    lines = [line.strip() for line in state["raw_assessment"].splitlines() if line.strip()]
    score = 5
    for token in (lines[0] if lines else "").replace("/", " ").split():
        if token.isdigit():
            score = int(token)
            break
    return {"score": min(score, 10), "issues": [line.lstrip("-* ") for line in lines[1:4]]}


review = StateGraph(ReviewInternal, input_schema=ReviewInput, output_schema=ReviewOutput)
review.add_sequence([("assess", assess), ("parse", parse_assessment)])
review.add_edge(START, "assess")
review.add_edge("parse", END)
review_graph = review.compile(name="reviewer")

print("child input :", list(ReviewInput.__annotations__))
print("child output:", list(ReviewOutput.__annotations__))

# %%
class ArticleState(TypedDict):
    topic: str
    article: str
    quality_score: int
    revision_notes: Annotated[list[str], operator.add]


def write_article(state: ArticleState) -> dict:
    article = (ChatPromptTemplate.from_template(
        "Write a 100-word internal note about {topic}."
    ) | model | parser).invoke(state)
    return {"article": article}


def run_review(state: ArticleState) -> dict:
    """The wrapper: map parent state -> child input, child output -> parent state."""
    result = review_graph.invoke({
        "content": state["article"],
        "criteria": "clarity, specificity, no unsupported claims",
    })
    return {"quality_score": result["score"], "revision_notes": result["issues"]}


article = StateGraph(ArticleState)
article.add_node("write", write_article)
article.add_node("review", run_review)
article.add_edge(START, "write")
article.add_edge("write", "review")
article.add_edge("review", END)
article_graph = article.compile()

outcome = article_graph.invoke({"topic": "why we cap agent tool calls", "article": "",
                                "quality_score": 0, "revision_notes": []})
print(f"score: {outcome['quality_score']}/10")
for note in outcome["revision_notes"]:
    print(f"  - {note[:90]}")

# %% [markdown]
# **Which form to use:**
#
# | | Compiled graph as a node | Wrapper function |
# |---|---|---|
# | Schemas | Must share the keys that flow | Independent |
# | Coupling | Tight - child sees parent state | Loose - explicit contract |
# | Streaming | `subgraphs=True` shows internals | Opaque unless you stream inside |
# | Checkpoints | Child state is checkpointed | Child run is a single parent step |
# | Reuse across projects | Harder | **Easier** |
#
# Use the direct form for splitting one application into readable pieces. Use the
# wrapper form for genuinely reusable components.

# %% [markdown]
# ## 3. Streaming through subgraphs
#
# By default the parent's stream shows the subgraph as a single node update.

# %%
print("without subgraphs=True:")
for chunk in pipeline.stream({"text": RAW, "cleaned": "", "findings": []}, stream_mode="updates"):
    print("  ", list(chunk))

print("\nwith subgraphs=True:")
for namespace, chunk in pipeline.stream(
    {"text": RAW, "cleaned": "", "findings": []}, stream_mode="updates", subgraphs=True
):
    location = ".".join(part.split(":")[0] for part in namespace) or "parent"
    print(f"   {location:16} {list(chunk)}")

# %% [markdown]
# The namespace tuple tells you which subgraph instance produced the chunk -
# which matters when the same subgraph is used twice.

# %% [markdown]
# ## 4. Checkpointing and subgraphs
#
# A subgraph **inherits the parent's checkpointer**. Do not give it its own; if
# you do, compile it without one and let the parent provide it.

# %%
checkpointed = parent.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "sub-1"}}
checkpointed.invoke({"text": RAW, "cleaned": "", "findings": []}, config)

print("parent state keys:", sorted(checkpointed.get_state(config).values))

deep = checkpointed.get_state(config, subgraphs=True)
print("subgraphs=True gives access to nested task state:", deep.next or "(finished)")

# %% [markdown]
# ## 5. Interrupts inside a subgraph
#
# An `interrupt()` in a child propagates all the way up and is resumed from the
# parent. This is what makes approval steps composable.

# %%
from langgraph.types import Command, interrupt


class PaymentState(TypedDict):
    amount: float
    approved: bool
    log: Annotated[list[str], operator.add]


def approval_gate(state: PaymentState) -> dict:
    decision = interrupt({"amount": state["amount"], "question": "Approve this payment?"})
    return {"approved": decision == "approve", "log": [f"gate:{decision}"]}


gate = StateGraph(PaymentState)
gate.add_node("gate", approval_gate)
gate.add_edge(START, "gate")
gate.add_edge("gate", END)
gate_graph = gate.compile(name="approval_gate")


def execute_payment(state: PaymentState) -> dict:
    return {"log": [f"PAID {state['amount']}" if state["approved"] else "cancelled"]}


payment = StateGraph(PaymentState)
payment.add_node("approve", gate_graph)
payment.add_node("execute", execute_payment)
payment.add_edge(START, "approve")
payment.add_edge("approve", "execute")
payment.add_edge("execute", END)
payment_graph = payment.compile(checkpointer=InMemorySaver())

cfg = {"configurable": {"thread_id": "pay-1"}}
paused = payment_graph.invoke({"amount": 18400.0, "approved": False, "log": []}, cfg)
print("parent saw the child's interrupt:", paused["__interrupt__"][0].value)

resumed = payment_graph.invoke(Command(resume="approve"), cfg)
print("after resume:", resumed["log"])

# %% [markdown]
# The parent did not need to know the gate exists. You can drop the same
# `approval_gate` subgraph into any workflow that needs it.

# %% [markdown]
# ## 6. Reuse: the same subgraph twice

# %%
class MultiDocState(TypedDict):
    policy_text: str
    faq_text: str
    policy_findings: list[str]
    faq_findings: list[str]


def check(field: str, output_key: str):
    def node(state: MultiDocState) -> dict:
        result = quality_graph.invoke({"text": state[field], "cleaned": "", "findings": []})
        return {output_key: result["findings"]}
    return node


multi = StateGraph(MultiDocState)
multi.add_node("check_policy", check("policy_text", "policy_findings"))
multi.add_node("check_faq", check("faq_text", "faq_findings"))
multi.add_edge(START, "check_policy")
multi.add_edge(START, "check_faq")
multi.add_edge("check_policy", END)
multi.add_edge("check_faq", END)

outcome = multi.compile().invoke({
    "policy_text": Path(ctx.data("leave_policy.txt")).read_text(encoding="utf-8")[:600],
    "faq_text": Path(ctx.data("product_faq.md")).read_text(encoding="utf-8")[:600],
    "policy_findings": [], "faq_findings": [],
})
print("policy:", outcome["policy_findings"])
print("faq   :", outcome["faq_findings"])

# %% [markdown]
# Both ran in parallel, each with isolated child state, because the wrapper form
# keeps them independent.

# %% [markdown]
# ## 7. Testing subgraphs in isolation
#
# The biggest practical benefit. A child graph is a unit you can test without
# standing up the parent.

# %%
def test_strips_boilerplate() -> None:
    result = quality_graph.invoke({
        "text": "Confidential - internal only\nReal content here with enough words to pass the length check easily.",
        "cleaned": "", "findings": [],
    })
    assert "Confidential" not in result["cleaned"]
    assert any("stripped" in f for f in result["findings"])


def test_flags_short_documents() -> None:
    result = quality_graph.invoke({"text": "Too short.", "cleaned": "", "findings": []})
    assert any("too short" in f for f in result["findings"])


for test in (test_strips_boilerplate, test_flags_short_documents):
    test()
    print(f"  ok  {test.__name__}")

# %% [markdown]
# ## 8. When to extract a subgraph
#
# | Extract when | Keep inline when |
# |---|---|
# | The same steps appear in two workflows | It is used once |
# | A section has its own clear contract | The boundary would be arbitrary |
# | You want to unit-test a stage | The parent is under ~10 nodes |
# | A team owns that stage independently | You are still designing the flow |
# | The parent diagram no longer fits on screen | The diagram is still readable |
#
# The mistake to avoid is extracting too early. A subgraph adds a schema
# boundary; if the boundary is wrong you now maintain the wrong abstraction plus
# the translation code.

# %% [markdown]
# ## Try it yourself
#
# 1. **Extract a RAG subgraph** (retrieve, rerank, format) with its own
#    input/output schemas, and use it from two different parents.
# 2. **Nest three levels deep** and verify `xray=True` renders the whole thing.
# 3. **Stream from a nested interrupt.** Put the approval gate two levels down and
#    confirm `__interrupt__` still surfaces at the top.
# 4. **Write real tests.** Convert the two assertions above into a `pytest` file
#    that runs without any API key by stubbing the model.

# %% [markdown]
# ## Recap
#
# | Concept | Takeaway |
# |---|---|
# | Compiled graph as a node | Works directly when schemas share keys |
# | Wrapper function | Translates between independent schemas; looser coupling |
# | `input_schema` / `output_schema` | Give a child a clean public contract |
# | `xray=True` | Expand subgraphs in the diagram |
# | `subgraphs=True` on `stream` | Namespaced chunks from inside children |
# | Checkpointer | Inherited from the parent - do not give the child its own |
# | `interrupt()` in a child | Propagates to the parent; resume normally |
# | Reuse | The same subgraph can run twice, in parallel, with isolated state |
# | Testability | The main practical win - test a stage without the parent |
#
# ## Next
#
# -> [41_multi_agent_handoffs.ipynb](41_multi_agent_handoffs.ipynb)
