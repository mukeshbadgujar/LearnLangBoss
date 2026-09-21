# %% [markdown]
# # 48 - Self-Reflective RAG
#
# | | |
# |---|---|
# | **Level** | Advanced (LangGraph) |
# | **Time** | 55 minutes |
# | **Prerequisites** | `47_token_limits_and_summarization_nodes`, `15_advanced_rag` |
# | **Checklist ID** | `48_self_reflective_rag` |
#
# ## Why this matters
#
# Standard RAG retrieves once and answers. When retrieval fails - wrong
# documents, no documents, or documents that look relevant but are not - the
# model answers anyway, confidently, from whatever it got.
#
# Self-reflective RAG adds explicit checkpoints: *are these documents actually
# relevant? is my answer grounded in them? did I answer the question?* Each
# check is a node, and a failed check routes to a repair. This is the
# architecture behind CRAG and Self-RAG, and it is what turns a demo into
# something you can point at an employee handbook.

# %%
from pathlib import Path
import sys

_root = Path.cwd()
while not (_root / "shared").is_dir() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from shared.notebook_setup import setup, require  # noqa: E402

ctx = setup("48_self_reflective_rag")

# %%
import operator
from typing import Annotated, Literal, TypedDict

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from shared.llm import get_chat_model, get_embeddings

model = get_chat_model()
parser = StrOutputParser()

# %% [markdown]
# ## 1. The corpus and a baseline

# %%
def build_corpus() -> list[Document]:
    docs: list[Document] = []
    for name in ("company_handbook.md", "product_faq.md"):
        text = Path(ctx.data(name)).read_text(encoding="utf-8")
        for section in MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "doc"), ("##", "section")], strip_headers=False
        ).split_text(text):
            section.metadata["source"] = name
            docs.append(section)
    policy = Path(ctx.data("leave_policy.txt")).read_text(encoding="utf-8")
    for chunk in RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=70).split_text(policy):
        docs.append(Document(chunk, metadata={"source": "leave_policy.txt"}))
    return RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=70).split_documents(docs)


store = FAISS.from_documents(build_corpus(), get_embeddings())
retriever = store.as_retriever(search_kwargs={"k": 4})


def format_docs(docs) -> str:
    return "\n\n".join(f"[{d.metadata.get('source')}] {d.page_content}" for d in docs)


ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Answer using only the context. Cite the source filename. "
               "If the context does not contain the answer, say so plainly."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])

QUESTIONS = [
    "How many days of casual leave are allowed?",
    "What is our policy on pet insurance?",          # not in the corpus
    "What is the thing about the days off?",          # badly phrased
]

for question in QUESTIONS:
    docs = retriever.invoke(question)
    answer = (ANSWER_PROMPT | model | parser).invoke({"context": format_docs(docs), "question": question})
    print(f"\nQ: {question}")
    print(f"   sources: {[d.metadata['source'] for d in docs]}")
    print(f"   {answer.strip()[:150]}")

# %% [markdown]
# The first is fine. The second may or may not refuse. The third retrieves
# something vaguely leave-related and answers as if that were what was asked.
# Reflection addresses all three.

# %% [markdown]
# ## 2. The graders
#
# Four small structured-output calls. Each answers exactly one question.

# %%
class Relevance(BaseModel):
    """Is this single document relevant to the question?"""

    relevant: bool = Field(description="True only if the document helps answer the question")
    reason: str = Field(description="Under 15 words")


class Grounded(BaseModel):
    """Is the answer supported by the documents?"""

    grounded: bool = Field(description="False if the answer states anything not in the documents")
    unsupported_claim: str = Field(default="", description="The first unsupported claim, if any")


class Answers(BaseModel):
    """Does the answer address the question that was asked?"""

    addresses_question: bool
    missing: str = Field(default="", description="What the question asked for but the answer omitted")


class Rewrite(BaseModel):
    """A better search query."""

    query: str = Field(description="A rewritten query using vocabulary likely to appear in the documents")


relevance_grader = ChatPromptTemplate.from_messages([
    ("system", "Grade document relevance. Be strict: topical overlap is not relevance. "
               "The document must contain information that helps answer the question."),
    ("human", "Question: {question}\n\nDocument:\n{document}"),
]) | model.with_structured_output(Relevance)

grounding_grader = ChatPromptTemplate.from_messages([
    ("system", "Check whether every factual claim in the answer appears in the documents. "
               "A claim that is true in the real world but absent from the documents is NOT grounded."),
    ("human", "Documents:\n{context}\n\nAnswer:\n{answer}"),
]) | model.with_structured_output(Grounded)

usefulness_grader = ChatPromptTemplate.from_messages([
    ("system", "Check whether the answer addresses what was asked. A correct refusal counts as addressing it."),
    ("human", "Question: {question}\n\nAnswer:\n{answer}"),
]) | model.with_structured_output(Answers)

query_rewriter = ChatPromptTemplate.from_messages([
    ("system", "Rewrite the user's question as a search query that will match formal policy "
               "documentation. Expand vague terms into the formal vocabulary a handbook would use."),
    ("human", "{question}"),
]) | model.with_structured_output(Rewrite)

# %%
docs = retriever.invoke("What is our policy on pet insurance?")
for doc in docs[:3]:
    verdict = relevance_grader.invoke({"question": "What is our policy on pet insurance?",
                                       "document": doc.page_content})
    print(f"  relevant={verdict.relevant!s:5} [{doc.metadata['source']:20}] {verdict.reason}")

# %% [markdown]
# ## 3. The reflective graph
#
# ```
#            retrieve
#               |
#          grade_documents
#          /            \
#   (none relevant)   (some relevant)
#         |                  |
#    rewrite_query       generate
#         |              /    |    \
#     retrieve   (not grounded) (doesn't answer) (good)
#                    |            |               |
#                 generate   rewrite_query       END
# ```

# %%
class ReflectiveState(TypedDict):
    question: str
    search_query: str
    documents: list[Document]
    answer: str
    retries: Annotated[int, operator.add]
    generations: Annotated[int, operator.add]
    trace: Annotated[list[str], operator.add]


MAX_RETRIES = 2
MAX_GENERATIONS = 2


def retrieve_node(state: ReflectiveState) -> dict:
    query = state["search_query"] or state["question"]
    docs = retriever.invoke(query)
    return {"documents": docs, "trace": [f"retrieve({query[:40]}) -> {len(docs)} docs"]}


def grade_documents(state: ReflectiveState) -> dict:
    kept = []
    for doc in state["documents"]:
        verdict = relevance_grader.invoke({"question": state["question"], "document": doc.page_content})
        if verdict.relevant:
            kept.append(doc)
    return {"documents": kept,
            "trace": [f"grade -> {len(kept)}/{len(state['documents'])} relevant"]}


def rewrite_query(state: ReflectiveState) -> dict:
    better = query_rewriter.invoke({"question": state["question"]})
    return {"search_query": better.query, "retries": 1, "trace": [f"rewrite -> {better.query[:48]}"]}


def generate(state: ReflectiveState) -> dict:
    answer = (ANSWER_PROMPT | model | parser).invoke({
        "context": format_docs(state["documents"]), "question": state["question"]
    })
    return {"answer": answer, "generations": 1, "trace": ["generate"]}


def give_up(state: ReflectiveState) -> dict:
    return {"answer": "I could not find this in the documents I have access to. "
                      "Please contact the HR or support team directly.",
            "trace": ["give_up: no relevant documents after retries"]}


# --- routing (deterministic, per notebook 43) ---------------------------- #
def after_grading(state: ReflectiveState) -> Literal["generate", "rewrite", "give_up"]:
    if state["documents"]:
        return "generate"
    if state["retries"] >= MAX_RETRIES:
        return "give_up"
    return "rewrite"


def after_generation(state: ReflectiveState) -> Literal["generate", "rewrite", "__end__"]:
    if state["generations"] >= MAX_GENERATIONS:
        return END

    grounded = grounding_grader.invoke({"context": format_docs(state["documents"]), "answer": state["answer"]})
    if not grounded.grounded:
        return "generate"          # same docs, try again - the answer drifted

    useful = usefulness_grader.invoke({"question": state["question"], "answer": state["answer"]})
    if not useful.addresses_question and state["retries"] < MAX_RETRIES:
        return "rewrite"           # docs were fine but did not cover the question

    return END


builder = StateGraph(ReflectiveState)
builder.add_node("retrieve", retrieve_node)
builder.add_node("grade", grade_documents)
builder.add_node("rewrite", rewrite_query)
builder.add_node("generate", generate)
builder.add_node("give_up", give_up)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "grade")
builder.add_conditional_edges("grade", after_grading,
                              {"generate": "generate", "rewrite": "rewrite", "give_up": "give_up"})
builder.add_edge("rewrite", "retrieve")
builder.add_conditional_edges("generate", after_generation,
                              {"generate": "generate", "rewrite": "rewrite", END: END})
builder.add_edge("give_up", END)

reflective = builder.compile()
print(reflective.get_graph().draw_ascii())

# %%
def ask(question: str) -> dict:
    return reflective.invoke({"question": question, "search_query": "", "documents": [],
                              "answer": "", "retries": 0, "generations": 0, "trace": []})


for question in QUESTIONS:
    outcome = ask(question)
    print(f"\nQ: {question}")
    for step in outcome["trace"]:
        print(f"    {step}")
    print(f"  -> {outcome['answer'].strip()[:180]}")

# %% [markdown]
# Three different paths through the same graph:
#
# - The good question retrieved, graded, generated, and stopped.
# - The unanswerable question found nothing relevant, rewrote, still found
#   nothing, and refused honestly rather than inventing a pet insurance policy.
# - The vague question was rewritten into policy vocabulary and then succeeded.

# %% [markdown]
# ## 4. Grading in parallel
#
# Grading four documents sequentially is four round trips. Fan out instead.

# %%
from langgraph.types import Send


class ParallelGradeState(TypedDict):
    question: str
    documents: list[Document]
    relevant: Annotated[list[Document], operator.add]
    answer: str
    trace: Annotated[list[str], operator.add]


def fan_out_grading(state: ParallelGradeState) -> list[Send]:
    return [Send("grade_one", {"question": state["question"], "document": doc})
            for doc in state["documents"]]


def grade_one(state: dict) -> dict:
    verdict = relevance_grader.invoke({"question": state["question"],
                                       "document": state["document"].page_content})
    return {"relevant": [state["document"]] if verdict.relevant else []}


def generate_parallel(state: ParallelGradeState) -> dict:
    if not state["relevant"]:
        return {"answer": "Not covered in the documents I have.", "trace": ["no relevant documents"]}
    answer = (ANSWER_PROMPT | model | parser).invoke({
        "context": format_docs(state["relevant"]), "question": state["question"]
    })
    return {"answer": answer, "trace": [f"generated from {len(state['relevant'])} docs"]}


builder2 = StateGraph(ParallelGradeState)
builder2.add_node("retrieve", lambda s: {"documents": retriever.invoke(s["question"]),
                                         "trace": ["retrieve"]})
builder2.add_node("grade_one", grade_one)
builder2.add_node("generate", generate_parallel)
builder2.add_edge(START, "retrieve")
builder2.add_conditional_edges("retrieve", fan_out_grading, ["grade_one"])
builder2.add_edge("grade_one", "generate")
builder2.add_edge("generate", END)
parallel_graded = builder2.compile()

import time

question = "How many days of casual leave are allowed?"
start = time.perf_counter()
outcome = parallel_graded.invoke({"question": question, "documents": [], "relevant": [],
                                  "answer": "", "trace": []})
print(f"parallel grading: {time.perf_counter() - start:.1f}s, {outcome['trace']}")
print(f"  {outcome['answer'].strip()[:160]}")

# %% [markdown]
# ## 5. CRAG - falling back to another source
#
# Corrective RAG adds a third outcome to grading: not just relevant/irrelevant,
# but *ambiguous*, which triggers a fallback to an external source.

# %%
class CRAGState(TypedDict):
    question: str
    documents: list[Document]
    confidence: str
    supplemented: bool
    answer: str
    trace: Annotated[list[str], operator.add]


def assess_corpus(state: CRAGState) -> dict:
    """Score the retrieval as a whole, not document by document."""
    docs = retriever.invoke(state["question"])
    relevant = []
    for doc in docs:
        if relevance_grader.invoke({"question": state["question"],
                                    "document": doc.page_content}).relevant:
            relevant.append(doc)

    ratio = len(relevant) / max(len(docs), 1)
    confidence = "high" if ratio >= 0.5 else ("ambiguous" if relevant else "low")
    return {"documents": relevant, "confidence": confidence,
            "trace": [f"assess: {len(relevant)}/{len(docs)} relevant -> {confidence}"]}


def supplement(state: CRAGState) -> dict:
    """Fallback source. Use a real web search tool here in production."""
    if require("TAVILY_API_KEY", feature="live web fallback"):
        from langchain_tavily import TavilySearch

        hits = TavilySearch(max_results=2).invoke({"query": state["question"]})
        extra = [Document(str(h.get("content", ""))[:600], metadata={"source": "web"})
                 for h in (hits.get("results") or [])]
    else:
        extra = [Document(
            "General guidance: where internal policy is silent, the statutory minimum applies and "
            "employees should confirm with HR before relying on any entitlement.",
            metadata={"source": "general-guidance"},
        )]
    return {"documents": state["documents"] + extra, "supplemented": True,
            "trace": [f"supplemented with {len(extra)} external source(s)"]}


def crag_answer(state: CRAGState) -> dict:
    if not state["documents"]:
        return {"answer": "I have no reliable source for this. Please ask HR directly.",
                "trace": ["refused"]}
    caveat = "\n\nNote: this draws partly on external sources, not only internal policy." \
        if state["supplemented"] else ""
    answer = (ANSWER_PROMPT | model | parser).invoke({
        "context": format_docs(state["documents"]), "question": state["question"]
    })
    return {"answer": answer.strip() + caveat, "trace": ["answered"]}


def crag_route(state: CRAGState) -> Literal["supplement", "answer"]:
    return "answer" if state["confidence"] == "high" else "supplement"


builder3 = StateGraph(CRAGState)
builder3.add_node("assess", assess_corpus)
builder3.add_node("supplement", supplement)
builder3.add_node("answer", crag_answer)
builder3.add_edge(START, "assess")
builder3.add_conditional_edges("assess", crag_route, {"supplement": "supplement", "answer": "answer"})
builder3.add_edge("supplement", "answer")
builder3.add_edge("answer", END)
crag = builder3.compile()

for question in ["How many days of casual leave are allowed?",
                 "What happens to my leave if I go on unpaid sabbatical?"]:
    outcome = crag.invoke({"question": question, "documents": [], "confidence": "",
                           "supplemented": False, "answer": "", "trace": []})
    print(f"\nQ: {question}")
    for step in outcome["trace"]:
        print(f"    {step}")
    print(f"  -> {outcome['answer'].strip()[:200]}")

# %% [markdown]
# **The caveat matters.** An answer partly built on external sources must say so
# - otherwise you have quietly turned a policy lookup into a guess.

# %% [markdown]
# ## 6. The cost of reflection
#
# Reflection is not free. Measure it before shipping.

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


def plain_rag(question: str, config=None):
    docs = retriever.invoke(question)
    return (ANSWER_PROMPT | model | parser).invoke(
        {"context": format_docs(docs), "question": question}, config=config
    )


print(f"{'approach':18} {'calls':>6} {'in tokens':>10} {'seconds':>8}")
for label, run in [
    ("plain RAG", lambda cb: plain_rag(QUESTIONS[0], {"callbacks": [cb]})),
    ("reflective", lambda cb: reflective.invoke(
        {"question": QUESTIONS[0], "search_query": "", "documents": [], "answer": "",
         "retries": 0, "generations": 0, "trace": []}, {"callbacks": [cb]})),
    ("parallel-graded", lambda cb: parallel_graded.invoke(
        {"question": QUESTIONS[0], "documents": [], "relevant": [], "answer": "", "trace": []},
        {"callbacks": [cb]})),
]:
    meter = Meter()
    start = time.perf_counter()
    run(meter)
    print(f"{label:18} {meter.calls:>6} {meter.input_tokens:>10,} {time.perf_counter() - start:>8.1f}")

# %% [markdown]
# Reflection typically costs 4-8x plain RAG. Two ways to make that acceptable:

# %%
# 1. Use a small, cheap model for the graders.
from shared.llm import available_providers

try:
    grader_model = (get_chat_model("llama-3.1-8b-instant")
                    if available_providers() and available_providers()[0] == "groq" else model)
except Exception:
    grader_model = model
print("grader model configured (graders are classification tasks - a small model is enough)")

# 2. Only reflect when the retrieval looks weak.
def retrieval_confidence(question: str, k: int = 4, threshold: float = 0.75) -> tuple[str, float]:
    """Use the vector store's own scores to decide whether reflection is needed."""
    scored = store.similarity_search_with_score(question, k=k)
    best = min(score for _, score in scored)        # FAISS L2: lower is better
    return ("high" if best < threshold else "low"), best


for question in QUESTIONS:
    level, score = retrieval_confidence(question)
    print(f"  {level:5} (score {score:.3f})  reflect={level == 'low'}  {question[:48]}")

# %% [markdown]
# That is the practical pattern: **cheap signal first, expensive reflection only
# when the cheap signal says something is wrong.** Most questions skip the
# graders entirely.

# %%
class AdaptiveState(TypedDict):
    question: str
    search_query: str
    documents: list[Document]
    answer: str
    retries: Annotated[int, operator.add]
    generations: Annotated[int, operator.add]
    reflected: bool
    trace: Annotated[list[str], operator.add]


def triage_retrieval(state: AdaptiveState) -> dict:
    level, score = retrieval_confidence(state["question"])
    return {"reflected": level == "low", "trace": [f"triage: {level} ({score:.3f})"]}


def fast_path(state: AdaptiveState) -> dict:
    docs = retriever.invoke(state["question"])
    answer = (ANSWER_PROMPT | model | parser).invoke({"context": format_docs(docs),
                                                      "question": state["question"]})
    return {"documents": docs, "answer": answer, "trace": ["fast path"]}


def slow_path(state: AdaptiveState) -> dict:
    outcome = reflective.invoke({"question": state["question"], "search_query": "", "documents": [],
                                 "answer": "", "retries": 0, "generations": 0, "trace": []})
    return {"answer": outcome["answer"], "documents": outcome["documents"],
            "trace": ["slow path"] + outcome["trace"]}


builder4 = StateGraph(AdaptiveState)
builder4.add_node("triage", triage_retrieval)
builder4.add_node("fast", fast_path)
builder4.add_node("slow", slow_path)
builder4.add_edge(START, "triage")
builder4.add_conditional_edges("triage", lambda s: "slow" if s["reflected"] else "fast",
                               {"fast": "fast", "slow": "slow"})
builder4.add_edge("fast", END)
builder4.add_edge("slow", END)
adaptive = builder4.compile()

for question in QUESTIONS:
    meter = Meter()
    outcome = adaptive.invoke({"question": question, "search_query": "", "documents": [], "answer": "",
                               "retries": 0, "generations": 0, "reflected": False, "trace": []},
                              {"callbacks": [meter]})
    print(f"\n{'SLOW' if outcome['reflected'] else 'FAST'} ({meter.calls} calls)  {question[:46]}")
    print(f"  {outcome['answer'].strip()[:140]}")

# %% [markdown]
# ## 7. Measuring whether reflection helps
#
# It does not always. Measure before you keep it.

# %%
EVAL = [
    ("How many days of annual leave do employees get?", "24", True),
    ("How many sick leave days per year?", "12", True),
    ("What is the notice period for an L5 engineer?", "90", True),
    ("What is the API rate limit on Growth?", "600", True),
    ("What is our policy on pet insurance?", None, False),
    ("How many employees does Northwind have?", None, False),
]
REFUSALS = ["not covered", "could not find", "do not have", "don't have", "no reliable source", "contact"]


def score(run_fn, label: str) -> None:
    facts = refusals = 0
    answerable = sum(1 for _, _, a in EVAL if a)
    unanswerable = len(EVAL) - answerable

    for question, fact, answerable_flag in EVAL:
        answer = run_fn(question).lower()
        if answerable_flag:
            facts += fact in answer
        else:
            refusals += any(marker in answer for marker in REFUSALS)

    print(f"{label:16} facts {facts}/{answerable}   correct refusals {refusals}/{unanswerable}")


print(f"{'approach':16} accuracy")
score(lambda q: plain_rag(q), "plain RAG")
score(lambda q: ask(q)["answer"], "reflective")

# %% [markdown]
# The number to watch is **correct refusals**. That is where reflection earns
# its cost: a plain pipeline answers the pet insurance question, and a
# reflective one declines.

# %% [markdown]
# ## 8. Design notes
#
# **Cap every loop.** `MAX_RETRIES` and `MAX_GENERATIONS` are not optional; a
# grader that never says "good enough" will spend your budget.
#
# **Grade documents individually, corpus-wide decisions separately.** Per-document
# relevance filters noise; a corpus-level confidence score decides whether to
# fall back. They answer different questions.
#
# **A refusal is a success.** Do not tune the graders until refusals disappear -
# you will have tuned away the feature.
#
# **Log every grader decision.** The `trace` field is how you find out that your
# relevance grader rejects 80% of a perfectly good corpus.

# %%
def audit_grader(questions: list[str]) -> None:
    """Sanity-check the relevance grader before trusting it in the loop."""
    total = kept = 0
    for question in questions:
        for doc in retriever.invoke(question):
            total += 1
            kept += relevance_grader.invoke({"question": question,
                                             "document": doc.page_content}).relevant
    rate = kept / max(total, 1)
    verdict = ("too strict - it will trigger needless rewrites" if rate < 0.2
               else "too lenient - it is not filtering anything" if rate > 0.9
               else "reasonable")
    print(f"relevance grader keeps {kept}/{total} documents ({rate:.0%}) - {verdict}")


audit_grader([q for q, _, a in EVAL if a])

# %% [markdown]
# ## Try it yourself
#
# 1. **Add a hallucination repair loop** that, when grounding fails, regenerates
#    with an explicit instruction naming the unsupported claim.
# 2. **Self-RAG style retrieval decision.** Add a node that decides whether to
#    retrieve at all - some questions ("hello") need no documents.
# 3. **Tune the confidence threshold.** Sweep the FAISS score threshold and plot
#    reflection rate against correct refusals.
# 4. **Swap in a small grader model** and measure whether accuracy drops. Usually
#    it does not, and the cost halves.

# %% [markdown]
# ## Recap
#
# | Concept | Takeaway |
# |---|---|
# | Reflection points | Document relevance, answer grounding, answer usefulness |
# | Structured graders | One small model call per question; `with_structured_output` |
# | Query rewriting | Expand vague wording into document vocabulary, then retry |
# | Loop caps | `MAX_RETRIES` and `MAX_GENERATIONS` are mandatory |
# | Parallel grading | `Send` fan-out turns n round trips into one step |
# | CRAG | Corpus-level confidence decides whether to supplement externally |
# | Caveat on fallback | Say when an answer used external sources |
# | Cost | 4-8x plain RAG - use a small grader model |
# | **Adaptive reflection** | Cheap retrieval score first; reflect only when it looks weak |
# | Correct refusals | The metric reflection actually improves - protect it |
# | Audit the graders | A too-strict grader silently breaks the whole pipeline |
#
# ## Next
#
# -> [48a_reflection_and_reflexion.ipynb](48a_reflection_and_reflexion.ipynb)
