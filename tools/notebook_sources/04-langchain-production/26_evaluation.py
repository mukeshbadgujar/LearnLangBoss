# %% [markdown]
# # 26 - Evaluation
#
# | | |
# |---|---|
# | **Level** | Advanced |
# | **Time** | 55 minutes |
# | **Prerequisites** | `25_routing_and_handoffs` |
# | **Checklist ID** | `26_evaluation` |
#
# ## Why this matters
#
# You cannot improve what you cannot measure, and LLM systems are unusually easy
# to fool yourself about. A prompt change *feels* better on the three examples you
# tried. Then it breaks four cases you were not looking at.
#
# This notebook builds a real evaluation harness: deterministic metrics where they
# apply, LLM-as-judge where they do not, and the RAG-specific triad that tells you
# whether retrieval or generation is the problem.

# %%
from pathlib import Path
import sys

_root = Path.cwd()
while not (_root / "shared").is_dir() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from shared.notebook_setup import setup, require  # noqa: E402

ctx = setup("26_evaluation")

# %%
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from shared.llm import get_chat_model

model = get_chat_model()
parser = StrOutputParser()

# %% [markdown]
# ## 1. What to measure
#
# | Layer | Question it answers | Metrics |
# |---|---|---|
# | **Retrieval** | Did we find the right context? | hit@k, MRR, recall@k |
# | **Generation** | Is the answer good given that context? | faithfulness, relevance, correctness |
# | **End to end** | Did the user get what they needed? | task success, escalation rate |
# | **Operational** | Can we afford it? | latency, tokens, cost, error rate |
#
# Measure retrieval and generation **separately**. A bad answer from good context
# is a prompt problem; a bad answer from bad context is a retrieval problem, and
# you will waste days if you optimise the wrong one.

# %% [markdown]
# ## 2. Build the system under test

# %%
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from shared.llm import get_embeddings

embeddings = get_embeddings()


def build_corpus() -> list[Document]:
    docs: list[Document] = []
    for file_name in ("company_handbook.md", "product_faq.md"):
        text = Path(ctx.data(file_name)).read_text(encoding="utf-8")
        sections = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "document"), ("##", "section")], strip_headers=False
        ).split_text(text)
        for doc in sections:
            doc.metadata["source"] = file_name
        docs.extend(sections)
    policy = Path(ctx.data("leave_policy.txt")).read_text(encoding="utf-8")
    for chunk in RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=70).split_text(policy):
        docs.append(Document(chunk, metadata={"source": "leave_policy.txt"}))
    return RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=70).split_documents(docs)


store = FAISS.from_documents(build_corpus(), embeddings)
retriever = store.as_retriever(search_kwargs={"k": 3})


def format_docs(docs) -> str:
    return "\n\n".join(f"[{d.metadata.get('source')}] {d.page_content}" for d in docs)


rag = RunnableParallel(context=retriever, question=RunnablePassthrough()).assign(
    answer=(
        {"context": lambda x: format_docs(x["context"]), "question": lambda x: x["question"]}
        | ChatPromptTemplate.from_template(
            "Answer using only the context. If the context does not contain the answer, say "
            "'That is not covered in the documents I have.'\n\nContext:\n{context}\n\nQ: {question}"
        )
        | model
        | parser
    )
)

print(rag.invoke("How many days of casual leave?")["answer"].strip())

# %% [markdown]
# ## 3. The evaluation set
#
# Twenty examples beats two thousand you never look at. Include the hard cases and
# - importantly - questions the system *should refuse*.

# %%
EVAL_SET = [
    # (question, expected_source, reference_answer_facts, should_answer)
    ("How many days of annual leave do employees get?", "leave_policy.txt", ["24"], True),
    ("How many sick leave days per year?", "leave_policy.txt", ["12"], True),
    ("How many days of casual leave?", "leave_policy.txt", ["6"], True),
    ("How much paternity leave is available?", "leave_policy.txt", ["15"], True),
    ("Can unused annual leave be carried forward?", "leave_policy.txt", ["12", "30 June"], True),
    ("When is a medical certificate required?", "leave_policy.txt", ["3"], True),
    ("What is the notice period for an L5 engineer?", "company_handbook.md", ["90"], True),
    ("What is the hotel limit for domestic travel?", "company_handbook.md", ["6,000", "6000"], True),
    ("What is the annual learning budget?", "company_handbook.md", ["60,000", "60000"], True),
    ("How many office days per month are required?", "company_handbook.md", ["8"], True),
    ("What are the API rate limits on Growth?", "product_faq.md", ["600"], True),
    ("Where is customer data stored by default?", "product_faq.md", ["ap-south-1", "Mumbai"], True),
    ("What is the support SLA for Enterprise?", "product_faq.md", ["1 hour", "1-hour"], True),
    ("How long is data retained after cancellation?", "product_faq.md", ["30"], True),
    # Questions the corpus cannot answer - refusal is the correct behaviour.
    ("What is the company policy on pet insurance?", None, [], False),
    ("How many employees does Northwind have?", None, [], False),
    ("What is the CEO's home address?", None, [], False),
]

print(f"{len(EVAL_SET)} examples ({sum(1 for e in EVAL_SET if not e[3])} should be refused)")

# %% [markdown]
# ## 4. Deterministic metrics first
#
# Cheap, fast and completely reproducible. Use them wherever you can - an
# LLM judge should only be used for things you genuinely cannot check with code.

# %%
def evaluate_retrieval(retriever, eval_set, k: int = 3) -> dict:
    """hit@k and MRR - does the right source appear, and how highly ranked?"""
    answerable = [e for e in eval_set if e[3]]
    hits = 0
    reciprocal_ranks = []

    for question, expected_source, _, _ in answerable:
        sources = [d.metadata.get("source") for d in retriever.invoke(question)]
        if expected_source in sources:
            hits += 1
            reciprocal_ranks.append(1 / (sources.index(expected_source) + 1))
        else:
            reciprocal_ranks.append(0.0)
            print(f"  MISS  {question[:52]:54} got {set(sources)}")

    return {
        f"hit@{k}": hits / len(answerable),
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "n": len(answerable),
    }


print("retrieval:", evaluate_retrieval(retriever, EVAL_SET))

# %%
REFUSAL_MARKERS = ["not covered", "do not have", "don't have", "no information", "cannot find"]


def evaluate_generation(chain, eval_set) -> dict:
    """Fact recall on answerable questions, refusal rate on unanswerable ones."""
    results = chain.batch([e[0] for e in eval_set], config={"max_concurrency": 4})

    fact_hits = 0
    answerable_count = 0
    correct_refusals = 0
    unanswerable_count = 0
    failures = []

    for (question, _, facts, should_answer), outcome in zip(eval_set, results):
        answer = outcome["answer"].lower()
        refused = any(marker in answer for marker in REFUSAL_MARKERS)

        if should_answer:
            answerable_count += 1
            found = any(str(fact).lower() in answer for fact in facts)
            fact_hits += found
            if not found:
                failures.append(("missing fact", question, outcome["answer"][:90]))
        else:
            unanswerable_count += 1
            correct_refusals += refused
            if not refused:
                failures.append(("hallucinated", question, outcome["answer"][:90]))

    for kind, question, snippet in failures:
        print(f"  {kind:14} {question[:44]:46} -> {snippet}")

    return {
        "fact_recall": fact_hits / answerable_count,
        "refusal_accuracy": correct_refusals / unanswerable_count,
        "failures": len(failures),
    }


print("\ngeneration:", evaluate_generation(rag, EVAL_SET))

# %% [markdown]
# **`refusal_accuracy` is the metric teams forget and regret.** A system that
# answers everything scores well on fact recall and is dangerous in production,
# because it confidently invents a pet insurance policy.

# %% [markdown]
# ## 5. LLM-as-judge
#
# For qualities code cannot check - is this answer faithful, helpful, correctly
# toned? - use a model as the grader.

# %%
from typing import Literal

from pydantic import BaseModel, Field


class Judgement(BaseModel):
    """A grader's verdict on one answer."""

    verdict: Literal["pass", "fail"] = Field(description="Whether the answer meets the criterion")
    score: int = Field(description="Score from 1 (terrible) to 5 (excellent)", ge=1, le=5)
    reason: str = Field(description="One-sentence justification citing specific evidence")


def make_judge(criterion: str, instructions: str):
    """Build a grader for one specific criterion."""
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         f"You are a strict evaluator grading the criterion: {criterion}.\n{instructions}\n"
         "Be harsh. A 5 means flawless. Most acceptable answers are a 3 or 4."),
        ("human", "QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}"),
    ])
    return prompt | model.with_structured_output(Judgement)


faithfulness_judge = make_judge(
    "faithfulness",
    "Fail if the answer states any fact not present in the context, even if that fact is "
    "true in the real world. Adding outside knowledge is a failure.",
)

relevance_judge = make_judge(
    "relevance",
    "Fail if the answer does not directly address what was asked, or buries the answer "
    "in irrelevant material.",
)

completeness_judge = make_judge(
    "completeness",
    "Fail if the context contained information needed to answer fully but the answer omitted it.",
)

# %%
sample = rag.invoke("Can unused annual leave be carried forward?")
payload = {
    "question": "Can unused annual leave be carried forward?",
    "context": format_docs(sample["context"]),
    "answer": sample["answer"],
}

for name, judge in [("faithfulness", faithfulness_judge), ("relevance", relevance_judge), ("completeness", completeness_judge)]:
    verdict = judge.invoke(payload)
    print(f"{name:13} {verdict.verdict:5} {verdict.score}/5  {verdict.reason}")

# %% [markdown]
# ### Catching an unfaithful answer
#
# Feed the judge an answer that is true in reality but absent from the context.

# %%
unfaithful = {
    "question": "Can unused annual leave be carried forward?",
    "context": format_docs(sample["context"]),
    "answer": (
        "Yes, up to 12 days can be carried forward. Note that under Indian labour law "
        "employers must allow at least 30 days of carry-forward, and Northwind also offers "
        "a cash-out option in December."
    ),
}
verdict = faithfulness_judge.invoke(unfaithful)
print(f"{verdict.verdict} {verdict.score}/5 - {verdict.reason}")

# %% [markdown]
# The judge caught invented claims that sound plausible. That is exactly the
# failure mode a human reviewer misses when skimming.

# %% [markdown]
# ## 6. The RAG triad
#
# Three relationships, three failure modes. Together they localise the problem.
#
# ```
#            context relevance
#   QUESTION -------------------- CONTEXT
#       |                            |
#       |  answer                    |  faithfulness
#       |  relevance                 |
#       +---------- ANSWER ----------+
# ```
#
# | Metric | Low score means |
# |---|---|
# | **Context relevance** | Retrieval is broken - fix chunking, embeddings, hybrid search |
# | **Faithfulness** | Generation is hallucinating - fix the prompt, add grounding rules |
# | **Answer relevance** | The model is answering a different question - fix the prompt |

# %%
context_relevance_judge = make_judge(
    "context relevance",
    "Judge whether the retrieved context actually contains information needed to answer "
    "the question. Ignore the answer entirely - grade the context only.",
)


def rag_triad(chain, question: str) -> dict:
    outcome = chain.invoke(question)
    payload = {"question": question, "context": format_docs(outcome["context"]), "answer": outcome["answer"]}
    return {
        "context_relevance": context_relevance_judge.invoke(payload).score,
        "faithfulness": faithfulness_judge.invoke(payload).score,
        "answer_relevance": relevance_judge.invoke(payload).score,
        "answer": outcome["answer"],
    }


for question in [
    "What is the hotel limit for international travel?",
    "What is the company policy on pet insurance?",
]:
    scores = rag_triad(rag, question)
    print(f"\n{question}")
    print(f"  context_relevance={scores['context_relevance']}  "
          f"faithfulness={scores['faithfulness']}  answer_relevance={scores['answer_relevance']}")
    print(f"  {scores['answer'].strip()[:150]}")

# %% [markdown]
# ## 7. A full harness
#
# Put it together into something you can run on every change.

# %%
import statistics
import time


def run_evaluation(chain, eval_set, label: str, judge_sample: int = 6) -> dict:
    """Deterministic metrics on everything, LLM judging on a sample (judging is not free)."""
    start = time.perf_counter()

    retrieval = evaluate_retrieval(retriever, eval_set)
    generation = evaluate_generation(chain, eval_set)

    judged = []
    for question, _, _, should_answer in eval_set[:judge_sample]:
        if not should_answer:
            continue
        judged.append(rag_triad(chain, question))

    elapsed = time.perf_counter() - start
    report = {
        "label": label,
        "hit@3": round(retrieval["hit@3"], 3),
        "mrr": round(retrieval["mrr"], 3),
        "fact_recall": round(generation["fact_recall"], 3),
        "refusal_accuracy": round(generation["refusal_accuracy"], 3),
        "faithfulness": round(statistics.mean(j["faithfulness"] for j in judged), 2),
        "answer_relevance": round(statistics.mean(j["answer_relevance"] for j in judged), 2),
        "seconds": round(elapsed, 1),
    }
    return report


baseline = run_evaluation(rag, EVAL_SET, "baseline k=3")
print()
for key, value in baseline.items():
    print(f"  {key:18} {value}")

# %% [markdown]
# ### Comparing a change
#
# Now change one thing and re-run. This is the workflow that prevents regressions.

# %%
wider_retriever = store.as_retriever(search_kwargs={"k": 6})
wider_rag = RunnableParallel(context=wider_retriever, question=RunnablePassthrough()).assign(
    answer=(
        {"context": lambda x: format_docs(x["context"]), "question": lambda x: x["question"]}
        | ChatPromptTemplate.from_template(
            "Answer using only the context. If the context does not contain the answer, say "
            "'That is not covered in the documents I have.'\n\nContext:\n{context}\n\nQ: {question}"
        )
        | model
        | parser
    )
)

variant = run_evaluation(wider_rag, EVAL_SET, "wider k=6")

print(f"\n{'metric':18} {'baseline':>10} {'k=6':>10} {'delta':>8}")
for key in ["hit@3", "mrr", "fact_recall", "refusal_accuracy", "faithfulness", "answer_relevance"]:
    before, after = baseline[key], variant[key]
    arrow = "+" if after > before else ("-" if after < before else "=")
    print(f"{key:18} {before:>10} {after:>10} {arrow:>8}")

# %% [markdown]
# Watch for the classic trade: more context often raises fact recall while
# *lowering* refusal accuracy, because marginally-related chunks give the model
# something to latch onto. Whether that trade is acceptable is a product decision,
# not a technical one - but at least now it is visible.

# %% [markdown]
# ## 8. Pairwise comparison
#
# Absolute scores drift between judge runs. Asking "which of these two is better?"
# is far more stable, which is why it is the standard for prompt A/B tests.

# %%
class Preference(BaseModel):
    """Which of two answers is better?"""

    winner: Literal["A", "B", "tie"] = Field(description="Which answer is better")
    reason: str = Field(description="One sentence explaining the choice")


comparator = (
    ChatPromptTemplate.from_messages([
        ("system",
         "Compare two answers to the same question against the same context. Prefer the answer "
         "that is more faithful to the context, more directly responsive, and more concise. "
         "Ignore length and formatting differences unless they affect clarity."),
        ("human", "QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\nANSWER A:\n{a}\n\nANSWER B:\n{b}"),
    ])
    | model.with_structured_output(Preference)
)


def pairwise(chain_a, chain_b, questions: list[str]) -> dict:
    tally = {"A": 0, "B": 0, "tie": 0}
    for question in questions:
        out_a, out_b = chain_a.invoke(question), chain_b.invoke(question)
        # Run both orderings to cancel position bias.
        first = comparator.invoke({
            "question": question, "context": format_docs(out_a["context"]),
            "a": out_a["answer"], "b": out_b["answer"],
        }).winner
        second = comparator.invoke({
            "question": question, "context": format_docs(out_b["context"]),
            "a": out_b["answer"], "b": out_a["answer"],
        }).winner
        flipped = {"A": "B", "B": "A", "tie": "tie"}[second]
        tally[first if first == flipped else "tie"] += 1
    return tally


sample_questions = [e[0] for e in EVAL_SET[:5]]
print("A = k=3, B = k=6:", pairwise(rag, wider_rag, sample_questions))

# %% [markdown]
# Running both orderings and only counting a win when the judge agrees with itself
# removes **position bias** - judges systematically favour whichever answer is
# shown first. Without this correction your A/B results are noise.

# %% [markdown]
# ## 9. LLM judges are imperfect - know how
#
# | Bias | Effect | Mitigation |
# |---|---|---|
# | Position | Prefers the first answer | Run both orderings |
# | Verbosity | Prefers longer answers | Say "ignore length" in the rubric |
# | Self-preference | Prefers its own model family's output | Use a different model as judge |
# | Sycophancy | Agrees with a confident tone | Demand evidence in `reason` |
# | Leniency | Gives 4/5 to everything | Anchor the scale: "most acceptable answers are 3" |
#
# **Calibrate the judge against humans.** Grade 20 examples yourself, run the
# judge on the same 20, and measure agreement. If agreement is below ~80%, fix the
# rubric before trusting any number it produces.

# %%
HUMAN_LABELS = [
    ("How many days of annual leave do employees get?", "pass"),
    ("What is the company policy on pet insurance?", "pass"),   # correct refusal = pass
    ("What is the hotel limit for domestic travel?", "pass"),
]

agreements = 0
for question, human in HUMAN_LABELS:
    outcome = rag.invoke(question)
    judge_verdict = faithfulness_judge.invoke({
        "question": question, "context": format_docs(outcome["context"]), "answer": outcome["answer"]
    }).verdict
    agree = judge_verdict == human
    agreements += agree
    print(f"{'agree' if agree else 'DISAGREE':9} human={human:5} judge={judge_verdict:5} {question[:48]}")

print(f"\nagreement: {agreements}/{len(HUMAN_LABELS)} (do this with 20+ examples in practice)")

# %% [markdown]
# ## 10. Operational metrics
#
# Quality is only half of it. Track cost and latency in the same harness or you
# will ship an accurate system nobody can afford.

# %%
from langchain_core.callbacks import BaseCallbackHandler


class UsageMeter(BaseCallbackHandler):
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0

    def on_llm_end(self, response, **kwargs):
        self.calls += 1
        try:
            usage = response.generations[0][0].message.usage_metadata or {}
        except (AttributeError, IndexError):
            return
        self.input_tokens += usage.get("input_tokens", 0)
        self.output_tokens += usage.get("output_tokens", 0)


def operational_profile(chain, questions: list[str], label: str) -> None:
    meter = UsageMeter()
    latencies = []
    for question in questions:
        start = time.perf_counter()
        chain.invoke(question, config={"callbacks": [meter]})
        latencies.append(time.perf_counter() - start)

    latencies.sort()
    print(f"{label:14} calls={meter.calls:3}  "
          f"tokens={meter.input_tokens + meter.output_tokens:6}  "
          f"p50={latencies[len(latencies) // 2]:.2f}s  p95={latencies[int(len(latencies) * 0.95) - 1]:.2f}s")


questions = [e[0] for e in EVAL_SET[:8]]
operational_profile(rag, questions, "k=3")
operational_profile(wider_rag, questions, "k=6")

# %% [markdown]
# ## Try it yourself
#
# 1. **Grow the eval set to 30**, with at least 8 unanswerable questions and 5
#    multi-hop questions that need two sources.
# 2. **Use a different judge model.** Run faithfulness judging with a second
#    provider and measure how often the two judges disagree.
# 3. **Regression gate.** Write a function that fails (raises) if `hit@3` drops
#    below 0.85 or `refusal_accuracy` below 0.9, and wire it into a script you
#    could run in CI.
# 4. **Find the real bottleneck.** Deliberately break retrieval (`k=1`, tiny
#    chunks), confirm the triad points at context relevance rather than
#    faithfulness, then break the prompt instead and confirm it points elsewhere.

# %% [markdown]
# ## Recap
#
# | Concept | Takeaway |
# |---|---|
# | Separate layers | Measure retrieval and generation independently |
# | `hit@k`, MRR | Deterministic retrieval metrics; cheap, run on everything |
# | Fact recall | String-match known facts - no judge needed |
# | **Refusal accuracy** | The metric teams forget; catches confident hallucination |
# | LLM-as-judge | For faithfulness, relevance, completeness; use structured output |
# | RAG triad | Context relevance / faithfulness / answer relevance localises the fault |
# | Pairwise comparison | More stable than absolute scores; run both orderings |
# | Judge bias | Position, verbosity, self-preference, leniency - all correctable |
# | Calibration | Agree with human labels on 20 examples before trusting the judge |
# | Operational metrics | Tokens, p50/p95 latency, call count belong in the same harness |
#
# ## Next
#
# -> [26a_unit_testing_chains_and_graphs.ipynb](26a_unit_testing_chains_and_graphs.ipynb)
