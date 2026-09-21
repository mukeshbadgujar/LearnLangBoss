# %% [markdown]
# # 08 - Sequential and Custom Chains
#
# | | |
# |---|---|
# | **Level** | Intermediate |
# | **Time** | 45 minutes |
# | **Prerequisites** | `07_lcel` |
# | **Checklist ID** | `08_chains_sequential_and_custom` |
#
# ## Why this matters
#
# Real tasks are rarely one model call. "Draft a customer reply" is actually:
# classify the issue, pull the relevant policy, draft the reply, check it against
# tone rules, then rewrite if needed.
#
# This notebook covers how to build those multi-step flows - the modern LCEL way,
# and the legacy `SimpleSequentialChain`/`SequentialChain` way you will meet in
# older code. It also covers `@chain` and `RunnableLambda` for putting arbitrary
# Python in the middle of a pipeline.

# %%
from pathlib import Path
import sys

_root = Path.cwd()
while not (_root / "shared").is_dir() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from shared.notebook_setup import setup, require  # noqa: E402

ctx = setup("08_chains_sequential_and_custom")

# %%
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from shared.llm import get_chat_model

model = get_chat_model()
parser = StrOutputParser()

# %% [markdown]
# ## 1. The legacy API, for reading old code
#
# ```python
# # LangChain <= 0.2 - now in langchain-classic
# from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
#
# summarise = LLMChain(llm=llm, prompt=summary_prompt)
# reply     = LLMChain(llm=llm, prompt=reply_prompt)
#
# # SimpleSequentialChain: exactly one input, one output, passed positionally
# pipeline = SimpleSequentialChain(chains=[summarise, reply])
# pipeline.run("customer message")
#
# # SequentialChain: named variables, so steps can use earlier outputs
# pipeline = SequentialChain(
#     chains=[summarise, reply],
#     input_variables=["ticket"],
#     output_variables=["summary", "draft"],
# )
# ```
#
# Why LCEL replaced it:
#
# | Problem with `SequentialChain` | LCEL answer |
# |---|---|
# | String-keyed wiring, errors only at runtime | Real Python objects and types |
# | No streaming | `stream` on every chain |
# | No concurrency | `RunnableParallel` |
# | Hard to insert plain Python | `RunnableLambda` / `@chain` |
# | Opaque in traces | Each step is a named span |

# %% [markdown]
# ## 2. The same thing in LCEL: two steps
#
# Business case: turn a raw customer email into a draft reply.

# %%
extract_prompt = ChatPromptTemplate.from_template(
    "Extract the core problem from this customer message in one factual sentence. "
    "Do not add advice.\n\nMessage: {message}"
)

reply_prompt = ChatPromptTemplate.from_template(
    "You are a Northwind support agent. Write a 3-sentence reply to a customer whose "
    "problem is:\n\n{problem}\n\nBe specific and end with a concrete next step."
)

two_step = extract_prompt | model | parser | reply_prompt | model | parser

customer_email = (
    "hi, since your january release our CSV exports are missing the last row whenever we "
    "apply a filter. we noticed it during month-end close and had to redo two reports. "
    "we're on starter. can you look into it quickly please"
)

print(two_step.invoke({"message": customer_email}))

# %% [markdown]
# Notice `parser | reply_prompt` works because the second prompt has exactly one
# input variable - a bare string is accepted. With two or more variables you must
# build a dict, which is the next pattern.

# %% [markdown]
# ## 3. Multi-step with named intermediates
#
# This is the LCEL equivalent of `SequentialChain`: use `assign` to accumulate
# named values as the data flows.

# %%
from langchain_core.runnables import RunnablePassthrough

classify_prompt = ChatPromptTemplate.from_template(
    "One word category (billing/integration/performance/bug/security/howto) for: {message}"
)
severity_prompt = ChatPromptTemplate.from_template(
    "One word severity (low/medium/high/critical) for: {message}"
)
draft_prompt = ChatPromptTemplate.from_template(
    "Write a 3-sentence support reply.\n"
    "Category: {category}\nSeverity: {severity}\nProblem: {problem}\n"
    "Match the urgency to the severity."
)

triage_pipeline = (
    RunnablePassthrough.assign(problem=extract_prompt | model | parser)
    .assign(
        category=classify_prompt | model | parser,
        severity=severity_prompt | model | parser,
    )
    .assign(draft=draft_prompt | model | parser)
)

result = triage_pipeline.invoke({"message": customer_email})

for key in ("category", "severity", "problem"):
    print(f"{key:9}: {result[key].strip()}")
print("\ndraft:\n", result["draft"].strip())

# %% [markdown]
# Two things to notice:
#
# 1. `category` and `severity` are in the **same** `assign` call, so they run
#    **concurrently**. `SequentialChain` could never do that.
# 2. Every intermediate value is in the output dict - you can log it, show it in a
#    UI, or store it for evaluation.

# %% [markdown]
# ## 4. `@chain`: turn a function into a first-class Runnable
#
# When logic needs real control flow - loops, conditionals, early returns - write
# a function and decorate it. It becomes a Runnable with `invoke`, `batch`,
# `stream` and tracing.

# %%
from langchain_core.runnables import chain


@chain
def summarise_ticket(ticket: dict) -> dict:
    """Custom logic: short tickets skip the LLM entirely."""
    text = ticket["message"]

    if len(text.split()) < 8:
        return {"summary": text.strip(), "method": "passthrough (too short to summarise)"}

    summary = (extract_prompt | model | parser).invoke({"message": text})
    return {"summary": summary.strip(), "method": "llm"}


print(summarise_ticket.invoke({"message": "cant login"}))
print()
print(summarise_ticket.invoke({"message": customer_email}))

# %% [markdown]
# That early return saved an API call. Multiply it by 10,000 short tickets a day
# and it is a real cost line. Deterministic short-circuits belong in your pipeline.

# %%
# Because it is a Runnable, it composes and batches like anything else.
print(summarise_ticket.batch([
    {"message": "invoice wrong"},
    {"message": customer_email},
]))

# %% [markdown]
# ## 5. `RunnableLambda` vs `@chain`
#
# | | `RunnableLambda(fn)` | `@chain` decorator |
# |---|---|---|
# | Syntax | wrap at use site | declare once at definition |
# | Best for | small inline transforms | reusable named steps |
# | Tracing name | `RunnableLambda` unless configured | the function name |
# | Streaming | supported if the function yields | same |
#
# They produce the same kind of object. Prefer `@chain` for anything you will
# reuse, because traces read far better.

# %%
from langchain_core.runnables import RunnableLambda

inline = RunnableLambda(lambda d: {**d, "word_count": len(d["message"].split())})
print(inline.invoke({"message": customer_email})["word_count"], "words")

# %% [markdown]
# ### A streaming custom step
#
# If your function is a generator, LCEL streams it.

# %%
@chain
def sentence_by_sentence(text: str):
    """Re-chunk a stream into whole sentences - nicer for text-to-speech."""
    buffer = ""
    for character in text:
        buffer += character
        if character in ".!?" and len(buffer.strip()) > 10:
            yield buffer.strip() + " "
            buffer = ""
    if buffer.strip():
        yield buffer.strip()


for sentence in sentence_by_sentence.stream(
    "First point here. Second point follows. Third and final point lands."
):
    print(f"-> {sentence}")

# %% [markdown]
# ## 6. A realistic pipeline with a quality gate
#
# Draft, critique, and rewrite only when needed. This is the "reflection" pattern -
# genuinely useful, and the seed of the self-correcting graphs in notebook 48.

# %%
critique_prompt = ChatPromptTemplate.from_template(
    "You are a support QA reviewer. Rules:\n"
    "- must not promise a fix date\n"
    "- must not exceed 4 sentences\n"
    "- must name a concrete next step\n\n"
    "Reply to review:\n{draft}\n\n"
    "Respond with exactly 'PASS' or 'FAIL: <reason>'."
)

rewrite_prompt = ChatPromptTemplate.from_template(
    "Rewrite this support reply to fix the reviewer's objection.\n\n"
    "Reply:\n{draft}\n\nObjection: {critique}\n\nRewritten reply:"
)


@chain
def draft_review_rewrite(inputs: dict) -> dict:
    draft = (draft_prompt | model | parser).invoke(inputs).strip()

    for attempt in range(1, 3):
        verdict = (critique_prompt | model | parser).invoke({"draft": draft}).strip()
        if verdict.upper().startswith("PASS"):
            return {"reply": draft, "attempts": attempt, "verdict": "PASS"}
        draft = (rewrite_prompt | model | parser).invoke(
            {"draft": draft, "critique": verdict}
        ).strip()

    return {"reply": draft, "attempts": 2, "verdict": "max attempts reached"}


quality_pipeline = (
    RunnablePassthrough.assign(problem=extract_prompt | model | parser)
    .assign(category=classify_prompt | model | parser, severity=severity_prompt | model | parser)
    | draft_review_rewrite
)

outcome = quality_pipeline.invoke({"message": customer_email})
print(f"verdict: {outcome['verdict']}  |  attempts: {outcome['attempts']}\n")
print(outcome["reply"])

# %% [markdown]
# **This loop is the boundary of LCEL.** It works, but the retry logic is buried
# inside a Python function: you cannot inspect it mid-flight, pause it for human
# approval, resume it after a crash, or see the loop in a trace as a loop.
#
# That list of limitations is precisely the argument for LangGraph. Notebook 18
# rebuilds this exact pipeline as a graph so you can feel the difference.

# %% [markdown]
# ## 7. Branching pipelines
#
# Different categories deserve different pipelines, not one prompt with a lot of
# "if the question is about billing..." text.

# %%
from langchain_core.runnables import RunnableBranch

billing_flow = ChatPromptTemplate.from_template(
    "You are a billing specialist. Quote policy exactly. Question: {problem}"
) | model | parser

security_flow = ChatPromptTemplate.from_template(
    "You are a security officer. Never confirm details you cannot verify. "
    "Escalate to the compliance team. Question: {problem}"
) | model | parser

default_flow = draft_prompt | model | parser


def is_category(name: str):
    return lambda x: name in x["category"].lower()


category_router = RunnableBranch(
    (is_category("billing"), billing_flow),
    (is_category("security"), security_flow),
    default_flow,
)

full_pipeline = (
    RunnablePassthrough.assign(problem=extract_prompt | model | parser)
    .assign(category=classify_prompt | model | parser, severity=severity_prompt | model | parser)
    .assign(reply=category_router)
)

for message in [
    "you billed us twice in january and nobody has replied to our email",
    "we need your SOC2 type II report before we can renew",
]:
    out = full_pipeline.invoke({"message": message})
    print(f"[{out['category'].strip():12}] {out['reply'].strip()[:220]}\n")

# %% [markdown]
# ## 8. Batch a whole pipeline over real data

# %%
import csv

with open(ctx.data("support_tickets.csv"), newline="", encoding="utf-8") as handle:
    tickets = list(csv.DictReader(handle))[:5]

batched = full_pipeline.batch(
    [{"message": t["summary"]} for t in tickets],
    config={"max_concurrency": 3},
)

for ticket, out in zip(tickets, batched):
    predicted = out["category"].strip().lower()
    match = "OK " if predicted.startswith(ticket["category"][:4]) else "diff"
    print(f"{ticket['ticket_id']} {match} actual={ticket['category']:12} predicted={predicted[:14]:14} {out['severity'].strip()[:8]}")

# %% [markdown]
# ## Try it yourself
#
# 1. **Add a translation step** so the final reply is produced in the customer's
#    language, with the language itself detected by an earlier step.
# 2. **Short-circuit on cost.** Extend `summarise_ticket` so any ticket whose
#    category is `howto` skips the critique loop entirely. Measure the call count
#    before and after.
# 3. **Rebuild with the legacy API.** Install `langchain-classic`, express the
#    two-step pipeline with `SequentialChain`, then list three things you cannot
#    do with it that you can do with LCEL.
# 4. **Find the loop limit.** Make the critique rule impossible to satisfy (for
#    example "must be exactly 7 words") and watch the pipeline burn both attempts.
#    Note that you have no way to intervene mid-loop - remember this in Track 05.

# %% [markdown]
# ## Recap
#
# | Concept | Takeaway |
# |---|---|
# | `SimpleSequentialChain` / `SequentialChain` | Legacy; string-wired, no streaming or concurrency |
# | `a \| b \| c` | Linear pipeline; single-variable prompts accept a bare string |
# | `.assign(...)` | Named intermediates; keys in one call run concurrently |
# | `@chain` | A function becomes a named, traceable, batchable Runnable |
# | `RunnableLambda` | Same idea, inline |
# | Generator + `@chain` | Custom streaming transforms |
# | Draft/critique/rewrite | Powerful, but the loop is opaque - a LangGraph signal |
# | `RunnableBranch` | Route to specialised sub-pipelines |
#
# ## Next
#
# -> [09_document_loaders.ipynb](../02-langchain-rag/09_document_loaders.ipynb)
