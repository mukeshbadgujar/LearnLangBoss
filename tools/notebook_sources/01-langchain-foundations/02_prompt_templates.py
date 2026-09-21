# %% [markdown]
# # 02 - Prompt Templates
#
# | | |
# |---|---|
# | **Level** | Beginner |
# | **Time** | 40 minutes |
# | **Prerequisites** | `01_models_messages` |
# | **Checklist ID** | `02_prompt_templates` |
#
# ## Why this matters
#
# Your support assistant needs to summarise a ticket. The naive version is an
# f-string:
#
# ```python
# prompt = f"Summarise this ticket for {customer}: {summary}"
# ```
#
# That works until the day a customer name is empty, or a ticket body contains a
# `{`, or a colleague needs the same prompt in a different service. Prompt
# templates solve four real problems: **missing-variable detection, reuse,
# composability, and traceability** (LangSmith shows the template and the values
# separately, which is enormously useful when debugging).

# %%
from pathlib import Path
import sys

_root = Path.cwd()
while not (_root / "shared").is_dir() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from shared.notebook_setup import setup, require  # noqa: E402

ctx = setup("02_prompt_templates")

# %%
from shared.llm import get_chat_model

model = get_chat_model()

# %% [markdown]
# ## 1. `PromptTemplate`: a string with declared inputs

# %%
from langchain_core.prompts import PromptTemplate

template = PromptTemplate.from_template(
    "You are a support analyst at Northwind Analytics.\n"
    "Summarise this ticket in one sentence for an internal standup.\n\n"
    "Customer: {customer}\n"
    "Plan: {plan}\n"
    "Ticket: {summary}"
)

print("Declared inputs:", template.input_variables)

filled = template.format(
    customer="Bluepeak Health",
    plan="Growth",
    summary="Webhook deliveries failing with 401 after key rotation",
)
print("\n--- rendered prompt ---")
print(filled)

# %% [markdown]
# ### `format` vs `invoke`
#
# - `.format(**kwargs)` -> a plain **string**
# - `.invoke({...})` -> a **`PromptValue`**, which is what you pipe into a model
#
# Use `invoke` in real code; `format` is for eyeballing the result.

# %%
value = template.invoke(
    {"customer": "Acme Retail", "plan": "Enterprise", "summary": "Dashboard load exceeds 30 seconds"}
)
print(type(value).__name__)
print(value.to_messages())
print("\nModel output:", model.invoke(value).content.strip())

# %% [markdown]
# ## 2. Input validation: the actual reason to bother
#
# An f-string with a typo produces a `NameError` at best, or silently renders
# `None` at worst. A template refuses to run.

# %%
try:
    template.invoke({"customer": "Cedar Logistics", "plan": "Starter"})  # 'summary' missing
except KeyError as exc:
    print("KeyError as expected ->", exc)

# %% [markdown]
# You can also validate *before* you have values, which is what you want in a unit
# test or at service start-up.

# %%
def validate_inputs(prompt_template, provided: dict) -> list[str]:
    """Return the list of declared variables that `provided` does not cover."""
    return sorted(set(prompt_template.input_variables) - set(provided))


candidate = {"customer": "Granite Foods", "plan": "Starter"}
print("Missing:", validate_inputs(template, candidate))
print("Missing:", validate_inputs(template, {**candidate, "summary": "Downgrade request"}))

# %% [markdown]
# ## 3. Partial formatting: lock in what you already know
#
# Some variables are known at wiring time (tenant, language, today's date) and
# some only at request time. `.partial()` splits those two lifetimes.

# %%
escalation = PromptTemplate.from_template(
    "Today is {today}.\n"
    "You are the {tier} support tier at Northwind Analytics.\n"
    "SLA for this tier: {sla}.\n\n"
    "Decide whether this ticket breaches SLA and reply with BREACH or OK plus a short reason.\n"
    "Ticket opened: {opened_on}\n"
    "Ticket: {summary}"
)

from datetime import date

# Bake in what the service knows at start-up.
enterprise_prompt = escalation.partial(
    tier="Enterprise",
    sla="1 hour first response for critical, 24x7",
    today=date(2026, 1, 26).isoformat(),
)

print("Originally required:", escalation.input_variables)
print("Still required     :", enterprise_prompt.input_variables)

print("\n", model.invoke(enterprise_prompt.invoke({
    "opened_on": "2026-01-16",
    "summary": "Request SOC2 report and data residency confirmation (critical, still open)",
})).content.strip())

# %% [markdown]
# ### Partials can be **callables** for values that must stay fresh
#
# Passing `date.today().isoformat()` freezes the date at import time - a bug that
# shows up the next morning. Pass the *function* instead and it is evaluated on
# every render.

# %%
def today_iso() -> str:
    return date.today().isoformat()


live_date_prompt = PromptTemplate.from_template(
    "Today is {today}. Answer briefly: {question}"
).partial(today=today_iso)

print(live_date_prompt.format(question="What quarter are we in?"))

# %% [markdown]
# ## 4. Escaping literal braces
#
# JSON examples inside prompts are the usual trap. `{` and `}` are template
# syntax, so you double them to mean a literal brace.

# %%
json_prompt = PromptTemplate.from_template(
    "Extract the fields and reply with JSON only.\n"
    'Shape: {{"customer": string, "category": string, "priority": string}}\n\n'
    "Ticket: {ticket}"
)

print(json_prompt.format(ticket="Acme Retail, Enterprise: dashboard load time exceeds 30 seconds"))
print("\nModel:", model.invoke(json_prompt.invoke({"ticket": "Granite Foods wants to downgrade mid-cycle"})).content.strip())

# %% [markdown]
# Forgetting to double them produces a confusing error, so recognise it:

# %%
try:
    PromptTemplate.from_template('Reply as {"answer": "..."} for {question}').format(question="hi")
except (KeyError, ValueError) as exc:
    print(f"{type(exc).__name__}: {exc}  <- single braces were parsed as a variable")

# %% [markdown]
# ## 5. Composing templates
#
# Prompts grow. Keep the pieces separate and combine them, rather than maintaining
# one 60-line string.

# %%
identity = PromptTemplate.from_template("You are {persona} at Northwind Analytics.")
rules = PromptTemplate.from_template("Rules:\n- Answer in at most {max_sentences} sentences.\n- If unsure, say so.")
task = PromptTemplate.from_template("Task: {task}\nInput: {input_text}")

combined = identity + "\n\n" + rules + "\n\n" + task
print("Combined inputs:", combined.input_variables)
print("\n--- rendered ---")
print(combined.format(
    persona="a billing specialist",
    max_sentences=2,
    task="Explain the refund position",
    input_text="Customer downgraded on day 12 of a 30-day cycle and wants money back",
))

# %% [markdown]
# ## 6. Your first chain: template into model
#
# The `|` operator connects Runnables. Notebook 07 covers this properly; here just
# notice how natural it is once the prompt declares its own inputs.

# %%
from langchain_core.output_parsers import StrOutputParser

summarise = template | model | StrOutputParser()

print(summarise.invoke({
    "customer": "Harlow Energy",
    "plan": "Enterprise",
    "summary": "Scheduled reports sent twice to all recipients",
}))

# %% [markdown]
# And because it is a Runnable, `batch` works with no extra effort:

# %%
import csv

with open(ctx.data("support_tickets.csv"), newline="", encoding="utf-8") as handle:
    tickets = list(csv.DictReader(handle))[:4]

payloads = [{"customer": t["customer"], "plan": t["plan"], "summary": t["summary"]} for t in tickets]
for ticket, line in zip(tickets, summarise.batch(payloads)):
    print(f"{ticket['ticket_id']}: {line.strip()}")

# %% [markdown]
# ## 7. A prompt template is a versionable asset
#
# Treat prompts like code: keep them in one module, give them names, and review
# changes. This is what makes LangChain Hub (notebook 28) and prompt A/B testing
# possible later.

# %%
PROMPTS = {
    "ticket_summary_v1": PromptTemplate.from_template(
        "Summarise this ticket in one sentence.\nTicket: {summary}"
    ),
    "ticket_summary_v2": PromptTemplate.from_template(
        "You are a support analyst. Summarise this ticket in one sentence for an "
        "engineering standup. Lead with the impact, not the symptom.\n"
        "Customer: {customer} ({plan})\nTicket: {summary}"
    ),
}

sample = {"customer": "Everline Bank", "plan": "Enterprise", "summary": "API p99 latency spiked to 4 seconds during EU business hours"}

for name, prompt in PROMPTS.items():
    needed = {k: v for k, v in sample.items() if k in prompt.input_variables}
    print(f"[{name}] {(prompt | model | StrOutputParser()).invoke(needed).strip()}\n")

# %% [markdown]
# ## Try it yourself
#
# 1. **Build a `policy_answer` template** taking `{policy_excerpt}`, `{question}`
#    and `{tone}`, with a rule that the model must answer *only* from the excerpt
#    and reply "Not covered by policy." otherwise. Test it with a real paragraph
#    from `shared/sample_data/leave_policy.txt` and with a question the policy does
#    not answer.
# 2. **Add a live partial** for `{today}` and write a template that computes
#    whether an expense claim is inside the 30-day window (see the handbook).
# 3. **Break and fix escaping.** Write a template whose output must be
#    `{"status": "ok"}` literally, get the `KeyError`, then fix it.

# %% [markdown]
# ## Recap
#
# | Concept | Takeaway |
# |---|---|
# | `PromptTemplate.from_template` | Declares inputs, so missing values fail loudly |
# | `.format()` vs `.invoke()` | String for humans, `PromptValue` for chains |
# | `.partial()` | Freeze wiring-time values; pass a *callable* for live values |
# | `{{` `}}` | Literal braces - the JSON-in-prompt trap |
# | Template `+` template | Compose small pieces instead of one giant string |
# | `prompt \| model \| parser` | The canonical LCEL chain |
#
# **When not to use `PromptTemplate`:** as soon as you need distinct system/user
# roles or chat history - use `ChatPromptTemplate`, which is the next notebook.
#
# ## Next
#
# -> [03_chat_prompt_templates.ipynb](03_chat_prompt_templates.ipynb)
