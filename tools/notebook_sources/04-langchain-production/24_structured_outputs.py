# %% [markdown]
# # 24 - Structured Outputs
#
# | | |
# |---|---|
# | **Level** | Intermediate |
# | **Time** | 45 minutes |
# | **Prerequisites** | `23_caching` |
# | **Checklist ID** | `24_structured_outputs` |
#
# ## Why this matters
#
# Notebook 05 taught output parsers: describe a schema in the prompt, hope the
# model complies, parse the text, repair it when it does not. That was the only
# option in 2023.
#
# Today providers support **constrained decoding**: you hand them a JSON schema
# and the model is restricted to producing output that matches it. The difference
# is between *asking* for JSON and *guaranteeing* it.
#
# This is the single most important technique for putting an LLM inside a real
# system, because it turns a text generator into a typed function.

# %%
from pathlib import Path
import sys

_root = Path.cwd()
while not (_root / "shared").is_dir() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from shared.notebook_setup import setup, require  # noqa: E402

ctx = setup("24_structured_outputs")

# %%
from shared.llm import get_chat_model

model = get_chat_model()

# %% [markdown]
# ## 1. `.with_structured_output()`

# %%
from typing import Literal

from pydantic import BaseModel, Field


class TicketTriage(BaseModel):
    """Structured triage of an inbound support ticket."""

    category: Literal["billing", "integration", "performance", "bug", "security", "howto"] = Field(
        description="The single best-fitting category"
    )
    priority: Literal["low", "medium", "high", "critical"] = Field(
        description="Urgency based on customer impact and plan tier"
    )
    summary: str = Field(description="One-line internal summary, at most 90 characters")
    needs_specialist: bool = Field(description="True if a human specialist must be involved")
    suggested_team: str = Field(description="Team that should own this, e.g. 'billing-ops'")


structured = model.with_structured_output(TicketTriage)

result = structured.invoke(
    "Everline Bank (Enterprise) reports API p99 latency of 4 seconds during EU business hours. "
    "Their trading dashboard is unusable in the morning."
)

print(type(result).__name__)
print(result.model_dump_json(indent=2))

# %% [markdown]
# No prompt engineering, no format instructions, no parsing. The model returned a
# validated Pydantic object, and `result.priority` is guaranteed to be one of four
# values.

# %%
tickets = [
    "how do I rename a data source",
    "we were charged twice for january seats and nobody replied to our email",
    "need your SOC2 type II report before we can renew next month",
    "csv export drops the last row when a filter is applied",
]

for ticket in tickets:
    triage = structured.invoke(ticket)
    flag = "SPECIALIST" if triage.needs_specialist else "auto      "
    print(f"[{flag}] {triage.category:12} {triage.priority:8} {triage.suggested_team:16} {triage.summary[:50]}")

# %% [markdown]
# ## 2. Three ways to define the schema

# %%
# (a) Pydantic - validation, defaults, descriptions. The default choice.
class ExpenseClaim(BaseModel):
    """An expense claim extracted from free text."""

    amount: float = Field(description="Total amount claimed", gt=0)
    currency: Literal["INR", "USD", "EUR", "GBP"] = Field(description="Currency code")
    category: Literal["travel", "meals", "software", "conference", "equipment"]
    description: str = Field(description="What the expense was for")
    receipt_attached: bool = Field(default=False, description="Whether a receipt was mentioned")


print(model.with_structured_output(ExpenseClaim).invoke(
    "I paid 7,400 rupees a night for a hotel in Bengaluru during the customer visit, receipt attached."
))

# %%
# (b) TypedDict - lighter, no runtime validation.
from typing import Annotated, TypedDict


class QuickTriage(TypedDict):
    """Minimal ticket triage."""

    category: Annotated[str, ..., "One of: billing, integration, performance, bug, security, howto"]
    priority: Annotated[str, ..., "One of: low, medium, high, critical"]


print(model.with_structured_output(QuickTriage).invoke("webhooks return 401 after key rotation"))

# %%
# (c) Raw JSON schema - when the schema comes from config or another system.
json_schema = {
    "title": "Escalation",
    "description": "Decision about escalating a support ticket",
    "type": "object",
    "properties": {
        "escalate": {"type": "boolean", "description": "Whether to escalate now"},
        "reason": {"type": "string", "description": "One sentence justification"},
        "sla_hours": {"type": "integer", "description": "Hours until the SLA breaches"},
    },
    "required": ["escalate", "reason", "sla_hours"],
}

print(model.with_structured_output(json_schema).invoke(
    "Enterprise customer, critical security request opened 5 days ago, still unassigned."
))

# %% [markdown]
# | Schema type | Returns | Use when |
# |---|---|---|
# | Pydantic `BaseModel` | validated instance | **Default.** You want validation and attribute access |
# | `TypedDict` | plain `dict` | Lightweight, you trust the model, dicts are convenient |
# | JSON schema `dict` | plain `dict` | Schema is dynamic or comes from elsewhere |

# %% [markdown]
# ## 3. `ToolStrategy` vs `ProviderStrategy`
#
# Under the hood there are two mechanisms, and knowing which one is running
# explains a lot of otherwise confusing behaviour.

# %%
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy

# ToolStrategy: the schema becomes a tool; the model "calls" it.
# Works anywhere tool calling works. This is the portable option.
tool_based = model.with_structured_output(TicketTriage, method="function_calling")
print("function_calling ->", tool_based.invoke("dashboards are slow on big datasets").category)

# %%
# ProviderStrategy: native JSON-schema mode with constrained decoding.
# Strongest guarantee, but not every provider/model supports it.
try:
    native = model.with_structured_output(TicketTriage, method="json_schema")
    print("json_schema      ->", native.invoke("dashboards are slow on big datasets").category)
except Exception as exc:
    print(f"json_schema not supported here ({type(exc).__name__}) - function_calling is the fallback")

# %% [markdown]
# | | `ToolStrategy` (`function_calling`) | `ProviderStrategy` (`json_schema`) |
# |---|---|---|
# | Mechanism | schema exposed as a tool | provider-enforced decoding |
# | Support | anywhere tool calling works | OpenAI, some others |
# | Guarantee | very high | **absolute** |
# | Coexists with real tools | needs care | yes |
# | Nested/complex schemas | good | best |
#
# Leave `method` unset and LangChain picks the best available. Set it explicitly
# when you need reproducible behaviour across environments.

# %% [markdown]
# ## 4. Getting the raw response too
#
# In production you usually want the parsed object *and* the token usage, and you
# do not want a validation error to throw away the whole response.

# %%
robust = model.with_structured_output(TicketTriage, include_raw=True)
outcome = robust.invoke("customer cannot log in since this morning, entire team blocked, enterprise plan")

print("keys         :", list(outcome))
print("parsing error:", outcome["parsing_error"])
print("tokens       :", outcome["raw"].usage_metadata)
print("parsed       :", outcome["parsed"].category, outcome["parsed"].priority)

# %%
def safe_triage(text: str) -> TicketTriage:
    """Never raises: falls back to a manual-review verdict."""
    outcome = robust.invoke(text)
    if outcome["parsing_error"] is not None or outcome["parsed"] is None:
        print(f"  [fallback] {outcome['parsing_error']}")
        return TicketTriage(
            category="howto",
            priority="low",
            summary="Automatic triage failed - needs manual review",
            needs_specialist=True,
            suggested_team="support-triage",
        )
    return outcome["parsed"]


print(safe_triage("...").summary)

# %% [markdown]
# ## 5. Rich schemas
#
# Nested models, lists and optional fields all work. This is where structured
# output stops being a formatting trick and becomes a data pipeline.

# %%
from typing import Optional


class Customer(BaseModel):
    """Customer identified in a message."""

    name: str = Field(description="Company name")
    plan: Optional[Literal["Starter", "Growth", "Enterprise"]] = Field(
        default=None, description="Plan tier if stated"
    )


class ActionItem(BaseModel):
    """A single follow-up action."""

    action: str = Field(description="What needs to be done")
    owner: str = Field(description="Team or role responsible")
    due_in_hours: int = Field(description="Hours until this should be done", ge=1, le=720)


class TicketAnalysis(BaseModel):
    """Complete analysis of a support conversation."""

    customer: Customer
    category: Literal["billing", "integration", "performance", "bug", "security", "howto"]
    priority: Literal["low", "medium", "high", "critical"]
    sentiment: Literal["angry", "frustrated", "neutral", "satisfied"]
    summary: str = Field(description="Two-sentence summary for an internal handover")
    action_items: list[ActionItem] = Field(description="Concrete next steps, at least one")
    mentioned_products: list[str] = Field(description="Product areas referenced")
    escalate: bool = Field(description="Whether this needs immediate escalation")


analyser = model.with_structured_output(TicketAnalysis)

conversation = """
Customer (Everline Bank, Enterprise): This is the third time this month our API latency has
spiked during EU hours. Our traders cannot load positions. We raised TCK-1011 two weeks ago
and were told it was resolved. It is not resolved. We are reviewing our renewal.

Agent: I'm sorry - let me escalate this.

Customer: We also still have not received the SOC2 report our procurement team asked for.
"""

analysis = analyser.invoke(conversation)
print(analysis.model_dump_json(indent=2))

# %%
print(f"{analysis.customer.name} ({analysis.customer.plan}) - {analysis.sentiment}, escalate={analysis.escalate}\n")
for item in analysis.action_items:
    print(f"  [{item.due_in_hours:>3}h] {item.owner:18} {item.action}")

# %% [markdown]
# One model call turned an unstructured conversation into a typed object you can
# insert into a database, route to a queue, and assert on in a test.

# %% [markdown]
# ## 6. Validation that runs on your side
#
# The provider enforces the *shape*. Pydantic validators enforce your *rules*.

# %%
from pydantic import field_validator, model_validator


class ValidatedTriage(BaseModel):
    """Triage with business rules enforced locally."""

    category: Literal["billing", "integration", "performance", "bug", "security", "howto"]
    priority: Literal["low", "medium", "high", "critical"]
    summary: str
    needs_specialist: bool

    @field_validator("summary")
    @classmethod
    def summary_length(cls, value: str) -> str:
        if len(value) > 90:
            raise ValueError(f"summary must be <= 90 characters, got {len(value)}")
        return value.strip()

    @model_validator(mode="after")
    def security_always_escalates(self):
        if self.category == "security" and not self.needs_specialist:
            raise ValueError("security tickets must set needs_specialist=True")
        return self


validated = model.with_structured_output(ValidatedTriage, include_raw=True)
outcome = validated.invoke("we need your SOC2 report and a signed DPA before renewal")

if outcome["parsing_error"]:
    print("validation failed:", outcome["parsing_error"])
else:
    print("passed:", outcome["parsed"])

# %% [markdown]
# `model_validator` is where domain rules belong. The model cannot be relied on to
# remember "security always escalates" - so make it structurally impossible to
# produce an object that violates it.

# %% [markdown]
# ## 7. Structured output in chains and agents

# %%
from langchain_core.prompts import ChatPromptTemplate

triage_chain = (
    ChatPromptTemplate.from_messages([
        ("system",
         "You triage Northwind support tickets. Enterprise customers get one priority level "
         "higher than the raw impact would suggest. Never mark a security request as low."),
        ("human", "Customer: {customer} ({plan})\nTicket: {ticket}"),
    ])
    | model.with_structured_output(TicketTriage)
)

for payload in [
    {"customer": "Cedar Logistics", "plan": "Starter", "ticket": "how do I export to PDF"},
    {"customer": "Everline Bank", "plan": "Enterprise", "ticket": "how do I export to PDF"},
]:
    triage = triage_chain.invoke(payload)
    print(f"{payload['plan']:10} -> {triage.priority:8} {triage.category}")

# %%
# Batch it over the real ticket file.
import csv

with open(ctx.data("support_tickets.csv"), newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))[:6]

results = triage_chain.batch(
    [{"customer": r["customer"], "plan": r["plan"], "ticket": r["summary"]} for r in rows],
    config={"max_concurrency": 3},
)

correct = 0
for row, triage in zip(rows, results):
    match = triage.category == row["category"]
    correct += match
    print(f"{row['ticket_id']} {'OK ' if match else 'X  '} actual={row['category']:12} predicted={triage.category:12} {triage.priority}")
print(f"\ncategory accuracy: {correct}/{len(rows)}")

# %%
# In an agent, use response_format.
import sqlite3
from contextlib import closing

from langchain.agents import create_agent
from langchain.tools import tool

from shared.sample_data import ensure_all

DB_PATH = str(ensure_all()["sqlite"])


@tool
def ticket_history(customer: str) -> str:
    """Look up all past tickets for a customer, including resolved ones."""
    with closing(sqlite3.connect(DB_PATH)) as conn:
        rows = conn.execute(
            "SELECT ticket_id, category, priority, status, summary FROM tickets WHERE customer LIKE ?",
            (f"%{customer}%",),
        ).fetchall()
    return "\n".join(f"{r[0]} [{r[1]}/{r[2]}/{r[3]}] {r[4]}" for r in rows) or "No tickets found."


agent = create_agent(
    model=model,
    tools=[ticket_history],
    system_prompt="You triage tickets. Always check the customer's history before deciding priority.",
    response_format=ToolStrategy(TicketAnalysis),
)

outcome = agent.invoke({"messages": [{"role": "user", "content":
    "Acme Retail says their dashboards are still slow. Triage this."}]})

report = outcome["structured_response"]
print(f"{report.customer.name}: {report.priority} / {report.sentiment}, escalate={report.escalate}")
print(f"summary: {report.summary}")
for item in report.action_items:
    print(f"  - [{item.due_in_hours}h] {item.owner}: {item.action}")

# %% [markdown]
# ## 8. Choosing between the approaches
#
# | Situation | Approach |
# |---|---|
# | Provider supports structured output | **`.with_structured_output()`** |
# | Local/open model, no tool calling | `PydanticOutputParser` (notebook 05) |
# | Need progressive partial objects for a UI | `JsonOutputParser` streaming |
# | Output is not JSON (a date, a list, an enum) | matching built-in parser |
# | Agent's final answer must be typed | `response_format=ToolStrategy(Schema)` |
# | Occasional failures must not crash | `include_raw=True` + fallback |

# %% [markdown]
# ## 9. Schema design rules
#
# 1. **Describe every field.** `description` is prompt text the model reads.
# 2. **`Literal` over `str`** for anything with a fixed set of values.
# 3. **Keep it flat where you can.** Deeply nested schemas degrade accuracy.
# 4. **Order fields so reasoning comes first.** A model filling `reason` before
#    `decision` produces better decisions - the schema controls generation order.
# 5. **Avoid more than ~15 fields per call.** Split into two calls instead.
# 6. **The class docstring matters** - it becomes the schema description.

# %%
class ReasonedDecision(BaseModel):
    """Approve or reject a leave request, with the reasoning recorded first."""

    # These three come first on purpose: the model "thinks" as it fills them.
    policy_rules_considered: list[str] = Field(description="Relevant policy rules")
    balance_check: str = Field(description="Whether the employee has enough balance, and the numbers")
    risk_notes: str = Field(description="Anything that complicates the decision")
    # ...and only then commits to an answer.
    decision: Literal["approve", "reject", "needs_manager"] = Field(description="Final decision")
    message_to_employee: str = Field(description="Two-sentence reply to the employee")


decider = model.with_structured_output(ReasonedDecision)
decision = decider.invoke(
    "Daniel Fernandes (E-104) has 21 annual leave days remaining and is requesting 10 consecutive "
    "days starting 2 March 2026, submitted today 26 January 2026. Policy: 7 days notice for 3+ days, "
    "documented handover for 5+ days."
)

print("rules     :", decision.policy_rules_considered)
print("balance   :", decision.balance_check)
print("risk      :", decision.risk_notes)
print("DECISION  :", decision.decision)
print("message   :", decision.message_to_employee)

# %% [markdown]
# That field ordering is a cheap, reliable accuracy improvement - the structured
# equivalent of "think step by step".

# %% [markdown]
# ## Try it yourself
#
# 1. **Reorder and measure.** Move `decision` to the *first* field in
#    `ReasonedDecision`, run both versions over 10 scenarios, and compare quality.
# 2. **Force a validation failure.** Make `summary` `max_length=20` and find a
#    ticket that breaks it. Handle it with `include_raw=True`.
# 3. **Flat vs nested.** Build a flat 12-field version of `TicketAnalysis` and
#    compare accuracy and latency against the nested one.
# 4. **Full pipeline.** Combine notebook 22 and this one: receipt image ->
#    `ExpenseClaim` -> policy check -> structured approval decision.

# %% [markdown]
# ## Recap
#
# | Concept | Takeaway |
# |---|---|
# | `.with_structured_output(Schema)` | Turns the model into a typed function |
# | Pydantic / `TypedDict` / JSON schema | Validated object / dict / dict |
# | `method="function_calling"` vs `"json_schema"` | Portable vs strongest guarantee |
# | `include_raw=True` | Get `parsed`, `raw` and `parsing_error` - required in production |
# | Nested schemas | One call can produce a whole analysis object |
# | `field_validator` / `model_validator` | Where your business rules live |
# | `response_format=ToolStrategy(...)` | Typed final answer from an agent |
# | Field order | Put reasoning fields before the decision field |
#
# ## Next
#
# -> [25_routing_and_handoffs.ipynb](25_routing_and_handoffs.ipynb)
