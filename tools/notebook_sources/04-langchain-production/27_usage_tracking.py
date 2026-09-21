# %% [markdown]
# # 27 - Usage and Cost Tracking
#
# | | |
# |---|---|
# | **Level** | Intermediate |
# | **Time** | 40 minutes |
# | **Prerequisites** | `26_evaluation` |
# | **Checklist ID** | `27_usage_tracking` |
#
# ## Why this matters
#
# The first surprise bill is a rite of passage. A RAG chatbot that costs a
# fraction of a cent per question feels free - until 5,000 employees use it daily
# and someone pastes a 40-page contract into the box.
#
# You need three things before launch: **visibility** (what does each request
# cost?), **attribution** (which team or feature is spending it?) and **limits**
# (what stops a runaway loop?). This notebook builds all three.

# %%
from pathlib import Path
import sys

_root = Path.cwd()
while not (_root / "shared").is_dir() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from shared.notebook_setup import setup, require  # noqa: E402

ctx = setup("27_usage_tracking")

# %%
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from shared.llm import get_chat_model, model_name_for, active_provider

model = get_chat_model()
parser = StrOutputParser()

print(f"provider={active_provider()} model={model_name_for(active_provider())}")

# %% [markdown]
# ## 1. `usage_metadata` - the source of truth
#
# Every modern chat model returns token counts on the response itself. This is
# the provider's own count, not an estimate, so it is what you should bill
# against.

# %%
response = model.invoke("Explain vector embeddings in two sentences.")

print(response.content.strip())
print("\nusage_metadata:", response.usage_metadata)

# %% [markdown]
# The shape is standardised across providers:
#
# ```python
# {
#     "input_tokens": 12,
#     "output_tokens": 48,
#     "total_tokens": 60,
#     "input_token_details": {"cache_read": 0},      # provider-dependent
#     "output_token_details": {"reasoning": 0},      # reasoning models only
# }
# ```
#
# The `*_details` sub-dicts matter for cost: **cached input tokens are typically
# billed at 10% of the normal rate**, and reasoning tokens are billed as output
# even though you never see them.

# %%
usage = response.usage_metadata or {}
print(f"input   : {usage.get('input_tokens', 0)}")
print(f"output  : {usage.get('output_tokens', 0)}")
print(f"cached  : {(usage.get('input_token_details') or {}).get('cache_read', 0)}")
print(f"reasoning: {(usage.get('output_token_details') or {}).get('reasoning', 0)}")

# %% [markdown]
# ## 2. Counting tokens *before* you send
#
# Useful for guardrails: reject or truncate an oversized request rather than
# paying for it and then failing on a context-length error.

# %%
long_text = (Path(ctx.data("company_handbook.md")).read_text(encoding="utf-8")) * 3

try:
    estimated = model.get_num_tokens(long_text)
    print(f"model.get_num_tokens -> {estimated:,} tokens")
except Exception as exc:
    print(f"model.get_num_tokens unavailable ({type(exc).__name__}); falling back to tiktoken")
    import tiktoken

    estimated = len(tiktoken.get_encoding("cl100k_base").encode(long_text))
    print(f"tiktoken cl100k_base -> {estimated:,} tokens")

print(f"characters per token: {len(long_text) / max(estimated, 1):.2f}")

# %% [markdown]
# The ~4 characters-per-token rule of thumb holds for English prose. It breaks
# badly for code, JSON, non-Latin scripts and long identifiers - so estimate,
# never assume.

# %%
samples = {
    "english prose": "The quarterly revenue increased by fifteen percent compared to last year.",
    "json": '{"customer_id":"CUST-48213","plan":"enterprise","seats":240,"mrr_usd":18400}',
    "python code": "def compute_rank(scores: dict[str, float]) -> list[str]:\n    return sorted(scores, key=scores.get, reverse=True)",
    "hindi": "कर्मचारियों को प्रति वर्ष चौबीस दिन का वार्षिक अवकाश मिलता है।",
}

print(f"{'kind':14} {'chars':>6} {'tokens':>7} {'chars/token':>12}")
for kind, text in samples.items():
    n = model.get_num_tokens(text)
    print(f"{kind:14} {len(text):>6} {n:>7} {len(text) / n:>12.2f}")

# %% [markdown]
# Note how much more expensive non-Latin scripts and JSON are per character. If
# your product serves Hindi or Japanese users, your per-request cost is
# materially higher than your English test data suggested.

# %% [markdown]
# ## 3. A cost tracking callback
#
# `usage_metadata` on a single response is easy. Real chains make several calls,
# and agents make an unpredictable number - so accumulate with a callback.

# %%
import time
from dataclasses import dataclass

from langchain_core.callbacks import BaseCallbackHandler

#: USD per 1 million tokens. Update these from your provider's pricing page.
PRICING: dict[str, tuple[float, float]] = {
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "claude-sonnet-4-5": (3.00, 15.00),
}
DEFAULT_PRICE = (0.50, 1.50)
CACHED_INPUT_DISCOUNT = 0.10


@dataclass
class CallRecord:
    model: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    seconds: float

    @property
    def cost_usd(self) -> float:
        in_price, out_price = PRICING.get(self.model, DEFAULT_PRICE)
        billable_input = self.input_tokens - self.cached_tokens
        return (
            billable_input * in_price
            + self.cached_tokens * in_price * CACHED_INPUT_DISCOUNT
            + self.output_tokens * out_price
        ) / 1_000_000


class UsageTracker(BaseCallbackHandler):
    """Accumulates tokens, cost and latency across every model call in a run."""

    def __init__(self, label: str = "run"):
        self.label = label
        self.records: list[CallRecord] = []
        self._starts: dict[str, float] = {}

    # --- callback hooks -------------------------------------------------- #
    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
        self._starts[str(run_id)] = time.perf_counter()

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        self._starts[str(run_id)] = time.perf_counter()

    def on_llm_end(self, response, *, run_id=None, **kwargs):
        elapsed = time.perf_counter() - self._starts.pop(str(run_id), time.perf_counter())
        try:
            message = response.generations[0][0].message
        except (AttributeError, IndexError):
            return
        usage = getattr(message, "usage_metadata", None) or {}
        self.records.append(
            CallRecord(
                model=(response.llm_output or {}).get("model_name")
                or message.response_metadata.get("model_name", "unknown"),
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                cached_tokens=(usage.get("input_token_details") or {}).get("cache_read", 0),
                seconds=elapsed,
            )
        )

    # --- reporting -------------------------------------------------------- #
    @property
    def total_tokens(self) -> int:
        return sum(r.input_tokens + r.output_tokens for r in self.records)

    @property
    def total_cost(self) -> float:
        return sum(r.cost_usd for r in self.records)

    def report(self) -> str:
        if not self.records:
            return f"{self.label}: no model calls"
        lines = [f"{self.label}: {len(self.records)} call(s)"]
        for i, r in enumerate(self.records, 1):
            lines.append(
                f"  {i}. {r.model[:28]:30} in={r.input_tokens:<6} out={r.output_tokens:<5} "
                f"{r.seconds:>5.2f}s  ${r.cost_usd:.6f}"
            )
        lines.append(f"  TOTAL {self.total_tokens:,} tokens  ${self.total_cost:.6f}")
        return "\n".join(lines)

# %%
summarise = ChatPromptTemplate.from_template("Summarise in one sentence: {text}") | model | parser
critique = ChatPromptTemplate.from_template("Give one improvement for this summary: {summary}") | model | parser

tracker = UsageTracker("summarise + critique")
summary = summarise.invoke({"text": Path(ctx.data("leave_policy.txt")).read_text(encoding="utf-8")},
                           config={"callbacks": [tracker]})
critique.invoke({"summary": summary}, config={"callbacks": [tracker]})

print(tracker.report())

# %% [markdown]
# Attaching the tracker at **invocation** scope means it sees every nested call
# made during that invocation - including tool-calling loops inside an agent,
# where you cannot predict the call count in advance.

# %% [markdown]
# ## 4. Tracking an agent
#
# This is where cost surprises live. Each tool call costs another round trip with
# the full conversation resent as input.

# %%
from langchain.agents import create_agent
from langchain_core.tools import tool


@tool
def get_ticket_count(team: str) -> int:
    """Return the number of open support tickets for a team."""
    return {"billing": 14, "platform": 37, "onboarding": 5}.get(team.lower(), 0)


@tool
def get_headcount(team: str) -> int:
    """Return the number of engineers on a team."""
    return {"billing": 4, "platform": 9, "onboarding": 3}.get(team.lower(), 0)


agent = create_agent(model, [get_ticket_count, get_headcount])

agent_tracker = UsageTracker("agent")
result = agent.invoke(
    {"messages": [("user", "Which team has the worst ticket-per-engineer ratio: billing, platform or onboarding?")]},
    config={"callbacks": [agent_tracker]},
)

print(result["messages"][-1].content.strip()[:220])
print()
print(agent_tracker.report())

# %% [markdown]
# Look at the input token counts climbing across calls. The conversation is
# resent in full every turn, so **agent cost grows quadratically with the number
# of tool calls**, not linearly. Capping iterations is a cost control, not just a
# safety measure.

# %% [markdown]
# ## 5. Attribution: who spent it?
#
# A single total is useless for chargeback. Tag every request with tenant,
# feature and user so you can slice the spend.

# %%
from collections import defaultdict


class AttributedTracker(UsageTracker):
    """Usage tracker that groups spend by arbitrary tags."""

    def __init__(self):
        super().__init__("attributed")
        self.by_tag: dict[str, dict[str, float]] = defaultdict(lambda: {"tokens": 0, "cost": 0.0, "calls": 0})
        self._current_tags: dict[str, str] = {}

    def set_tags(self, **tags: str) -> "AttributedTracker":
        self._current_tags = tags
        return self

    def on_llm_end(self, response, *, run_id=None, **kwargs):
        before = len(self.records)
        super().on_llm_end(response, run_id=run_id, **kwargs)
        if len(self.records) == before:
            return
        record = self.records[-1]
        for key, value in self._current_tags.items():
            bucket = self.by_tag[f"{key}={value}"]
            bucket["tokens"] += record.input_tokens + record.output_tokens
            bucket["cost"] += record.cost_usd
            bucket["calls"] += 1

    def breakdown(self) -> str:
        rows = sorted(self.by_tag.items(), key=lambda kv: kv[1]["cost"], reverse=True)
        lines = [f"{'tag':28} {'calls':>6} {'tokens':>9} {'cost':>12}"]
        for tag, stats in rows:
            lines.append(f"{tag:28} {stats['calls']:>6} {int(stats['tokens']):>9,} ${stats['cost']:>11.6f}")
        return "\n".join(lines)


billed = AttributedTracker()

WORKLOAD = [
    ("acme", "policy_qa", "How many sick leave days are allowed?"),
    ("acme", "policy_qa", "What is the notice period for L5?"),
    ("acme", "ticket_triage", "Customer reports dashboards loading slowly since the update."),
    ("globex", "policy_qa", "Can I carry forward annual leave?"),
    ("globex", "ticket_triage", "Double charged for January seats, very upset."),
]

qa = ChatPromptTemplate.from_template("Answer briefly: {text}") | model | parser

for tenant, feature, text in WORKLOAD:
    billed.set_tags(tenant=tenant, feature=feature)
    qa.invoke({"text": text}, config={"callbacks": [billed]})

print(billed.breakdown())
print(f"\ngrand total: ${billed.total_cost:.6f}")

# %% [markdown]
# ### The same thing, natively, via LangSmith metadata
#
# If you have LangSmith (notebook 19), attach the same tags as run metadata and
# get this breakdown in the UI with no custom code.

# %%
if require("LANGSMITH_API_KEY", feature="LangSmith cost attribution"):
    qa.invoke(
        {"text": "What is the annual learning budget?"},
        config={
            "run_name": "policy_qa",
            "tags": ["tenant:acme", "feature:policy_qa"],
            "metadata": {"tenant_id": "acme", "user_id": "u-4821", "feature": "policy_qa"},
        },
    )
    print("Traced. Filter by metadata in LangSmith and group by cost.")

# %% [markdown]
# ## 6. Budgets and hard limits
#
# Tracking tells you what happened. Limits stop it happening again.

# %%
class BudgetExceeded(RuntimeError):
    """Raised when a run exceeds its allotted spend."""


class BudgetGuard(UsageTracker):
    """Aborts the run once cumulative cost or call count crosses a ceiling."""

    def __init__(self, max_usd: float = 0.01, max_calls: int = 10):
        super().__init__("budget")
        self.max_usd = max_usd
        self.max_calls = max_calls

    def on_llm_end(self, response, *, run_id=None, **kwargs):
        super().on_llm_end(response, run_id=run_id, **kwargs)
        if len(self.records) > self.max_calls:
            raise BudgetExceeded(f"call limit {self.max_calls} exceeded")
        if self.total_cost > self.max_usd:
            raise BudgetExceeded(f"budget ${self.max_usd} exceeded (spent ${self.total_cost:.6f})")


guard = BudgetGuard(max_usd=0.000_02, max_calls=10)
try:
    agent.invoke(
        {"messages": [("user", "Compare ticket load per engineer for all three teams and rank them.")]},
        config={"callbacks": [guard]},
    )
except BudgetExceeded as exc:
    print(f"stopped: {exc}")
    print(guard.report())

# %% [markdown]
# Raising from a callback aborts the whole run. That is deliberate here, but note
# the trade-off from notebook 20: an exception in a callback is *not* isolated.
# Use it only for genuine circuit breakers, never for logging.
#
# A gentler alternative for agents is `ModelCallLimitMiddleware`, which ends the
# loop cleanly and lets the agent return what it has so far.

# %%
from langchain.agents.middleware import ModelCallLimitMiddleware

capped_agent = create_agent(
    model,
    [get_ticket_count, get_headcount],
    middleware=[ModelCallLimitMiddleware(thread_limit=3, exit_behavior="end")],
)

capped_tracker = UsageTracker("capped agent")
outcome = capped_agent.invoke(
    {"messages": [("user", "Rank all three teams by tickets per engineer, with reasoning.")]},
    config={"callbacks": [capped_tracker]},
)
print(outcome["messages"][-1].content.strip()[:200])
print()
print(capped_tracker.report())

# %% [markdown]
# ## 7. Cost modelling before you launch
#
# Measure one representative request, then extrapolate. This five-line model has
# prevented more bad architecture decisions than any benchmark.

# %%
def project_monthly_cost(cost_per_request: float, requests_per_day: int, label: str = "") -> None:
    daily = cost_per_request * requests_per_day
    print(
        f"{label:24} ${cost_per_request:.6f}/req  "
        f"${daily:>8.2f}/day  ${daily * 30:>9.2f}/month  ${daily * 365:>10.2f}/year"
    )


simple_tracker = UsageTracker()
qa.invoke({"text": "How many casual leave days are allowed?"}, config={"callbacks": [simple_tracker]})
simple_cost = simple_tracker.total_cost

agent_cost = agent_tracker.total_cost

print(f"{'workload':24} {'per request':>13} {'per day':>12} {'per month':>12} {'per year':>13}")
for requests in (1_000, 10_000):
    print(f"\n-- {requests:,} requests/day --")
    project_monthly_cost(simple_cost, requests, "simple Q&A")
    project_monthly_cost(agent_cost, requests, "agent (multi tool call)")
    project_monthly_cost(simple_cost * 4, requests, "RAG with reranking")

# %% [markdown]
# The gap between the simple path and the agent path is usually 10-30x. That is
# the entire argument for the routing work in notebook 25: send the easy 80% down
# the cheap path and reserve the agent for cases that need it.

# %% [markdown]
# ## 8. The levers that actually reduce cost
#
# | Lever | Typical saving | Cost |
# |---|---|---|
# | **Route to a small model** | 70-90% on routed traffic | Slightly worse quality on hard cases |
# | **Cache** (notebook 23) | 100% on repeats | Staleness risk |
# | **Provider prompt caching** | 90% on the cached prefix | Requires a stable prompt prefix |
# | **Trim conversation history** | Grows with turn count | Loses distant context |
# | **Retrieve fewer chunks** | Linear in `k` | Lower recall - measure it (notebook 26) |
# | **Shorten the system prompt** | Small but on every call | Engineering time |
# | **Cap `max_tokens`** | Bounds the worst case | Truncated answers |
#
# Measure before and after each one with the tracker above. "Obvious"
# optimisations frequently move cost by less than 2%.

# %%
from langchain_core.messages import trim_messages

long_history = []
for i in range(12):
    long_history.append(("user", f"Question {i}: tell me about the leave policy clause {i}."))
    long_history.append(("assistant", f"Clause {i} covers eligibility, accrual, approval and carry-forward rules."))

full_prompt = ChatPromptTemplate.from_messages([("system", "You are an HR assistant.")] + long_history + [("user", "{q}")])
untrimmed = UsageTracker("full history")
(full_prompt | model | parser).invoke({"q": "Summarise what we discussed."}, config={"callbacks": [untrimmed]})

trimmed_messages = trim_messages(
    full_prompt.invoke({"q": "Summarise what we discussed."}).to_messages(),
    max_tokens=400,
    token_counter=model,
    strategy="last",
    include_system=True,
    start_on="human",
)
trimmed = UsageTracker("trimmed history")
model.invoke(trimmed_messages, config={"callbacks": [trimmed]})

saving = 1 - (trimmed.total_cost / untrimmed.total_cost) if untrimmed.total_cost else 0
print(untrimmed.report())
print(trimmed.report())
print(f"\nsaving: {saving:.0%}")

# %% [markdown]
# ## Try it yourself
#
# 1. **Update `PRICING`** with your provider's real numbers and re-run the
#    projection. The absolute figures only matter once they are accurate.
# 2. **Per-tenant monthly caps.** Extend `AttributedTracker` with a
#    `limits: dict[str, float]` and raise when a tenant crosses its cap.
# 3. **Measure the routing saving.** Combine notebook 25's cascading router with
#    this tracker and report the cost of routed versus always-large-model on a
#    20-question workload.
# 4. **Find your worst request.** Run 20 varied questions, sort by cost, and look
#    at the top three - the distribution is usually far more skewed than expected.

# %% [markdown]
# ## Recap
#
# | Concept | Takeaway |
# |---|---|
# | `usage_metadata` | Provider's own token counts on every response - bill against this |
# | `input_token_details.cache_read` | Cached input is ~10% price; track it separately |
# | `get_num_tokens` | Estimate before sending to guard against oversized input |
# | Chars per token | ~4 for English; far worse for JSON, code and non-Latin scripts |
# | `BaseCallbackHandler` | Accumulate usage across nested and unpredictable calls |
# | Invocation-scope callbacks | Capture agent loops where call count is unknown up front |
# | Attribution tags | Tenant/feature/user tags make chargeback possible |
# | Agent cost growth | Quadratic in tool calls - history is resent every turn |
# | `BudgetGuard` | Raise from a callback as a circuit breaker only |
# | `ModelCallLimitMiddleware` | Graceful cap that still returns partial work |
# | Cost projection | Measure one request, extrapolate, decide architecture before launch |
#
# ## Next
#
# -> [27a_llm_gateways.ipynb](27a_llm_gateways.ipynb)
