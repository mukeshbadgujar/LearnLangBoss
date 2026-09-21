# %% [markdown]
# # 27a - LLM Gateways: Fallbacks, Routing and Load Balancing
#
# | | |
# |---|---|
# | **Level** | Intermediate to Advanced |
# | **Time** | 45 minutes |
# | **Prerequisites** | `00_environment_providers_and_keys`, `27_usage_tracking` |
# | **Checklist ID** | `27a_llm_gateways` |
# | **Sourced from** | `LLG/llm_gateway_tutorial.ipynb` |
#
# ## Why this matters
#
# `shared/llm.py` picks one provider at import time. That is fine for learning.
# In production you want **per-call** decisions:
#
# - fallback when Groq is down
# - route cheap classification to a small model and hard reasoning to a large one
# - spread load across keys / regions
# - one place that accounts for cost
#
# That layer is an **LLM gateway**. LiteLLM is the open-source one most teams
# reach for; LangChain talks to it like any OpenAI-compatible endpoint. This
# lesson shows the patterns; it does not require you to deploy a gateway.

# %%
from pathlib import Path
import sys

_root = Path.cwd()
while not (_root / "shared").is_dir() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from shared.notebook_setup import setup, require_package  # noqa: E402

ctx = setup("27a_llm_gateways")

# %% [markdown]
# ## 1. What our helper already does (and does not)
#
# `get_chat_model()` resolves Groq → OpenRouter → OpenAI → Anthropic **once**,
# when the process starts. If Groq goes down mid-request, every call fails until
# you restart. A gateway retries and fails over inside the same call.

# %%
from shared.llm import available_providers, describe_environment, get_chat_model

print(describe_environment())
print("available:", available_providers())

# Built-in LCEL fallback - the poor man's gateway, no extra dependency.
primary = get_chat_model()
fallback_chain = primary.with_fallbacks(
    [get_chat_model()]  # same provider here; in production, pass a second provider
)
print(fallback_chain.invoke("Say OK in one word.").content)

# %% [markdown]
# `with_fallbacks` covers the happy case of "provider A is down, try B". It
# does not do cost-based routing, key pooling, or spend dashboards. That is
# what a gateway adds.

# %% [markdown]
# ## 2. LiteLLM as a unified router (optional)
#
# Install with `python -m pip install litellm`. One client, many providers,
# model strings like `groq/llama-3.3-70b-versatile`.

# %%
if require_package("litellm", feature="LiteLLM gateway patterns", pip="litellm"):
    import litellm

    # Per-call fallbacks: try Groq, then OpenRouter, then OpenAI.
    # Keys are read from the same env vars our shared.llm uses.
    try:
        response = litellm.completion(
            model="groq/llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Say OK in one word."}],
            fallbacks=[
                "openrouter/openai/gpt-4o-mini",
                "openai/gpt-4o-mini",
            ],
            max_tokens=16,
        )
        print("gateway reply:", response.choices[0].message.content)
        print("model used   :", response.model)
    except Exception as exc:  # noqa: BLE001
        print(f"gateway call failed (keys may be missing): {type(exc).__name__}: {exc}")

# %% [markdown]
# ## 3. Cost-aware routing
#
# Classification and extraction rarely need a frontier model. Route by task.

# %%
ROUTING_TABLE = {
    "classify": "groq/llama-3.1-8b-instant",      # cheap, fast
    "extract": "groq/llama-3.1-8b-instant",
    "reason": "groq/llama-3.3-70b-versatile",     # stronger
    "write": "openrouter/openai/gpt-4o-mini",    # quality for prose
}


def model_for(task: str) -> str:
    return ROUTING_TABLE.get(task, ROUTING_TABLE["reason"])


for task in ("classify", "reason", "write"):
    print(f"  {task:10} -> {model_for(task)}")

# %% [markdown]
# Wire this into LangChain by constructing the chat model per call:
#
# ```python
# from langchain_openai import ChatOpenAI
# # Point at LiteLLM's OpenAI-compatible proxy if you run one:
# chat = ChatOpenAI(base_url="http://localhost:4000/v1", api_key="sk-...",
#                   model=model_for("classify"))
# ```
#
# Or keep using `shared.llm.get_chat_model(model=...)` and let the gateway sit
# in front of every provider.

# %% [markdown]
# ## 4. Load balancing and key pooling
#
# When one key is rate-limited, a gateway rotates across a pool. Conceptually:

# %%
from itertools import cycle

GROQ_KEYS = ["gsk_primary", "gsk_secondary", "gsk_tertiary"]  # placeholders
key_ring = cycle(GROQ_KEYS)


def next_key() -> str:
    return next(key_ring)


print([next_key() for _ in range(5)])

# %% [markdown]
# In LiteLLM this is `Router` with `Deployment` entries. In our curriculum the
# important idea is: **keys are a pool, not a singleton**. Putting the only
# production key in `.env` on one box is how outages happen.

# %% [markdown]
# ## 5. Central cost accounting
#
# Notebook 27 tracks cost per request with a callback. A gateway does it once
# for every service that talks to it - which is why finance prefers a gateway
# over twelve services each inventing their own meter.

# %%
class GatewayMeter:
    def __init__(self):
        self.by_model: dict[str, dict] = {}

    def record(self, model: str, input_tokens: int, output_tokens: int, usd: float):
        bucket = self.by_model.setdefault(model, {"calls": 0, "in": 0, "out": 0, "usd": 0.0})
        bucket["calls"] += 1
        bucket["in"] += input_tokens
        bucket["out"] += output_tokens
        bucket["usd"] += usd


meter = GatewayMeter()
meter.record("groq/llama-3.1-8b-instant", 120, 40, 0.0001)
meter.record("groq/llama-3.3-70b-versatile", 800, 200, 0.002)
meter.record("groq/llama-3.1-8b-instant", 90, 30, 0.00008)
print(meter.by_model)

# %% [markdown]
# ## 6. Choosing: helper, LCEL fallback, or gateway
#
# | Need | Use |
# |---|---|
# | Learning / single-box demos | `shared.llm.get_chat_model()` |
# | "If A fails, try B" inside one chain | `.with_fallbacks([...])` |
# | Multi-service, cost routing, key pools, spend caps | An LLM gateway (LiteLLM, Portkey, Helicone, provider gateways) |
#
# Do not start with a gateway. Start with the helper. Add fallbacks when you
# feel the first outage. Add a gateway when a second service needs the same
# routing rules.

# %% [markdown]
# ## Try it yourself
#
# 1. Point `ChatOpenAI(base_url=...)` at a local LiteLLM proxy and run one
#    notebook through it unchanged.
# 2. Add a `BudgetGuard` (notebook 27) that reads from `GatewayMeter` and
#    refuses calls after a daily USD cap.
# 3. Measure p50 latency of `llama-3.1-8b-instant` vs `llama-3.3-70b-versatile`
#    on a 20-example classification set. Put the winner in `ROUTING_TABLE`.
#
# ## Recap
#
# | Concept | Takeaway |
# |---|---|
# | Import-time vs per-call | `shared.llm` picks once; a gateway picks every call |
# | `.with_fallbacks` | The smallest resilience upgrade |
# | Task-based routing | Small model for classify/extract, large for reason/write |
# | Key pooling | Rate limits become a fleet problem, not a single-key problem |
# | Central metering | One dashboard beats twelve home-grown counters |
# | When to graduate | Second service sharing the same routing rules |
#
# ## Next
#
# -> [28_langchain_hub.ipynb](28_langchain_hub.ipynb)
