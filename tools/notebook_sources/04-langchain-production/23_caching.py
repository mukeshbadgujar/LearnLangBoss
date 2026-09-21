# %% [markdown]
# # 23 - Caching: Model, Semantic and Prompt
#
# | | |
# |---|---|
# | **Level** | Intermediate |
# | **Time** | 35 minutes |
# | **Prerequisites** | `22_multimodal` |
# | **Checklist ID** | `23_caching` |
#
# ## Why this matters
#
# Support assistants get the same questions constantly. "What are the API rate
# limits?" arrives forty times a day in thirty different phrasings. Paying for
# forty identical generations is waste you can remove in three lines.
#
# There are three distinct caches, and people conflate them:
#
# | Cache | Keyed on | Removes |
# |---|---|---|
# | **Model cache** | exact prompt + model + params | duplicate generations |
# | **Semantic cache** | embedding similarity | *near*-duplicate generations |
# | **Prompt caching** (provider-side) | shared prompt prefix | re-processing a long system prompt |

# %%
from pathlib import Path
import sys

_root = Path.cwd()
while not (_root / "shared").is_dir() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from shared.notebook_setup import setup, require  # noqa: E402

ctx = setup("23_caching")

# %%
import time

from shared.llm import get_chat_model

model = get_chat_model()


def timed(label: str, fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    print(f"{label:28} {time.perf_counter() - start:6.3f}s")
    return result

# %% [markdown]
# ## 1. `InMemoryCache`: the simplest win

# %%
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache

set_llm_cache(InMemoryCache())

question = "What is the difference between a chain and an agent? Two sentences."

first = timed("first call (miss)", model.invoke, question)
second = timed("second call (hit)", model.invoke, question)

print("\nidentical content:", first.content == second.content)
print("identical id     :", first.id == second.id)

# %% [markdown]
# The second call never reached the provider. Note it returned the **exact same
# response object** - a cache hit replays a previous generation, it does not
# regenerate.

# %% [markdown]
# ### What the cache key includes
#
# Any change misses. That is correct behaviour but surprises people.

# %%
probes = [
    ("same prompt", lambda: model.invoke(question)),
    ("trailing space", lambda: model.invoke(question + " ")),
    ("different case", lambda: model.invoke(question.upper())),
    ("different temperature", lambda: get_chat_model(temperature=0.7).invoke(question)),
]

for label, call in probes:
    timed(label, call)

# %% [markdown]
# The key is `(prompt string, model name, all model parameters)`. A single
# trailing space is a different key. That fragility is exactly why semantic
# caching exists.

# %% [markdown]
# ## 2. `SQLiteCache`: survives a restart
#
# `InMemoryCache` dies with the process, which makes it useless for a service that
# restarts on every deploy.

# %%
from langchain_community.cache import SQLiteCache

cache_path = ctx.artifact("llm_cache.sqlite")
set_llm_cache(SQLiteCache(database_path=str(cache_path)))

prompt = "Summarise the Northwind hybrid work policy in one sentence."

timed("sqlite miss", model.invoke, prompt)
timed("sqlite hit ", model.invoke, prompt)

print(f"\ncache file: {cache_path} ({cache_path.stat().st_size:,} bytes)")
print("This survives a kernel restart - re-run the cell after restarting to prove it.")

# %%
import sqlite3

with sqlite3.connect(cache_path) as conn:
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    count = conn.execute("SELECT COUNT(*) FROM full_llm_cache").fetchone()[0] if "full_llm_cache" in tables else 0
print("tables:", tables, "| cached entries:", count)

# %% [markdown]
# ## 3. When caching is wrong
#
# Caching is not free correctness. Turn it off when:
#
# - **the answer depends on time or state** - "how many tickets are open?" must
#   not be served from a cache
# - **you want variety** - a brainstorming tool returning the same three ideas
#   forever is broken
# - **responses are user-specific** - a shared cache across tenants is a data leak
# - **the prompt embeds retrieved content that changes** - the cache key includes
#   it, so this is usually safe, but verify

# %%
set_llm_cache(None)   # disable globally
print("global cache off")

# Or disable for one model instance while others keep caching.
set_llm_cache(InMemoryCache())
uncached_model = get_chat_model(cache=False)

timed("cached model   miss", model.invoke, "Name one benefit of caching.")
timed("cached model   hit ", model.invoke, "Name one benefit of caching.")
timed("uncached model 1st ", uncached_model.invoke, "Name one benefit of caching.")
timed("uncached model 2nd ", uncached_model.invoke, "Name one benefit of caching.")

# %% [markdown]
# ### Per-tenant cache keys
#
# If you must cache user-specific content, put the tenant in the prompt or use
# separate cache instances. Never share one cache across tenants.

# %%
set_llm_cache(None)

TENANT_CACHES: dict[str, InMemoryCache] = {}


def model_for_tenant(tenant_id: str):
    """One cache per tenant - the only safe way to cache user-specific output."""
    if tenant_id not in TENANT_CACHES:
        TENANT_CACHES[tenant_id] = InMemoryCache()
    return get_chat_model(cache=TENANT_CACHES[tenant_id])


timed("tenant A miss", model_for_tenant("acme").invoke, "Summarise our plan limits.")
timed("tenant A hit ", model_for_tenant("acme").invoke, "Summarise our plan limits.")
timed("tenant B miss", model_for_tenant("granite").invoke, "Summarise our plan limits.")

print("\ncaches held:", list(TENANT_CACHES))

# %% [markdown]
# ## 4. Semantic caching
#
# Exact-match caching misses "What are the API rate limits?" versus "how many
# requests per minute can we send?" - semantically identical, textually different.
#
# A semantic cache embeds the query and returns a cached answer when similarity
# crosses a threshold.

# %%
from typing import Any

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document

from shared.llm import get_embeddings


class SemanticCache:
    """Return a cached answer for a semantically similar previous question."""

    def __init__(self, embeddings, threshold: float = 0.88):
        self.store = InMemoryVectorStore(embeddings)
        self.threshold = threshold
        self.answers: dict[str, str] = {}
        self.stats = {"hits": 0, "misses": 0}

    def lookup(self, question: str) -> str | None:
        results = self.store.similarity_search_with_score(question, k=1)
        if not results:
            self.stats["misses"] += 1
            return None
        document, score = results[0]
        # InMemoryVectorStore returns cosine SIMILARITY: higher is better.
        if score >= self.threshold:
            self.stats["hits"] += 1
            return self.answers[document.page_content]
        self.stats["misses"] += 1
        return None

    def store_answer(self, question: str, answer: str) -> None:
        self.store.add_documents([Document(question)])
        self.answers[question] = answer

    def ask(self, model: Any, question: str) -> tuple[str, bool]:
        cached = self.lookup(question)
        if cached is not None:
            return cached, True
        answer = model.invoke(question).content
        self.store_answer(question, answer)
        return answer, False


semantic_cache = SemanticCache(get_embeddings(), threshold=0.85)
uncached = get_chat_model(cache=False)

questions = [
    "What are the API rate limits on the Growth plan?",
    "How many requests per minute can a Growth customer send?",   # paraphrase
    "What's the rate limit for Growth?",                          # paraphrase
    "How long is the notice period for L4 engineers?",            # different topic
    "What notice period applies to an L4 engineer?",              # paraphrase
]

for question in questions:
    start = time.perf_counter()
    answer, was_hit = semantic_cache.ask(uncached, question)
    elapsed = time.perf_counter() - start
    print(f"{'HIT ' if was_hit else 'MISS'} {elapsed:5.2f}s  {question}")

print(f"\n{semantic_cache.stats['hits']} hits / {sum(semantic_cache.stats.values())} queries")

# %% [markdown]
# ### Choosing the threshold
#
# This is the whole design decision, and it is a trade-off with real consequences.

# %%
probe_pairs = [
    ("What are the API rate limits?", "What is the rate limit?", "should hit"),
    ("What are the API rate limits?", "What are the storage limits?", "should MISS"),
    ("How much annual leave?", "How much sick leave?", "should MISS"),
    ("How much annual leave?", "How many vacation days?", "should hit"),
    ("Can I work remotely?", "Can I not work remotely?", "should MISS - negation"),
]

embeddings = get_embeddings()
print(f"{'similarity':>11}  {'expectation':22} pair")
for left, right, expectation in probe_pairs:
    a, b = embeddings.embed_documents([left, right])
    import math

    similarity = sum(x * y for x, y in zip(a, b)) / (
        math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    )
    print(f"{similarity:>11.4f}  {expectation:22} {left!r} / {right!r}")

# %% [markdown]
# Look at the negation pair: "Can I work remotely?" and "Can I **not** work
# remotely?" score very high. A semantic cache will happily serve the wrong
# answer. And "annual leave" vs "sick leave" may score above a careless threshold.
#
# **Practical rules:**
#
# 1. Start at 0.95 and lower only with evidence.
# 2. Never semantically cache anything where a near-miss is expensive - pricing,
#    legal, medical, security.
# 3. Log every hit with both questions so you can audit false positives.
# 4. Set a TTL. A cached answer from before a policy change is now a wrong answer.

# %%
if require("REDIS_URL", feature="production semantic cache (Redis)"):
    import os

    from langchain_redis import RedisSemanticCache

    set_llm_cache(
        RedisSemanticCache(
            redis_url=os.environ["REDIS_URL"],
            embeddings=get_embeddings(),
            distance_threshold=0.1,   # distance, so LOWER is stricter
            ttl=3600,
        )
    )
    timed("redis semantic miss", model.invoke, "What are the Growth plan rate limits?")
    timed("redis semantic hit ", model.invoke, "What is the rate limit on Growth?")
    set_llm_cache(None)

# %% [markdown]
# ## 5. Provider prompt caching
#
# Different mechanism entirely. Instead of caching the *answer*, the provider
# caches its internal processing of a repeated **prompt prefix** - a long system
# prompt, a large document, a big tool schema.
#
# You still get a fresh generation. You just pay much less for the prefix.

# %%
LONG_SYSTEM_PROMPT = (
    Path(ctx.data("company_handbook.md")).read_text(encoding="utf-8")
    + "\n\n"
    + Path(ctx.data("leave_policy.txt")).read_text(encoding="utf-8")
    + "\n\nYou are the Northwind HR assistant. Answer strictly from the policy above."
)

print(f"system prompt: {len(LONG_SYSTEM_PROMPT):,} chars (~{len(LONG_SYSTEM_PROMPT) // 4:,} tokens)")
print("Sent on EVERY request. Prompt caching is what makes this affordable.")

# %%
from langchain.messages import HumanMessage, SystemMessage

set_llm_cache(None)
fresh_model = get_chat_model(cache=False)

for question in ["How many casual leave days?", "What is the hotel limit in India?"]:
    response = fresh_model.invoke([SystemMessage(LONG_SYSTEM_PROMPT), HumanMessage(question)])
    usage = response.usage_metadata or {}
    cached_tokens = (usage.get("input_token_details") or {}).get("cache_read", 0)
    print(f"in={usage.get('input_tokens'):>6}  cached={cached_tokens:>6}  out={usage.get('output_tokens'):>4}  {question}")

# %% [markdown]
# If `cached` is greater than zero on the second call, the provider reused the
# prefix. Support varies:
#
# | Provider | How |
# |---|---|
# | **OpenAI** | Automatic for prompts over ~1024 tokens; 50% discount on cached input |
# | **Anthropic** | Explicit `cache_control` breakpoints; ~90% discount on reads, small write cost |
# | **Groq** | Automatic on supported models |
# | **Google** | Explicit context caching API |
#
# **The rule that makes it work: put static content first.** Prefix caching only
# helps if the beginning of the prompt is byte-identical between requests.

# %%
# Wrong: variable content early destroys the shared prefix.
bad = f"User {'{user_id}'} asked at {'{timestamp}'}.\n\n{{long_policy}}\n\nQuestion: {{q}}"

# Right: static bulk first, variable content last.
good = f"{{long_policy}}\n\nUser: {'{user_id}'}\nTime: {'{timestamp}'}\nQuestion: {{q}}"

print("BAD  ordering:", bad[:70], "...")
print("GOOD ordering:", good[:70], "...")
print("\nOnly the second shares a cacheable prefix across users.")

# %%
# Anthropic requires explicit markers:
#
# messages = [
#     SystemMessage(content=[{
#         "type": "text",
#         "text": LONG_SYSTEM_PROMPT,
#         "cache_control": {"type": "ephemeral"},   # <- cache everything up to here
#     }]),
#     HumanMessage("How many casual leave days?"),
# ]
print("(Anthropic uses explicit cache_control breakpoints - see the comment above.)")

# %% [markdown]
# ## 6. Measuring the benefit
#
# Decide with numbers, not vibes.

# %%
set_llm_cache(InMemoryCache())
metered = get_chat_model()

traffic = [
    "What are the API rate limits?",
    "How many sick days do I get?",
    "What are the API rate limits?",
    "What is the notice period for L4?",
    "How many sick days do I get?",
    "What are the API rate limits?",
    "What is the expense claim deadline?",
    "How many sick days do I get?",
]

start = time.perf_counter()
for question in traffic:
    metered.invoke(question)
cached_total = time.perf_counter() - start

set_llm_cache(None)
plain = get_chat_model(cache=False)
start = time.perf_counter()
for question in traffic:
    plain.invoke(question)
plain_total = time.perf_counter() - start

unique = len(set(traffic))
print(f"{len(traffic)} requests, {unique} unique ({(1 - unique / len(traffic)):.0%} duplicate)")
print(f"with cache   : {cached_total:5.2f}s")
print(f"without cache: {plain_total:5.2f}s")
print(f"saved        : {plain_total - cached_total:5.2f}s and {len(traffic) - unique} provider calls")

# %% [markdown]
# ## 7. Cache invalidation
#
# The hard part. A cached answer is a snapshot of a policy that may have changed.

# %%
set_llm_cache(None)


class VersionedCache:
    """Include a corpus version in the key so a policy update invalidates everything."""

    def __init__(self, version: str):
        self.version = version
        self.entries: dict[tuple[str, str], str] = {}

    def key(self, question: str) -> tuple[str, str]:
        return (self.version, question)

    def get(self, question: str) -> str | None:
        return self.entries.get(self.key(question))

    def set(self, question: str, answer: str) -> None:
        self.entries[self.key(question)] = answer

    def bump(self, new_version: str) -> None:
        print(f"corpus {self.version} -> {new_version}: {len(self.entries)} entries now unreachable")
        self.version = new_version


cache = VersionedCache("policy-2026-01")
cache.set("How many casual leave days?", "6 days per calendar year.")
print("hit:", cache.get("How many casual leave days?"))

cache.bump("policy-2026-04")
print("after policy update:", cache.get("How many casual leave days?"))

# %% [markdown]
# **Invalidation strategies, in order of preference:**
#
# 1. **Version the key** - include a corpus or prompt version; a bump invalidates
#    everything atomically. Simple and correct.
# 2. **TTL** - expire after an hour or a day. Bounds staleness without tracking
#    dependencies.
# 3. **Explicit deletion** - precise but requires knowing which entries depend on
#    which document. Usually not worth the complexity.

# %% [markdown]
# ## 8. What to cache where
#
# | Layer | Cache | Why |
# |---|---|---|
# | Embeddings | `CacheBackedEmbeddings` (notebook 11) | Deterministic; always cache |
# | Retrieval results | your own dict/Redis with TTL | Same query, same docs until the corpus changes |
# | Model responses | `SQLiteCache` / Redis | Exact repeats |
# | Near-duplicate questions | semantic cache, high threshold | FAQ traffic |
# | Long static prompts | provider prompt caching | Big, automatic saving |
# | Final rendered answers | your application cache | Cheapest hit of all |
#
# Start at the top of that list. Embedding caching is free correctness; semantic
# caching needs care.

# %% [markdown]
# ## Try it yourself
#
# 1. **Hit-rate simulation.** Generate 200 queries from a Zipf distribution over
#    20 unique questions and measure exact-match versus semantic hit rates.
# 2. **Find a false positive.** Lower the semantic threshold until a wrong answer
#    is served, and record the threshold at which it happened.
# 3. **Add a TTL** to `SemanticCache` (store a timestamp with each answer) and
#    prove that a stale entry is not served.
# 4. **Measure prompt caching.** With an OpenAI or Anthropic key, send the long
#    system prompt five times and plot `cache_read` tokens per call.

# %% [markdown]
# ## Recap
#
# | Concept | Takeaway |
# |---|---|
# | `set_llm_cache(InMemoryCache())` | One line; exact-match only, dies with the process |
# | `SQLiteCache` | Survives restarts; the sensible local default |
# | Cache key | Prompt + model + every parameter. A trailing space is a miss |
# | `get_chat_model(cache=False)` | Opt out per model instance |
# | Per-tenant caches | Never share a cache across tenants |
# | Semantic cache | Catches paraphrases; **dangerous with negation and near-synonyms** |
# | Threshold | Start at 0.95, lower only with evidence, log every hit |
# | Prompt caching | Provider-side; put static content **first** |
# | Invalidation | Version the key; TTL as a backstop |
#
# ## Next
#
# -> [24_structured_outputs.ipynb](24_structured_outputs.ipynb)
