# %% [markdown]
# # 00 - Environment, Providers and Keys
#
# | | |
# |---|---|
# | **Level** | Beginner |
# | **Time** | 30-45 minutes |
# | **Prerequisites** | Python 3.10+ installed |
# | **Checklist ID** | `00_environment_providers_and_keys` |
#
# ## Why this matters
#
# Nearly every "LangChain doesn't work" problem a new developer hits is really an
# environment problem: the wrong Python, a key in the wrong file, or a package
# version mismatch between `langchain` 0.x tutorials and the 1.x code you just
# installed. Fixing that once, here, saves you hours across the next 52 notebooks.
#
# By the end of this notebook you will have a working setup that can talk to
# **Groq, OpenRouter and OpenAI through the same two lines of code**, so no lesson
# ever locks you into one vendor.

# %% [markdown]
# ## 1. Install the dependencies
#
# Run this **once**, in a terminal, from the repository root:
#
# ```powershell
# # Create an isolated environment so this course never breaks your other projects
# python -m venv .venv
# .\.venv\Scripts\Activate.ps1        # Windows PowerShell
# # source .venv/bin/activate         # macOS / Linux
#
# python -m pip install --upgrade pip
# python -m pip install -r requirements.txt
# python -m ipykernel install --user --name genai-mastery --display-name "GenAI Mastery"
# ```
#
# Then pick the **GenAI Mastery** kernel in the top-right of this notebook.
#
# > **Windows tip:** always use `python -m pip ...` rather than bare `pip`. On many
# > Windows machines `python` and `pip` resolve to *different* installations, which
# > is how you end up installing a package and still getting `ModuleNotFoundError`.

# %% [markdown]
# ## 2. Understand the key hierarchy before you paste anything
#
# This course resolves providers in a fixed order. The first one with a key wins:
#
# | Order | Provider | Why it is here | Get a key |
# |---|---|---|---|
# | 1 | **Groq** | Very fast, generous free tier, great for learning loops | <https://console.groq.com/keys> |
# | 2 | **OpenRouter** | One key, hundreds of models, OpenAI-compatible API | <https://openrouter.ai/keys> |
# | 3 | **OpenAI** | The reference implementation most docs assume | <https://platform.openai.com/api-keys> |
# | 4 | Anthropic | Optional extra provider | <https://console.anthropic.com/> |
#
# **You only need one to begin.** Groq is recommended because the free tier is
# enough for this entire curriculum.
#
# ### Rules for handling keys
#
# 1. Keys live in `.env` at the repo root. Never in a notebook cell, never in git.
# 2. `.env` is already listed in `.gitignore`. Verify that before your first commit.
# 3. If a key ever appears in a screenshot, a chat message, or a pushed commit,
#    **rotate it immediately** - treat it as public from that moment.
# 4. Keys are per-environment. Your laptop key should never be the production key.

# %% [markdown]
# ## 3. Create your `.env`
#
# Copy the template and fill in what you have:
#
# ```powershell
# Copy-Item .env.example .env      # Windows PowerShell
# # cp .env.example .env           # macOS / Linux
# ```
#
# Minimum viable `.env`:
#
# ```ini
# GROQ_API_KEY=gsk_your_key_here
# GROQ_MODEL=llama-3.3-70b-versatile
# ```
#
# Now run the cell below. It is the **standard bootstrap cell** that opens every
# notebook in this course.

# %%
# Standard bootstrap - works no matter how deep the notebook folder is nested.
from pathlib import Path
import sys

_root = Path.cwd()
while not (_root / "shared").is_dir() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from shared.notebook_setup import setup, require  # noqa: E402

ctx = setup("00_environment_providers_and_keys")

# %% [markdown]
# If you see `Chat provider : NONE CONFIGURED`, stop and fix `.env` before going on.
# Everything after this point needs a working provider.
#
# ## 4. Your first model call
#
# Two lines. Notice there is no provider name anywhere in the lesson code - that is
# the whole point of `get_chat_model()`.

# %%
from shared.llm import get_chat_model

model = get_chat_model()
response = model.invoke("In one sentence, what problem does LangChain solve?")

print(type(response).__name__)
print(response.content)

# %% [markdown]
# `invoke()` returned an **`AIMessage`** object, not a string. That matters: the
# message also carries metadata you will use constantly for debugging and costing.

# %%
print("Model used :", response.response_metadata.get("model_name") or response.response_metadata.get("model"))
print("Finish      :", response.response_metadata.get("finish_reason"))
print("Token usage :", response.usage_metadata)

# %% [markdown]
# ## 5. Proving provider independence
#
# The same prompt, routed to every provider you have configured. If you only have
# one key, only one row prints - that is expected, not an error.

# %%
from shared.llm import available_providers, model_name_for

prompt = "Reply with exactly three words describing yourself."

for provider in available_providers():
    try:
        alt_model = get_chat_model(provider=provider)
        answer = alt_model.invoke(prompt)
        print(f"{provider:11} | {model_name_for(provider):38} | {answer.content.strip()[:60]}")
    except Exception as exc:
        print(f"{provider:11} | FAILED: {type(exc).__name__}: {exc}")

# %% [markdown]
# ### How to switch the default provider
#
# Edit one line in `.env` - no notebook changes:
#
# ```ini
# LLM_PROVIDER=openrouter
# ```
#
# Or override per call when you want to compare behaviour:
#
# ```python
# fast   = get_chat_model(provider="groq")
# strong = get_chat_model("openai/gpt-4o", provider="openrouter")
# ```

# %% [markdown]
# ## 6. `init_chat_model`: the idiomatic LangChain way
#
# `shared.llm` wraps LangChain's own `init_chat_model`, which accepts a
# `"provider:model"` string. You will see this all over the official docs, so know
# both forms.

# %%
from langchain.chat_models import init_chat_model
from shared.llm import active_provider, model_name_for

provider = active_provider()
spec = f"{provider}:{model_name_for(provider)}"

# OpenRouter isn't a native init_chat_model provider, so it routes through the
# OpenAI-compatible path inside shared.llm instead.
if provider == "openrouter":
    print("OpenRouter is used via ChatOpenAI(base_url=...) - see shared/llm.py")
    direct = get_chat_model()
else:
    print("init_chat_model spec:", spec)
    direct = init_chat_model(spec, temperature=0)

print(direct.invoke("Say OK").content)

# %% [markdown]
# ## 7. Embeddings are configured separately
#
# Embeddings are a different model type from chat, with different providers and
# different pricing, so `EMBEDDING_PROVIDER` resolves independently of
# `LLM_PROVIDER`. The default is `auto`, which picks the first that works,
# cheapest first:
#
# | Provider | Dims | Cost | Trade-off |
# |---|---|---|---|
# | **`groq`** | 768 | free tier | Reuses `GROQ_API_KEY`, nothing to install. Rate limited. |
# | `huggingface` | 384 | free | Local and offline, no limits - but needs torch (~2 GB) |
# | `openai` | 1536 | paid | Strongest quality baseline |
# | `fake` | 384 | free | Deterministic nonsense so notebooks still run |
#
# **If `get_embeddings()` reports `fake`, stop and fix it before Track 02.**
# Every retrieval result you see will be random, which makes the RAG lessons
# actively misleading. The cheapest fix is a free `GROQ_API_KEY`.

# %%
from shared.llm import embedding_provider, get_embeddings

embeddings = get_embeddings()
vector = embeddings.embed_query("How many annual leave days do I get?")

print("Resolved to     :", embedding_provider())
print("Embedding class :", type(embeddings).__name__)
print("Dimensions      :", len(vector))

# %% [markdown]
# ### Are these embeddings real?
#
# A dimension count proves nothing - the fake embedder returns 384 numbers too.
# The test that matters is whether *related* text scores higher than unrelated
# text. Run this before trusting anything in Track 02.

# %%
import math


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


question = embeddings.embed_query("How much annual leave do employees get?")
related, unrelated = embeddings.embed_documents([
    "Full-time employees receive 24 days of paid annual leave each year.",
    "The office coffee machine is on the third floor near the lifts.",
])

related_score, unrelated_score = cosine(question, related), cosine(question, unrelated)
print(f"related   {related_score:+.4f}")
print(f"unrelated {unrelated_score:+.4f}")
print("\nreal embeddings" if related_score > unrelated_score + 0.05
      else "\nFAKE OR BROKEN - related text did not score higher. Fix before Track 02.")
print("First 5 values  :", [round(v, 4) for v in vector[:5]])

# %% [markdown]
# ## 8. Turn on LangSmith tracing (optional but do it)
#
# LangSmith records every model call, prompt, tool invocation and latency figure.
# Debugging an agent without it is like debugging a web app with no network tab.
#
# 1. Sign up at <https://smith.langchain.com> and create an API key
# 2. Add to `.env`:
#    ```ini
#    LANGSMITH_TRACING=true
#    LANGSMITH_API_KEY=lsv2_...
#    LANGSMITH_PROJECT=genai-mastery
#    ```
# 3. Re-run the bootstrap cell
#
# Notebook 19 covers LangSmith properly. For now just confirm whether it is on.

# %%
if require("LANGSMITH_API_KEY", feature="LangSmith tracing"):
    traced = get_chat_model()
    traced.invoke("Trace check: reply with the single word 'traced'.")
    print("Sent. Open https://smith.langchain.com and look in project:",
          __import__("os").environ.get("LANGSMITH_PROJECT"))

# %% [markdown]
# ## 9. Generate the sample data
#
# The RAG, SQL and agent lessons all use one fictional company, **Northwind
# Analytics**. Two assets are generated locally rather than committed as binaries.

# %%
from shared.sample_data import ensure_all, list_files

created = ensure_all()
print("Generated:")
for name, path in created.items():
    print(f"  {name:8} {path.name}  ({path.stat().st_size:,} bytes)")

print("\nFull sample corpus:")
for path in list_files():
    print(f"  {path.relative_to(ctx.repo_root)}")

# %% [markdown]
# ## 10. Full environment health check
#
# Run this whenever something breaks. It tells you exactly which layer failed.

# %%
import importlib

checks = [
    ("langchain", "core framework"),
    ("langchain_core", "messages, runnables"),
    ("langgraph", "graph runtime"),
    ("langchain_groq", "Groq provider"),
    ("langchain_openai", "OpenAI + OpenRouter provider"),
    ("langchain_community", "loaders and community tools"),
    ("langchain_huggingface", "local embeddings"),
    ("langchain_chroma", "Chroma vector store"),
    ("faiss", "FAISS vector store"),
    ("pypdf", "PDF loading"),
    ("bs4", "HTML loading"),
    ("pandas", "dataframes"),
    ("fastapi", "serving lessons"),
]

print(f"Python {sys.version.split()[0]}  ({sys.executable})\n")
for module_name, purpose in checks:
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "unknown")
        print(f"  OK       {module_name:24} {version:12} {purpose}")
    except ImportError:
        print(f"  MISSING  {module_name:24} {'':12} {purpose}")

# %% [markdown]
# ## Try it yourself
#
# 1. **Force a provider.** Set `LLM_PROVIDER=openai` (or any provider you have),
#    restart the kernel, re-run the bootstrap cell and confirm the summary changes.
#    Then blank it out again.
# 2. **Break it on purpose.** Temporarily rename `GROQ_API_KEY` to `GROQ_KEY` in
#    `.env`, restart, and read the error message. Recognising
#    `NoProviderConfigured` early will save you later.
# 3. **Compare two models.** Ask the same question to a small and a large model
#    (for example `llama-3.1-8b-instant` vs `llama-3.3-70b-versatile` on Groq) and
#    note the difference in answer quality and latency using `%%time`.

# %% [markdown]
# ## Recap
#
# | Concept | Takeaway |
# |---|---|
# | `get_chat_model()` | Provider-agnostic chat model. Order: Groq -> OpenRouter -> OpenAI -> Anthropic |
# | `init_chat_model("provider:model")` | LangChain's native equivalent, used throughout official docs |
# | `.env` | The only place keys live. Never in notebooks, never in git |
# | `AIMessage` | `invoke()` returns an object with `.content`, `.response_metadata`, `.usage_metadata` |
# | Embeddings | Configured separately because Groq has no embedding endpoint |
# | `require("KEY")` | Lets optional cells skip cleanly instead of crashing |
#
# **When to deviate:** in production you would inject the model from config rather
# than resolve it from key presence, and keys come from a secret manager (AWS
# Secrets Manager, Vault) rather than `.env`. The pattern is the same; the source
# of the value changes.
#
# ## Next
#
# -> [01_models_messages.ipynb](../01-langchain-foundations/01_models_messages.ipynb)
