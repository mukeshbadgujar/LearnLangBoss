# %% [markdown]
# # 11 - Embeddings and Embedding Caches
#
# | | |
# |---|---|
# | **Level** | Intermediate |
# | **Time** | 40 minutes |
# | **Prerequisites** | `10_text_splitters` |
# | **Checklist ID** | `11_embeddings_and_caching` |
#
# ## Why this matters
#
# An embedding turns text into a list of numbers such that **similar meaning gives
# similar numbers**. That single property is what makes semantic search possible:
# a user asking "can I take time off for a funeral?" finds the *bereavement leave*
# section even though neither phrase shares a word with the other.
#
# The cost angle matters too. Re-embedding an unchanged 10,000-document corpus on
# every deploy is pure waste - `CacheBackedEmbeddings` removes it.

# %%
from pathlib import Path
import sys

_root = Path.cwd()
while not (_root / "shared").is_dir() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from shared.notebook_setup import setup, require  # noqa: E402

ctx = setup("11_embeddings_and_caching")

# %%
from shared.llm import get_embeddings

embeddings = get_embeddings()
print("Embedding model:", type(embeddings).__name__)

# %% [markdown]
# ## 1. What an embedding actually is

# %%
vector = embeddings.embed_query("How many days of annual leave do I get?")

print("type       :", type(vector).__name__)
print("dimensions :", len(vector))
print("first 8    :", [round(v, 4) for v in vector[:8]])
print("magnitude  :", round(sum(v * v for v in vector) ** 0.5, 4))

# %% [markdown]
# Dimensions are fixed per model (384 for MiniLM, 1536 for `text-embedding-3-small`,
# 3072 for `text-embedding-3-large`). **You cannot mix models in one index** -
# the numbers mean different things and the dimensions usually differ anyway.

# %% [markdown]
# ## 2. `embed_query` vs `embed_documents`
#
# Two methods, and the distinction is not cosmetic. Some models are trained
# asymmetrically: queries and documents get different prefixes internally because
# a short question and a long passage have different statistics.

# %%
doc_vectors = embeddings.embed_documents([
    "Annual leave accrues at 2 days per completed month of service.",
    "Sick leave is 12 days per calendar year and does not carry forward.",
    "Paternity leave is 15 working days, usable within 6 months of birth.",
])

print(f"embed_documents -> {len(doc_vectors)} vectors of {len(doc_vectors[0])} dims")
print(f"embed_query     -> 1 vector of {len(vector)} dims")

# %% [markdown]
# ## 3. Similarity: seeing meaning become geometry

# %%
import math


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


sentences = [
    "How many vacation days am I entitled to?",
    "What is my annual leave allowance?",
    "Can I work from home three days a week?",
    "The expense claim deadline is 30 days.",
    "Our API rate limit on Growth is 600 requests per minute.",
]
vectors = embeddings.embed_documents(sentences)

base = vectors[0]
print(f"Reference: {sentences[0]}\n")
for sentence, vec in zip(sentences[1:], vectors[1:]):
    print(f"  {cosine_similarity(base, vec):+.4f}  {sentence}")

# %% [markdown]
# Sentence 2 scores highest despite sharing almost no vocabulary with the
# reference - "vacation days" and "annual leave allowance" are near-synonyms in
# embedding space. That is the entire value proposition over keyword search.
#
# **Reading cosine scores** (for a typical model):
#
# | Range | Interpretation |
# |---|---|
# | 0.85 - 1.00 | Paraphrase or near-duplicate |
# | 0.65 - 0.85 | Same topic, related |
# | 0.40 - 0.65 | Loosely related |
# | < 0.40 | Unrelated |
#
# These bands are model-specific. Calibrate on your own data before hard-coding a
# threshold.

# %% [markdown]
# ## 4. Semantic search over the real policy
#
# Let us build the smallest possible retrieval system by hand, so the vector store
# in notebook 12 holds no mystery.

# %%
from langchain_text_splitters import RecursiveCharacterTextSplitter

policy_text = Path(ctx.data("leave_policy.txt")).read_text(encoding="utf-8")
chunks = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60).split_text(policy_text)
chunk_vectors = embeddings.embed_documents(chunks)

print(f"indexed {len(chunks)} chunks\n")


def search(question: str, k: int = 2) -> None:
    query_vector = embeddings.embed_query(question)
    scored = sorted(
        ((cosine_similarity(query_vector, v), c) for v, c in zip(chunk_vectors, chunks)),
        reverse=True,
        key=lambda pair: pair[0],
    )
    print(f"Q: {question}")
    for score, chunk in scored[:k]:
        print(f"   {score:.4f}  {' '.join(chunk.split())[:120]}...")
    print()


search("Can I take time off when a family member passes away?")
search("What happens to leave I do not use by December?")
search("Do I need a doctor's note?")

# %% [markdown]
# Notice the first query: the word "bereavement" never appears in the question,
# yet the bereavement section is retrieved. A keyword search would have returned
# nothing.
#
# That whole `search` function is what a vector store does - plus indexing
# structures that make it fast at a million documents instead of twenty.

# %% [markdown]
# ## 5. Where embeddings fail
#
# Knowing the failure modes is what separates a working RAG system from a demo.

# %%
tricky_pairs = [
    ("Employees may work remotely.", "Employees may not work remotely.", "negation"),
    ("Notice period is 60 days.", "Notice period is 90 days.", "numbers differ"),
    ("Send the invoice to finance.", "Send the invoice to legal.", "one word changes meaning"),
    ("The API limit is 600 rpm.", "The API limit is 60 rpm.", "order of magnitude"),
]

for left, right, label in tricky_pairs:
    score = cosine_similarity(*embeddings.embed_documents([left, right]))
    print(f"  {score:.4f}  [{label}]  {left!r} vs {right!r}")

# %% [markdown]
# All of these score **very high** even though they mean opposite or materially
# different things. Embeddings capture topic, not logic.
#
# Practical consequences:
#
# 1. Retrieval can hand the model a chunk that says the opposite of what is needed.
# 2. Numbers and negations must be verified by the model reading the chunk - never
#    trust similarity alone for a factual claim.
# 3. This is why **hybrid search** (keyword + vector, notebook 15) and
#    **reranking** exist.

# %% [markdown]
# ## 6. Choosing a model
#
# | Model | Dims | Cost | Notes |
# |---|---|---|---|
# | `nomic-embed-text-v1_5` (Groq) | 768 | free tier | **This course's default.** No install, reuses your chat key. Rate limited. |
# | `all-MiniLM-L6-v2` (HF, local) | 384 | free | Offline, no limits. Needs torch (~2 GB). Best for bulk indexing. |
# | `bge-small-en-v1.5` (HF, local) | 384 | free | Often better than MiniLM on retrieval benchmarks |
# | `text-embedding-3-small` (OpenAI) | 1536 | low | Strong general baseline |
# | `text-embedding-3-large` (OpenAI) | 3072 | ~6x small | Best quality; supports dimension reduction |
# | `embed-english-v3.0` (Cohere) | 1024 | low | Has a dedicated query/document mode |
#
# Embeddings resolve independently of the chat provider - they are a different
# model type with different pricing - which is why `EMBEDDING_PROVIDER` is its
# own setting in `.env`. It happens that Groq serves both.
#
# **A warning about bulk indexing.** Groq's free tier is rate limited, and
# `embed_documents` on a few hundred chunks can hit it. If you see 429s while
# building an index, set `EMBEDDING_PROVIDER=huggingface` (after installing
# `langchain-huggingface[full]`) - local embedding has no quota.
#
# **Selection checklist**
#
# 1. Does it run where you need it (offline? on-prem? CPU only?)
# 2. Cost at your corpus size and query volume
# 3. Dimensions - larger costs more to store and search, and gains flatten
# 4. Domain - a legal/medical corpus may justify a specialised model
# 5. **Migration cost** - changing models means re-embedding everything

# %%
if require("OPENAI_API_KEY", feature="comparing a hosted embedding model"):
    from shared.llm import get_embeddings as _ge

    openai_embeddings = _ge(provider="openai")
    a, b = openai_embeddings.embed_documents([sentences[0], sentences[1]])
    print("OpenAI dims      :", len(a))
    print("OpenAI similarity:", round(cosine_similarity(a, b), 4))
    print("Local  similarity:", round(cosine_similarity(vectors[0], vectors[1]), 4))
    print("\nScores are not comparable across models - only rankings within one model are.")

# %% [markdown]
# ## 7. `CacheBackedEmbeddings`: stop paying twice
#
# Embedding is deterministic: the same text always produces the same vector. So
# cache it, keyed by a hash of the text.

# %%
import time

from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_core.stores import InMemoryByteStore

store = InMemoryByteStore()
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=embeddings,
    document_embedding_cache=store,
    namespace="minilm-v1",   # include the model name so a model swap misses the cache
)

start = time.perf_counter()
cached_embeddings.embed_documents(chunks)
cold = time.perf_counter() - start

start = time.perf_counter()
cached_embeddings.embed_documents(chunks)
warm = time.perf_counter() - start

print(f"cold: {cold:.3f}s")
print(f"warm: {warm:.3f}s   ({cold / max(warm, 0.0001):.0f}x faster)")
print(f"cache entries: {len(list(store.yield_keys()))}")

# %% [markdown]
# ### The `namespace` argument is not optional
#
# Without it, switching embedding models silently returns vectors from the *old*
# model. Your retrieval quality collapses and nothing errors. Always namespace by
# model name and version.

# %%
print("cache keys are hashes of the text:")
for key in list(store.yield_keys())[:3]:
    print("  ", key)

# %% [markdown]
# ### Persist the cache across restarts
#
# `InMemoryByteStore` dies with the process. For a real pipeline use a file store
# so a re-run of your ingestion job costs nothing for unchanged documents.

# %%
from langchain_classic.storage import LocalFileStore

cache_dir = ctx.artifact("embedding_cache")
file_store = LocalFileStore(str(cache_dir))

persistent_embeddings = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=embeddings,
    document_embedding_cache=file_store,
    namespace="minilm-v1",
)

start = time.perf_counter()
persistent_embeddings.embed_documents(chunks[:5])
first_run = time.perf_counter() - start

start = time.perf_counter()
persistent_embeddings.embed_documents(chunks[:5])
second_run = time.perf_counter() - start

print(f"first run : {first_run:.3f}s")
print(f"second run: {second_run:.3f}s")
print(f"cache dir : {cache_dir}")
print(f"files     : {len(list(cache_dir.glob('*')))}  (survives a kernel restart)")

# %% [markdown]
# ### Query caching is opt-in, and usually a mistake to enable blindly

# %%
query_cache = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=embeddings,
    document_embedding_cache=InMemoryByteStore(),
    namespace="minilm-v1",
    query_embedding_cache=True,   # off by default
)

start = time.perf_counter()
query_cache.embed_query("How many sick days do I get?")
print(f"first  : {time.perf_counter() - start:.4f}s")

start = time.perf_counter()
query_cache.embed_query("How many sick days do I get?")
print(f"cached : {time.perf_counter() - start:.4f}s")

# %% [markdown]
# Only enable query caching when queries genuinely repeat (an FAQ bot, a
# dashboard). For open-ended chat, the hit rate is near zero and you just grow
# memory forever.

# %% [markdown]
# ## 8. Cost model for a real corpus

# %%
import tiktoken

encoder = tiktoken.get_encoding("cl100k_base")
tokens_per_chunk = sum(len(encoder.encode(c)) for c in chunks) / len(chunks)

for corpus_docs in (1_000, 100_000, 1_000_000):
    chunks_total = corpus_docs * 8          # assume ~8 chunks per document
    tokens_total = chunks_total * tokens_per_chunk
    openai_small = tokens_total / 1_000_000 * 0.02   # USD per 1M tokens, indicative
    print(f"{corpus_docs:>9,} docs -> {chunks_total:>10,} chunks, "
          f"{tokens_total / 1e6:>7.1f}M tokens, ~${openai_small:>7.2f} hosted, $0.00 local")

print(f"\n(measured {tokens_per_chunk:.0f} tokens per chunk on this corpus)")
print("Re-embedding without a cache means paying this again on every deploy.")

# %% [markdown]
# ## Try it yourself
#
# 1. **Calibrate your thresholds.** Embed 10 pairs from the handbook that you
#    judge related and 10 unrelated, and find the cosine value that separates them
#    for *your* model. That number is your retrieval floor.
# 2. **Break the namespace.** Cache with `namespace="v1"`, then create a second
#    `CacheBackedEmbeddings` with a *different* underlying model but the same
#    namespace. Confirm you get stale vectors back and no warning.
# 3. **Multilingual check.** Embed "How many leave days?" in English and Hindi and
#    measure similarity. Decide whether the default model is usable for a
#    multilingual workforce.
# 4. **Incremental ingest.** Using `LocalFileStore`, embed the corpus, add one new
#    chunk, re-run, and verify only the new chunk was computed.

# %% [markdown]
# ## Recap
#
# | Concept | Takeaway |
# |---|---|
# | Embedding | Text -> fixed-length vector; similar meaning, similar vector |
# | `embed_query` vs `embed_documents` | Asymmetric in some models; use the right one |
# | Cosine similarity | The standard distance; calibrate thresholds per model |
# | Failure modes | Negation, numbers and antonyms score high - embeddings capture topic, not logic |
# | Model choice | Dimensions drive storage and search cost; changing model = full re-index |
# | `CacheBackedEmbeddings` | Deterministic output means caching is free accuracy-wise |
# | `namespace` | Must include the model name, or a model swap returns stale vectors silently |
# | `LocalFileStore` | Cache that survives restarts; makes incremental ingestion cheap |
#
# ## Next
#
# -> [12_vector_stores.ipynb](12_vector_stores.ipynb)
