# %% [markdown]
# # 13 - Retrievers: Standard and Advanced
#
# | | |
# |---|---|
# | **Level** | Intermediate |
# | **Time** | 50 minutes |
# | **Prerequisites** | `12_vector_stores` |
# | **Checklist ID** | `13_retrievers` |
#
# ## Why this matters
#
# Retrieval quality sets the ceiling on answer quality. If the right chunk is not
# in the context window, no prompt engineering and no bigger model will save you -
# the model will either refuse or hallucinate.
#
# A `Retriever` is a Runnable: `str` in, `list[Document]` out. That uniform shape
# means you can swap naive similarity search for a five-stage hybrid pipeline
# without touching the rest of your chain.

# %%
from pathlib import Path
import sys

_root = Path.cwd()
while not (_root / "shared").is_dir() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from shared.notebook_setup import setup, require  # noqa: E402

ctx = setup("13_retrievers")

# %%
from collections import Counter

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from shared.llm import get_chat_model, get_embeddings

model = get_chat_model()
embeddings = get_embeddings()


def build_corpus() -> list[Document]:
    docs: list[Document] = []
    for file_name, domain, sensitivity in [
        ("company_handbook.md", "hr", "internal"),
        ("product_faq.md", "product", "public"),
    ]:
        text = Path(ctx.data(file_name)).read_text(encoding="utf-8")
        sections = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "document"), ("##", "section")], strip_headers=False
        ).split_text(text)
        for doc in sections:
            doc.metadata.update({"source": file_name, "domain": domain, "sensitivity": sensitivity})
        docs.extend(sections)

    policy = Path(ctx.data("leave_policy.txt")).read_text(encoding="utf-8")
    for chunk in RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80).split_text(policy):
        docs.append(Document(chunk, metadata={"source": "leave_policy.txt", "domain": "hr", "sensitivity": "internal"}))

    return RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80).split_documents(docs)


corpus = build_corpus()
vector_store = FAISS.from_documents(corpus, embeddings)
print(f"{len(corpus)} chunks indexed:", dict(Counter(d.metadata["source"] for d in corpus)))


def preview(docs, label: str, width: int = 95) -> None:
    print(f"\n{label}  ({len(docs)} docs)")
    for doc in docs:
        source = doc.metadata.get("source", "?")
        section = doc.metadata.get("section", "-")
        print(f"   [{source:22} | {str(section)[:26]:26}] {' '.join(doc.page_content.split())[:width]}")

# %% [markdown]
# ## 1. The base case: `.as_retriever()`

# %%
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

docs = retriever.invoke("How many office days per month are required?")
preview(docs, "similarity k=3")

# %% [markdown]
# It is a Runnable, so `batch`, `stream` and `|` all work:

# %%
batched = retriever.batch(["expense limits", "notice period"])
print([len(group) for group in batched], "docs per query")

# %% [markdown]
# ## 2. Search types: similarity, MMR, threshold

# %%
question = "what are the rules about taking leave?"

preview(vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4}).invoke(question),
        "similarity")

preview(vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 15, "lambda_mult": 0.4},
).invoke(question), "mmr (diverse)")

# %%
# Threshold retrieval: return nothing rather than something irrelevant.
strict = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 4, "score_threshold": 0.5},
)

preview(strict.invoke("what is the leave carry forward limit?"), "on-topic question")
preview(strict.invoke("what is the company's pet insurance policy?"), "off-topic question")

# %% [markdown]
# An empty result is a **feature**. It lets your chain say "I don't have that"
# instead of feeding the model loosely-related text that invites a hallucination.

# %% [markdown]
# ## 3. Metadata filters at retrieval time

# %%
public_only = vector_store.as_retriever(
    search_kwargs={"k": 3, "filter": {"sensitivity": "public"}}
)
preview(public_only.invoke("what are the support SLAs?"), "public documents only")

# %% [markdown]
# > FAISS filters *after* fetching, so ask for a larger `fetch_k` when the filter
# > is selective. Chroma, Pinecone and pgvector filter inside the index, which is
# > both faster and more accurate - another reason to prefer them at scale.

# %% [markdown]
# ## 4. `MultiQueryRetriever`: fix the vocabulary mismatch
#
# The problem: a user's phrasing may not match the document's. "Can I get money
# back for my train ticket?" versus "Reimbursements are processed in the payroll
# cycle following approval."
#
# The fix: have an LLM generate several rephrasings, retrieve for each, and union
# the results.

# %%
import logging

from langchain_classic.retrievers.multi_query import MultiQueryRetriever

logging.getLogger("langchain_classic.retrievers.multi_query").setLevel(logging.INFO)
logging.basicConfig()

multi_query = MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
    llm=model,
)

vague = "can I get money back for my train ticket?"

preview(vector_store.as_retriever(search_kwargs={"k": 3}).invoke(vague), "plain similarity")
preview(multi_query.invoke(vague), "multi-query union")

# %% [markdown]
# The log lines show the generated variations. Costs: one extra LLM call plus N
# retrievals. Worth it when users phrase things unpredictably; wasteful for a
# controlled internal tool where queries are uniform.

# %% [markdown]
# ## 5. `ContextualCompressionRetriever`: shrink what you send
#
# Retrieved chunks contain relevant *and* irrelevant sentences. Compression
# filters the noise before it reaches the model - cutting cost and reducing the
# chance the model latches onto the wrong sentence.

# %%
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(model)
compressed = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
)

precise_question = "What is the hotel limit for domestic travel?"

raw_docs = vector_store.as_retriever(search_kwargs={"k": 3}).invoke(precise_question)
small_docs = compressed.invoke(precise_question)

print(f"raw       : {len(raw_docs)} docs, {sum(len(d.page_content) for d in raw_docs):,} chars")
print(f"compressed: {len(small_docs)} docs, {sum(len(d.page_content) for d in small_docs):,} chars")
preview(small_docs, "extracted sentences only", width=200)

# %% [markdown]
# ### Cheaper compression without an LLM
#
# `EmbeddingsFilter` drops chunks below a similarity threshold. No LLM call, so it
# is nearly free - a good first filter in a pipeline.

# %%
from langchain_classic.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
)
from langchain_text_splitters import CharacterTextSplitter

pipeline_compressor = DocumentCompressorPipeline(
    transformers=[
        CharacterTextSplitter(chunk_size=250, chunk_overlap=0, separator=". "),
        EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.3),
    ]
)

cheap = ContextualCompressionRetriever(
    base_compressor=pipeline_compressor,
    base_retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
)
preview(cheap.invoke(precise_question), "split then embedding-filter", width=140)

# %% [markdown]
# ## 6. `BM25Retriever`: keyword search still matters
#
# Embeddings are bad at exact tokens: error codes, SKUs, names, "SOC 2", "L4".
# BM25 is classic keyword ranking and nails precisely those.

# %%
from langchain_community.retrievers import BM25Retriever

bm25 = BM25Retriever.from_documents(corpus)
bm25.k = 3

exact_term = "SOC 2"
preview(vector_store.as_retriever(search_kwargs={"k": 3}).invoke(exact_term), "vector search for 'SOC 2'")
preview(bm25.invoke(exact_term), "BM25 for 'SOC 2'")

# %%
conceptual = "what happens if someone dies in my family"
preview(bm25.invoke(conceptual), "BM25 on a conceptual question (weak)")
preview(vector_store.as_retriever(search_kwargs={"k": 2}).invoke(conceptual), "vector on the same (strong)")

# %% [markdown]
# Each method fails where the other succeeds. That observation is the entire
# argument for combining them.

# %% [markdown]
# ## 7. `EnsembleRetriever`: hybrid search
#
# Runs several retrievers and fuses their rankings with **Reciprocal Rank Fusion** -
# a document ranked highly by either method rises, and documents ranked well by
# both rise most.

# %%
from langchain_classic.retrievers import EnsembleRetriever

hybrid = EnsembleRetriever(
    retrievers=[bm25, vector_store.as_retriever(search_kwargs={"k": 3})],
    weights=[0.4, 0.6],   # tune on your own data
)

for query in ["SOC 2", "what happens if someone dies in my family", "L4 notice period"]:
    preview(hybrid.invoke(query), f"hybrid: {query!r}", width=90)

# %% [markdown]
# RRF needs no score normalisation, which is why it is the standard fusion method:
# it only uses rank positions, so incompatible scoring scales stop being a problem.
#
# **Hybrid search is the single highest-value retrieval upgrade for most internal
# corpora.** Start here before reaching for anything exotic.

# %% [markdown]
# ## 8. `ParentDocumentRetriever`: search small, return large
#
# The chunking tension from notebook 10 has a clever resolution: index *small*
# chunks for precise matching, but return their *larger parent* so the model gets
# full context.

# %%
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.stores import InMemoryStore
from langchain_chroma import Chroma

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=0)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=30)

child_store = Chroma(
    collection_name="parent_child_demo",
    embedding_function=embeddings,
)

parent_retriever = ParentDocumentRetriever(
    vectorstore=child_store,
    docstore=InMemoryStore(),
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

source_docs = [
    Document(
        Path(ctx.data("leave_policy.txt")).read_text(encoding="utf-8"),
        metadata={"source": "leave_policy.txt"},
    ),
    Document(
        Path(ctx.data("company_handbook.md")).read_text(encoding="utf-8"),
        metadata={"source": "company_handbook.md"},
    ),
]
parent_retriever.add_documents(source_docs)

matched_children = child_store.similarity_search("medical certificate", k=2)
returned_parents = parent_retriever.invoke("medical certificate")

print(f"matched child chunks : {len(matched_children)}, avg {sum(len(d.page_content) for d in matched_children) // len(matched_children)} chars")
print(f"returned parent docs : {len(returned_parents)}, avg {sum(len(d.page_content) for d in returned_parents) // len(returned_parents)} chars")
print("\nparent text starts:\n", " ".join(returned_parents[0].page_content.split())[:300])

# %% [markdown]
# ## 9. Writing a custom retriever
#
# Sometimes the right retrieval is business logic, not similarity. Subclass
# `BaseRetriever` and you get `invoke`, `batch`, tracing and composability free.

# %%
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever


class ClearanceRetriever(BaseRetriever):
    """Retrieve only what the requesting user is cleared to see."""

    base_store: Any
    clearance: str = "employee"
    k: int = 3

    _levels = {
        "public": {"public"},
        "employee": {"public", "internal"},
        "manager": {"public", "internal", "confidential"},
        "security": {"public", "internal", "confidential", "restricted"},
    }

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        allowed = self._levels[self.clearance]
        candidates = self.base_store.similarity_search(query, k=self.k * 4)
        permitted = [d for d in candidates if d.metadata.get("sensitivity", "internal") in allowed]
        return permitted[: self.k]


for clearance in ["public", "employee"]:
    found = ClearanceRetriever(base_store=vector_store, clearance=clearance, k=3).invoke(
        "how much leave do I get?"
    )
    print(f"{clearance:9} -> {sorted({d.metadata['source'] for d in found})}")

# %% [markdown]
# ## 10. Comparing retrievers on questions you care about
#
# Build a tiny evaluation set. This takes 15 minutes and beats months of intuition.

# %%
eval_set = [
    ("How many days of sick leave per year?", "leave_policy.txt"),
    ("What is the hotel limit for international travel?", "company_handbook.md"),
    ("What are the API rate limits?", "product_faq.md"),
    ("Do you have SOC 2?", "product_faq.md"),
    ("How long is the notice period for L4?", "company_handbook.md"),
    ("Can I carry forward unused annual leave?", "leave_policy.txt"),
]

candidates = {
    "similarity k=3": vector_store.as_retriever(search_kwargs={"k": 3}),
    "mmr k=3": vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 12}),
    "bm25 k=3": bm25,
    "hybrid": hybrid,
}

print(f"{'retriever':16} {'hit@3':>6}  details")
for name, candidate in candidates.items():
    hits = []
    for question, expected_source in eval_set:
        found_sources = {d.metadata.get("source") for d in candidate.invoke(question)}
        hits.append(expected_source in found_sources)
    print(f"{name:16} {sum(hits)}/{len(hits):<4} {''.join('Y' if h else '.' for h in hits)}")

# %% [markdown]
# `hit@3` asks: did the correct source document appear in the top 3? Run this
# against your own questions before and after every retrieval change. If a change
# does not move this number, it is not an improvement.

# %% [markdown]
# ## 11. Decision guide
#
# | Symptom | Try |
# |---|---|
# | Results are on-topic but repetitive | MMR |
# | Users phrase things unpredictably | `MultiQueryRetriever` |
# | Exact terms/codes/names are missed | BM25, then `EnsembleRetriever` |
# | Context window is full of noise | `ContextualCompressionRetriever` |
# | Chunks are too small to answer, too big to match | `ParentDocumentRetriever` |
# | Off-topic questions return junk | `similarity_score_threshold` |
# | Wrong documents for this user | metadata filter / custom retriever |
# | Still wrong after all of the above | reranking (notebook 15) |
#
# **Order of adoption:** similarity -> hybrid -> reranking -> query transformation.
# Each step adds latency and cost, so earn each one with a measured improvement.

# %% [markdown]
# ## Try it yourself
#
# 1. **Tune the ensemble.** Sweep `weights` from `[0.2, 0.8]` to `[0.8, 0.2]` and
#    find the best `hit@3` on the eval set above.
# 2. **Extend the eval set** to 15 questions, including three the corpus cannot
#    answer. Add a "correctly returned nothing" metric for the threshold retriever.
# 3. **Compression cost.** Measure tokens sent to the model with and without
#    `LLMChainExtractor` across all eval questions, and decide whether the saving
#    covers the extra LLM call.
# 4. **Custom retriever.** Write one that boosts recently-updated documents by
#    sorting on an `updated_at` metadata field after retrieval.

# %% [markdown]
# ## Recap
#
# | Retriever | What it fixes |
# |---|---|
# | `.as_retriever()` | Baseline similarity; `k`, `filter`, `search_type` |
# | MMR | Redundant near-duplicate results |
# | `similarity_score_threshold` | Returns nothing rather than junk |
# | `MultiQueryRetriever` | Vocabulary mismatch between user and documents |
# | `ContextualCompressionRetriever` | Noisy context, wasted tokens |
# | `EmbeddingsFilter` | Same, without an LLM call |
# | `BM25Retriever` | Exact terms, codes, names |
# | `EnsembleRetriever` | Combines both with RRF - the best default upgrade |
# | `ParentDocumentRetriever` | Small-chunk precision with large-chunk context |
# | `BaseRetriever` subclass | Business rules such as access control |
# | `hit@k` | The metric that tells you whether any of it helped |
#
# ## Next
#
# -> [14_retrieval_chains.ipynb](14_retrieval_chains.ipynb)
