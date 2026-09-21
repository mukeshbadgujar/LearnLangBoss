# %% [markdown]
# # 12 - Vector Stores
#
# | | |
# |---|---|
# | **Level** | Intermediate |
# | **Time** | 45 minutes |
# | **Prerequisites** | `11_embeddings_and_caching` |
# | **Checklist ID** | `12_vector_stores` |
#
# ## Why this matters
#
# In notebook 11 you searched 20 chunks by comparing against every single one.
# That is O(n) per query. At 20 chunks it is instant; at 5 million it is
# unusable.
#
# A vector store is a database built for this: it indexes vectors so search is
# sub-linear, it stores metadata alongside so you can filter, and it persists so
# you do not re-embed on every restart. This notebook covers the two local stores
# you will use daily and the managed ones you will meet in production.

# %%
from pathlib import Path
import sys

_root = Path.cwd()
while not (_root / "shared").is_dir() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from shared.notebook_setup import setup, require  # noqa: E402

ctx = setup("12_vector_stores")

# %% [markdown]
# ## 1. Build the corpus once, reuse it everywhere
#
# Loaders (notebook 9) plus splitters (notebook 10), with the two-pass pattern.

# %%
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from shared.llm import get_embeddings

embeddings = get_embeddings()


def build_corpus() -> list[Document]:
    docs: list[Document] = []

    # Markdown handbook: split by heading, then by size, keeping section metadata.
    handbook = Path(ctx.data("company_handbook.md")).read_text(encoding="utf-8")
    sections = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "document"), ("##", "section")], strip_headers=False
    ).split_text(handbook)
    for doc in sections:
        doc.metadata.update({"source": "company_handbook.md", "domain": "hr", "sensitivity": "internal"})
    docs.extend(sections)

    # Plain-text leave policy.
    policy = Path(ctx.data("leave_policy.txt")).read_text(encoding="utf-8")
    for chunk in RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80).split_text(policy):
        docs.append(Document(chunk, metadata={"source": "leave_policy.txt", "domain": "hr", "sensitivity": "internal"}))

    # Product FAQ.
    faq = Path(ctx.data("product_faq.md")).read_text(encoding="utf-8")
    faq_sections = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "document"), ("##", "section")], strip_headers=False
    ).split_text(faq)
    for doc in faq_sections:
        doc.metadata.update({"source": "product_faq.md", "domain": "product", "sensitivity": "public"})
    docs.extend(faq_sections)

    return RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80).split_documents(docs)


corpus = build_corpus()
print(f"{len(corpus)} chunks")
from collections import Counter

print("by source     :", dict(Counter(d.metadata["source"] for d in corpus)))
print("by sensitivity:", dict(Counter(d.metadata["sensitivity"] for d in corpus)))

# %% [markdown]
# ## 2. `InMemoryVectorStore`: the one with no dependencies
#
# Perfect for tests and notebooks. Nothing persists.

# %%
from langchain_core.vectorstores import InMemoryVectorStore

memory_store = InMemoryVectorStore.from_documents(corpus, embeddings)

hits = memory_store.similarity_search("How long is the notice period for senior engineers?", k=2)
for doc in hits:
    print(f"[{doc.metadata['source']}] {' '.join(doc.page_content.split())[:150]}...\n")

# %% [markdown]
# ## 3. FAISS: fast, local, file-backed
#
# FAISS is Meta's similarity-search library. It is a *library*, not a server - your
# process owns the index and you save/load it as files. Excellent default for
# single-node apps and batch pipelines.

# %%
from langchain_community.vectorstores import FAISS

faiss_store = FAISS.from_documents(corpus, embeddings)
print("vectors indexed:", faiss_store.index.ntotal)
print("dimensions     :", faiss_store.index.d)

results = faiss_store.similarity_search("what is the expense claim deadline?", k=3)
for doc in results:
    print(f"\n[{doc.metadata['source']} / {doc.metadata.get('section', '-')}]")
    print(" ", " ".join(doc.page_content.split())[:180])

# %% [markdown]
# ### Scores, and the distance/similarity trap

# %%
scored = faiss_store.similarity_search_with_score("what is the expense claim deadline?", k=3)
for doc, score in scored:
    print(f"  {score:.4f}  {doc.metadata['source']:24} {' '.join(doc.page_content.split())[:80]}")

print("\nFAISS returns L2 DISTANCE by default: LOWER is better.")
print("Chroma returns a distance too. Pinecone returns SIMILARITY: HIGHER is better.")
print("Never hard-code a threshold without checking which one your store uses.")

# %%
# Normalised relevance scores (0-1, higher better) give you a portable threshold.
normalised = faiss_store.similarity_search_with_relevance_scores(
    "what is the expense claim deadline?", k=3
)
for doc, score in normalised:
    print(f"  relevance {score:.4f}  {doc.metadata['source']}")

# %% [markdown]
# ### Persisting to disk

# %%
index_path = ctx.artifact("faiss_index")
faiss_store.save_local(str(index_path))
print("saved:", sorted(p.name for p in index_path.iterdir()))

reloaded = FAISS.load_local(
    str(index_path),
    embeddings,
    allow_dangerous_deserialization=True,   # required: the docstore is a pickle
)
print("reloaded vectors:", reloaded.index.ntotal)
print("search still works:", reloaded.similarity_search("notice period", k=1)[0].metadata["source"])

# %% [markdown]
# > **`allow_dangerous_deserialization=True` is a real warning.** FAISS stores
# > documents in a pickle file, and unpickling untrusted data executes code. Only
# > load indexes you built yourself or received over a trusted channel.

# %% [markdown]
# ## 4. Chroma: local store with a proper database underneath
#
# Chroma persists to SQLite/DuckDB, supports real metadata filtering, and can run
# as a server. It is the better choice when you need filtering and incremental
# updates rather than a static index.

# %%
chroma_available = True
try:
    from langchain_chroma import Chroma
except ImportError:
    chroma_available = False
    print("[skipped] pip install langchain-chroma to run this section")

# %%
if chroma_available:
    chroma_dir = ctx.artifact("chroma_db")
    chroma_store = Chroma.from_documents(
        documents=corpus,
        embedding=embeddings,
        collection_name="northwind",
        persist_directory=str(chroma_dir),
    )
    print("collection count:", chroma_store._collection.count())

    for doc in chroma_store.similarity_search("how often does data refresh?", k=2):
        print(f"\n[{doc.metadata['source']}] {' '.join(doc.page_content.split())[:150]}")

# %% [markdown]
# ## 5. Metadata filtering: the feature that makes this production-grade
#
# This is where a vector store beats the hand-rolled search from notebook 11.
# "Answer only from public documents" becomes a filter, not a prompt instruction
# you hope the model obeys.

# %%
if chroma_available:
    question = "what are the support response times?"

    print("UNFILTERED:")
    for doc in chroma_store.similarity_search(question, k=3):
        print(f"   [{doc.metadata['sensitivity']:10}] {doc.metadata['source']}")

    print("\nFILTERED to public only:")
    for doc in chroma_store.similarity_search(question, k=3, filter={"sensitivity": "public"}):
        print(f"   [{doc.metadata['sensitivity']:10}] {doc.metadata['source']}")

# %%
if chroma_available:
    # Chroma supports operators: $eq $ne $in $nin $gt $gte $lt $lte $and $or
    combined = chroma_store.similarity_search(
        "leave and time off rules",
        k=3,
        filter={"$and": [{"domain": {"$eq": "hr"}}, {"source": {"$eq": "leave_policy.txt"}}]},
    )
    for doc in combined:
        print(f"[{doc.metadata['source']}] {' '.join(doc.page_content.split())[:110]}")

# %% [markdown]
# ### Filtering is your access-control layer
#
# A user's permissions become a filter applied server-side. Never rely on a system
# prompt saying "do not reveal restricted documents" - if a restricted chunk
# reaches the context window, it has already leaked.

# %%
if chroma_available:
    def answer_scope(user_clearance: str) -> dict:
        """Map a clearance level to the sensitivities that user may retrieve."""
        allowed = {
            "public": ["public"],
            "employee": ["public", "internal"],
            "manager": ["public", "internal", "confidential"],
            "security": ["public", "internal", "confidential", "restricted"],
        }[user_clearance]
        return {"sensitivity": {"$in": allowed}}

    for clearance in ["public", "employee"]:
        found = chroma_store.similarity_search(
            "how many leave days do employees get?", k=3, filter=answer_scope(clearance)
        )
        sources = {d.metadata["source"] for d in found}
        print(f"{clearance:9} sees: {sorted(sources)}")

# %% [markdown]
# ## 6. Search strategies: similarity vs MMR
#
# Plain similarity can return three chunks that all say the same thing. **Maximal
# Marginal Relevance** trades a little relevance for diversity.

# %%
question = "what are the rules about leave?"

print("SIMILARITY (may repeat):")
for doc in faiss_store.similarity_search(question, k=4):
    print("  ", " ".join(doc.page_content.split())[:95])

print("\nMMR (diverse):")
for doc in faiss_store.max_marginal_relevance_search(question, k=4, fetch_k=12, lambda_mult=0.5):
    print("  ", " ".join(doc.page_content.split())[:95])

# %% [markdown]
# `lambda_mult` controls the trade-off: `1.0` = pure relevance, `0.0` = maximum
# diversity. `0.5` is a sensible default. `fetch_k` is how many candidates are
# considered before diversification.

# %% [markdown]
# ## 7. Updating an index
#
# Documents change. You need add, update and delete - not just create.

# %%
if chroma_available:
    new_doc = Document(
        page_content=(
            "Sabbatical policy (new, effective March 2026): after 5 years of continuous "
            "service employees may take up to 8 weeks of unpaid sabbatical with 90 days notice."
        ),
        metadata={"source": "leave_policy.txt", "domain": "hr", "sensitivity": "internal"},
    )

    before = chroma_store._collection.count()
    added_ids = chroma_store.add_documents([new_doc], ids=["sabbatical-v1"])
    print(f"count {before} -> {chroma_store._collection.count()}  (ids={added_ids})")

    print("\nsearchable immediately:")
    print(" ", chroma_store.similarity_search("can I take a sabbatical?", k=1)[0].page_content[:130])

# %%
if chroma_available:
    # Update = add with the same id.
    chroma_store.add_documents(
        [Document(
            page_content=(
                "Sabbatical policy (revised): after 4 years of service employees may take up "
                "to 12 weeks of unpaid sabbatical with 60 days notice."
            ),
            metadata={"source": "leave_policy.txt", "domain": "hr", "sensitivity": "internal"},
        )],
        ids=["sabbatical-v1"],
    )
    print("after update:", chroma_store.similarity_search("sabbatical", k=1)[0].page_content[:130])

    chroma_store.delete(ids=["sabbatical-v1"])
    print("\nafter delete, count:", chroma_store._collection.count())

# %% [markdown]
# **Design your ids deliberately.** A stable id like `f"{source}:{section}:{n}"`
# lets a re-ingest overwrite rather than duplicate. Without stable ids, running
# your pipeline twice doubles the corpus and retrieval starts returning the same
# chunk twice.

# %% [markdown]
# ## 8. Managed stores: Pinecone, Weaviate, pgvector
#
# The interface is identical - that is the point of the `VectorStore` abstraction.
# What changes is operations.
#
# | Store | Runs | Filtering | Best for |
# |---|---|---|---|
# | `InMemoryVectorStore` | in process | basic | tests |
# | **FAISS** | in process, file-backed | manual | single node, static index, batch |
# | **Chroma** | embedded or server | rich | local dev, small-to-mid production |
# | **pgvector** | your Postgres | full SQL | you already run Postgres - often the best answer |
# | **Pinecone** | managed SaaS | rich | large scale, no ops team |
# | **Weaviate** | self-host or SaaS | rich + hybrid | built-in hybrid search |
# | **OpenSearch** | self-host or AWS | full | you already run it for logs |
#
# The code below is a template - it only runs if you have the key.

# %%
if require("PINECONE_API_KEY", feature="Pinecone managed vector store"):
    import os

    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone, ServerlessSpec

    index_name = os.environ.get("PINECONE_INDEX", "genai-mastery")
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    if index_name not in [i["name"] for i in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=len(embeddings.embed_query("dimension probe")),
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    pinecone_store = PineconeVectorStore.from_documents(corpus, embeddings, index_name=index_name)
    for doc in pinecone_store.similarity_search("expense claim deadline", k=2):
        print(f"[{doc.metadata['source']}] {doc.page_content[:110]}")

# %% [markdown]
# ```python
# # Weaviate sketch - same VectorStore interface
# from langchain_weaviate import WeaviateVectorStore
# import weaviate
#
# client = weaviate.connect_to_local()          # or connect_to_weaviate_cloud(...)
# store = WeaviateVectorStore.from_documents(corpus, embeddings, client=client,
#                                            index_name="Northwind")
# ```
#
# Because everything speaks the same interface, migrating stores is a one-line
# change in your factory function - which is exactly how you should structure it.

# %%
def make_vector_store(kind: str, documents: list[Document]):
    """The shape your production code should have: one place that knows the store."""
    if kind == "memory":
        return InMemoryVectorStore.from_documents(documents, embeddings)
    if kind == "faiss":
        return FAISS.from_documents(documents, embeddings)
    if kind == "chroma" and chroma_available:
        return Chroma.from_documents(documents, embeddings, collection_name="swap-demo")
    raise ValueError(f"unsupported store: {kind}")


for kind in ["memory", "faiss"] + (["chroma"] if chroma_available else []):
    store = make_vector_store(kind, corpus)
    top = store.similarity_search("home office stipend", k=1)[0]
    print(f"{kind:8} -> {' '.join(top.page_content.split())[:90]}")

# %% [markdown]
# ## 9. Every vector store is a retriever
#
# `.as_retriever()` converts a store into a `Runnable` you can pipe into an LCEL
# chain. Notebook 13 explores the options; here is the bridge.

# %%
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from shared.llm import get_chat_model

model = get_chat_model()
retriever = faiss_store.as_retriever(search_kwargs={"k": 3})

rag_prompt = ChatPromptTemplate.from_template(
    "Answer using only the context. If the context lacks the answer, say "
    "'That is not covered in the documents I have.'\n\nContext:\n{context}\n\nQuestion: {question}"
)


def format_docs(docs) -> str:
    return "\n\n".join(f"[{d.metadata.get('source')}] {d.page_content}" for d in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | model
    | StrOutputParser()
)

print(rag_chain.invoke("How many office days per month are required under the hybrid policy?"))
print()
print(rag_chain.invoke("What is the company's policy on pet insurance?"))

# %% [markdown]
# The second answer is the important one: it declined instead of inventing a
# policy. That behaviour comes from the prompt plus grounded context - and it is
# what makes RAG trustworthy enough to deploy.

# %% [markdown]
# ## Try it yourself
#
# 1. **Stable ids.** Re-run `Chroma.from_documents` twice without ids and count the
#    collection. Then add deterministic ids and confirm the count stays flat.
# 2. **Filter beats prompting.** Ask a question whose best answer lives in a
#    `confidential` document, once with a `public`-only filter and once with a
#    prompt instruction not to use confidential material. Note which one actually
#    holds.
# 3. **MMR tuning.** Run the same query at `lambda_mult` 0.1, 0.5 and 0.9 and
#    describe the change in the result set.
# 4. **Benchmark.** Time 50 searches against `InMemoryVectorStore` and FAISS with
#    the corpus duplicated 100x, and note where the gap appears.

# %% [markdown]
# ## Recap
#
# | Concept | Takeaway |
# |---|---|
# | `InMemoryVectorStore` | Zero setup, no persistence - tests only |
# | FAISS | Library not server; `save_local`/`load_local`; pickle warning is real |
# | Chroma | Persistent, rich metadata filtering, add/update/delete |
# | Score direction | FAISS/Chroma return distance (lower better), Pinecone similarity (higher better) |
# | Metadata filters | Your access-control layer; enforce before retrieval, not in the prompt |
# | MMR | `lambda_mult` trades relevance for diversity |
# | Stable ids | Prevent duplicate chunks on re-ingest |
# | `.as_retriever()` | Turns any store into an LCEL Runnable |
#
# ## Next
#
# -> [13_retrievers.ipynb](13_retrievers.ipynb)
