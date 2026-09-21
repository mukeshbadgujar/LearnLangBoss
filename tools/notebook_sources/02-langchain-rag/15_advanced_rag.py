# %% [markdown]
# # 15 - Advanced RAG: Query Transformation, Hybrid Search, Reranking
#
# | | |
# |---|---|
# | **Level** | Advanced |
# | **Time** | 60 minutes |
# | **Prerequisites** | `14_retrieval_chains` |
# | **Checklist ID** | `15_advanced_rag` |
#
# ## Why this matters
#
# Basic RAG gets you to maybe 70% accuracy on a real corpus. The remaining 30% is
# where users lose trust. This notebook covers the techniques that close the gap,
# organised by **which failure they fix**:
#
# | Failure | Technique |
# |---|---|
# | User's words differ from the document's words | Multi-query, HyDE, step-back |
# | Question bundles several sub-questions | Query decomposition |
# | Exact terms and codes are missed | Hybrid search (BM25 + vector) |
# | Right document retrieved but ranked 8th | Reranking |
# | Too much context, model picks the wrong sentence | Compression + reranking |
#
# Each technique adds latency and cost. Adopt them one at a time, and measure.

# %%
from pathlib import Path
import sys

_root = Path.cwd()
while not (_root / "shared").is_dir() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from shared.notebook_setup import setup, require  # noqa: E402

ctx = setup("15_advanced_rag")

# %%
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from shared.llm import get_chat_model, get_embeddings

model = get_chat_model()
embeddings = get_embeddings()


def build_corpus() -> list[Document]:
    docs: list[Document] = []
    for file_name, domain in [("company_handbook.md", "hr"), ("product_faq.md", "product")]:
        text = Path(ctx.data(file_name)).read_text(encoding="utf-8")
        sections = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "document"), ("##", "section")], strip_headers=False
        ).split_text(text)
        for doc in sections:
            doc.metadata.update({"source": file_name, "domain": domain})
        docs.extend(sections)

    policy = Path(ctx.data("leave_policy.txt")).read_text(encoding="utf-8")
    for chunk in RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=70).split_text(policy):
        docs.append(Document(chunk, metadata={"source": "leave_policy.txt", "domain": "hr"}))

    return RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=70).split_documents(docs)


corpus = build_corpus()
vector_store = FAISS.from_documents(corpus, embeddings)
base_retriever = vector_store.as_retriever(search_kwargs={"k": 4})
print(f"{len(corpus)} chunks indexed")


def show(docs, label: str, width: int = 90) -> None:
    print(f"\n{label}")
    for i, doc in enumerate(docs, 1):
        print(f"  {i}. [{doc.metadata.get('source', '?'):22}] {' '.join(doc.page_content.split())[:width]}")

# %% [markdown]
# ## 1. HyDE - Hypothetical Document Embeddings
#
# The insight: a **question** and an **answer passage** look different in
# embedding space. A question is short and interrogative; a policy paragraph is
# long and declarative. So instead of embedding the question, ask the model to
# *invent* the answer, and embed **that**.
#
# The invented answer may be factually wrong - it does not matter. It only has to
# be *shaped* like the real document.

# %%
hyde_prompt = ChatPromptTemplate.from_template(
    "Write a short passage from a company policy document that would answer this "
    "question. Write it in the confident declarative style of an official policy. "
    "Two or three sentences. Do not hedge, do not say you are unsure.\n\n"
    "Question: {question}"
)
hyde_generator = hyde_prompt | model | StrOutputParser()

question = "if I'm sick for a long time do I need to prove I'm fit before coming back?"

hypothetical = hyde_generator.invoke({"question": question})
print("QUESTION:\n ", question)
print("\nHYPOTHETICAL DOCUMENT (invented, possibly wrong):\n ", hypothetical.strip())

# %%
show(base_retriever.invoke(question), "retrieved by embedding the QUESTION")
show(vector_store.similarity_search(hypothetical, k=4), "retrieved by embedding the HYPOTHETICAL ANSWER")

# %% [markdown]
# ### HyDE as a reusable retriever

# %%
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever


class HyDERetriever(BaseRetriever):
    """Embed a generated hypothetical answer instead of the raw question."""

    vectorstore: Any
    generator: Any
    k: int = 4

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        hypothetical_doc = self.generator.invoke({"question": query})
        return self.vectorstore.similarity_search(hypothetical_doc, k=self.k)


hyde_retriever = HyDERetriever(vectorstore=vector_store, generator=hyde_generator, k=3)
show(hyde_retriever.invoke("what happens to my leave if I quit?"), "HyDE retriever")

# %% [markdown]
# **When HyDE helps:** short or colloquial questions against formal documents.
# **When it hurts:** the model has no idea what a plausible answer looks like
# (highly specialised domains), or the question contains exact identifiers that
# the hallucinated passage will not preserve.

# %% [markdown]
# ## 2. Step-back prompting
#
# Sometimes the question is too specific to match anything. Generalise it first,
# retrieve the broader principle, then answer.

# %%
step_back_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You rewrite specific questions into a more general question about the "
     "underlying policy area. Reply with the general question only."),
    ("human", "Can Rahul, an L3 engineer in Pune, take 11 days off starting next Monday?"),
    ("ai", "What are the rules for applying for annual leave?"),
    ("human", "{question}"),
])
step_back = step_back_prompt | model | StrOutputParser()

specific = "If I expense a 7,500 rupee hotel in Bengaluru on a Tuesday, who signs it off?"
general = step_back.invoke({"question": specific}).strip()

print("specific:", specific)
print("general :", general)

show(base_retriever.invoke(specific), "retrieved with the SPECIFIC question")
show(base_retriever.invoke(general), "retrieved with the STEP-BACK question")

# %%
# Best results come from using both sets of documents.
combined_docs = {d.page_content: d for d in base_retriever.invoke(specific) + base_retriever.invoke(general)}

step_back_answer = (
    ChatPromptTemplate.from_template(
        "Answer the specific question using the context.\n\n"
        "Context:\n{context}\n\nSpecific question: {question}"
    )
    | model
    | StrOutputParser()
).invoke({
    "context": "\n\n".join(d.page_content for d in combined_docs.values()),
    "question": specific,
})
print("\nANSWER:\n", step_back_answer.strip())

# %% [markdown]
# ## 3. Query decomposition
#
# Compound questions need compound retrieval. Split, retrieve per sub-question,
# then synthesise.

# %%
decompose_prompt = ChatPromptTemplate.from_template(
    "Break this question into 2-4 independent sub-questions that can each be "
    "answered separately. One per line, no numbering, no extra text.\n\n"
    "Question: {question}"
)
decompose = decompose_prompt | model | StrOutputParser()

compound = (
    "I'm an L4 joining a new team - how much leave do I get, what's my notice period, "
    "and how many office days are required?"
)

sub_questions = [q.strip("-* ").strip() for q in decompose.invoke({"question": compound}).splitlines() if q.strip()]
print("sub-questions:")
for sub in sub_questions:
    print("  -", sub)

# %%
sub_answers = []
for sub in sub_questions:
    docs = base_retriever.invoke(sub)
    answer = (
        ChatPromptTemplate.from_template(
            "Answer in one sentence from the context. If absent, say 'not found'.\n\n"
            "Context:\n{context}\n\nQuestion: {question}"
        )
        | model
        | StrOutputParser()
    ).invoke({"context": "\n\n".join(d.page_content for d in docs), "question": sub})
    sub_answers.append(f"Q: {sub}\nA: {answer.strip()}")
    print(f"\n{sub}\n  -> {answer.strip()[:160]}")

# %%
final = (
    ChatPromptTemplate.from_template(
        "Combine these sub-answers into one coherent reply to the original question.\n\n"
        "Original: {question}\n\nSub-answers:\n{subs}"
    )
    | model
    | StrOutputParser()
).invoke({"question": compound, "subs": "\n\n".join(sub_answers)})

print("\nFINAL ANSWER:\n", final.strip())

# %% [markdown]
# ## 4. Hybrid search with Reciprocal Rank Fusion
#
# Notebook 13 introduced `EnsembleRetriever`. Here is RRF implemented by hand, so
# the fusion is not a black box, plus a measured comparison.

# %%
bm25 = BM25Retriever.from_documents(corpus)
bm25.k = 6


def reciprocal_rank_fusion(ranked_lists: list[list[Document]], k: int = 60, top_n: int = 4) -> list[Document]:
    """Fuse ranked lists by 1/(k + rank). Uses positions only, so scales never clash."""
    scores: dict[str, float] = {}
    lookup: dict[str, Document] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            key = doc.page_content
            lookup[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)

    ordered = sorted(scores, key=scores.get, reverse=True)
    return [lookup[key] for key in ordered[:top_n]]


probe = "SOC 2 data residency for EU customers"

dense = vector_store.similarity_search(probe, k=6)
sparse = bm25.invoke(probe)
fused = reciprocal_rank_fusion([dense, sparse])

show(dense[:3], "dense only")
show(sparse[:3], "BM25 only")
show(fused, "RRF fused")

# %%
from langchain_classic.retrievers import EnsembleRetriever

hybrid = EnsembleRetriever(
    retrievers=[bm25, vector_store.as_retriever(search_kwargs={"k": 4})],
    weights=[0.4, 0.6],
)
show(hybrid.invoke(probe), "EnsembleRetriever (same idea, built in)")

# %% [markdown]
# ## 5. Reranking
#
# Retrieval optimises for **recall** at speed: get the right document into the top
# 20. Reranking optimises for **precision**: a slower, smarter model re-scores
# those 20 and puts the best at position 1.
#
# A cross-encoder reads the query and the document *together*, so it catches
# nuance a single embedding cannot - including negation and numbers, exactly the
# weaknesses from notebook 11.

# %%
if require("COHERE_API_KEY", feature="Cohere reranking"):
    from langchain_classic.retrievers import ContextualCompressionRetriever
    from langchain_cohere import CohereRerank

    reranked = ContextualCompressionRetriever(
        base_compressor=CohereRerank(model="rerank-english-v3.0", top_n=3),
        base_retriever=vector_store.as_retriever(search_kwargs={"k": 20}),
    )
    show(reranked.invoke("what proof do I need for a long illness?"), "Cohere rerank (20 -> 3)")

# %% [markdown]
# ### A free LLM-based reranker
#
# No Cohere key? Score documents with your existing model. Slower and less
# accurate than a purpose-built cross-encoder, but it demonstrates the mechanism
# and is genuinely usable at small `k`.

# %%
import re

rerank_prompt = ChatPromptTemplate.from_template(
    "Rate how well this document answers the question, from 0 to 10.\n"
    "Reply with the number only.\n\n"
    "Question: {question}\n\nDocument:\n{document}"
)
scorer = rerank_prompt | model | StrOutputParser()


def llm_rerank(question: str, docs: list[Document], top_n: int = 3) -> list[tuple[float, Document]]:
    scored: list[tuple[float, Document]] = []
    raw_scores = scorer.batch(
        [{"question": question, "document": d.page_content} for d in docs],
        config={"max_concurrency": 5},
    )
    for doc, raw in zip(docs, raw_scores):
        match = re.search(r"\d+(?:\.\d+)?", raw)
        scored.append((float(match.group()) if match else 0.0, doc))
    return sorted(scored, key=lambda pair: pair[0], reverse=True)[:top_n]


tricky = "what proof do I need if I'm off sick for two weeks?"
candidates = vector_store.similarity_search(tricky, k=8)

print("BEFORE rerank (embedding order):")
for i, doc in enumerate(candidates[:4], 1):
    print(f"  {i}. {' '.join(doc.page_content.split())[:95]}")

print("\nAFTER LLM rerank:")
for score, doc in llm_rerank(tricky, candidates, top_n=4):
    print(f"  {score:4.1f}  {' '.join(doc.page_content.split())[:95]}")

# %% [markdown]
# ### Local cross-encoder (no API, no cost)
#
# If `sentence-transformers` is installed, this is the practical free option:

# %%
try:
    from langchain_classic.retrievers import ContextualCompressionRetriever
    from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder

    cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    local_rerank = ContextualCompressionRetriever(
        base_compressor=CrossEncoderReranker(model=cross_encoder, top_n=3),
        base_retriever=vector_store.as_retriever(search_kwargs={"k": 15}),
    )
    show(local_rerank.invoke(tricky), "local cross-encoder rerank")
except Exception as exc:
    print(f"[skipped] local cross-encoder unavailable ({type(exc).__name__}).")
    print("          pip install 'langchain-huggingface[full]' to run this section.")

# %% [markdown]
# ## 6. Putting it together: a production-shaped pipeline
#
# ```
# question
#    |
#    +-- query transformation (multi-query / HyDE)   <- recall
#    |
#    +-- hybrid retrieval: BM25 + dense, RRF fused   <- recall
#    |
#    +-- rerank top 20 -> top 4                      <- precision
#    |
#    +-- compress (optional)                         <- cost
#    |
#    +-- generate with grounding rules               <- trust
# ```

# %%
class AdvancedRagRetriever(BaseRetriever):
    """Multi-query -> hybrid -> RRF -> LLM rerank."""

    vectorstore: Any
    keyword_retriever: Any
    query_generator: Any
    reranker: Any
    top_n: int = 4

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        variants = [query] + [
            v.strip("-* ").strip()
            for v in self.query_generator.invoke({"question": query}).splitlines()
            if v.strip()
        ][:2]

        ranked_lists: list[list[Document]] = []
        for variant in variants:
            ranked_lists.append(self.vectorstore.similarity_search(variant, k=6))
            ranked_lists.append(self.keyword_retriever.invoke(variant))

        fused = reciprocal_rank_fusion(ranked_lists, top_n=10)
        return [doc for _, doc in self.reranker(query, fused, self.top_n)]


variation_prompt = ChatPromptTemplate.from_template(
    "Write 2 alternative phrasings of this question, one per line, no numbering.\n\n{question}"
)

advanced_retriever = AdvancedRagRetriever(
    vectorstore=vector_store,
    keyword_retriever=bm25,
    query_generator=variation_prompt | model | StrOutputParser(),
    reranker=llm_rerank,
    top_n=3,
)

show(advanced_retriever.invoke("do I need a doctor's note for a long illness?"), "full advanced pipeline")

# %% [markdown]
# ## 7. Measure it, or you are guessing
#
# Every technique above costs latency and money. Prove each one earns its place.

# %%
import time

eval_set = [
    ("How many days of sick leave per year?", "leave_policy.txt"),
    ("What proof is needed for a long illness?", "leave_policy.txt"),
    ("What is the international hotel limit?", "company_handbook.md"),
    ("Do you have SOC 2?", "product_faq.md"),
    ("What are the API rate limits on Growth?", "product_faq.md"),
    ("Can unused leave be carried forward?", "leave_policy.txt"),
    ("How many office days per month?", "company_handbook.md"),
    ("What happens when a connector credential expires?", "product_faq.md"),
]

strategies = {
    "dense k=3": vector_store.as_retriever(search_kwargs={"k": 3}),
    "bm25 k=3": bm25,
    "hybrid": hybrid,
    "hyde": hyde_retriever,
    "advanced": advanced_retriever,
}

print(f"{'strategy':12} {'hit@k':>6} {'sec/query':>10}")
for name, strategy in strategies.items():
    start = time.perf_counter()
    hits = 0
    for question, expected in eval_set:
        sources = {d.metadata.get("source") for d in strategy.invoke(question)}
        hits += expected in sources
    elapsed = (time.perf_counter() - start) / len(eval_set)
    print(f"{name:12} {hits}/{len(eval_set):<4} {elapsed:>10.2f}")

# %% [markdown]
# Read this table honestly. On a small, clean corpus like ours, plain dense
# retrieval often ties the advanced pipeline at a fraction of the cost. The
# advanced techniques pay off as the corpus grows, gets noisier, and mixes
# jargon with natural language.
#
# **Do not adopt complexity you cannot measure a benefit from.**

# %% [markdown]
# ## Try it yourself
#
# 1. **Make the corpus hard.** Duplicate the corpus 20 times with small wording
#    changes, then re-run the comparison. Watch reranking start to matter.
# 2. **Add adversarial questions** with exact identifiers (`TCK-1006`, `E-104`,
#    `ap-south-1`) and confirm BM25 and hybrid beat dense retrieval clearly.
# 3. **Cost accounting.** Count LLM calls per query for each strategy. The
#    advanced pipeline makes several - decide what it is worth per 1,000 queries.
# 4. **HyDE ablation.** Find two questions where HyDE beats dense retrieval and
#    two where it loses, and articulate the pattern.

# %% [markdown]
# ## Recap
#
# | Technique | Fixes | Cost |
# |---|---|---|
# | Multi-query | vocabulary mismatch | 1 LLM call + N searches |
# | **HyDE** | question/answer shape mismatch | 1 LLM call |
# | Step-back | question too specific to match | 1 LLM call |
# | Decomposition | compound questions | N+1 LLM calls |
# | **Hybrid + RRF** | exact terms missed | ~free, biggest win |
# | **Reranking** | right doc ranked low | 1 rerank call; large precision gain |
# | Compression | noisy context, token cost | 1 LLM call, saves tokens |
# | `hit@k` measurement | not knowing if any of it helped | minutes, mandatory |
#
# **Adoption order:** hybrid search -> reranking -> query transformation. Stop as
# soon as your evaluation set is satisfied.
#
# ### Names you will see in other courses
#
# Two named pipelines show up constantly in Eden Marco's and Kris Naik's material.
# They are compositions of techniques you already have, not new primitives:
#
# | Named pipeline | What it actually is (in this curriculum) |
# |---|---|
# | **CRAG** (Corrective RAG) | Retrieval grader + fallback to web/another corpus when confidence is low — built end-to-end in notebook 48 |
# | **Adaptive RAG** | A router that sends the question to vectorstore, web search, or "no retrieval" — the routing half of notebooks 25 and 48 |
# | **Self-RAG** | Generate, then grade the generation for grounding and usefulness, looping if needed — also notebook 48 |
#
# When a course says "build CRAG", they mean: wire a grader (this notebook's
# spirit) into a graph that can rewrite or fall back (notebook 48).
#
# ## Next
#
# -> [16_tools_builtin_custom_toolkits.ipynb](../03-langchain-agents/16_tools_builtin_custom_toolkits.ipynb)
