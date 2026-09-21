# %% [markdown]
# # 10 - Text Splitters and Chunking Strategy
#
# | | |
# |---|---|
# | **Level** | Beginner to Intermediate |
# | **Time** | 45 minutes |
# | **Prerequisites** | `09_document_loaders` |
# | **Checklist ID** | `10_text_splitters` |
#
# ## Why this matters
#
# Chunking is the highest-leverage decision in a RAG system, and the one people
# skip. Here is the tension:
#
# - **Chunks too large** -> the embedding averages several topics together, so
#   similarity search gets vague and you waste context tokens on irrelevant text.
# - **Chunks too small** -> a fact gets severed from the condition that qualifies
#   it, and the model confidently gives a wrong answer.
#
# A concrete failure from our handbook: split
# *"Reimbursements are processed within 15 business days"* away from
# *"Claims submitted after 60 days are rejected"*, and your assistant will promise
# a refund that policy forbids.

# %%
from pathlib import Path
import sys

_root = Path.cwd()
while not (_root / "shared").is_dir() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from shared.notebook_setup import setup, require  # noqa: E402

ctx = setup("10_text_splitters")

# %%
policy_text = Path(ctx.data("leave_policy.txt")).read_text(encoding="utf-8")
handbook_text = Path(ctx.data("company_handbook.md")).read_text(encoding="utf-8")

print(f"leave_policy.txt   : {len(policy_text):,} chars")
print(f"company_handbook.md: {len(handbook_text):,} chars")

# %% [markdown]
# ## 1. `CharacterTextSplitter`: the naive baseline
#
# Splits on a single separator and only falls back to a hard cut when a piece is
# still too long. Simple, and usually the wrong choice - but it shows the problem
# clearly.

# %%
from langchain_text_splitters import CharacterTextSplitter

char_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=400,
    chunk_overlap=0,
    length_function=len,
)
char_chunks = char_splitter.split_text(policy_text)

print(f"{len(char_chunks)} chunks")
sizes = [len(c) for c in char_chunks]
print(f"sizes: min={min(sizes)} max={max(sizes)} mean={sum(sizes) // len(sizes)}")
print(f"\nchunks exceeding the 400 limit: {sum(1 for s in sizes if s > 400)}")

# %%
for i, chunk in enumerate(char_chunks[:3]):
    print(f"--- chunk {i} ({len(chunk)} chars) ---")
    print(chunk[:200].strip(), "...\n")

# %% [markdown]
# Notice chunks can **exceed** `chunk_size`. `CharacterTextSplitter` will not break
# a paragraph that has no `\n\n` inside it. That surprises people and silently
# blows context budgets.

# %% [markdown]
# ## 2. `RecursiveCharacterTextSplitter`: the default you should use
#
# It tries a list of separators in order - paragraphs, then lines, then sentences,
# then words, then characters - descending only when a piece is still too big.
# The effect is that it respects natural structure whenever it can.

# %%
from langchain_text_splitters import RecursiveCharacterTextSplitter

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=60,
    separators=["\n\n", "\n", ". ", " ", ""],   # this order is the whole trick
    length_function=len,
)
recursive_chunks = recursive_splitter.split_text(policy_text)

sizes = [len(c) for c in recursive_chunks]
print(f"{len(recursive_chunks)} chunks")
print(f"sizes: min={min(sizes)} max={max(sizes)} mean={sum(sizes) // len(sizes)}")
print(f"chunks over limit: {sum(1 for s in sizes if s > 400)}")

# %%
for i, chunk in enumerate(recursive_chunks[:3]):
    print(f"--- chunk {i} ({len(chunk)} chars) ---")
    print(chunk.strip()[:220], "...\n")

# %% [markdown]
# ### Why overlap exists
#
# Overlap duplicates the tail of one chunk at the head of the next, so a sentence
# that straddles a boundary still appears intact somewhere.

# %%
a, b = recursive_chunks[0], recursive_chunks[1]
tail = a[-60:]
print("end of chunk 0  :", repr(tail))
print("start of chunk 1:", repr(b[:60]))
print("\noverlap present:", tail.strip()[:30] in b)

# %% [markdown]
# **Rule of thumb: overlap = 10-20% of `chunk_size`.** More than that and you pay
# to store and search near-duplicate vectors; less and you risk severing facts.

# %% [markdown]
# ## 3. `TokenTextSplitter`: count what the model actually counts
#
# Characters are a proxy. Models charge by tokens, and the ratio varies with
# language and content: English prose is roughly 4 chars per token, but code,
# tables and non-Latin scripts are far denser.

# %%
from langchain_text_splitters import TokenTextSplitter

token_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=15)
token_chunks = token_splitter.split_text(policy_text)

print(f"{len(token_chunks)} chunks of ~100 tokens")
for chunk in token_chunks[:2]:
    print(f"\n({len(chunk)} chars) {chunk.strip()[:180]}...")

# %%
# Compare character-based vs token-based measurement on different content types.
import tiktoken

encoder = tiktoken.get_encoding("cl100k_base")

samples = {
    "english prose": "Employees earn twenty four days of annual leave each calendar year.",
    "dense table  ": "| Domestic travel | INR 6,000 | INR 8,000 |",
    "code         ": 'splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60)',
    "hindi        ": "कर्मचारियों को प्रति वर्ष चौबीस दिन का अवकाश मिलता है।",
}

print(f"{'sample':16} {'chars':>6} {'tokens':>7} {'chars/token':>12}")
for label, text in samples.items():
    n_chars, n_tokens = len(text), len(encoder.encode(text))
    print(f"{label:16} {n_chars:>6} {n_tokens:>7} {n_chars / n_tokens:>12.1f}")

# %% [markdown]
# The Hindi line uses far fewer characters per token. If you size chunks in
# characters and your corpus is multilingual, some chunks will silently be 3x your
# intended token budget.
#
# **Best practice:** use `RecursiveCharacterTextSplitter` (good boundaries) but
# measure in tokens via `from_tiktoken_encoder` - you get both.

# %%
hybrid_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=120,       # now measured in TOKENS
    chunk_overlap=20,
)
hybrid_chunks = hybrid_splitter.split_text(policy_text)

token_counts = [len(encoder.encode(c)) for c in hybrid_chunks]
print(f"{len(hybrid_chunks)} chunks")
print(f"tokens: min={min(token_counts)} max={max(token_counts)} mean={sum(token_counts) // len(token_counts)}")
print("all within budget:", max(token_counts) <= 120)

# %% [markdown]
# ## 4. Structure-aware splitting for Markdown
#
# Our handbook has headings. Splitting on them preserves meaning *and* gives you
# free metadata.

# %%
from langchain_text_splitters import MarkdownHeaderTextSplitter

md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "document"), ("##", "section"), ("###", "subsection")],
    strip_headers=False,
)
md_chunks = md_splitter.split_text(handbook_text)

print(f"{len(md_chunks)} sections\n")
for doc in md_chunks:
    section = doc.metadata.get("section", "(root)")
    print(f"  {section:42} {len(doc.page_content):5} chars")

# %% [markdown]
# Each chunk now carries its heading in metadata. That means a citation can say
# *"Expense Policy"* rather than *"chunk 7"*, and you can filter retrieval to a
# single section.

# %%
# Sections vary wildly in size, so do a second pass to cap the long ones.
two_pass = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", chunk_size=200, chunk_overlap=30
)
final_chunks = two_pass.split_documents(md_chunks)

print(f"{len(md_chunks)} sections -> {len(final_chunks)} final chunks")
print("\nmetadata survives the second pass:")
for doc in final_chunks[:4]:
    print(" ", {k: v for k, v in doc.metadata.items()}, f"({len(doc.page_content)} chars)")

# %% [markdown]
# This **two-pass pattern** - structure split, then size split - is the one to
# reach for with any structured document. It is what most production pipelines do.

# %% [markdown]
# ## 5. Code and other languages

# %%
from langchain_text_splitters import Language

python_source = '''
def calculate_leave_balance(employee_id: str, year: int) -> int:
    """Annual leave accrues at 2 days per completed month."""
    months = completed_months(employee_id, year)
    return min(months * 2, 24)


def completed_months(employee_id: str, year: int) -> int:
    joined = employee_join_date(employee_id)
    if joined.year < year:
        return 12
    return 12 - joined.month + 1


class LeaveRequest:
    """A single request for time off."""

    def __init__(self, employee_id: str, days: int):
        self.employee_id = employee_id
        self.days = days

    def requires_handover(self) -> bool:
        return self.days >= 5
'''

code_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=300, chunk_overlap=0
)
for i, chunk in enumerate(code_splitter.split_text(python_source)):
    print(f"--- chunk {i} ---")
    print(chunk.strip()[:220])
    print()

# %%
print("Languages with tuned separators:")
print(", ".join(sorted(lang.value for lang in Language))[:600])

# %% [markdown]
# ## 6. Measuring chunk quality
#
# Do not guess. Score your splitter against questions you care about.

# %%
def chunk_report(chunks, name: str, encoder=encoder) -> dict:
    texts = [c.page_content if hasattr(c, "page_content") else c for c in chunks]
    tokens = [len(encoder.encode(t)) for t in texts]
    return {
        "splitter": name,
        "chunks": len(texts),
        "min_tok": min(tokens),
        "max_tok": max(tokens),
        "mean_tok": sum(tokens) // len(tokens),
        "tiny(<20)": sum(1 for t in tokens if t < 20),
    }


import json

for report in [
    chunk_report(char_chunks, "Character(400)"),
    chunk_report(recursive_chunks, "Recursive(400c)"),
    chunk_report(hybrid_chunks, "Recursive(120tok)"),
    chunk_report(final_chunks, "Markdown+Recursive"),
]:
    print(json.dumps(report))

# %% [markdown]
# `tiny(<20)` matters: fragment chunks like a lone heading embed to noise and
# pollute retrieval. If you have many, raise `chunk_size` or merge small pieces.

# %% [markdown]
# ## 7. The real test: can a chunk answer the question alone?
#
# This is the only evaluation that counts. Take questions your users will ask and
# check whether **one** chunk contains the full answer.

# %%
questions_and_keys = [
    ("How many annual leave days do I get?", ["24 days", "2 days per completed month"]),
    ("Can I carry forward unused leave?", ["12 unused annual leave", "30 June"]),
    ("When do I need a medical certificate?", ["3 or more consecutive days", "medical certificate"]),
    ("How much paternity leave is there?", ["15 working days", "3 blocks"]),
    ("How is leave encashment calculated?", ["basic salary", "capped at 30 days"]),
]


def self_contained_score(chunks, label: str) -> None:
    texts = [c.page_content if hasattr(c, "page_content") else c for c in chunks]
    hits = 0
    for question, keys in questions_and_keys:
        found = any(all(key.lower() in text.lower() for key in keys) for text in texts)
        hits += found
        marker = "OK  " if found else "MISS"
        print(f"  {marker} {question}")
    print(f"  => {label}: {hits}/{len(questions_and_keys)} answerable from a single chunk\n")


self_contained_score(recursive_splitter.split_text(policy_text), "Recursive(400 chars, 60 overlap)")
self_contained_score(
    RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=0).split_text(policy_text),
    "Recursive(150 chars, no overlap)",
)
self_contained_score(
    RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=100).split_text(policy_text),
    "Recursive(900 chars, 100 overlap)",
)

# %% [markdown]
# Small chunks fragment the answer; very large chunks contain it but bury it. The
# middle setting usually wins - and now you have a way to prove it rather than
# guess.

# %% [markdown]
# ## 8. Choosing your settings
#
# | Content | Splitter | chunk_size | overlap |
# |---|---|---|---|
# | Policies, FAQs, articles | `RecursiveCharacterTextSplitter.from_tiktoken_encoder` | 300-500 tokens | 10-15% |
# | Markdown docs | `MarkdownHeaderTextSplitter` -> recursive | section, then 300 tokens | 10% |
# | Source code | `from_language(Language.PYTHON)` | 500-1000 tokens | 0 |
# | Chat transcripts | recursive on `\n` | 200-400 tokens | 1-2 turns |
# | Legal contracts | recursive, clause separators | 800-1500 tokens | 20% |
# | Tables/CSV | one row per document (no splitter) | n/a | n/a |
#
# ### Advanced: semantic chunking
#
# `SemanticChunker` (in `langchain-experimental`) splits where the *meaning* shifts
# by comparing embeddings of adjacent sentences. Higher quality, much slower, and
# it costs an embedding call per sentence. Worth trying when recursive splitting
# demonstrably fails - not as a default.
#
# ```python
# from langchain_experimental.text_splitter import SemanticChunker
# from shared.llm import get_embeddings
#
# chunks = SemanticChunker(get_embeddings(), breakpoint_threshold_type="percentile").split_text(text)
# ```

# %% [markdown]
# ## Try it yourself
#
# 1. **Find the cliff.** Run `self_contained_score` with chunk sizes 100, 200, 400,
#    800, 1600 and plot the score. Where does it peak for this policy?
# 2. **Overlap sweep.** Fix `chunk_size=400` and try overlap 0, 40, 80, 160. Count
#    total stored characters and note the storage cost of each point of accuracy.
# 3. **Split the PDF.** Load the security policy PDF, apply the two-pass pattern,
#    and confirm `page` metadata survives into the final chunks - you need it for
#    citations.
# 4. **Write your own.** Implement a splitter for `leave_policy.txt` that splits on
#    the numbered headings (`1.`, `2.`, ...) and puts the heading in metadata.

# %% [markdown]
# ## Recap
#
# | Concept | Takeaway |
# |---|---|
# | `CharacterTextSplitter` | Single separator; can exceed `chunk_size`. Rarely the right pick |
# | `RecursiveCharacterTextSplitter` | **The default.** Separator ladder respects structure |
# | `chunk_overlap` | 10-20% of size; stops facts being severed at boundaries |
# | `TokenTextSplitter` | Counts what the model bills; essential for multilingual corpora |
# | `from_tiktoken_encoder` | Best of both - good boundaries, token-accurate sizing |
# | `MarkdownHeaderTextSplitter` | Structure split first, size split second; free metadata |
# | `from_language(...)` | Code-aware separators |
# | Self-containment test | The only chunk metric that predicts answer quality |
#
# ## Next
#
# -> [11_embeddings_and_caching.ipynb](11_embeddings_and_caching.ipynb)
