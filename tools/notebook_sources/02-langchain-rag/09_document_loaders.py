# %% [markdown]
# # 09 - Document Loaders
#
# | | |
# |---|---|
# | **Level** | Beginner |
# | **Time** | 40 minutes |
# | **Prerequisites** | `08_chains_sequential_and_custom` |
# | **Checklist ID** | `09_document_loaders` |
#
# ## Why this matters
#
# The running project for this track: **"Chat with the Northwind employee
# handbook"** - an internal assistant that answers leave, expense and security
# questions from real company documents instead of the model's imagination.
#
# Every RAG system starts here, and loading is where most of the quality is won
# or lost. Garbage in the loader means garbage retrieval no matter how good your
# embeddings are. The two things you must get right are:
#
# 1. **Text fidelity** - did the loader actually recover the content?
# 2. **Metadata** - can you tell the user *where* an answer came from?

# %%
from pathlib import Path
import sys

_root = Path.cwd()
while not (_root / "shared").is_dir() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from shared.notebook_setup import setup, require  # noqa: E402

ctx = setup("09_document_loaders")

# %%
from shared.sample_data import ensure_all

assets = ensure_all()
print("Sample corpus ready:")
for name, path in assets.items():
    print(f"  {name:8} {path.name}")

# %% [markdown]
# ## 1. The `Document`: the unit of everything downstream
#
# Every loader, splitter, vector store and retriever speaks `Document`. It has
# exactly two parts.

# %%
from langchain_core.documents import Document

example = Document(
    page_content="Annual leave of 3 or more consecutive days must be applied for 7 days in advance.",
    metadata={"source": "leave_policy.txt", "section": "1. Annual Leave", "page": 1},
)

print("page_content:", example.page_content)
print("metadata    :", example.metadata)

# %% [markdown]
# **Metadata is not optional in a real system.** It is what lets you cite sources,
# filter by department, restrict by access level, and show "from page 4 of the
# security policy" in the UI. Plan it before you load anything.

# %% [markdown]
# ## 2. `TextLoader`: plain text

# %%
from langchain_community.document_loaders import TextLoader

loader = TextLoader(str(ctx.data("leave_policy.txt")), encoding="utf-8")
docs = loader.load()

print(f"{len(docs)} document(s)")
print("metadata:", docs[0].metadata)
print("characters:", len(docs[0].page_content))
print("\nfirst 250 chars:\n", docs[0].page_content[:250])

# %% [markdown]
# Note: `TextLoader` returns **one document for the whole file**. Splitting into
# chunks is a separate step (notebook 10), and that separation is deliberate -
# loading and chunking have different failure modes.
#
# > **Windows gotcha:** always pass `encoding="utf-8"`. Without it Python uses the
# > system code page (often cp1252), and any curly quote or em-dash raises
# > `UnicodeDecodeError`.

# %% [markdown]
# ## 3. `CSVLoader`: one row, one document

# %%
from langchain_community.document_loaders import CSVLoader

csv_loader = CSVLoader(
    file_path=str(ctx.data("support_tickets.csv")),
    encoding="utf-8",
    csv_args={"delimiter": ","},
)
ticket_docs = csv_loader.load()

print(f"{len(ticket_docs)} documents (one per row)\n")
print(ticket_docs[0].page_content)
print("\nmetadata:", ticket_docs[0].metadata)

# %% [markdown]
# The default `page_content` is `column: value` on each line. That is verbose and
# embeds poorly. Better: control which columns become content and which become
# metadata.

# %%
targeted = CSVLoader(
    file_path=str(ctx.data("support_tickets.csv")),
    encoding="utf-8",
    content_columns=["summary"],                       # embed only the meaningful text
    metadata_columns=["ticket_id", "customer", "plan", "category", "priority", "status"],
)
clean_docs = targeted.load()

print(clean_docs[0].page_content)
print("metadata:", clean_docs[0].metadata)

# %% [markdown]
# Now the embedding sees only the ticket summary, while customer and priority stay
# available as **filterable** metadata. This one change usually improves retrieval
# quality more than swapping the embedding model.

# %%
# Older versions of CSVLoader used source_column/metadata_columns only.
# If content_columns is unavailable on your version, build documents manually:
import csv

with open(ctx.data("support_tickets.csv"), newline="", encoding="utf-8") as handle:
    manual_docs = [
        Document(
            page_content=row["summary"],
            metadata={k: v for k, v in row.items() if k != "summary"},
        )
        for row in csv.DictReader(handle)
    ]

print(f"{len(manual_docs)} manual documents, e.g.:")
print(" ", manual_docs[3].page_content)
print(" ", manual_docs[3].metadata)

# %% [markdown]
# Writing the loader yourself is completely legitimate. Loaders are a convenience,
# not a requirement - a `Document` is just a dataclass.

# %% [markdown]
# ## 4. `PyPDFLoader`: the format that causes the most pain

# %%
from langchain_community.document_loaders import PyPDFLoader

pdf_loader = PyPDFLoader(str(assets["pdf"]))
pdf_docs = pdf_loader.load()

print(f"{len(pdf_docs)} documents (one per page)\n")
for doc in pdf_docs:
    page = doc.metadata.get("page", "?")
    print(f"page {page}: {len(doc.page_content):4} chars | starts: {doc.page_content[:60].strip()!r}")

# %%
print(pdf_docs[1].page_content[:600])

# %% [markdown]
# ### PDF reality check
#
# PDFs describe *where ink goes*, not what the text means. Expect:
#
# | Problem | Symptom | Mitigation |
# |---|---|---|
# | Multi-column layout | Sentences interleave between columns | Use a layout-aware parser |
# | Tables | Collapse into unreadable token soup | Extract tables separately |
# | Scanned pages | `page_content` is empty | OCR required |
# | Headers/footers | Repeat in every chunk, pollute embeddings | Strip with a regex after loading |
# | Ligatures | "ﬁ" instead of "fi" | Normalise with `unicodedata` |
#
# **Always print the extracted text before trusting it.** A silent empty extraction
# is the most common cause of "my RAG returns nothing".

# %%
def audit_extraction(documents, min_chars: int = 100) -> None:
    """Catch empty or suspiciously short pages before they poison an index."""
    for doc in documents:
        page = doc.metadata.get("page", doc.metadata.get("source", "?"))
        length = len(doc.page_content.strip())
        status = "EMPTY - needs OCR" if length == 0 else ("short" if length < min_chars else "ok")
        print(f"  page {page}: {length:5} chars  [{status}]")


audit_extraction(pdf_docs)

# %%
# Other PDF loaders, in increasing order of capability and cost:
#
#   PyPDFLoader           fast, simple, no layout awareness      (pypdf)
#   PyMuPDFLoader         fast, better layout + image extraction (pymupdf)
#   PDFPlumberLoader      good tables                            (pdfplumber)
#   UnstructuredPDFLoader layout model, best quality, slowest    (unstructured)
#   AzureAIDocumentIntelligenceLoader / AWS Textract  - cloud OCR for scans
#
# Start with PyPDFLoader. Escalate only when the audit above shows a problem.

# %% [markdown]
# ## 5. `WebBaseLoader`: scraping HTML

# %%
from langchain_community.document_loaders import WebBaseLoader

try:
    web_loader = WebBaseLoader(
        web_paths=["https://docs.langchain.com/oss/python/langgraph/overview"],
        requests_kwargs={"timeout": 20},
    )
    web_docs = web_loader.load()
    text = web_docs[0].page_content
    print("source   :", web_docs[0].metadata.get("source"))
    print("title    :", web_docs[0].metadata.get("title"))
    print("chars    :", len(text))
    print("\nfirst 300 chars of raw extraction:\n", " ".join(text.split())[:300])
except Exception as exc:
    print(f"[skipped] no network access here ({type(exc).__name__}). "
          "The technique below still applies when you run this online.")

# %% [markdown]
# Raw web extraction includes navigation, cookie banners and footers. Narrow it
# with a BeautifulSoup selector so you only embed the article body:
#
# ```python
# import bs4
#
# loader = WebBaseLoader(
#     web_paths=["https://example.com/article"],
#     bs_kwargs={"parse_only": bs4.SoupStrainer(class_=("prose", "article-body"))},
# )
# ```
#
# Before scraping anything: check `robots.txt`, set a real user agent, and rate
# limit. `WebBaseLoader` accepts `header_template` and `requests_per_second`.

# %% [markdown]
# ## 6. `DirectoryLoader`: a whole folder at once

# %%
from langchain_community.document_loaders import DirectoryLoader

dir_loader = DirectoryLoader(
    str(ctx.data()),
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
    show_progress=False,
)
folder_docs = dir_loader.load()

print(f"{len(folder_docs)} .txt documents")
for doc in folder_docs:
    print(f"  {Path(doc.metadata['source']).name:28} {len(doc.page_content):6} chars")

# %% [markdown]
# ### Mixed folders need a per-extension mapping
#
# One `DirectoryLoader` call cannot use different parsers per file type, so route
# by suffix yourself. This function is the practical shape of a real ingestion job.

# %%
def load_corpus(folder: Path) -> list[Document]:
    """Load a mixed folder, choosing the right loader per extension."""
    loaders = {
        ".txt": lambda p: TextLoader(str(p), encoding="utf-8"),
        ".md": lambda p: TextLoader(str(p), encoding="utf-8"),
        ".csv": lambda p: CSVLoader(str(p), encoding="utf-8"),
        ".pdf": lambda p: PyPDFLoader(str(p)),
    }

    collected: list[Document] = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        build = loaders.get(path.suffix.lower())
        if build is None:
            continue
        try:
            loaded = build(path).load()
        except Exception as exc:  # one bad file must not kill the whole ingest
            print(f"  ERROR {path.name}: {type(exc).__name__}: {exc}")
            continue

        for doc in loaded:
            doc.metadata.setdefault("source", str(path))
            doc.metadata["file_name"] = path.name
            doc.metadata["file_type"] = path.suffix.lstrip(".")
        collected.extend(loaded)
        print(f"  loaded {path.name:36} -> {len(loaded):3} doc(s)")

    return collected


corpus = load_corpus(ctx.data())
print(f"\nTotal: {len(corpus)} documents from {len({d.metadata['file_name'] for d in corpus})} files")

# %% [markdown]
# The `try/except` per file matters. In production one corrupt PDF out of 5,000
# should be logged and skipped, not crash the nightly ingestion.

# %% [markdown]
# ## 7. Enriching metadata at load time
#
# Metadata you do not add at ingestion is metadata you cannot filter on later.
# Re-indexing a large corpus is expensive, so think about this now.

# %%
from datetime import datetime, timezone


def enrich(documents: list[Document]) -> list[Document]:
    """Add the fields the retrieval layer will actually want to filter on."""
    classification = {
        "northwind_security_policy.pdf": ("security", "restricted"),
        "leave_policy.txt": ("hr", "internal"),
        "company_handbook.md": ("hr", "internal"),
        "product_faq.md": ("product", "public"),
        "support_tickets.csv": ("support", "internal"),
        "employees.csv": ("hr", "confidential"),
    }

    for doc in documents:
        name = doc.metadata.get("file_name", "")
        domain, sensitivity = classification.get(name, ("general", "internal"))
        doc.metadata.update(
            {
                "domain": domain,
                "sensitivity": sensitivity,
                "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "char_count": len(doc.page_content),
            }
        )
    return documents


enriched = enrich(corpus)
for doc in enriched[:3]:
    print({k: v for k, v in doc.metadata.items() if k != "source"})

# %%
from collections import Counter

print("documents by domain     :", dict(Counter(d.metadata["domain"] for d in enriched)))
print("documents by sensitivity:", dict(Counter(d.metadata["sensitivity"] for d in enriched)))

# %% [markdown]
# That `sensitivity` field is how you later stop a Starter-plan user's question
# from retrieving a `restricted` security document. Access control belongs in
# metadata, enforced at retrieval time.

# %% [markdown]
# ## 8. Lazy loading for large corpora
#
# `.load()` puts everything in memory. For thousands of files use `.lazy_load()`,
# which yields documents one at a time.

# %%
streamed = 0
total_chars = 0
for doc in TextLoader(str(ctx.data("leave_policy.txt")), encoding="utf-8").lazy_load():
    streamed += 1
    total_chars += len(doc.page_content)

print(f"streamed {streamed} document(s), {total_chars} chars, constant memory")

# %% [markdown]
# ## 9. Sanity-check the corpus before indexing
#
# Five minutes here saves a day of debugging bad answers.

# %%
def corpus_report(documents: list[Document]) -> None:
    lengths = [len(d.page_content) for d in documents]
    empties = [d for d in documents if not d.page_content.strip()]
    duplicates = len(documents) - len({d.page_content for d in documents})

    print(f"documents     : {len(documents)}")
    print(f"total chars   : {sum(lengths):,}")
    print(f"shortest      : {min(lengths)} chars")
    print(f"longest       : {max(lengths):,} chars")
    print(f"median        : {sorted(lengths)[len(lengths) // 2]} chars")
    print(f"empty         : {len(empties)}  <- must be 0 before indexing")
    print(f"exact dupes   : {duplicates}")
    missing_source = [d for d in documents if "source" not in d.metadata]
    print(f"no source meta: {len(missing_source)}  <- breaks citations")


corpus_report(enriched)

# %% [markdown]
# ## Try it yourself
#
# 1. **Header/footer strip.** Write a function that removes any line appearing on
#    more than half the PDF pages, and confirm it cleans the extraction.
# 2. **Sensitivity filter.** Write `load_corpus(folder, max_sensitivity="internal")`
#    that refuses to load `confidential` or `restricted` files, and prove it
#    excludes `employees.csv`.
# 3. **Your own loader.** Build a `Document` list from
#    `shared/sample_data/company_handbook.md` where each `##` section becomes one
#    document with the heading in metadata. (`MarkdownHeaderTextSplitter` in
#    notebook 10 does this - try it by hand first.)
# 4. **Break it.** Feed `TextLoader` a file without `encoding="utf-8"` on Windows
#    and read the traceback so you recognise it later.

# %% [markdown]
# ## Recap
#
# | Concept | Takeaway |
# |---|---|
# | `Document` | `page_content` + `metadata`; the currency of all RAG |
# | `TextLoader` | Always pass `encoding="utf-8"` on Windows |
# | `CSVLoader` | Use `content_columns` / `metadata_columns`; do not embed column names |
# | `PyPDFLoader` | One doc per page; **audit the extraction**, empty pages are silent |
# | `WebBaseLoader` | Narrow with a `SoupStrainer`; respect robots.txt |
# | `DirectoryLoader` | Per-extension routing + per-file error handling |
# | Metadata | Add domain/sensitivity/source at ingest - you cannot filter on what you did not store |
# | `lazy_load()` | Constant memory for large corpora |
#
# ## Next
#
# -> [10_text_splitters.ipynb](10_text_splitters.ipynb)
