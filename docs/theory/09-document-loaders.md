# Document Loaders

## 30-second answer

A loader turns a file or URL into `Document` objects. Each one has `page_content` (the text you will embed) and `metadata` (where it came from, what you can filter on later). Loading and chunking stay separate on purpose. Bad text or missing metadata here caps retrieval forever, no matter how good the embedding model is.

## Tiny example

Priya loads the Northwind leave policy for the HR chat bot.

1. She points `TextLoader` at `leave_policy.txt` with `encoding="utf-8"`.
2. `.load()` returns one `Document` for the whole file.
3. She stamps `domain="hr"` and `sensitivity="internal"` into `metadata`.
4. She prints the first 250 characters and checks the page is not empty.
5. Only then does she hand the list to the splitter in notebook 10.

```mermaid
flowchart LR
  file[leave_policy.txt] --> loader[TextLoader]
  loader --> doc[Document]
  doc --> audit[Check length and source]
  audit --> next[Ready for splitters]
```

*Picture: a file goes in; a `Document` with text plus tags comes out; you audit before you index.*

## Why it exists

The Northwind assistant must answer leave, expense, and security questions from real company files—not from the model's guesses. Two things matter at ingest: **text fidelity** (did we recover what a human reads?) and **metadata** (can we cite the source and enforce access later?). Mistakes here multiply cost when you re-embed later.

## Runtime

1. Pick a loader by format: `TextLoader`, `CSVLoader`, `PyPDFLoader`, `WebBaseLoader`, or a per-extension map for mixed folders.
2. Call `.load()` (or `.lazy_load()` for huge corpora). You always get `list[Document]`.
3. Inspect right away: `metadata`, character count, first ~250 chars of `page_content`.
4. Enrich fields you will filter on: `domain`, `sensitivity`, `file_name`, `file_type`, `ingested_at`, `char_count`.
5. Audit: empties, exact duplicates, missing `source`. Empties must be zero before indexing.
6. Hand off to splitters. Do not chunk inside the loader.

Cardinality is part of the contract:

- `TextLoader` → one document for the whole file
- `CSVLoader` → one document per row
- `PyPDFLoader` → one document per page

## Objects, fields, and merge rules

| Object / field | In plain words |
|---|---|
| `Document.page_content` | The string the embedder and LLM will see |
| `Document.metadata` | Dict that must survive splitters and stores |
| `source` | Path or URL; required for citations |
| `page` | PDF page index from `PyPDFLoader` |
| `content_columns` / `metadata_columns` | CSV: what becomes text vs filterable tags |
| `encoding="utf-8"` | Required on Windows; avoids decode errors |
| `bs_kwargs` / `SoupStrainer` | Web: keep only the article body |
| `loader_cls` / `loader_kwargs` | DirectoryLoader: which loader per match |
| `.lazy_load()` | Yield one doc at a time; steady memory |

**Merge rules:** enrich with `setdefault("source", ...)`—do not wipe loader keys like `page`. Catch errors per file so one bad PDF does not kill the job. Metadata you skip at ingest cannot be filtered later without a full re-index. Access control lives in metadata and is enforced at retrieval, not in a system prompt. Hand-built `Document(...)` lists are fine; loaders are convenience.

**PDF escalation (lab):** `PyPDFLoader` (fast start) → `PyMuPDFLoader` (layout) → `PDFPlumberLoader` (tables) → `UnstructuredPDFLoader` (best, slowest) → cloud OCR for scans.

## Control surface

| Knob | Effect |
|---|---|
| `TextLoader(..., encoding="utf-8")` | Stops Windows `UnicodeDecodeError` on curly quotes |
| `CSVLoader(content_columns=..., metadata_columns=...)` | Embed useful text; keep ids filterable |
| `WebBaseLoader` + `SoupStrainer` | Drop nav, banners, footers |
| `DirectoryLoader(glob=..., loader_cls=...)` | Bulk one-extension folders |
| Sensitivity map at ingest | Later ACL via `sensitivity` |
| `min_chars` in audit | Flag empty/short PDF pages before index |

## Failure anatomy

| Symptom | Cause | Fix |
|---|---|---|
| `UnicodeDecodeError` on plain text | Missing `encoding="utf-8"` on Windows | Always pass utf-8 |
| RAG returns nothing; index "has" docs | Empty PDF `page_content` (scans) | Audit lengths; use OCR |
| Multi-column sentences interleaved | Layout-blind PDF parser | Escalate to layout-aware loader |
| Tables become token soup | Ink-position PDF, not a real table | Extract tables / PDFPlumber |
| Headers/footers in every chunk | Repeated chrome per page | Strip lines that appear on most pages |
| Web page is mostly chrome | Raw HTML scrape | `SoupStrainer` on article classes |
| Nightly job dies on one corrupt PDF | Uncaught exception in folder walk | Per-file try/except |
| Cannot filter by department later | Metadata never stored | Enrich before indexing |
| Citations broken | Missing `source` | Corpus report: zero missing |

Cost at this stage: zero LLM and zero embedding calls—CPU/IO only.

## Keywords

In plain words:

- **Document** — unit of RAG: `page_content` + `metadata`.
- **Loader** — `.load()` / `.lazy_load()` → `list[Document]`.
- **Text fidelity** — extracted text matches what a human reads.
- **Metadata enrichment** — add filter fields (`domain`, `sensitivity`) at ingest.
- **Lazy load** — stream docs; roughly one in memory at a time.
- **Corpus audit** — length, empty, duplicate, missing-source checks.
- **Sensitivity** — access tag enforced at retrieval, never only in a prompt.

## Minimal fragment

```python
from langchain_community.document_loaders import TextLoader, CSVLoader, PyPDFLoader

docs = TextLoader("leave_policy.txt", encoding="utf-8").load()
tickets = CSVLoader(
    "support_tickets.csv",
    encoding="utf-8",
    content_columns=["summary"],
    metadata_columns=["ticket_id", "priority", "status"],
).load()
pdf_pages = PyPDFLoader("policy.pdf").load()
for page in pdf_pages:
    n = len(page.page_content.strip())
    assert n > 0, f"empty page {page.metadata.get('page')} — needs OCR"
```

## Interview traps

**Shallow answer.** A document loader just reads files into strings.

**Better answer.** It produces `Document` objects with a metadata contract. `TextLoader` is one doc per file; `CSVLoader` one per row; `PyPDFLoader` one per page. Metadata planned at ingest is the only filterable surface later.

**Shallow answer.** If the PDF loads without an exception, the text is fine.

**Better answer.** PDFs describe ink positions, not meaning. Empty scanned pages, interleaved columns, and repeated headers all "succeed" while poisoning the index. Always audit lengths before embedding.

**Shallow answer.** Access control can be a system prompt that says ignore restricted docs.

**Better answer.** If a restricted chunk reaches the context window, it already leaked. Put `sensitivity` in metadata and filter server-side at retrieval.

## Lab

[09_document_loaders.ipynb](../../02-langchain-rag/09_document_loaders.ipynb)
