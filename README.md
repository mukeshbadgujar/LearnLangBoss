# GenAI Mastery: LangChain + LangGraph

A complete, runnable curriculum that takes you from "what is a chat model" to
shipping a multi-agent LangGraph service behind FastAPI. **54 Jupyter
notebooks**, every one executable on Windows, macOS or Linux, using free model
providers.

Built against **LangChain 1.x** and **LangGraph 1.x** - not the 0.x tutorials
you will find everywhere else. Where your search results show a deprecated API,
the notebooks show the current one and a short "if you see this in old repos"
note so you can still read the old code.

## Quick start

```powershell
# 1. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate

# 2. Install
python -m pip install -r requirements.txt

# 3. Add at least one API key
copy .env.example .env                # macOS/Linux: cp .env.example .env
#    then edit .env and set GROQ_API_KEY (free tier, fastest to get)

# 4. Verify the setup
python -c "from shared.llm import describe_environment; print(describe_environment())"

# 5. Start
jupyter lab 00-setup/00_environment_providers_and_keys.ipynb
```

If step 4 prints at least one provider, you are ready. If it prints none, open
[`00-setup/00_environment_providers_and_keys.ipynb`](00-setup/00_environment_providers_and_keys.ipynb)
- it diagnoses the problem.

> **Use `python -m pip`, not bare `pip`.** On many Windows setups they point at
> different interpreters, which is the most common cause of
> "I installed it but the notebook says it is missing".

## You only need one API key

Every notebook calls `shared.llm.get_chat_model()`, which picks the first
provider you have configured:

| Order | Provider | Key | Why it is first |
|---|---|---|---|
| 1 | **Groq** | `GROQ_API_KEY` | Free tier, very fast, no card required |
| 2 | OpenRouter | `OPENROUTER_API_KEY` | One key, many models |
| 3 | OpenAI | `OPENAI_API_KEY` | Best structured-output and vision support |
| 4 | Anthropic | `ANTHROPIC_API_KEY` | Long context, strong reasoning |

**Embeddings follow the same free-first policy**, resolved independently via
`EMBEDDING_PROVIDER=auto`:

| Order | Provider | Dims | Notes |
|---|---|---|---|
| 1 | **Groq** (`nomic-embed-text-v1_5`) | 768 | Free tier, nothing to install, reuses `GROQ_API_KEY` |
| 2 | HuggingFace (local) | 384 | Offline, no rate limits, but needs torch (~2 GB) |
| 3 | OpenAI | 1536 | Paid, strongest baseline |
| 4 | Fake | 384 | Last resort so notebooks still run - **scores are meaningless** |

Vector stores (FAISS, Chroma) run locally, so RAG costs nothing beyond
embedding calls. Cells needing a paid extra - Cohere rerank, Pinecone, Tavily,
Redis, LangGraph Platform - are guarded with `require(...)` and skip cleanly
with an explanation when the key is absent. **No notebook is blocked by a
missing optional key.**

> Groq's free tier is rate limited. If you hit 429s while indexing a large
> corpus, install `langchain-huggingface[full]` and set
> `EMBEDDING_PROVIDER=huggingface` for unlimited local embedding.

## Layout

```
00-setup/                   environment, providers, keys, tracing
01-langchain-foundations/   models, prompts, parsers, memory, LCEL, chains
02-langchain-rag/           loaders, splitters, embeddings, stores, retrieval
03-langchain-agents/        tools, agents, and why LangGraph exists
04-langchain-production/    tracing, callbacks, streaming, caching, cost, eval
05-langgraph-beginner/      state, nodes, edges, routing
06-langgraph-intermediate/  checkpoints, human-in-the-loop, parallelism
07-langgraph-advanced/      time travel, subgraphs, multi-agent, deep agents
08-capstones/               three complete, deployable products

shared/                     llm.py (provider selection), notebook_setup.py, sample_data/
references/                 official-sources.md, course-crosswalk.md
tools/                      notebook generator, static checker, runtime smoke tests
artifacts/                  git-ignored scratch space the notebooks write to
```

- **[LEARNING_PATH.md](LEARNING_PATH.md)** - what to study in what order,
  including a 5-week overlay if you are learning this alongside a job.
- **[CHECKLIST.md](CHECKLIST.md)** - mark each notebook Done or Pending as you
  verify it.
- **[references/official-sources.md](references/official-sources.md)** - the
  documentation page behind each notebook, for when the libraries move.
- **[references/course-crosswalk.md](references/course-crosswalk.md)** - the
  three public course repos and the `LLG/` archive mapped onto these notebooks,
  plus the topics they cover that we do not.

## How each notebook is built

The same seven-part shape, every time:

1. **Header** - level, time, prerequisites, checklist id
2. **Why this matters** - a real situation where you would reach for this
3. **Concept** - plain-language explanation before any code
4. **Code** - runnable cells, output you can compare against
5. **Try it yourself** - exercises that change something and observe the result
6. **Recap** - a table of the decisions and why they went that way
7. **Next** - a link onwards

Notebooks build on a running example rather than toy strings: a company
handbook and leave policy for RAG, a support desk for agents, and a research
desk for multi-agent work. The sample corpus lives in `shared/sample_data/`;
the PDF and SQLite files are **generated on first use** so nothing binary is
committed.

## The three capstones

| Notebook | What you build |
|---|---|
| [51 - Policy RAG chatbot](08-capstones/51_policy_rag_chatbot.ipynb) | Hybrid retrieval, history-aware follow-ups, grounding verification, citations, escalation, an evaluation harness |
| [52 - Multi-agent research desk](08-capstones/52_multi_agent_research_desk.ipynb) | Supervisor, researcher, writer and reviewer with enforced budgets and a human approval gate |
| [53 - FastAPI + LangGraph service](08-capstones/53_fastapi_langgraph_service.ipynb) | Deployable HTTP service: auth, thread isolation, SSE streaming, concurrency, health checks, Docker |

## Verifying your work

Two scripts, neither of which needs an API key:

```powershell
python tools\check_notebooks.py        # JSON validity, syntax, imports, structure
python tools\smoke_lg.py               # runtime behaviour of LangGraph patterns
python tools\smoke_lg2.py
python tools\smoke_lg3.py
python tools\smoke_capstones.py        # capstone patterns, including the FastAPI service
```

`tools/nbgen.py` regenerates every `.ipynb` from the percent-format sources in
`tools/notebook_sources/`. **Edit the `.py` source and regenerate** rather than
editing notebooks directly, if you want your changes to survive.

## Known rough edges

- With **no** provider configured at all, `get_embeddings()` falls back to a
  deterministic fake. Everything still runs, but similarity scores are
  meaningless. `describe_environment()` says so explicitly, and notebook 00 has
  a cell that proves whether your embeddings are real. Set `GROQ_API_KEY` and
  this does not arise.
- Groq has no vision model in the free tier, so notebook 22 needs OpenAI,
  OpenRouter or Anthropic for the image cells.
- LangGraph Studio and LangGraph Platform cells in notebooks 45 and 46 are
  explained but not executed - they need a LangSmith account.

## Licence and provenance

Original material, written against the official LangChain and LangGraph
documentation and reference repositories. See
[references/official-sources.md](references/official-sources.md) for the full
source list.
