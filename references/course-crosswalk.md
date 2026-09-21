# Course Crosswalk

Three public course repositories and the `LLG/` archive of purchased course
material were read end to end and mapped against this curriculum. This file
answers two questions:

1. **Where does each external asset land in our learning path?** So when you
   watch a video or read a branch, you know which notebook it reinforces.
2. **What did they teach that we do not?** So the gaps are explicit instead of
   discovered six months later in an interview.

Reproduce the clones with:

```powershell
mkdir .upstream; cd .upstream
git clone https://github.com/emarco177/langchain-course
git clone https://github.com/emarco177/langgraph-course
git clone https://github.com/krishnaik06/Deep-agents-With-Langchain
```

Both `emarco177` repos keep one project per branch and one lesson per commit, so
`git log --oneline` on a `project/*` branch is the lesson list. `.upstream/` and
`LLG/_extracted/` are gitignored.

---

## 1. What was analysed

### emarco177/langchain-course — 9 project branches

| Branch | Contents | Our lesson |
|---|---|---|
| `project/hello-world` | `init_chat_model`, first invoke | 00, 01 |
| `project/search-agent` | `create_agent` + Tavily | 16, 17 |
| `project/ReAct-search-agent` | `create_agent` with `response_format=AgentResponse` | 17, 24 |
| `project/ReAct-Algo` | hand-rolled `bind_tools` loop + `BaseCallbackHandler` | 17, 20 |
| `project/agents-under-the-hood` | the same agent at three abstraction levels | **gap — see 17a** |
| `project/agent-harnesses` | `create_deep_agent` knob by knob (01–05) | 50, **gap — see 50a** |
| `project/rag-gist` | ingest a blog post, `create_retrieval_chain` | 09–14 |
| `project/chat-wth-your-pdf` | `PyPDFLoader` → FAISS → retrieval chain | 09, 12, 14 |
| `project/code-interpreter` | `PythonREPLTool` + CSV agent + router agent | **gap — see 16a** |

### emarco177/langgraph-course — 6 project branches

| Branch | Contents | Our lesson |
|---|---|---|
| `project/ReAct-agent` | ReAct loop as an explicit graph | 18, 29 |
| `project/ReAct-Agent-Function-Calling` | same, via native tool calling | 18, 31 |
| `project/agentic-rag` | router → retrieve → grade → websearch → generate, plus graders and a `pytest` suite | 15, 48, **gap — 26a testing** |
| `project/reflection` / `project/reflection-agent` | generate ↔ critique loop over `MessagesState` | **gap — see 48a** |
| `project/reflexion-agent` | structured self-critique (`AnswerQuestion` / `ReviseAnswer`) driving tool-executed research | **gap — see 48a** |
| `project/search-agent` | Tavily agent | 16 |

The `agentic-rag` branch is the closest match to anything we built — its graders
map almost one for one onto notebook 48's relevance, grounding and usefulness
graders. What it adds is **repo layout** (`graph/chains/`, `graph/nodes/`,
`graph/state.py`, `graph/consts.py`) and a **`pytest` suite that unit-tests each
grader**, which we only do for the FastAPI capstone.

### krishnaik06/Deep-agents-With-Langchain — `main` only

| Asset | Contents | Our lesson |
|---|---|---|
| `1-basicsdeepagent.ipynb` | `create_deep_agent`, `result["files"]` | 50 |
| `2-contextengineering.ipynb` | `AGENTS.md` context file, backends, `skills=` | **gap — see 50a** |
| `3-backends.ipynb` | `StateBackend` / `FilesystemBackend` / `StoreBackend` compared | **gap — see 50a** |
| `4-subagents.ipynb` | subagents + structured output from a subagent | 41, 50 |
| `skills/*/SKILL.md` | progressive-disclosure skill files | **gap — see 50a** |
| `streamlit_app.py` | Streamlit front end | 53 covers FastAPI instead |

### LLG/ — purchased course archive

| Asset | Contents | Our lesson |
|---|---|---|
| `1-Basics+Of+Langchain.zip` | models, ingestion, splitters, embeddings, FAISS, Chroma | 01, 09–12 |
| `LCEL.zip` | LCEL chain + **LangServe** `add_routes` + Streamlit client | 07, 53 (we use plain FastAPI) |
| `langchainupdated.zip` | intro, models, tools, messages, structured output, **middleware** | 01, 03, 16, 24, 36 |
| `4-+Workflows.zip` | prompt chaining, parallelization, routing, **orchestrator-worker**, **evaluator-optimizer** | 32, 38, 43, **gap — see 43a** |
| `5-HumanintheLoop.zip` | interrupt / resume | 36 |
| `1-simplegraph`, `3-DataclassStateSchema`, `4-pydantic` | state schema variants | 30 |
| `5-ChainsLangGraph`, `1-chatbots`, `2-chatbot`, `6-chatbotswithmultipletools` | graph basics and tool chatbots | 29–32 |
| `1-streaming.ipynb` | stream modes | 33, 42 |
| `7-ReActAgents.ipynb` | ReAct in LangGraph | 18 |
| `1-AgenticRAG`, `2-CorrectiveRAG`, `4-AdaptiveRAG` | CRAG and adaptive routing | 15, 48 |
| `mcplanggraph.zip` | **FastMCP servers (stdio + streamable-http) + `MultiServerMCPClient`** | **gap — see 16b** |
| `langchain_guardrails_crash_course.ipynb` | **`PIIMiddleware`, before/after-agent hooks, layered guardrails** | **gap — see 20a** |
| `llm_gateway_tutorial.ipynb` | **LiteLLM gateway: fallbacks, routing, load balancing, cost** | **gap — see 27a** |
| `AINEWSAgentic.zip`, `BAsicChatbot.zip`, `bloggeneration.zip`, `Chatbot_with_Web.zip` | Streamlit + `src/langgraphagenticai/` project layout | 51–53 |
| `Complete-Python-Bootcamp-main.zip` | Python prerequisites | prerequisite, not in scope |
| `deepagents.pdf`, `claude.pdf`, `claudecode.zip`, `mcp.pdf` | slide decks for the above | reading material |

---

## 2. Gaps — closed

Nine topics appeared in the external material and were absent (or only
mentioned) in the original 54 notebooks. Each is now a letter-suffixed lesson
sitting next to the notebook it extends. Status below is current as of the
crosswalk implementation.

| New | Folder | Title | Status | Sourced from |
|---|---|---|---|---|
| `16a` | 03-langchain-agents | Code Execution Tools and Sandboxing | **Added** | `langchain-course:project/code-interpreter` |
| `16b` | 03-langchain-agents | MCP: Model Context Protocol Servers and Clients | **Added** | `LLG/mcplanggraph` |
| `17a` | 03-langchain-agents | The Agent Loop From Scratch | **Added** | `langchain-course:project/agents-under-the-hood`, `project/ReAct-Algo` |
| `20a` | 04-langchain-production | Guardrails: PII, Injection and Safety Middleware | **Added** | `LLG/langchain_guardrails_crash_course.ipynb` |
| `26a` | 04-langchain-production | Unit-Testing Chains, Graders and Graphs | **Added** | `langgraph-course:project/agentic-rag` tests |
| `27a` | 04-langchain-production | LLM Gateways: Fallbacks, Routing and Load Balancing | **Added** | `LLG/llm_gateway_tutorial.ipynb` |
| `43a` | 07-langgraph-advanced | The Five Workflow Patterns | **Added** | `LLG/4-Workflows` |
| `48a` | 07-langgraph-advanced | Reflection and Reflexion Agents | **Added** | `langgraph-course:project/reflection-agent`, `project/reflexion-agent` |
| `50a` | 07-langgraph-advanced | Deep Agent Backends, Skills and AGENTS.md | **Added** | `langchain-course:project/agent-harnesses`, `Deep-agents-With-Langchain` |

Curriculum size: **54 → 63 notebooks**. Next-link chain and
[CHECKLIST.md](../CHECKLIST.md) / [LEARNING_PATH.md](../LEARNING_PATH.md) are
updated.

## 3. Upgrades to existing notebooks

Smaller than a new lesson. Applied:

| Lesson | Change | Status |
|---|---|---|
| 12 Vector Stores | `FAISS.save_local` / `load_local` and `allow_dangerous_deserialization` | Already present |
| 15 Advanced RAG | Name CRAG / Adaptive RAG / Self-RAG and point at notebooks 25 and 48 | **Done** |
| 17 Agents | Cross-link to `17a`; mention prompt-printing callbacks | **Done** |
| 24 Structured Outputs | `create_agent(..., response_format=...)` | Already present |
| 41 Multi-Agent Handoffs | Structured output from a subagent; only final message returns | **Done** |
| 46 Deployment | LangServe historical note vs FastAPI / LangGraph Platform | **Done** |
| 53 FastAPI Capstone | Project-layout alternatives + optional Streamlit client | **Done** |

## 4. What we already do better

Worth knowing so you do not rewrite what you have.

- **Provider independence.** Every external repo hardcodes `ChatOpenAI` or an
  OpenAI model string. `shared/llm.py` resolves Groq → OpenRouter → OpenAI →
  Anthropic and does the same for embeddings.
- **Runnable without keys.** Our smoke tests exercise the non-LLM logic with
  fake models. None of the courses can run offline.
- **Current APIs.** `project/code-interpreter` still uses `AgentExecutor` and
  `create_react_agent` from `langchain.agents`; `chat-wth-your-pdf` hardcodes an
  API key in source. We target LangChain v1 throughout.
- **Explicit failure modes.** The courses show the happy path. Our notebooks
  spend real space on what breaks — recursion limits, silent state drops,
  `InMemoryStore` TTL, supervisor loops.

## 5. How to study alongside those courses

| When you watch / read… | Run our notebook… | Then deepen with… |
|---|---|---|
| Eden Marco hello-world / search-agent | 00, 01, 16, 17 | 17a |
| Eden Marco code-interpreter | 16a | 20a (sandbox + safety) |
| Eden Marco agents-under-the-hood | 17a | 18 |
| Eden Marco agentic-RAG | 15, 48 | 26a, 53 project layout |
| Eden Marco reflection / reflexion | 48a | 43a evaluator-optimizer |
| Eden Marco / Kris Naik deep agents | 50 | 50a |
| LLG workflows zip | 43a | 32, 38, 43 |
| LLG MCP zip | 16b | 16 |
| LLG guardrails notebook | 20a | 36 (HITL) |
| LLG LiteLLM gateway | 27a | 27 |
| Kris Naik Streamlit front ends | 53 (FastAPI first) | optional Streamlit client |
