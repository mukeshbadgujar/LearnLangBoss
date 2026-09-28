# System design prompts

Talk through the design in this order: what you store in state, how control flows, what you save to disk, how you catch wrong answers, and what it costs. Open the linked chapters when the interviewer wants more detail.

## Contents

- [51 — Policy RAG design](#51--policy-rag-design)
- [52 — Multi-agent research desk](#52--multi-agent-research-desk)
- [53 — HTTP service in front of a graph](#53--http-service-in-front-of-a-graph)
- [Why not just one prompt? (29, 43)](#why-not-just-one-prompt-29-43)
- [Chunking and how you prove retrieval worked (10, 13, 26)](#chunking-and-how-you-prove-retrieval-worked-10-13-26)
- [When you refuse a multi-agent design (29, 41, 52)](#when-you-refuse-a-multi-agent-design-29-41-52)
- [How you prove users cannot read each other's threads (53, 34)](#how-you-prove-users-cannot-read-each-others-threads-53-34)
- [Migrating a graph that has paused threads (46, 35)](#migrating-a-graph-that-has-paused-threads-46-35)
- [Human approval versus automatic guardrails (36, 20a)](#human-approval-versus-automatic-guardrails-36-20a)
- [What a revise loop costs (48, 48a, 47, 45)](#what-a-revise-loop-costs-48-48a-47-45)
- [Where long-term memory lives versus the checkpoint (49, 34, 06)](#where-long-term-memory-lives-versus-the-checkpoint-49-34-06)
- [Deep agent: when the package, when primitives, where files live (50, 50a)](#deep-agent-when-the-package-when-primitives-where-files-live-50-50a)
- [Code execution in production (16a, 20a)](#code-execution-in-production-16a-20a)
- [MCP: when tools live in another process (16b, 16)](#mcp-when-tools-live-in-another-process-16b-16)
- [Topic index](#topic-index)

## 51 — Policy RAG design

**What they are asking.** Answers based on a handbook, with citations, and a clear refusal when the docs do not cover the question.

**Walkthrough.**

1. Split the policy into chunks. Give each chunk a stable id and section metadata. Embed and store locally (FAISS or Chroma). Filter by document version.
2. Turn follow-up questions into standalone queries. Retrieve (rerank if useful). Put chunk ids in the prompt and require the model to cite them.
3. After generate, check grounding and citation validity. If chunks are irrelevant or the answer invents policy, refuse or escalate. Do not fill gaps from memory. That branch is reflective RAG, not a longer prompt.
4. Save the thread with a checkpointer so "what about contractors?" still works. Cap history without breaking tool-call pairs.
5. Escalate when retrieval is thin or checks fail. A correct refusal is a product feature, not a soft fail.
6. Keep a small set of questions with expected chunk ids, plus known unanswerables. Log chunk ids with every answer. Protect the hallucinated metric when you change prompts.

**Cost / failure.** Baseline is one embed, one search, one generate. Extra tricks (HyDE, multi-query, graders) add model calls — measure hit rate and refusal quality before you add them. The quiet failure is a confident answer from empty or wrong chunks.

**Chapters.** [15](../theory/15-advanced-rag.md), [48](../theory/48-self-reflective-rag.md), [51](../theory/51-policy-rag.md)

## 52 — Multi-agent research desk

**What they are asking.** Several specialists, a spend budget, and a human before anything goes out.

**Walkthrough.**

1. Start with one agent. Split only when the tool list or the system prompt is the real problem — not because "multi-agent" sounds advanced.
2. If you split, use a supervisor that routes with a typed decision (`Command` / goto), a hop cap, and a shared scratchpad in state. Specialists should not freely call each other unless the loop has a hard bound.
3. Keep budgets and most routing rules in code (`MAX_RESEARCH_CALLS`, `MAX_REVISIONS`), not in the prompt. A recursion limit is a safety net, not the design.
4. Isolate the researcher: only structured findings reach the writer. The reviewer should not see the researcher's raw tool trace.
5. Put the human interrupt on the send/publish node (`interrupt()`), not on every search. Approve-with-changes sends the work back to revise.
6. Compare accuracy and cost to the single-agent version on a fixed question set before you keep the desk. Expect about 5–10× token cost if you are honest.

**Cost / failure.** Extra model calls per hop, reviewer/writer ping-pong, and open revise loops. The failure mode is a desk that costs more and scores worse than one good prompt plus tools.

**Chapters.** [41](../theory/41-multi-agent.md), [36](../theory/36-human-in-the-loop.md), [52](../theory/52-research-desk.md)

## 53 — HTTP service in front of a graph

**What they are asking.** Auth, streaming, deploys, and keeping users apart.

**Walkthrough.**

1. Build the graph once when the process starts. Do not compile it on every request. Health checks never call the model.
2. Build `thread_id` from `user_id` plus conversation id from auth. The client never sends a raw thread id. Another user's id must never open that checkpoint.
3. Missing or foreign conversation lookups return 404, not 403 — do not tell an attacker the resource exists.
4. Handlers are async. Use `ainvoke` or `astream`. SSE sends a final `done` (or `error`) event, stops when the client disconnects, and turns off proxy buffering.
5. Use Postgres as the checkpointer if you run more than one worker. Stamp a graph version in state and migrate old checkpoints when the schema changes. Do not rename a node that still has paused threads.
6. Rate limit and set a token budget per user. Trace with `prompt_version` and `user_id` in metadata. Sample production traces.
7. Tests must prove two users cannot read each other's threads. That test is a product requirement, not a nice-to-have.

**Cost / failure.** Compiling per request burns connection pools and adds latency. SQLite under many workers locks or corrupts who owns a thread. Streaming with no final event leaves the browser hanging. Cross-user leaks usually come from a bad `thread_id` design, not a missing ACL middleware.

**Chapters.** [53](../theory/53-fastapi.md), [46](../theory/46-deployment.md), [35](../theory/35-sqlite-postgres.md)

## Why not just one prompt? (29, 43)

**What they are asking.** Whether you reach for agents and graphs by habit, or only when one transform is not enough.

**Walkthrough.**

1. Name the task: one transform, retrieve-then-generate, a branching workflow, or an open tool loop. That choice drives the architecture.
2. Use one prompt when input maps to output in a fixed way and facts live in the model or the request. No retrieval, no pause, no resume.
3. Add retrieval when facts live outside the model and must be cited or versioned (policy, product docs).
4. Add a graph when the path branches, loops, must resume after a human, or must show which step failed.
5. Add agents when the next tool depends on the last result and you cannot write the branch table ahead of time.
6. Hybrid is common: a graph owns control flow; an agent node owns one open-ended step with a call budget.
7. Refuse "just make it an agent" when the steps are already known — you pay extra model calls and invent a loop you did not need.

**Cost / failure.** One prompt cannot retrieve, cannot pause for a human, cannot remember across processes, and cannot show which step failed. An agent on a fixed pipeline burns tokens choosing tools you already know.

**Chapters.** [29](../theory/29-graphs-vs-agents.md), [43](../theory/43-hybrid.md)

## Chunking and how you prove retrieval worked (10, 13, 26)

**What they are asking.** Whether chunk size is a guess, or a measured gate before you trust generation.

**Walkthrough.**

1. First split by structure (headers → section metadata). Then split by size (`RecursiveCharacterTextSplitter` / tiktoken) with about 10–20% overlap so facts on boundaries survive.
2. Measure chunk size in tokens for dense or multilingual text. Drop tiny noise chunks (lone headings) that pollute the index.
3. Give every chunk a stable id. Copy parent metadata (`section`, `source`, version) onto children — you need this for citations and filters.
4. Prove chunk quality with a self-containment set: question → keyphrases that must appear in one chunk alone. Tune size and overlap on that score.
5. Prove retrieval with a frozen question set and expected chunk ids (or sections). Score hit rate at k before you touch the generator.
6. Optionally filter by metadata (version, tenant) so similarity cannot return the wrong handbook edition.
7. Only then wire generate. If the right chunk was never retrieved, the answer cannot be grounded — fix the index first, not the prompt.
8. Regression: every prompt or splitter change re-runs the retrieval eval. Generation eval is a separate experiment.

**Cost / failure.** Chunks too large mush topics into one vague vector. Chunks too small cut a fact from its condition. Blind generation after bad retrieval ships confident wrong policy. Extra embed/search cost is cheap next to a wrong answer in production.

**Chapters.** [10](../theory/10-text-splitters.md), [13](../theory/13-retrievers.md), [26](../theory/26-evaluation.md)

## When you refuse a multi-agent design (29, 41, 52)

**What they are asking.** Whether you can say no to a supervisor plus specialists when a chain or single agent is enough.

**Walkthrough.**

1. List the steps. If they are known in advance, draw edges (chain or graph) — do not invent a router that picks among fixed stages.
2. Prefer one agent with a small tool list when the only unknown is which tool comes next, and a hop/call cap can bound it.
3. Split only when tool lists collide, system prompts fight, or one specialist must hide its working context from another.
4. If you still want multi-agent, demand a baseline: same questions, single agent or single prompt plus tools, measured accuracy and tokens.
5. Put budgets in code (`MAX_*`, recursion limit). Prompt-only "please stop looping" is not a control.
6. Put the human gate on irreversible publish/send, not on every hop — otherwise you built expensive babysitting.
7. Refuse the design in the interview when they cannot name what fails in the single-agent version. Multi-agent is a remedy, not a default.

**Cost / failure.** Supervisor hops and reviewer/writer ping-pong often cost 3–10× with equal or worse quality. Unbounded handoffs are the classic outage: token burn until the recursion limit.

**Chapters.** [29](../theory/29-graphs-vs-agents.md), [41](../theory/41-multi-agent.md), [52](../theory/52-research-desk.md)

## How you prove users cannot read each other's threads (53, 34)

**What they are asking.** Isolation as a primary-key design, not a permission check bolted on later.

**Walkthrough.**

1. Authenticate to a `user_id`. Never trust a client-supplied raw `thread_id`.
2. Derive the checkpoint key as `thread_id_for(user_id, conversation_id)` so Alice and Bob with the same conversation id get different keys.
3. Load checkpoints only through that derived id. Bob presenting Alice's conversation id finds empty or foreign state.
4. Return 404 on missing or foreign lookups, not 403, so you do not confirm existence to an attacker.
5. Use a durable shared checkpointer (Postgres under multi-worker). In-memory savers die on deploy and cannot prove isolation across processes.
6. Write an in-process test: create Alice's thread, attempt Bob's GET/stream with Alice's conversation id, assert 404 and no message bodies.
7. Trace metadata may include `user_id`, but traces are not the ACL — the checkpointer key is.

**Cost / failure.** A shared conversation id as the sole key is a cross-tenant leak. SQLite with multiple workers can scramble who owns which row. Skipping the two-user test means you discover the leak in production support tickets.

**Chapters.** [53](../theory/53-fastapi.md), [34](../theory/34-checkpointing.md)

## Migrating a graph that has paused threads (46, 35)

**What they are asking.** What happens when yesterday's interrupted run resumes against today's code.

**Walkthrough.**

1. Stamp `graph_version` early in the run so every checkpoint knows which shape produced it.
2. On resume, compare stamped version to current code. If they differ, run an explicit `migrate_state` (renames, defaults) before continuing.
3. Treat node rename/remove and reducer or required-key removal as breaking for in-flight threads — paused at the old node name cannot resume.
4. Safe changes: optional new state keys, new nodes off existing paths, prompt-only edits (output may differ; resume still works).
5. For breaking deploys: dual-version drain, or finish/expire paused threads, then ship. Do not hot-rename nodes under open interrupts.
6. Use one checkpointer singleton per process; Postgres when multiple workers must see the same paused threads. SQLite is a single-writer story.
7. Simulate with `update_state` / older checkpoint ids in tests before you deploy the rename.

**Cost / failure.** A deploy that renames an approval node strands every human-in-the-loop ticket. In-memory checkpointers lose paused work on restart. Skipping version stamps makes every resume a guess.

**Chapters.** [46](../theory/46-deployment.md), [35](../theory/35-sqlite-postgres.md)

## Human approval versus automatic guardrails (36, 20a)

**What they are asking.** When to pause a thread for a person, and when middleware should block or redact without waiting.

**Walkthrough.**

1. Draw the irreversible actions (publish, send email, write to prod, move money). Those get `interrupt()` / human-in-the-loop on that node.
2. Put automatic guardrails on every turn: PII middleware on input/output, cheap deterministic injection filters `before_agent`, call limits so attacks cannot spin.
3. Layer cheap checks first (regex/keywords, allow-lists). Use a model judge only for soft topic fences after the agent answers.
4. Guardrails return a refusal or redaction and continue or jump to end — they do not persist a pause. HITL ends the HTTP request and resumes later with the same `thread_id`.
5. Do not put a human on every search or every tool call; that becomes a queue, not a safety system.
6. Do not rely on prompt bans alone for code execution or PII — middleware and sandboxing are the hard controls.
7. Say in the interview: HITL for irreversible side effects; guardrails for content that must never reach the model or the user.

**Cost / failure.** HITL on every hop kills latency and staffing. Guardrails alone cannot approve a nuanced publish decision. Missing call limits turns a jailbreak into a token outage.

**Chapters.** [36](../theory/36-human-in-the-loop.md), [20a](../theory/20a-guardrails.md)

## What a revise loop costs (48, 48a, 47, 45)

**What they are asking.** Whether you treat graders and reflection as extra model calls with caps, not free quality.

**Walkthrough.**

1. Name each grader (relevance, grounded, useful) as its own structured call after retrieve and/or generate.
2. Count the happy path: retrieve + generate. Then add one call per document graded, plus rewrite and regenerate on failure.
3. Cap retries (`MAX_RETRIES`, `MAX_GENERATIONS`). Exhaustion routes to give-up or escalate — never an open while-loop.
4. Prefer adaptive paths: cheap score first; run the expensive reflective branch only when retrieval looks weak.
5. Reflection / revise on drafts (writer ↔ critic) is the same budget math: each round is another full generation plus a judge.
6. Context engineering matters: graders should see only what they need; stuffing full history into every judge call multiplies tokens (47).
7. Trace the loop in LangSmith/Studio: which node fired, how many rewrite hops, tokens per hop — so "quality" is a measured cost curve (45).
8. Ship a correct refusal when evidence is thin; that outcome is cheaper and safer than another revise round that invents grounding.

**Cost / failure.** Uncapped revise loops are a classic bill spike. Parallel grading helps latency but not total tokens. Without traces you cannot tell whether the loop helped or just burned.

**Chapters.** [48](../theory/48-self-reflective-rag.md), [48a](../theory/48a-reflection.md), [47](../theory/47-context.md), [45](../theory/45-langsmith-studio.md)

## Where long-term memory lives versus the checkpoint (49, 34, 06)

**What they are asking.** Thread state versus facts that cross threads, and why legacy chain memory is the wrong answer.

**Walkthrough.**

1. Checkpointer: whole graph state for one `thread_id`, written automatically each superstep. Same id continues the conversation; new id starts empty.
2. Store: facts you deliberately `put` under namespaces (user, org), optionally with semantic search. Survives new conversations.
3. Namespace with tenant/user from auth first so prefix search cannot leak across tenants.
4. Nodes recall from the store on a fresh thread, then extract durable facts and write back — do not save "user asked about France."
5. Deleting a thread does not erase store namespaces unless you also delete them; design TTL and forget tools on purpose.
6. Legacy `ConversationBufferMemory` on the chain object dies with the process and cannot isolate two users — prefer messages in state or a checkpointer (06, 34).
7. Say it cleanly: checkpoint = this conversation's stage; store = profile and preferences across conversations.

**Cost / failure.** Without a store, every new `thread_id` re-asks for the same profile. Putting cross-user facts in the checkpoint couples memory to one chat and leaks when you copy threads carelessly. `InMemoryStore` without TTL/durability is a demo, not production.

**Chapters.** [49](../theory/49-memory-store.md), [34](../theory/34-checkpointing.md), [06](../theory/06-memory.md)

## Deep agent: when the package, when primitives, where files live (50, 50a)

**What they are asking.** Whether "deep agent" means a smarter model, or a harness with plan, files, subagents, and caps.

**Walkthrough.**

1. Use a deep agent when a competent human would need a plan, a notebook, and handoffs — not for a single tool call or a fixed RAG path.
2. Pillars: planning (`write_todos`), workspace files, subagents with isolated histories (parent sees only the final summary), and a procedural harness prompt.
3. Cap model calls with middleware. Without a hard limit, the harness wanders until money runs out.
4. Prefer building from primitives when you must unit-test middleware and file reducers alone; use `create_deep_agent` when you want the bundled harness.
5. Decide the backend: files in graph state (merge reducer), a sandbox directory, or a store that survives threads (50a). That choice is persistence and isolation, not cosmetics.
6. Conversation messages are not storage — large findings go to files; synthesis reads files, not a 100k-token tool dump.
7. Refuse deep agents for FAQ/RAG with known steps: you will pay 5–20× a plain agent for little gain.

**Cost / failure.** Subagents and long plans multiply calls. Wrong backend (state-only files on a multi-worker deploy without a shared store) loses the workspace. Treating "deep" as model quality hides the real bill.

**Chapters.** [50](../theory/50-deep-agents.md), [50a](../theory/50a-backends-skills.md)

## Code execution in production (16a, 20a)

**What they are asking.** How you keep a model from running arbitrary Python on your host.

**Walkthrough.**

1. Prefer narrow tools: allow-listed `calculate`, typed DataFrame aggregations — never `exec` on model text by default.
2. If you must offer a REPL, sandbox it: separate process/container, no host FS, no network, allow-listed imports, CPU/memory/time limits.
3. Wire with `create_agent` plus `ModelCallLimitMiddleware`; treat legacy `AgentExecutor` demos as historical.
4. Add guardrails: block dangerous patterns on the way in; audit every snippet; human approval for writes that leave the sandbox (20a, 36).
5. Route specialists (calculator vs tickets) instead of one mega-agent that always has a REPL tool available.
6. Prompt bans ("do not import os") are soft. Containers and empty `__builtins__` are hard.
7. Say you would not ship notebook-style "write files into the working directory" patterns to production.

**Cost / failure.** Unsandboxed REPL reads secrets, calls the network, and writes the disk. Unbounded model/tool loops turn one prompt injection into a compute bill. Soft prompt policies fail the first determined user.

**Chapters.** [16a](../theory/16a-code-execution.md), [20a](../theory/20a-guardrails.md)

## MCP: when tools live in another process (16b, 16)

**What they are asking.** In-process `@tool` versus tools discovered over MCP from another team or language.

**Walkthrough.**

1. Use in-process `@tool` / `BaseTool` for agent-private helpers you own and deploy with the app (16).
2. Use MCP when another team owns the capability, you need process isolation, or the tool server is not Python.
3. Server: FastMCP exposes `@mcp.tool()` over stdio (local subprocess) or streamable-http (shared service).
4. Client: `MultiServerMCPClient` discovers tools; `get_tools()` yields LangChain tools for `create_agent`.
5. Prefer stdio for single-host demos; HTTP when multiple clients or remote services share one tool process.
6. Still apply call limits, auth to the tool server, and the same refusal rules you use for local tools — MCP is a transport, not a trust boundary by itself.
7. Adding a new MCP server should not require rewriting agent graph code — only client config — if discovery is done right.

**Cost / failure.** Extra hop latency and operational ownership of the tool process. stdio servers tied to one agent process do not scale like an HTTP tool service. Treating MCP tools as automatically safe skips the same sandbox and auth reviews as local tools.

**Chapters.** [16b](../theory/16b-mcp.md), [16](../theory/16-tools.md)

## Topic index

Every curriculum id and which interview file owns it.

| Id | Topic | Interview file |
|---|---|---|
| 00 | Environment | [langchain.md](langchain.md) |
| 01 | Models and messages | [langchain.md](langchain.md) |
| 02 | Prompt templates | [langchain.md](langchain.md) |
| 03 | Chat prompts | [langchain.md](langchain.md) |
| 04 | Few-shot | [langchain.md](langchain.md) |
| 05 | Output parsers | [langchain.md](langchain.md) |
| 06 | Memory | [langchain.md](langchain.md) |
| 07 | LCEL | [langchain.md](langchain.md) |
| 08 | Chains | [langchain.md](langchain.md) |
| 09 | Document loaders | [langchain.md](langchain.md) |
| 10 | Text splitters | [langchain.md](langchain.md) |
| 11 | Embeddings | [langchain.md](langchain.md) |
| 12 | Vector stores | [langchain.md](langchain.md) |
| 13 | Retrievers | [langchain.md](langchain.md) |
| 14 | Retrieval chains | [langchain.md](langchain.md) |
| 15 | Advanced RAG | [langchain.md](langchain.md) |
| 16 | Tools | [langchain.md](langchain.md) |
| 16a | Code execution | [langchain.md](langchain.md) |
| 16b | MCP | [langchain.md](langchain.md) |
| 17 | Agents | [langchain.md](langchain.md) |
| 17a | Agent loop | [langchain.md](langchain.md) |
| 18 | Graphs over agents | [langchain.md](langchain.md) |
| 19 | LangSmith | [langsmith-production.md](langsmith-production.md) |
| 20 | Callbacks | [langsmith-production.md](langsmith-production.md) |
| 20a | Guardrails | [langsmith-production.md](langsmith-production.md) |
| 21 | Streaming | [langsmith-production.md](langsmith-production.md) |
| 22 | Multimodal | [langchain.md](langchain.md) |
| 23 | Caching | [langchain.md](langchain.md) |
| 24 | Structured outputs | [langchain.md](langchain.md) |
| 25 | Routing | [langchain.md](langchain.md) |
| 26 | Evaluation | [langsmith-production.md](langsmith-production.md) |
| 26a | Unit testing | [langsmith-production.md](langsmith-production.md) |
| 27 | Usage tracking | [langsmith-production.md](langsmith-production.md) |
| 27a | Gateways | [langsmith-production.md](langsmith-production.md) |
| 28 | Hub | [langchain.md](langchain.md) |
| 29 | Graphs vs agents | [langgraph.md](langgraph.md) |
| 30 | State schemas | [langgraph.md](langgraph.md) |
| 31 | StateGraph | [langgraph.md](langgraph.md) |
| 32 | Conditional routing | [langgraph.md](langgraph.md) |
| 33 | Compile, invoke, stream | [langgraph.md](langgraph.md) |
| 34 | Checkpointing | [langgraph.md](langgraph.md) |
| 35 | SQLite and Postgres | [langgraph.md](langgraph.md) |
| 36 | Human-in-the-loop | [langgraph.md](langgraph.md) |
| 37 | Message history | [langgraph.md](langgraph.md) |
| 38 | Fan-out / fan-in | [langgraph.md](langgraph.md) |
| 39 | Time travel | [langgraph.md](langgraph.md) |
| 40 | Subgraphs | [langgraph.md](langgraph.md) |
| 41 | Multi-agent | [langgraph.md](langgraph.md) |
| 42 | Custom streams | [langgraph.md](langgraph.md) |
| 43 | Hybrid | [langgraph.md](langgraph.md) |
| 43a | Workflows | [langgraph.md](langgraph.md) |
| 44 | Retries | [langgraph.md](langgraph.md) |
| 45 | LangSmith Studio | [langsmith-production.md](langsmith-production.md) |
| 46 | Deployment | [langsmith-production.md](langsmith-production.md) |
| 47 | Context | [langgraph.md](langgraph.md) |
| 48 | Self-reflective RAG | [langgraph.md](langgraph.md) |
| 48a | Reflection | [langgraph.md](langgraph.md) |
| 49 | Memory store | [langgraph.md](langgraph.md) |
| 50 | Deep agents | [langgraph.md](langgraph.md) |
| 50a | Backends and skills | [langgraph.md](langgraph.md) |
| 51 | Policy RAG design | [system-design.md](system-design.md) |
| 52 | Research desk design | [system-design.md](system-design.md) |
| 53 | FastAPI graph service | Design walkthrough: [system-design.md](system-design.md). Production questions: [langsmith-production.md](langsmith-production.md) |
