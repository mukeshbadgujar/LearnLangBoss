# Learning Path

Two routes through the same 63 notebooks. Pick the one that matches your
situation, and track progress in [CHECKLIST.md](CHECKLIST.md).

- **[Route A - full mastery](#route-a--full-mastery)**: study in order,
beginner to expert. Roughly 70-80 hours.
- **[Route B - the 5-week work sprint](#route-b--the-5-week-work-sprint)**:
learn what you need to be productive on the job now, and fill in the depth
afterwards.

Letter-suffixed notebooks (`16a`, `20a`, …) sit next to the lesson they
extend. They come from the [course crosswalk](references/course-crosswalk.md)
against Eden Marco, Kris Naik, and the `LLG/` archive.

---

## Route A — full mastery

### Track 00 — Setup (1 notebook, ~1 hour)


| #   | Notebook                                                                            | You will be able to                       |
| --- | ----------------------------------------------------------------------------------- | ----------------------------------------- |
| 00  | [environment, providers and keys](00-setup/00_environment_providers_and_keys.ipynb) | Run every other notebook, with tracing on |




### Track 01 — LangChain foundations (8 notebooks, ~8 hours)


| #   | Notebook                                                                                           | You will be able to                                              |
| --- | -------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| 01  | [models and messages](01-langchain-foundations/01_models_messages.ipynb)                           | Hold a conversation, read `finish_reason` and token usage        |
| 02  | [prompt templates](01-langchain-foundations/02_prompt_templates.ipynb)                             | Build reusable prompts with validated inputs                     |
| 03  | [chat prompt templates](01-langchain-foundations/03_chat_prompt_templates.ipynb)                   | Manage multi-turn history with `MessagesPlaceholder`             |
| 04  | [few-shot and example selectors](01-langchain-foundations/04_few_shot_and_example_selectors.ipynb) | Teach a model a format by example, and pick examples dynamically |
| 05  | [output parsers](01-langchain-foundations/05_output_parsers.ipynb)                                 | Get structured data out, and recover when parsing fails          |
| 06  | [memory](01-langchain-foundations/06_memory.ipynb)                                                 | Understand the old memory classes and their modern replacements  |
| 07  | [LCEL](01-langchain-foundations/07_lcel.ipynb)                                                     | Compose, batch, stream, retry and fall back                      |
| 08  | [chains, sequential and custom](01-langchain-foundations/08_chains_sequential_and_custom.ipynb)    | Build multi-step pipelines with your own Python in the middle    |




### Track 02 — Retrieval-augmented generation (7 notebooks, ~10 hours)

Running example: **chat with the company handbook and leave policy**.


| #   | Notebook                                                                   | You will be able to                                             |
| --- | -------------------------------------------------------------------------- | --------------------------------------------------------------- |
| 09  | [document loaders](02-langchain-rag/09_document_loaders.ipynb)             | Load text, CSV, PDF, web and directories with useful metadata   |
| 10  | [text splitters](02-langchain-rag/10_text_splitters.ipynb)                 | Choose a chunking strategy and measure whether it worked        |
| 11  | [embeddings and caching](02-langchain-rag/11_embeddings_and_caching.ipynb) | Explain what embeddings do and do not capture; cache them       |
| 12  | [vector stores](02-langchain-rag/12_vector_stores.ipynb)                   | Use FAISS and Chroma locally; filter on metadata; persist       |
| 13  | [retrievers](02-langchain-rag/13_retrievers.ipynb)                         | MMR, multi-query, compression, BM25, ensemble, parent-document  |
| 14  | [retrieval chains](02-langchain-rag/14_retrieval_chains.ipynb)             | Answer follow-up questions correctly                            |
| 15  | [advanced RAG](02-langchain-rag/15_advanced_rag.ipynb)                     | HyDE, step-back, decomposition, RRF, reranking - and their cost |




### Track 03 — Agents and tools (6 notebooks, ~9 hours)

Running example: **an ops assistant over tickets, a calculator and docs**.


| #   | Notebook                                                                                         | You will be able to                                               |
| --- | ------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------- |
| 16  | [tools, built-in and custom](03-langchain-agents/16_tools_builtin_custom_toolkits.ipynb)         | Write tools with precise schemas; handle tool errors              |
| 16a | [code execution and sandboxing](03-langchain-agents/16a_code_execution_and_sandboxing.ipynb)    | Run `PythonREPLTool` safely; know the blast radius                |
| 16b | [MCP servers and clients](03-langchain-agents/16b_mcp_servers_and_clients.ipynb)                | Expose and consume tools over stdio / streamable-http             |
| 17  | [agents](03-langchain-agents/17_agents.ipynb)                                                    | Use `create_agent`, control the loop with middleware              |
| 17a | [the agent loop from scratch](03-langchain-agents/17a_agent_loop_from_scratch.ipynb)            | Rebuild the loop at three abstraction levels                      |
| 18  | [from AgentExecutor to graphs](03-langchain-agents/18_from_agentexecutor_to_graphs.ipynb)        | Explain exactly what LangGraph adds over an agent                 |




### Track 04 — Production LangChain (13 notebooks, ~16 hours)


| #   | Notebook                                                                            | You will be able to                                             |
| --- | ----------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| 19  | [LangSmith](04-langchain-production/19_langsmith.ipynb)                             | Trace, debug, build datasets, evaluate                          |
| 20  | [callbacks](04-langchain-production/20_callbacks.ipynb)                             | Hook every lifecycle event; track cost and tokens               |
| 20a | [guardrails, PII and safety](04-langchain-production/20a_guardrails_pii_and_safety.ipynb) | Redact PII, block injection, fence unsafe topics          |
| 21  | [streaming](04-langchain-production/21_streaming.ipynb)                             | Stream tokens, steps and events - and know what kills streaming |
| 22  | [multimodal](04-langchain-production/22_multimodal.ipynb)                           | Send images and extract structured data from them               |
| 23  | [caching](04-langchain-production/23_caching.ipynb)                                 | Exact, semantic and provider prompt caching                     |
| 24  | [structured outputs](04-langchain-production/24_structured_outputs.ipynb)           | `.with_structured_output()`, strategies, and schema design      |
| 25  | [routing and handoffs](04-langchain-production/25_routing_and_handoffs.ipynb)       | Keyword, semantic, LLM and cascade routing                      |
| 26  | [evaluation](04-langchain-production/26_evaluation.ipynb)                           | Measure retrieval and generation; LLM-as-judge; the RAG triad   |
| 26a | [unit-testing chains and graphs](04-langchain-production/26a_unit_testing_chains_and_graphs.ipynb) | pytest graders and routers in isolation               |
| 27  | [usage tracking](04-langchain-production/27_usage_tracking.ipynb)                   | Attribute cost per user and enforce budgets                     |
| 27a | [LLM gateways](04-langchain-production/27a_llm_gateways.ipynb)                      | Fallbacks, routing and load balancing at the gateway            |
| 28  | [LangChain Hub](04-langchain-production/28_langchain_hub.ipynb)                     | Pull, pin and version prompts                                   |




### Track 05 — LangGraph beginner (4 notebooks, ~6 hours)

Running example: **a support ticket router**.


| #   | Notebook                                                                             | You will be able to                                        |
| --- | ------------------------------------------------------------------------------------ | ---------------------------------------------------------- |
| 29  | [graphs vs agents](05-langgraph-beginner/29_graphs_vs_agents.ipynb)                  | Say when a graph is the right tool and when it is overkill |
| 30  | [state schemas](05-langgraph-beginner/30_state_schemas.ipynb)                        | Use `TypedDict`, Pydantic, reducers and `add_messages`     |
| 31  | [StateGraph, nodes and edges](05-langgraph-beginner/31_stategraph_nodes_edges.ipynb) | Build and debug graphs; avoid silently-dropped state keys  |
| 32  | [conditional routing](05-langgraph-beginner/32_conditional_routing.ipynb)            | Branch, loop safely, and route with `Command`              |




### Track 06 — LangGraph intermediate (6 notebooks, ~9 hours)


| #   | Notebook                                                                                        | You will be able to                                       |
| --- | ----------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| 33  | [compile, invoke, stream](06-langgraph-intermediate/33_compile_invoke_stream.ipynb)             | Use all seven stream modes deliberately                   |
| 34  | [checkpointing and thread ids](06-langgraph-intermediate/34_checkpointing_thread_ids.ipynb)     | Inspect, edit and resume conversation state               |
| 35  | [SQLite and Postgres savers](06-langgraph-intermediate/35_sqlite_postgres_savers.ipynb)         | Make state survive a restart and scale to many replicas   |
| 36  | [human in the loop](06-langgraph-intermediate/36_human_in_the_loop.ipynb)                       | Pause for approval, edit, and resume                      |
| 37  | [message history and deletion](06-langgraph-intermediate/37_message_history_and_deletion.ipynb) | Prune history without breaking tool-call pairs            |
| 38  | [parallel fan-out and fan-in](06-langgraph-intermediate/38_parallel_fanout_fanin.ipynb)         | Run branches concurrently; `Send`; handle partial failure |




### Track 07 — LangGraph advanced (15 notebooks, ~23 hours)


| #   | Notebook                                                                                              | You will be able to                                       |
| --- | ----------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| 39  | [time travel](07-langgraph-advanced/39_time_travel.ipynb)                                             | Replay and fork from any past checkpoint                  |
| 40  | [subgraphs](07-langgraph-advanced/40_subgraphs.ipynb)                                                 | Compose graphs with shared or translated state            |
| 41  | [multi-agent handoffs](07-langgraph-advanced/41_multi_agent_handoffs.ipynb)                           | Supervisor, network and hierarchical architectures        |
| 42  | [custom stream channels](07-langgraph-advanced/42_custom_stream_channels.ipynb)                       | Emit your own progress events, all the way to SSE         |
| 43  | [hybrid deterministic + LLM](07-langgraph-advanced/43_hybrid_deterministic_llm.ipynb)                 | Let the model interpret and the code decide               |
| 43a | [five workflow patterns](07-langgraph-advanced/43a_five_workflow_patterns.ipynb)                     | Chaining, parallel, routing, orchestrator-worker, evaluator-optimizer |
| 44  | [errors, retries, tool failures](07-langgraph-advanced/44_error_retries_tool_failures.ipynb)          | `RetryPolicy`, error handlers, timeouts, circuit breakers |
| 45  | [LangSmith and Studio](07-langgraph-advanced/45_langsmith_and_studio.ipynb)                           | Trace/debug graphs; LangSmith use cases; Studio           |
| 46  | [deployment and versioning](07-langgraph-advanced/46_deployment_and_versioning.ipynb)                 | Ship a graph and migrate in-flight threads                |
| 47  | [token limits and summarisation](07-langgraph-advanced/47_token_limits_and_summarization_nodes.ipynb) | Choose between stuff, map-reduce, refine and retrieval    |
| 48  | [self-reflective RAG](07-langgraph-advanced/48_self_reflective_rag.ipynb)                             | Grade documents and answers; CRAG; adaptive reflection    |
| 48a | [reflection and reflexion](07-langgraph-advanced/48a_reflection_and_reflexion.ipynb)                 | Critique drafts; drive research from structured critique  |
| 49  | [long-term semantic memory](07-langgraph-advanced/49_long_term_semantic_memory.ipynb)                 | Remember across conversations, and forget responsibly     |
| 50  | [deep agents](07-langgraph-advanced/50_deep_agents.ipynb)                                             | Planning, workspace, sub-agents and a real harness        |
| 50a | [deep agent backends and skills](07-langgraph-advanced/50a_deep_agent_backends_and_skills.ipynb)     | Backends, `SKILL.md`, and `AGENTS.md` standing context    |




### Track 08 — Capstones (3 notebooks, ~8 hours)


| #   | Notebook                                                                       | You will have built                                          |
| --- | ------------------------------------------------------------------------------ | ------------------------------------------------------------ |
| 51  | [policy RAG chatbot](08-capstones/51_policy_rag_chatbot.ipynb)                 | A grounded, cited, escalating assistant with an eval harness |
| 52  | [multi-agent research desk](08-capstones/52_multi_agent_research_desk.ipynb)   | A supervised desk with budgets and a human gate              |
| 53  | [FastAPI + LangGraph service](08-capstones/53_fastapi_langgraph_service.ipynb) | A deployable service with auth, streaming and tests          |


---



## Route B — the 5-week work sprint

For learning this alongside a job, where you need to be useful quickly. The
remaining notebooks are still worth doing - just afterwards, as depth.

### Week 1 — Get running

**Notebooks:** `00`

Goal: every subsequent notebook runs without fighting the environment, and you
can see your traces in LangSmith.

**Done when:** `describe_environment()` lists a provider, and a trace from
notebook 00 appears in your LangSmith project.

### Week 2 — LangChain fundamentals

**Notebooks:** `01`-`08`

Goal: build and compose chains fluently. LCEL (`07`) is the notebook that makes
everything afterwards feel easy; do not skim it.

**Done when:** you can write a three-step chain with structured output and a
fallback, without looking anything up.

### Week 3 — RAG and your first graphs

**Notebooks:** `09`-`15`, then `29`-`32`

Goal: ingest a real document set and answer questions about it, then express
the same flow as a graph.

**Done when:** you can explain why your retrieval returned a particular chunk,
and draw your pipeline as a state graph.

### Week 4 — Production and shipping

**Notebooks:** `19`-`21`, `24`, `27`, then capstone `53`

Goal: tracing, streaming, structured output, cost tracking - then wrap a graph
in FastAPI and serve it.

**Done when:** capstone 53's test suite passes and you can explain why the
thread key includes the user id.

### Week 5 — Human-in-the-loop, multi-agent, deep agents

**Notebooks:** `33`-`38`, `41`, `50`, then capstone `52`

Goal: pause for human approval, coordinate several agents, and understand what
a deep agent adds over a plain one.

**Done when:** you can pause a running graph, edit its state, resume it, and
say honestly whether the multi-agent version beat a single good prompt.

### Afterwards, as depth

`16`-`18` (agents in detail, including `16a`/`16b`/`17a`), `20a`, `22`-`23`,
`25`-`26a`, `27a`, `28`, `39`-`40`, `42`-`49` (including `43a`/`48a`), `50a`,
and capstone `51`.

---



## How to actually learn this

**Run every cell, then break it.** Change a parameter and predict the output
before re-running. The prediction is the learning; the output only confirms it.

**Do the exercises.** Each notebook ends with "Try it yourself". They are
where reading turns into knowing.

**Keep a decisions file.** When a notebook says "use X when Y", write it down
in your own words. Six weeks later your notes will be more useful than the
notebooks.

**Do not skip the recap tables.** They are the revision material.

**Mark honestly.** In [CHECKLIST.md](CHECKLIST.md), "Done" should mean you
could rebuild the core idea from scratch, not that the cells ran.