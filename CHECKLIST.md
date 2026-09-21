# Progress Checklist

63 notebooks. Mark each one as you verify it.

**Status values**

- `Pending` — not started, or started and not finished
- `Done` — you ran every cell, did at least one exercise, and could rebuild the
  core idea from scratch without the notebook open
- `Blocked` — you hit something that does not work; note why on the line

> "Done" is a claim about **you**, not about whether the cells executed. A
> notebook whose cells ran while you read passively is still `Pending`.

Run `python tools/list_lessons.py --check` to confirm every notebook still
declares the checklist id listed here.

---

## Track 00 — Setup

- [ ] `00_environment_providers_and_keys` — Environment, Providers and Keys — **Pending**

## Track 01 — LangChain foundations

- [ ] `01_models_messages` — Models and Messages — **Pending**
- [ ] `02_prompt_templates` — Prompt Templates — **Pending**
- [ ] `03_chat_prompt_templates` — ChatPromptTemplates and MessagesPlaceholder — **Pending**
- [ ] `04_few_shot_and_example_selectors` — Few-Shot Prompting and Example Selectors — **Pending**
- [ ] `05_output_parsers` — Output Parsers and Error Recovery — **Pending**
- [ ] `06_memory` — Memory: Classic Types and the Modern Replacement — **Pending**
- [ ] `07_lcel` — LangChain Expression Language (LCEL) — **Pending**
- [ ] `08_chains_sequential_and_custom` — Sequential and Custom Chains — **Pending**

## Track 02 — Retrieval-augmented generation

- [ ] `09_document_loaders` — Document Loaders — **Pending**
- [ ] `10_text_splitters` — Text Splitters and Chunking Strategy — **Pending**
- [ ] `11_embeddings_and_caching` — Embeddings and Embedding Caches — **Pending**
- [ ] `12_vector_stores` — Vector Stores — **Pending**
- [ ] `13_retrievers` — Retrievers: Standard and Advanced — **Pending**
- [ ] `14_retrieval_chains` — Retrieval Chains and History-Aware RAG — **Pending**
- [ ] `15_advanced_rag` — Advanced RAG: Query Transformation, Hybrid Search, Reranking — **Pending**

## Track 03 — Agents and tools

- [ ] `16_tools_builtin_custom_toolkits` — Tools: Built-in, Custom and Toolkits — **Pending**
- [ ] `16a_code_execution_and_sandboxing` — Code Execution Tools and Sandboxing — **Pending**
- [ ] `16b_mcp_servers_and_clients` — MCP: Model Context Protocol Servers and Clients — **Pending**
- [ ] `17_agents` — Agents: Types, Execution and Failure Modes — **Pending**
- [ ] `17a_agent_loop_from_scratch` — The Agent Loop From Scratch — **Pending**
- [ ] `18_from_agentexecutor_to_graphs` — From AgentExecutor to Graphs — **Pending**

## Track 04 — Production LangChain

- [ ] `19_langsmith` — LangSmith: Tracing, Debugging and Datasets — **Pending**
- [ ] `20_callbacks` — Callbacks — **Pending**
- [ ] `20a_guardrails_pii_and_safety` — Guardrails: PII, Injection and Safety Middleware — **Pending**
- [ ] `21_streaming` — Streaming — **Pending**
- [ ] `22_multimodal` — Multimodal Models: Images and Documents — **Pending**
- [ ] `23_caching` — Caching: Model, Semantic and Prompt — **Pending**
- [ ] `24_structured_outputs` — Structured Outputs — **Pending**
- [ ] `25_routing_and_handoffs` — Routing and Handoffs — **Pending**
- [ ] `26_evaluation` — Evaluation — **Pending**
- [ ] `26a_unit_testing_chains_and_graphs` — Unit-Testing Chains, Graders and Graphs — **Pending**
- [ ] `27_usage_tracking` — Usage and Cost Tracking — **Pending**
- [ ] `27a_llm_gateways` — LLM Gateways: Fallbacks, Routing and Load Balancing — **Pending**
- [ ] `28_langchain_hub` — LangChain Hub and Prompt Management — **Pending**

## Track 05 — LangGraph beginner

- [ ] `29_graphs_vs_agents` — Graphs vs Chains vs Agents — **Pending**
- [ ] `30_state_schemas` — State Schemas and Reducers — **Pending**
- [ ] `31_stategraph_nodes_edges` — StateGraph: Nodes and Edges — **Pending**
- [ ] `32_conditional_routing` — Conditional Routing — **Pending**

## Track 06 — LangGraph intermediate

- [ ] `33_compile_invoke_stream` — Compile, Invoke and Stream — **Pending**
- [ ] `34_checkpointing_thread_ids` — Checkpointing and Thread IDs — **Pending**
- [ ] `35_sqlite_postgres_savers` — SQLite and Postgres Savers — **Pending**
- [ ] `36_human_in_the_loop` — Human in the Loop — **Pending**
- [ ] `37_message_history_and_deletion` — Message History and Deletion — **Pending**
- [ ] `38_parallel_fanout_fanin` — Parallel Execution: Fan-out and Fan-in — **Pending**

## Track 07 — LangGraph advanced

- [ ] `39_time_travel` — Time Travel — **Pending**
- [ ] `40_subgraphs` — Subgraphs — **Pending**
- [ ] `41_multi_agent_handoffs` — Multi-Agent Systems and Handoffs — **Pending**
- [ ] `42_custom_stream_channels` — Custom Stream Channels — **Pending**
- [ ] `43_hybrid_deterministic_llm` — Hybrid Deterministic and LLM Workflows — **Pending**
- [ ] `43a_five_workflow_patterns` — The Five Workflow Patterns — **Pending**
- [ ] `44_error_retries_tool_failures` — Errors, Retries and Tool Failures — **Pending**
- [ ] `45_langsmith_and_studio` — LangSmith and LangGraph Studio — **Pending**
- [ ] `46_deployment_and_versioning` — Deployment and Versioning — **Pending**
- [ ] `47_token_limits_and_summarization_nodes` — Token Limits and Summarisation Nodes — **Pending**
- [ ] `48_self_reflective_rag` — Self-Reflective RAG — **Pending**
- [ ] `48a_reflection_and_reflexion` — Reflection and Reflexion Agents — **Pending**
- [ ] `49_long_term_semantic_memory` — Long-Term Semantic Memory — **Pending**
- [ ] `50_deep_agents` — Deep Agents — **Pending**
- [ ] `50a_deep_agent_backends_and_skills` — Deep Agent Backends, Skills and AGENTS.md — **Pending**

## Track 08 — Capstones

- [ ] `51_policy_rag_chatbot` — Capstone: Policy RAG Chatbot — **Pending**
- [ ] `52_multi_agent_research_desk` — Capstone: Multi-Agent Research Desk — **Pending**
- [ ] `53_fastapi_langgraph_service` — Capstone: FastAPI + LangGraph Service — **Pending**

---

## Milestones

Beyond individual notebooks, these are the things you should be able to do.
They are worth more than the tick boxes above.

- [ ] **Explain LCEL to someone else** without notes, including why `|` works
      and what `RunnablePassthrough.assign` is for. *(after 07)*
- [ ] **Diagnose a bad RAG answer** — say whether retrieval, chunking or the
      prompt was at fault, with evidence. *(after 13-15)*
- [ ] **Justify agent vs graph** for a specific task, naming what you gain and
      what it costs. *(after 18, 29)*
- [ ] **Pause, edit and resume** a running graph from a checkpoint. *(after 36)*
- [ ] **Quote your cost per 1,000 requests** for something you built, and name
      the biggest lever for reducing it. *(after 27)*
- [ ] **Ship a graph behind HTTP** with authentication, streaming and a test
      that proves users cannot read each other's threads. *(after 53)*
- [ ] **Say no to a multi-agent design** when a single prompt would do, with
      numbers from your own comparison. *(after 52)*

## Environment health

- [ ] `python tools/check_notebooks.py` passes (63/63)
- [ ] `python tools/list_lessons.py --check` passes
- [ ] `python tools/smoke_lg.py`, `smoke_lg2.py`, `smoke_lg3.py` pass
- [ ] `python tools/smoke_capstones.py` passes
- [ ] `describe_environment()` shows an embedding provider other than `fake`
      (Groq's free tier is enough), and notebook 00's "are these embeddings
      real?" cell reports **real embeddings**
- [ ] LangSmith tracing on, and you have found one of your own traces in the UI
