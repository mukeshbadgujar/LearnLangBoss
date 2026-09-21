# Official Sources

Where to look when a library moves, or when you want the authoritative version
of something a notebook summarised.

These notebooks were written against the official documentation and reference
repositories listed below. The content is original; the APIs, semantics and
recommended patterns come from these sources.

## Primary documentation

| Source | Use it for |
|---|---|
| [docs.langchain.com/oss/python](https://docs.langchain.com/oss/python/learn) | The current LangChain and LangGraph docs. Start here. |
| [LangChain API reference](https://reference.langchain.com/python/) | Exact signatures. The docs explain; the reference is authoritative. |
| [LangGraph concepts](https://docs.langchain.com/oss/python/langgraph/overview) | State, checkpointing, interrupts, streaming |
| [LangSmith docs](https://docs.langchain.com/langsmith/home) | Tracing, datasets, evaluation |
| [LangChain changelog](https://changelog.langchain.com/) | What broke and when - check this first when something stops working |

## Reference repositories

| Repository | Use it for |
|---|---|
| [langchain-ai/langchain](https://github.com/langchain-ai/langchain) | Source of truth. Read the code when the docs are ambiguous. |
| [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) | Graph internals, checkpointer and store implementations |
| [langchain-ai/langchain-academy](https://github.com/langchain-ai/langchain-academy) | The official LangGraph course notebooks |
| [langchain-ai/langgraph-101](https://github.com/langchain-ai/langgraph-101) | Short, focused LangGraph introductions |
| [langchain-ai/rag-from-scratch](https://github.com/langchain-ai/rag-from-scratch) | The query-transformation and reranking techniques in notebook 15 |
| [langchain-ai/deepagents](https://github.com/langchain-ai/deepagents) | The packaged deep-agent harness from notebook 50 |
| [LangChain-OpenTutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial) | A broad community tutorial set; useful for alternative explanations |
| [emarco177/langchain-course](https://github.com/emarco177/langchain-course) | One project per branch, one lesson per commit. See [course-crosswalk.md](course-crosswalk.md) |
| [emarco177/langgraph-course](https://github.com/emarco177/langgraph-course) | Agentic RAG, ReAct, reflection and reflexion agents, with a `pytest` suite |
| [krishnaik06/Deep-agents-With-Langchain](https://github.com/krishnaik06/Deep-agents-With-Langchain) | Deep-agent backends, skills and `AGENTS.md` context files |

## Courses

- [DeepLearning.AI — AI Agents in LangGraph](https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/)
- [LangChain Academy — Introduction to LangGraph](https://academy.langchain.com/courses/intro-to-langgraph)

## Provider documentation

| Provider | Docs | Notes |
|---|---|---|
| Groq | [console.groq.com/docs](https://console.groq.com/docs) | Default provider. Check the model list - ids change. |
| OpenRouter | [openrouter.ai/docs](https://openrouter.ai/docs) | OpenAI-compatible; one key, many models |
| OpenAI | [platform.openai.com/docs](https://platform.openai.com/docs) | Best structured-output and vision support |
| Anthropic | [docs.anthropic.com](https://docs.anthropic.com) | Long context, prompt caching |
| HuggingFace | [huggingface.co/docs](https://huggingface.co/docs) | Local embedding models |

---

## Notebook → documentation map

### Track 00 — Setup

| Notebook | Primary references |
|---|---|
| `00_environment_providers_and_keys` | [Install](https://docs.langchain.com/oss/python/langchain/install) · [`init_chat_model`](https://reference.langchain.com/python/langchain/models/) · [LangSmith setup](https://docs.langchain.com/langsmith/observability-quickstart) |

### Track 01 — LangChain foundations

| Notebook | Primary references |
|---|---|
| `01_models_messages` | [Chat models](https://docs.langchain.com/oss/python/langchain/models) · [Messages](https://docs.langchain.com/oss/python/langchain/messages) |
| `02_prompt_templates` | [Prompt templates](https://docs.langchain.com/oss/python/langchain/prompts) |
| `03_chat_prompt_templates` | [Prompt templates](https://docs.langchain.com/oss/python/langchain/prompts) · [`trim_messages`](https://reference.langchain.com/python/langchain_core/messages/) |
| `04_few_shot_and_example_selectors` | [Few-shot prompting](https://docs.langchain.com/oss/python/langchain/prompts) · [Example selectors](https://reference.langchain.com/python/langchain_core/example_selectors/) |
| `05_output_parsers` | [Output parsers](https://reference.langchain.com/python/langchain_core/output_parsers/) · [Structured output](https://docs.langchain.com/oss/python/langchain/structured-output) |
| `06_memory` | [Short-term memory](https://docs.langchain.com/oss/python/langchain/short-term-memory) · [Middleware](https://docs.langchain.com/oss/python/langchain/middleware) |
| `07_lcel` | [Runnable interface](https://docs.langchain.com/oss/python/langchain/runnables) |
| `08_chains_sequential_and_custom` | [Runnable interface](https://docs.langchain.com/oss/python/langchain/runnables) · [`langchain-classic`](https://reference.langchain.com/python/langchain_classic/) for the legacy chains |

### Track 02 — RAG

| Notebook | Primary references |
|---|---|
| `09_document_loaders` | [Document loaders](https://docs.langchain.com/oss/python/integrations/document_loaders) |
| `10_text_splitters` | [Text splitters](https://docs.langchain.com/oss/python/langchain/retrieval) · [`langchain-text-splitters`](https://reference.langchain.com/python/text_splitters/) |
| `11_embeddings_and_caching` | [Embedding models](https://docs.langchain.com/oss/python/integrations/text_embedding) · [`CacheBackedEmbeddings`](https://reference.langchain.com/python/langchain_classic/embeddings/) |
| `12_vector_stores` | [Vector stores](https://docs.langchain.com/oss/python/integrations/vectorstores) |
| `13_retrievers` | [Retrievers](https://docs.langchain.com/oss/python/langchain/retrieval) · [`langchain-classic` retrievers](https://reference.langchain.com/python/langchain_classic/retrievers/) |
| `14_retrieval_chains` | [RAG](https://docs.langchain.com/oss/python/langchain/rag) |
| `15_advanced_rag` | [rag-from-scratch](https://github.com/langchain-ai/rag-from-scratch) · [Rerankers](https://docs.langchain.com/oss/python/integrations/document_transformers) |

### Track 03 — Agents and tools

| Notebook | Primary references |
|---|---|
| `16_tools_builtin_custom_toolkits` | [Tools](https://docs.langchain.com/oss/python/langchain/tools) · [Tool integrations](https://docs.langchain.com/oss/python/integrations/tools) |
| `16a_code_execution_and_sandboxing` | [Python REPL tool](https://docs.langchain.com/oss/python/integrations/tools) · [emarco177 code-interpreter](https://github.com/emarco177/langchain-course/tree/project/code-interpreter) |
| `16b_mcp_servers_and_clients` | [MCP](https://docs.langchain.com/oss/python/langchain/mcp) · [Model Context Protocol](https://modelcontextprotocol.io/) |
| `17_agents` | [Agents](https://docs.langchain.com/oss/python/langchain/agents) · [`create_agent`](https://reference.langchain.com/python/langchain/agents/) · [Middleware](https://docs.langchain.com/oss/python/langchain/middleware) |
| `17a_agent_loop_from_scratch` | [ReAct](https://arxiv.org/abs/2210.03629) · [emarco177 agents-under-the-hood](https://github.com/emarco177/langchain-course/tree/project/agents-under-the-hood) |
| `18_from_agentexecutor_to_graphs` | [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) |

### Track 04 — Production LangChain

| Notebook | Primary references |
|---|---|
| `19_langsmith` | [Tracing](https://docs.langchain.com/langsmith/observability-quickstart) · [Evaluation](https://docs.langchain.com/langsmith/evaluation) |
| `20_callbacks` | [Callbacks](https://reference.langchain.com/python/langchain_core/callbacks/) |
| `20a_guardrails_pii_and_safety` | [Guardrails](https://docs.langchain.com/oss/python/langchain/guardrails) · [Middleware](https://docs.langchain.com/oss/python/langchain/middleware) |
| `21_streaming` | [Streaming](https://docs.langchain.com/oss/python/langchain/streaming) |
| `22_multimodal` | [Multimodal](https://docs.langchain.com/oss/python/langchain/models) · [Content blocks](https://docs.langchain.com/oss/python/langchain/messages) |
| `23_caching` | [Caching](https://reference.langchain.com/python/langchain_core/caches/) · provider prompt-caching docs |
| `24_structured_outputs` | [Structured output](https://docs.langchain.com/oss/python/langchain/structured-output) |
| `25_routing_and_handoffs` | [Runnable interface](https://docs.langchain.com/oss/python/langchain/runnables) · [Multi-agent](https://docs.langchain.com/oss/python/langgraph/multi-agent) |
| `26_evaluation` | [Evaluation](https://docs.langchain.com/langsmith/evaluation) · [LLM-as-judge](https://docs.langchain.com/langsmith/llm-as-judge) |
| `26a_unit_testing_chains_and_graphs` | [emarco177 agentic-rag tests](https://github.com/emarco177/langgraph-course/tree/project/agentic-rag) · pytest |
| `27_usage_tracking` | [Token usage](https://docs.langchain.com/oss/python/langchain/models) |
| `27a_llm_gateways` | [LiteLLM](https://docs.litellm.ai/) · OpenAI-compatible proxies |
| `28_langchain_hub` | [Prompt Hub](https://docs.langchain.com/langsmith/prompt-engineering-quickstart) |

### Track 05 — LangGraph beginner

| Notebook | Primary references |
|---|---|
| `29_graphs_vs_agents` | [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) · [langgraph-101](https://github.com/langchain-ai/langgraph-101) |
| `30_state_schemas` | [Graph state](https://docs.langchain.com/oss/python/langgraph/graph-api) · [Reducers](https://docs.langchain.com/oss/python/langgraph/graph-api) |
| `31_stategraph_nodes_edges` | [Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api) |
| `32_conditional_routing` | [Graph API — conditional edges](https://docs.langchain.com/oss/python/langgraph/graph-api) · [`Command`](https://reference.langchain.com/python/langgraph/types/) |

### Track 06 — LangGraph intermediate

| Notebook | Primary references |
|---|---|
| `33_compile_invoke_stream` | [Streaming](https://docs.langchain.com/oss/python/langgraph/streaming) |
| `34_checkpointing_thread_ids` | [Persistence](https://docs.langchain.com/oss/python/langgraph/persistence) |
| `35_sqlite_postgres_savers` | [Checkpointers](https://reference.langchain.com/python/langgraph/checkpointing/) |
| `36_human_in_the_loop` | [Human-in-the-loop](https://docs.langchain.com/oss/python/langgraph/add-human-in-the-loop) |
| `37_message_history_and_deletion` | [Manage memory](https://docs.langchain.com/oss/python/langgraph/add-memory) · [`RemoveMessage`](https://reference.langchain.com/python/langchain_core/messages/) |
| `38_parallel_fanout_fanin` | [Graph API — parallelism](https://docs.langchain.com/oss/python/langgraph/graph-api) · [`Send`](https://reference.langchain.com/python/langgraph/types/) |

### Track 07 — LangGraph advanced

| Notebook | Primary references |
|---|---|
| `39_time_travel` | [Time travel](https://docs.langchain.com/oss/python/langgraph/time-travel) |
| `40_subgraphs` | [Subgraphs](https://docs.langchain.com/oss/python/langgraph/use-subgraphs) |
| `41_multi_agent_handoffs` | [Multi-agent](https://docs.langchain.com/oss/python/langgraph/multi-agent) |
| `42_custom_stream_channels` | [Streaming — custom mode](https://docs.langchain.com/oss/python/langgraph/streaming) |
| `43_hybrid_deterministic_llm` | [Workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents) |
| `43a_five_workflow_patterns` | [Workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents) · Anthropic workflow patterns |
| `44_error_retries_tool_failures` | [Tool error handling](https://docs.langchain.com/oss/python/langchain/tools) · [`RetryPolicy`](https://reference.langchain.com/python/langgraph/types/) |
| `45_langsmith_and_studio` | [LangGraph Studio](https://docs.langchain.com/langsmith/langgraph-studio) |
| `46_deployment_and_versioning` | [Deployment](https://docs.langchain.com/langgraph-platform/deployment-quickstart) · [`langgraph.json`](https://docs.langchain.com/langgraph-platform/cli) |
| `47_token_limits_and_summarization_nodes` | [Manage context](https://docs.langchain.com/oss/python/langchain/context-engineering) · [`SummarizationMiddleware`](https://reference.langchain.com/python/langchain/agents/) |
| `48_self_reflective_rag` | [langchain-academy — reflection](https://github.com/langchain-ai/langchain-academy) · CRAG and Self-RAG papers |
| `48a_reflection_and_reflexion` | [Reflexion](https://arxiv.org/abs/2303.11366) · [emarco177 reflection-agent](https://github.com/emarco177/langgraph-course/tree/project/reflection-agent) |
| `49_long_term_semantic_memory` | [Long-term memory](https://docs.langchain.com/oss/python/langgraph/add-memory) · [Store API](https://reference.langchain.com/python/langgraph/store/) |
| `50_deep_agents` | [deepagents](https://github.com/langchain-ai/deepagents) · [Middleware](https://docs.langchain.com/oss/python/langchain/middleware) |
| `50a_deep_agent_backends_and_skills` | [deepagents](https://github.com/langchain-ai/deepagents) · [krishnaik06 Deep-agents](https://github.com/krishnaik06/Deep-agents-With-Langchain) |

### Track 08 — Capstones

| Notebook | Primary references |
|---|---|
| `51_policy_rag_chatbot` | [RAG](https://docs.langchain.com/oss/python/langchain/rag) · [Persistence](https://docs.langchain.com/oss/python/langgraph/persistence) |
| `52_multi_agent_research_desk` | [Multi-agent](https://docs.langchain.com/oss/python/langgraph/multi-agent) · [Human-in-the-loop](https://docs.langchain.com/oss/python/langgraph/add-human-in-the-loop) |
| `53_fastapi_langgraph_service` | [FastAPI](https://fastapi.tiangolo.com/) · [Streaming](https://docs.langchain.com/oss/python/langgraph/streaming) · [Deployment](https://docs.langchain.com/langgraph-platform/deployment-quickstart) |

---

## Papers worth reading

| Paper | Notebook | Why |
|---|---|---|
| [ReAct](https://arxiv.org/abs/2210.03629) | 17 | The reasoning-and-acting loop every agent descends from |
| [HyDE](https://arxiv.org/abs/2212.10496) | 15 | Hypothetical document embeddings |
| [Self-RAG](https://arxiv.org/abs/2310.11511) | 48 | Retrieve-and-critique |
| [Corrective RAG](https://arxiv.org/abs/2401.15884) | 48 | Falling back when retrieval is weak |
| [Reflexion](https://arxiv.org/abs/2303.11366) | 43a, 48a | Self-critique loops that drive research |
| [Lost in the Middle](https://arxiv.org/abs/2307.03172) | 13, 47 | Why where a fact sits in context matters |

## When something in these notebooks stops working

1. Check the [changelog](https://changelog.langchain.com/) for the package.
2. Check the [API reference](https://reference.langchain.com/python/) for the
   current signature - most breakage is a moved import or a renamed argument.
3. Run `python tools/check_notebooks.py`; it resolves every import in every
   notebook and will name the ones that no longer exist.
4. Fix the source in `tools/notebook_sources/`, then run
   `python tools/nbgen.py` to regenerate.

Version pins live in [`requirements.txt`](../requirements.txt). The curriculum
targets `langchain>=1.4`, `langgraph>=1.2`.
