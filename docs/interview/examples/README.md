# Interview mini-examples

One short Python file for each question in the interview Q&A. Files that call a model import [model.py](model.py). That is the only shared piece. Change the provider in `.env`, not in the examples.

| Guide | Questions | Folder |
|---|---|---|
| [LangChain Q&A](../langchain-qa.md) | 66 | [langchain/](langchain/) |
| [LangGraph Q&A](../langgraph-qa.md) | 75 | [langgraph/](langgraph/) |

The notebooks under `01-langchain-foundations/` through `08-capstones/` are the labs. These files are smaller. Each one makes a single interview question concrete.

## The story

Every example is the same shop: a small online bookstore support desk. Orders, refunds, shipping, policy pages, and tickets show up again and again, so a new idea (a retriever, a checkpoint, a human approval) sits on a scene you already know.

## How to read a file

The top of the file is the explanation:

1. **The problem** - the everyday situation that is broken or unclear.
2. **What we are going to solve** - the one decision this file demonstrates.
3. **What this example is about** - the bookstore scene.
4. **What it solves** - what is true after the code runs.

**Keywords** explains every term the file uses, in plain language. Comments in the code say what a line does and why this question needs it.

## How to run

Put a key in the repo `.env` file and pick the provider with one line:

```text
LLM_PROVIDER=groq
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

`LLM_PROVIDER` can be `groq`, `openrouter`, `openai`, or `anthropic`. Set the matching key and model name (`OPENAI_API_KEY` and `OPENAI_MODEL`, and the same pattern for the others). Leave `LLM_PROVIDER` empty to use the first key that is filled in.

From the repo root:

```text
python docs/interview/examples/langchain/q01_what_is_langchain.py
python docs/interview/examples/langgraph/q08_minimal_two_node_graph.py
```

`q01` prints the provider it is using, then answers with `model.invoke`. Search examples use `embedding_model()` from the same file, which follows `EMBEDDING_PROVIDER`. Graph examples that only show state, edges, or checkpoints do not call a model.

## LangChain

| # | Question | File |
|---|---|---|
| 1 | Q1. What is LangChain? | [q01_what_is_langchain.py](langchain/q01_what_is_langchain.py) |
| 2 | Q2. Why would you use LangChain instead of calling an LLM API directly? | [q02_why_not_call_the_api_directly.py](langchain/q02_why_not_call_the_api_directly.py) |
| 3 | Q3. What are the main components of LangChain? | [q03_main_components.py](langchain/q03_main_components.py) |
| 4 | Q4. What problems does LangChain solve? | [q04_problems_langchain_solves.py](langchain/q04_problems_langchain_solves.py) |
| 5 | Q5. What types of applications can be built with LangChain? | [q05_application_types.py](langchain/q05_application_types.py) |
| 6 | Q6. What are chains? | [q06_what_are_chains.py](langchain/q06_what_are_chains.py) |
| 7 | Q7. What are tools? | [q07_what_are_tools.py](langchain/q07_what_are_tools.py) |
| 8 | Q8. What are agents? | [q08_what_are_agents.py](langchain/q08_what_are_agents.py) |
| 9 | Q9. What are Runnables? | [q09_what_are_runnables.py](langchain/q09_what_are_runnables.py) |
| 10 | Q10. What are output parsers? | [q10_output_parsers.py](langchain/q10_output_parsers.py) |
| 11 | Q11. How does LangChain implement RAG? | [q11_how_rag_works.py](langchain/q11_how_rag_works.py) |
| 12 | Q12. What is a retriever? | [q12_what_is_a_retriever.py](langchain/q12_what_is_a_retriever.py) |
| 13 | Q13. How do vector databases fit into LangChain? | [q13_vector_databases.py](langchain/q13_vector_databases.py) |
| 14 | Q14. What embedding models can LangChain use? | [q14_embedding_models.py](langchain/q14_embedding_models.py) |
| 15 | Q15. How would you improve retrieval quality? | [q15_improve_retrieval.py](langchain/q15_improve_retrieval.py) |
| 16 | Q16. How do you reduce hallucinations in a RAG application? | [q16_reduce_hallucinations.py](langchain/q16_reduce_hallucinations.py) |
| 17 | Q17. What is an AI agent? | [q17_what_is_an_ai_agent.py](langchain/q17_what_is_an_ai_agent.py) |
| 18 | Q18. How do LangChain agents choose tools? | [q18_how_agents_choose_tools.py](langchain/q18_how_agents_choose_tools.py) |
| 19 | Q19. What is the difference between a chain and an agent? | [q19_chain_vs_agent.py](langchain/q19_chain_vs_agent.py) |
| 20 | Q20. When should you avoid agents? | [q20_when_to_avoid_agents.py](langchain/q20_when_to_avoid_agents.py) |
| 21 | Q21. How do you evaluate an agent? | [q21_evaluate_an_agent.py](langchain/q21_evaluate_an_agent.py) |
| 22 | Q22. What is memory in LangChain? | [q22_what_is_memory.py](langchain/q22_what_is_memory.py) |
| 23 | Q23. What memory types are available? | [q23_memory_types.py](langchain/q23_memory_types.py) |
| 24 | Q24. How would you summarize conversation history? | [q24_summarize_history.py](langchain/q24_summarize_history.py) |
| 25 | Q25. What problems arise with long conversations? | [q25_long_conversation_problems.py](langchain/q25_long_conversation_problems.py) |
| 26 | Q26. How would you reduce context growth? | [q26_reduce_context_growth.py](langchain/q26_reduce_context_growth.py) |
| 27 | Q27. Design a chatbot using LangChain. | [q27_design_a_chatbot.py](langchain/q27_design_a_chatbot.py) |
| 28 | Q28. Design a document question-answering system. | [q28_document_qa.py](langchain/q28_document_qa.py) |
| 29 | Q29. Design a customer support assistant. | [q29_customer_support_assistant.py](langchain/q29_customer_support_assistant.py) |
| 30 | Q30. How would you build a multi-step workflow? | [q30_multi_step_workflow.py](langchain/q30_multi_step_workflow.py) |
| 31 | Q31. How would you add human approval? | [q31_human_approval.py](langchain/q31_human_approval.py) |
| 32 | Q32. How would you debug complex chains? | [q32_debug_complex_chains.py](langchain/q32_debug_complex_chains.py) |
| 33 | Q33. How do you deploy LangChain applications? | [q33_deploy.py](langchain/q33_deploy.py) |
| 34 | Q34. How do you monitor LLM applications? | [q34_monitor.py](langchain/q34_monitor.py) |
| 35 | Q35. How do you cache responses? | [q35_cache_responses.py](langchain/q35_cache_responses.py) |
| 36 | Q36. How do you handle API failures? | [q36_api_failures.py](langchain/q36_api_failures.py) |
| 37 | Q37. How do you control costs? | [q37_control_costs.py](langchain/q37_control_costs.py) |
| 38 | Q38. How do you evaluate application quality? | [q38_evaluate_quality.py](langchain/q38_evaluate_quality.py) |
| 39 | Q39. How would you build a multi-agent system? | [q39_multi_agent.py](langchain/q39_multi_agent.py) |
| 40 | Q40. How would you implement retries and fallbacks? | [q40_retries_and_fallbacks.py](langchain/q40_retries_and_fallbacks.py) |
| 41 | Q41. How would you manage context efficiently? | [q41_manage_context.py](langchain/q41_manage_context.py) |
| 42 | Q42. How would you optimize latency? | [q42_optimize_latency.py](langchain/q42_optimize_latency.py) |
| 43 | Q43. How would you evaluate tool selection? | [q43_evaluate_tool_selection.py](langchain/q43_evaluate_tool_selection.py) |
| 44 | Q44. How would you build an enterprise RAG pipeline? | [q44_enterprise_rag.py](langchain/q44_enterprise_rag.py) |
| 45 | Q45. What are common production bottlenecks? | [q45_production_bottlenecks.py](langchain/q45_production_bottlenecks.py) |
| 46 | Q46. LangChain vs. LlamaIndex. | [q46_langchain_vs_llamaindex.py](langchain/q46_langchain_vs_llamaindex.py) |
| 47 | Q47. LangChain vs. Semantic Kernel. | [q47_langchain_vs_semantic_kernel.py](langchain/q47_langchain_vs_semantic_kernel.py) |
| 48 | Q48. LangChain vs. Haystack. | [q48_langchain_vs_haystack.py](langchain/q48_langchain_vs_haystack.py) |
| 49 | Q49. LangGraph vs. LangChain. | [q49_langgraph_vs_langchain.py](langchain/q49_langgraph_vs_langchain.py) |
| 50 | Q50. How have "Chains" evolved in LangChain from legacy versions to modern architecture? | [q50_chains_evolved.py](langchain/q50_chains_evolved.py) |
| 51 | Q51. What problems did LCEL and the Runnable interface solve? | [q51_lcel_problems_solved.py](langchain/q51_lcel_problems_solved.py) |
| 52 | Q52. What are the primary Runnable primitives used in LCEL pipelines? | [q52_runnable_primitives.py](langchain/q52_runnable_primitives.py) |
| 53 | Q53. What are "Deep Agents" and what specific problems do they solve? | [q53_deep_agents.py](langchain/q53_deep_agents.py) |
| 54 | Q54. What are the 4 core pillars of Deep Agent architecture? | [q54_deep_agent_pillars.py](langchain/q54_deep_agent_pillars.py) |
| 55 | Q55. How do Claude Skills integrate with Deep Agents compared to static instructions? | [q55_skills_vs_static_instructions.py](langchain/q55_skills_vs_static_instructions.py) |
| 56 | Q56. What is RecursiveCharacterTextSplitter and why is it the industry standard? | [q56_recursive_character_splitter.py](langchain/q56_recursive_character_splitter.py) |
| 57 | Q57. What are the major text splitting strategy types? | [q57_text_splitting_strategies.py](langchain/q57_text_splitting_strategies.py) |
| 58 | Q58. Outline the architecture for Project 1: Conversational FAQ & Ticket Escalation Agent. | [q58_faq_ticket_escalation.py](langchain/q58_faq_ticket_escalation.py) |
| 59 | Q59. Outline the architecture for Project 2: Interactive Resume Document Writing Agent. | [q59_resume_writing_agent.py](langchain/q59_resume_writing_agent.py) |
| 60 | Q60. Explain the 5 core components and data flow of a LangChain RAG pipeline. | [q60_rag_five_components.py](langchain/q60_rag_five_components.py) |
| 61 | Q61. What is the Runnable Interface, and how does it work under the hood? | [q61_runnable_interface.py](langchain/q61_runnable_interface.py) |
| 62 | Q62. How does Memory work in LangChain, and what are the architectural trade-offs between Buffer, Summary, and VectorStore memory? | [q62_memory_tradeoffs.py](langchain/q62_memory_tradeoffs.py) |
| 63 | Q63. What is a SKILL.md file, and how is it used in agentic architectures? | [q63_skill_md.py](langchain/q63_skill_md.py) |
| 64 | Q64. How do you approach Testing and Evaluation for non-deterministic AI agents and graphs? | [q64_testing_nondeterministic.py](langchain/q64_testing_nondeterministic.py) |
| 65 | Q65. How do you handle Observability and Tracing using LangSmith? | [q65_langsmith_tracing.py](langchain/q65_langsmith_tracing.py) |
| 66 | Q66. How do you approach Testing and Evaluation for AI agents? | [q66_testing_agents.py](langchain/q66_testing_agents.py) |

## LangGraph

| # | Question | File |
|---|---|---|
| 1 | Q1. What is LangGraph? | [q01_what_is_langgraph.py](langgraph/q01_what_is_langgraph.py) |
| 2 | Q2. LangGraph vs LCEL | [q02_langgraph_vs_lcel.py](langgraph/q02_langgraph_vs_lcel.py) |
| 3 | Q3. Three building blocks: State, Nodes, Edges | [q03_three_building_blocks.py](langgraph/q03_three_building_blocks.py) |
| 4 | Q4. What is a superstep? | [q04_superstep.py](langgraph/q04_superstep.py) |
| 5 | Q5. Cycles vs DAG (Airflow-style) | [q05_cycles_vs_dag.py](langgraph/q05_cycles_vs_dag.py) |
| 6 | Q6. MessagesState | [q06_messages_state.py](langgraph/q06_messages_state.py) |
| 7 | Q7. Languages and runtimes | [q07_languages_and_runtimes.py](langgraph/q07_languages_and_runtimes.py) |
| 8 | Q8. Minimal two-node graph | [q08_minimal_two_node_graph.py](langgraph/q08_minimal_two_node_graph.py) |
| 9 | Q9. What compile() does | [q09_what_compile_does.py](langgraph/q09_what_compile_does.py) |
| 10 | Q10. What is a reducer? | [q10_what_is_a_reducer.py](langgraph/q10_what_is_a_reducer.py) |
| 11 | Q11. add_messages vs operator.add | [q11_add_messages_vs_operator_add.py](langgraph/q11_add_messages_vs_operator_add.py) |
| 12 | Q12. TypedDict vs dataclass vs Pydantic for state | [q12_typeddict_dataclass_pydantic.py](langgraph/q12_typeddict_dataclass_pydantic.py) |
| 13 | Q13. input_schema and output_schema | [q13_input_and_output_schema.py](langgraph/q13_input_and_output_schema.py) |
| 14 | Q14. Keep large payloads out of checkpointed state | [q14_keep_large_payloads_out.py](langgraph/q14_keep_large_payloads_out.py) |
| 15 | Q15. Parallel write conflict (InvalidUpdateError) | [q15_parallel_write_conflict.py](langgraph/q15_parallel_write_conflict.py) |
| 16 | Q16. Private node / subgraph channel vs graph-level state | [q16_private_node_channel.py](langgraph/q16_private_node_channel.py) |
| 17 | Q17. Pydantic validation cost | [q17_pydantic_validation_cost.py](langgraph/q17_pydantic_validation_cost.py) |
| 18 | Q18. What is a checkpointer? | [q18_what_is_a_checkpointer.py](langgraph/q18_what_is_a_checkpointer.py) |
| 19 | Q19. thread_id | [q19_thread_id.py](langgraph/q19_thread_id.py) |
| 20 | Q20. MemorySaver vs SqliteSaver vs PostgresSaver | [q20_memory_sqlite_postgres.py](langgraph/q20_memory_sqlite_postgres.py) |
| 21 | Q21. Checkpointer vs Store | [q21_checkpointer_vs_store.py](langgraph/q21_checkpointer_vs_store.py) |
| 22 | Q22. Time travel | [q22_time_travel.py](langgraph/q22_time_travel.py) |
| 23 | Q23. Encrypt sensitive checkpoint fields | [q23_encrypt_checkpoint_fields.py](langgraph/q23_encrypt_checkpoint_fields.py) |
| 24 | Q24. Pending writes | [q24_pending_writes.py](langgraph/q24_pending_writes.py) |
| 25 | Q25. Conditional edges vs Command | [q25_conditional_edges_vs_command.py](langgraph/q25_conditional_edges_vs_command.py) |
| 26 | Q26. Command update and route together | [q26_command_update_and_route.py](langgraph/q26_command_update_and_route.py) |
| 27 | Q27. Command(graph=Command.PARENT) | [q27_command_parent.py](langgraph/q27_command_parent.py) |
| 28 | Q28. Tool returns Command | [q28_tool_returns_command.py](langgraph/q28_tool_returns_command.py) |
| 29 | Q29. Send API | [q29_send_api.py](langgraph/q29_send_api.py) |
| 30 | Q30. Fan-out / fan-in with Send | [q30_fanout_fanin_with_send.py](langgraph/q30_fanout_fanin_with_send.py) |
| 31 | Q31. Stop infinite retry | [q31_stop_infinite_retry.py](langgraph/q31_stop_infinite_retry.py) |
| 32 | Q32. Subgraph with isolated state | [q32_subgraph_isolated_state.py](langgraph/q32_subgraph_isolated_state.py) |
| 33 | Q33. Multi-agent topologies | [q33_multi_agent_topologies.py](langgraph/q33_multi_agent_topologies.py) |
| 34 | Q34. Supervisor pattern | [q34_supervisor_pattern.py](langgraph/q34_supervisor_pattern.py) |
| 35 | Q35. Swarm handoff | [q35_swarm_handoff.py](langgraph/q35_swarm_handoff.py) |
| 36 | Q36. Supervisor vs Swarm - when to pick which | [q36_supervisor_vs_swarm.py](langgraph/q36_supervisor_vs_swarm.py) |
| 37 | Q37. create_react_agent | [q37_create_react_agent.py](langgraph/q37_create_react_agent.py) |
| 38 | Q38. Supervisor context sharing vs isolation. | [q38_supervisor_context_sharing.py](langgraph/q38_supervisor_context_sharing.py) |
| 39 | Q39. LangGraph and MCP tool servers. | [q39_mcp_tools.py](langgraph/q39_mcp_tools.py) |
| 40 | Q40. Deep agents beyond a bare ReAct loop. | [q40_deep_agents.py](langgraph/q40_deep_agents.py) |
| 41 | Q41. Pause a graph for human approval before a high-risk action. | [q41_human_approval_pause.py](langgraph/q41_human_approval_pause.py) |
| 42 | Q42. Static interrupts vs dynamic interrupt(). | [q42_static_vs_dynamic_interrupt.py](langgraph/q42_static_vs_dynamic_interrupt.py) |
| 43 | Q43. Which stream mode - messages vs values (short version). | [q43_stream_modes.py](langgraph/q43_stream_modes.py) |
| 44 | Q44. Stream graph output toward a frontend. | [q44_stream_tokens_to_frontend.py](langgraph/q44_stream_tokens_to_frontend.py) |
| 45 | Q45. recursion_limit and runaway loops. | [q45_recursion_limit.py](langgraph/q45_recursion_limit.py) |
| 46 | Q46. Unit test a single LangGraph node in isolation. | [q46_unit_test_a_node.py](langgraph/q46_unit_test_a_node.py) |
| 47 | Q47. Observe a production LangGraph agent. | [q47_observe_production_agent.py](langgraph/q47_observe_production_agent.py) |
| 48 | Q48. Realistic LangGraph system-design prompt structure. | [q48_system_design_prompt.py](langgraph/q48_system_design_prompt.py) |
| 49 | Q49. LangGraph vs LangChain in 2026 - not a rivalry. | [q49_langgraph_vs_langchain_2026.py](langgraph/q49_langgraph_vs_langchain_2026.py) |
| 50 | Q50. LangGraph vs CrewAI vs AutoGen - one-line differentiators. | [q50_langgraph_vs_crewai_vs_autogen.py](langgraph/q50_langgraph_vs_crewai_vs_autogen.py) |
| 51 | Q51. Architectural difference: LangChain chains vs LangGraph state. | [q51_langchain_vs_langgraph_architecture.py](langgraph/q51_langchain_vs_langgraph_architecture.py) |
| 52 | Q52. Node and edge types in LangGraph. | [q52_node_and_edge_types.py](langgraph/q52_node_and_edge_types.py) |
| 53 | Q53. State management and reducers. | [q53_state_and_reducers.py](langgraph/q53_state_and_reducers.py) |
| 54 | Q54. Fan-out and fan-in execution. | [q54_fanout_fanin.py](langgraph/q54_fanout_fanin.py) |
| 55 | Q55. Checkpoint recovery after a mid-refund crash. | [q55_checkpoint_recovery.py](langgraph/q55_checkpoint_recovery.py) |
| 56 | Q56. Chain vs agent vs graph. | [q56_chain_vs_agent_vs_graph.py](langgraph/q56_chain_vs_agent_vs_graph.py) |
| 57 | Q57. What LangGraph adds over chains/agents. | [q57_what_langgraph_adds.py](langgraph/q57_what_langgraph_adds.py) |
| 58 | Q58. Nodes, edges, and conditional edges. | [q58_conditional_edges.py](langgraph/q58_conditional_edges.py) |
| 59 | Q59. Why reducers exist. | [q59_why_reducers.py](langgraph/q59_why_reducers.py) |
| 60 | Q60. Checkpointer and thread_id on a shipping ticket. | [q60_checkpointer_thread_id.py](langgraph/q60_checkpointer_thread_id.py) |
| 61 | Q61. HITL breakpoints with interrupt_before. | [q61_hitl_breakpoints.py](langgraph/q61_hitl_breakpoints.py) |
| 62 | Q62. Supersteps and fan-out/fan-in. | [q62_supersteps_fanout.py](langgraph/q62_supersteps_fanout.py) |
| 63 | Q63. Time-travel replay of a bad refund draft. | [q63_time_travel_replay.py](langgraph/q63_time_travel_replay.py) |
| 64 | Q64. Why nest subgraphs. | [q64_why_subgraphs.py](langgraph/q64_why_subgraphs.py) |
| 65 | Q65. Multi-agent handoffs. | [q65_multi_agent_handoffs.py](langgraph/q65_multi_agent_handoffs.py) |
| 66 | Q66. Node retries with RetryPolicy. | [q66_node_retries.py](langgraph/q66_node_retries.py) |
| 67 | Q67. Stream modes values vs updates vs messages on one graph. | [q67_stream_modes_values_updates_messages.py](langgraph/q67_stream_modes_values_updates_messages.py) |
| 68 | Q68. Token limits and production scaling habits. | [q68_token_limits.py](langgraph/q68_token_limits.py) |
| 69 | Q69. Command API for routing and state updates together. | [q69_command_api.py](langgraph/q69_command_api.py) |
| 70 | Q70. Observability with a callback handler on a graph. | [q70_langsmith_tracing.py](langgraph/q70_langsmith_tracing.py) |
| 71 | Q71. Testing non-deterministic graphs. | [q71_test_nondeterministic_graphs.py](langgraph/q71_test_nondeterministic_graphs.py) |
| 72 | Q72. When to choose LangGraph over LangChain. | [q72_when_langgraph_over_langchain.py](langgraph/q72_when_langgraph_over_langchain.py) |
| 73 | Q73. What to look at in LangSmith (local span list stand-in). | [q73_langsmith_observability.py](langgraph/q73_langsmith_observability.py) |
| 74 | Q74. Testing AI agents (practical layers). | [q74_test_agents.py](langgraph/q74_test_agents.py) |
| 75 | Q75. Prebuilt ReAct agents in LangGraph. | [q75_prebuilt_react_agent.py](langgraph/q75_prebuilt_react_agent.py) |
