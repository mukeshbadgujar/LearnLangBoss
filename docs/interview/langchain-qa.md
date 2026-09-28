# LangChain Interview Q&A

Questions and answers taken from `[docs/sampleQuestonsdata.md](../sampleQuestonsdata.md)` (primary article). Answers are kept in the original wording — not rewritten. Extra unique items from `[addonQA.md](addonQA.md)` are appended under **From addonQA (LangChain extras)**.

**Total questions:** 66

---

## Index

### Basics

- [Q1. What is LangChain?](#q1-what-is-langchain)
- [Q2. Why would you use LangChain instead of calling an LLM API directly?](#q2-why-would-you-use-langchain-instead-of-calling-an-llm-api-directly)
- [Q3. What are the main components of LangChain?](#q3-what-are-the-main-components-of-langchain)
- [Q4. What problems does LangChain solve?](#q4-what-problems-does-langchain-solve)
- [Q5. What types of applications can be built with LangChain?](#q5-what-types-of-applications-can-be-built-with-langchain)



### Core concepts

- [Q6. What are chains?](#q6-what-are-chains)
- [Q7. What are tools?](#q7-what-are-tools)
- [Q8. What are agents?](#q8-what-are-agents)
- [Q9. What are Runnables?](#q9-what-are-runnables)
- [Q10. What are output parsers?](#q10-what-are-output-parsers)



### RAG

- [Q11. How does LangChain implement RAG?](#q11-how-does-langchain-implement-rag)
- [Q12. What is a retriever?](#q12-what-is-a-retriever)
- [Q13. How do vector databases fit into LangChain?](#q13-how-do-vector-databases-fit-into-langchain)
- [Q14. What embedding models can LangChain use?](#q14-what-embedding-models-can-langchain-use)
- [Q15. How would you improve retrieval quality?](#q15-how-would-you-improve-retrieval-quality)
- [Q16. How do you reduce hallucinations in a RAG application?](#q16-how-do-you-reduce-hallucinations-in-a-rag-application)



### Agents

- [Q17. What is an AI agent?](#q17-what-is-an-ai-agent)
- [Q18. How do LangChain agents choose tools?](#q18-how-do-langchain-agents-choose-tools)
- [Q19. What is the difference between a chain and an agent?](#q19-what-is-the-difference-between-a-chain-and-an-agent)
- [Q20. When should you avoid agents?](#q20-when-should-you-avoid-agents)
- [Q21. How do you evaluate an agent?](#q21-how-do-you-evaluate-an-agent)



### Memory

- [Q22. What is memory in LangChain?](#q22-what-is-memory-in-langchain)
- [Q23. What memory types are available?](#q23-what-memory-types-are-available)
- [Q24. How would you summarize conversation history?](#q24-how-would-you-summarize-conversation-history)
- [Q25. What problems arise with long conversations?](#q25-what-problems-arise-with-long-conversations)
- [Q26. How would you reduce context growth?](#q26-how-would-you-reduce-context-growth)



### Architecture

- [Q27. Design a chatbot using LangChain](#q27-design-a-chatbot-using-langchain)
- [Q28. Design a document question-answering system](#q28-design-a-document-question-answering-system)
- [Q29. Design a customer support assistant](#q29-design-a-customer-support-assistant)
- [Q30. How would you build a multi-step workflow?](#q30-how-would-you-build-a-multi-step-workflow)
- [Q31. How would you add human approval?](#q31-how-would-you-add-human-approval)
- [Q32. How would you debug complex chains?](#q32-how-would-you-debug-complex-chains)



### Production

- [Q33. How do you deploy LangChain applications?](#q33-how-do-you-deploy-langchain-applications)
- [Q34. How do you monitor LLM applications?](#q34-how-do-you-monitor-llm-applications)
- [Q35. How do you cache responses?](#q35-how-do-you-cache-responses)
- [Q36. How do you handle API failures?](#q36-how-do-you-handle-api-failures)
- [Q37. How do you control costs?](#q37-how-do-you-control-costs)
- [Q38. How do you evaluate application quality?](#q38-how-do-you-evaluate-application-quality)



### Advanced

- [Q39. How would you build a multi-agent system?](#q39-how-would-you-build-a-multi-agent-system)
- [Q40. How would you implement retries and fallbacks?](#q40-how-would-you-implement-retries-and-fallbacks)
- [Q41. How would you manage context efficiently?](#q41-how-would-you-manage-context-efficiently)
- [Q42. How would you optimize latency?](#q42-how-would-you-optimize-latency)
- [Q43. How would you evaluate tool selection?](#q43-how-would-you-evaluate-tool-selection)
- [Q44. How would you build an enterprise RAG pipeline?](#q44-how-would-you-build-an-enterprise-rag-pipeline)
- [Q45. What are common production bottlenecks?](#q45-what-are-common-production-bottlenecks)



### Comparisons

- [Q46. LangChain vs. LlamaIndex](#q46-langchain-vs-llamaindex)
- [Q47. LangChain vs. Semantic Kernel](#q47-langchain-vs-semantic-kernel)
- [Q48. LangChain vs. Haystack](#q48-langchain-vs-haystack)
- [Q49. LangGraph vs. LangChain](#q49-langgraph-vs-langchain)



### From addonQA (LangChain extras)

- [Q50. How have "Chains" evolved in LangChain from legacy versions to modern architecture?](#q50-how-have-chains-evolved-in-langchain-from-legacy-versions-to-modern-ar)
- [Q51. What problems did LCEL and the Runnable interface solve?](#q51-what-problems-did-lcel-and-the-runnable-interface-solve)
- [Q52. What are the primary Runnable primitives used in LCEL pipelines?](#q52-what-are-the-primary-runnable-primitives-used-in-lcel-pipelines)
- [Q53. What are "Deep Agents" and what specific problems do they solve?](#q53-what-are-deep-agents-and-what-specific-problems-do-they-solve)
- [Q54. What are the 4 core pillars of Deep Agent architecture?](#q54-what-are-the-4-core-pillars-of-deep-agent-architecture)
- [Q55. How do Claude Skills integrate with Deep Agents compared to static instructions?](#q55-how-do-claude-skills-integrate-with-deep-agents-compared-to-static-ins)
- [Q56. What is](#q56-what-is-recursivecharactertextsplitter-and-why-is-it-the-industry-stan) `RecursiveCharacterTextSplitter` [and why is it the industry standard?](#q56-what-is-recursivecharactertextsplitter-and-why-is-it-the-industry-stan)
- [Q57. What are the major text splitting strategy types?](#q57-what-are-the-major-text-splitting-strategy-types)
- [Q58. Outline the architecture for Project 1: Conversational FAQ & Ticket Escalation Agent.?](#q58-outline-the-architecture-for-project-1-conversational-faq-ticket-escal)
- [Q59. Outline the architecture for Project 2: Interactive Resume Document Writing Agent.?](#q59-outline-the-architecture-for-project-2-interactive-resume-document-wri)
- [Q60. Explain the 5 core components and data flow of a LangChain RAG pipeline.?](#q60-explain-the-5-core-components-and-data-flow-of-a-langchain-rag-pipelin)
- [Q61. What is the Runnable Interface, and how does it work under the hood?](#q61-what-is-the-runnable-interface-and-how-does-it-work-under-the-hood)
- [Q62. How does Memory work in LangChain, and what are the architectural trade-offs between Buffer, Summary, and VectorStore memory?](#q62-how-does-memory-work-in-langchain-and-what-are-the-architectural-trade)
- [Q63. What is a](#q63-what-is-a-skillmd-file-and-how-is-it-used-in-agentic-architectures) `SKILL.md` [file, and how is it used in agentic architectures?](#q63-what-is-a-skillmd-file-and-how-is-it-used-in-agentic-architectures)
- [Q64. How do you approach Testing and Evaluation for non-deterministic AI agents and graphs?](#q64-how-do-you-approach-testing-and-evaluation-for-non-deterministic-ai-ag)
- [Q65. How do you handle Observability and Tracing using LangSmith?](#q65-how-do-you-handle-observability-and-tracing-using-langsmith)
- [Q66. How do you approach Testing and Evaluation for AI agents?](#q66-how-do-you-approach-testing-and-evaluation-for-ai-agents)

---



## Basics



### Q1. What is LangChain?

Example: [q01_what_is_langchain.py](examples/langchain/q01_what_is_langchain.py)

**Answer.**

1. **What is LangChain?**
  - It is an open-source framework designed to help developers build LLM-powered applications by providing standard, unified interfaces for models, tools, and external data sources.
  - Its primary benefit is **provider abstraction**: you can swap out your underlying LLM provider (e.g., moving from OpenAI to Anthropic or a local model via Ollama) without having to rewrite your core application logic.
2. **The LangChain 1.0 Evolution (Post-October 2025):**
  - The framework has matured significantly. LangChain 1.0 streamlined the ecosystem into a smaller, highly stable core explicitly optimized for **autonomous agents**.
  - The modern package centers around utility functions like `create_agent`, standardized message structures, and an execution **middleware system** that gives developers precise control over the agent loop.
3. **Handling Legacy Code in Interviews:**
  - Older components like `LLMChain` and `AgentExecutor` have been moved out into a separate package called `langchain-classic` and are officially **deprecated**.
  - *Interview pro-tip:* If an interviewer brings up older patterns, explicitly noting that they are deprecated shows you are up-to-date with modern production standards.



### 2. Real-World Interview Example

> *"If an interviewer asks how LangChain fits into our modern software stack, I explain that it acts as our foundational abstraction layer. Since LangChain 1.0 streamlined its core around agents and execution middleware, we use its standardized components to handle model calls and tool bindings cleanly. This protects our codebase from vendor lock-in—if we need to transition our backend agents from OpenAI to Claude, our core application code remains entirely untouched because LangChain normalizes the underlying provider APIs."*



### 3. Keywords Explained

- **Provider Abstraction:** A software design pattern that hides vendor-specific details behind a universal interface so you can switch tools or APIs seamlessly.
- **Agent Loop:** The continuous cycle where an LLM analyzes user input, decides whether to call a tool, reviews the tool output, and determines when the final answer has been reached.
- **Middleware System:** A modular layer of code that intercepts and controls requests or execution steps as they flow through your application loop (e.g., logging, rate-limiting, or modifying prompts on the fly).

---



### Q2. Why would you use LangChain instead of calling an LLM API directly?

Example: [q02_why_not_call_the_api_directly.py](examples/langchain/q02_why_not_call_the_api_directly.py)

**Answer.**

1. **The Rule for Simple Tasks:**
  - For a basic "single prompt in, single response out" task, **do not use LangChain**. A direct API call is leaner, faster, and avoids unnecessary framework overhead.
2. **When You Need an Orchestration Layer:**
  - The moment your application demands multi-step logic—such as querying external databases, managing conversation history across restarts, or enforcing strict data schemas—you are forced to build an **orchestration layer**.
3. **The Four Pillars LangChain Provides:**
  - **Provider Abstraction:** Swap between OpenAI, Anthropic, or local models (via Ollama) using unified code.
  - **Tool Calling:** A standardized format to declare functions and safely execute what the model selects.
  - **Persistence:** Preserving state across user requests and system restarts.
  - **Structured Output:** Forcing LLM responses to validate against strict schemas instead of fragile string parsing.
4. **The Honest Downside (Crucial for Interviews):**
  - LangChain introduces deep **abstraction layers**. When a bug occurs, tracing through framework abstractions takes significantly longer than debugging a raw, direct HTTP request. Acknowledging this tradeoff proves you are a pragmatic engineer, not a blind framework fanboy.



### 2. Real-World Interview Example

> *"If an interviewer asks why we shouldn't just write raw HTTP requests to OpenAI, I explain that for a simple text-generation endpoint, direct calls are fine. But for our enterprise applications—where the model needs to dynamically invoke tools, maintain state across multi-turn user threads, and return rigidly validated Pydantic models—building a custom orchestration layer from scratch is a waste of engineering time. LangChain provides those foundational abstractions out of the box. Of course, I also note the operational downside: abstraction layers can make deep debugging trickier, so we keep our implementation clean and intentional."*



### 3. Keywords Explained

- **Orchestration Layer:** The governing control logic that coordinates data flow, tool execution, memory management, and prompt handling around an LLM.
- **Structured Output:** A technique where the LLM is constrained to return responses formatted as strict data models (like JSON matching a schema) rather than unpredictable natural language text.
- **Abstraction Penalty:** The hidden engineering cost of using frameworks—while they speed up initial development, they can obscure underlying errors and make debugging more complex.

---



### Q3. What are the main components of LangChain?

Example: [q03_main_components.py](examples/langchain/q03_main_components.py)

**Answer.**

1. **The Six Pillars of LangChain:**
  - **Models:** Chat models (for generation) and embedding models (for vector representations) wrapped behind a single, unified interface.
  - **Prompts:** Reusable templates that dynamically inject variables to format the exact messages sent to an LLM.
  - **Tools:** Python functions wrapped in a format the model can read, allowing the LLM to request calculations, database lookups, or API calls.
  - **Agents:** The autonomous decision-making loop that evaluates tool outputs, decides what to run next, and determines when the task is complete.
  - **Retrievers:** Data-fetching interfaces designed to query vector stores and pull relevant documents for Retrieval-Augmented Generation (RAG).
  - **Runnables:** The core execution protocol that makes everything composable.
2. **The Interview Trap: Runnables vs. Chains:**
  - **Runnable (The Protocol):** An interface contract. Anything that implements `.invoke()`, `.stream()`, and `.batch()` is a Runnable.
  - **Chain (The Result):** The actual pipeline you get when you compose multiple Runnables together using LCEL pipes.



### 2. Real-World Interview Example

> *"If an interviewer asks me to break down LangChain's architecture, I point to its six foundational components. At the data layer, we use **Models**, **Prompts**, and **Retrievers** to feed context into our system. At the action layer, we use **Tools** and **Agents** to execute tasks dynamically. Finally, everything is tied together using **Runnables**—the universal interface protocol that lets us chain these components into clean, streamable pipelines using standard execution methods like* `.invoke()`*."*



### 3. Keywords Explained

- **Runnable Protocol:** A standardized programming contract in LangChain ensuring that any component can be invoked, streamed, or batched identically.
- **Embedding Model:** A machine learning model that converts text into high-dimensional numerical vectors, allowing computers to search documents based on semantic meaning rather than exact keyword matches.
- **Composition Interface:** A design pattern that allows small, independent software pieces to snap together cleanly to build larger, complex workflows.

---



### Q4. What problems does LangChain solve?

Example: [q04_problems_langchain_solves.py](examples/langchain/q04_problems_langchain_solves.py)

**Answer.**

### 1. Core Technical Explanation (Breaking Down the Paragraph)

Here is what this answer means, broken down into the three primary production problems LangChain and LangGraph solve:

1. **Provider Lock-In (Normalization):**
  - Every LLM provider (OpenAI, Anthropic, Google Gemini, or open-source models via Ollama) uses completely different JSON schemas, message formats, and streaming parameters.
  - LangChain acts as a translation layer, **normalizing those differences** so your core application code remains entirely agnostic to which model is powering it.
2. **State Management (Overcoming Stateless APIs):**
  - Raw LLM APIs have zero memory; every request is treated as a brand-new, isolated interaction.
  - Real-world software requires long-running conversation history and multi-turn data tracking, which LangChain handles seamlessly by integrating with **LangGraph persistence layers**.
3. **Complex Orchestration (Automating Control Flow):**
  - Writing multi-step agentic workflows by hand—managing conditional branching, error retries, parallel tool execution, and human-in-the-loop approval gates—requires writing a massive amount of brittle, custom state-machine code.
  - LangChain and LangGraph provide out-of-the-box orchestration structures so you don't have to reinvent the wheel.



### 2. Real-World Interview Example

> *"If an interviewer asks what core operational problems LangChain solves, I emphasize that it addresses the infrastructure surrounding the model, not the model itself. First, it eliminates **provider lock-in** by standardizing message schemas across OpenAI and Anthropic. Second, it bridges the gap for **stateless APIs** by pairing with LangGraph to maintain persistent conversation history. Third, it removes the nightmare of writing manual control flow for **orchestration**, letting us declaratively manage tool loops, retries, and human-in-the-loop pauses without building a custom state machine from scratch."*



### 3. Keywords Explained

- **Provider Agnostic:** Software architecture designed in a way that allows you to swap out infrastructure vendors (like moving from one AI provider to another) with zero code changes.
- **Stateless API:** A backend service that processes incoming requests independently without storing any memory or context about what happened in previous requests.
- **Declarative Control Flow:** A programming style where you define *what* the workflow should do (using graphs, nodes, and edges) rather than manually writing complex procedural `if/else` loops to track execution state.

---



### Q5. What types of applications can be built with LangChain?

Example: [q05_application_types.py](examples/langchain/q05_application_types.py)

**Answer.**

1. **Retrieval-Augmented Generation (RAG) Systems:**
  - Applications that ingest private corporate documents (PDFs, codebases, wikis), chunk them into vector databases, and allow an LLM to answer questions grounded in proprietary data rather than relying solely on its public training memory.
2. **Autonomous Agents:**
  - Intelligent workflows where the LLM dynamically decides which tools, external APIs, or SQL databases to query to complete a multi-step objective (e.g., investigating a production server crash or generating financial reports).
3. **Context-Aware Chatbots & Support Assistants:**
  - Conversational interfaces that maintain long-term multi-turn memory across user sessions, integrating with persistent checkpointers to handle complex customer service or IT helpdesk tasks.
4. **Structured Data Extraction Pipelines:**
  - Automated systems that take messy, unstructured text (like invoices, resumes, or medical notes) and use strict validation schemas (like Pydantic) to output clean, parsable JSON data.

*Interview Pro-Tip:* The formula for acing this question is **Category + Specific Example**. State the broad category, and immediately follow up with a concise 2-sentence description of a system you've personally built or worked on.

### 2. Real-World Interview Example

> *"When building with LangChain, I focus on four major application categories: RAG systems for private document search, autonomous agents that orchestrate external APIs, context-aware support chatbots, and structured data extraction pipelines. For instance, in a recent project, I built an **autonomous agent** for IT incident response: it took raw user tickets, queried our internal observability APIs, ran diagnostic tools, and generated a validated remediation plan before pausing for human approval."*



### 3. Keywords Explained

- **Retrieval-Augmented Generation (RAG):** An architectural pattern that combines external database retrieval with LLM text generation to provide accurate, up-to-date, and grounded answers.
- **Unstructured Text:** Raw human language data (such as emails, logs, or PDFs) that lacks a rigid database format, requiring intelligent parsing to be usable by backend software.
- **Grounded Output:** AI responses that are strictly tied and restricted to verified source documents or data provided in the prompt context, minimizing hallucinations.

---



## Core concepts



### Q6. What are chains?

Example: [q06_what_are_chains.py](examples/langchain/q06_what_are_chains.py)

**Answer.**

- **What is a Chain?**
  - A chain is a predictable, sequential pipeline where data flows step-by-step: the output of one component becomes the input of the next. For example: A prompt template feeds into an LLM, whose response feeds into an output parser.
- **LCEL (LangChain Expression Language):**
  - Modern LangChain builds these pipelines using the standard Unix-style pipe operator (`|`).
  - `chain = prompt | model | parser`
- **Every Piece is a Runnable:**
  - Because every component (`prompt`, `model`, `parser`) implements the **Runnable protocol**, they can snap together effortlessly. The combined chain *itself* becomes a new Runnable, meaning you can call `.invoke()`, `.stream()`, or `.batch()` on the whole pipeline just like any single piece.
- **When to Use a Chain (Fixed vs. Dynamic Paths):**
  - Use chains when your workflow is **fixed**—meaning you know every step in advance and the execution order never changes. They are clean, highly predictable, and easy to test. (If your path needs to branch dynamically based on LLM decisions or tool outputs, you graduate from a chain to a LangGraph agent).

```python
chain = prompt | model | parser
result = chain.invoke({"topic": "vector databases"})
```



### 2. Real-World Interview Example

> *"If an interviewer asks how we structure linear data processing tasks in LangChain, I explain that we use **LCEL (LangChain Expression Language)** to build clean, declarative chains. By leveraging the pipe operator (*`|`*), we compose prompts, chat models, and output parsers into a single unified Runnable pipeline. Since every component implements the Runnable interface, we can seamlessly invoke, stream, or batch the entire sequence. We use chains whenever the workflow path is fixed and predictable, reserving LangGraph for complex, multi-step loops and dynamic branching."*



### 3. Keywords Explained

- **LCEL (LangChain Expression Language):** A declarative syntax in LangChain used to compose Runnables together into production-ready chains using the pipe operator (`|`).
- **Declarative Pipeline:** A programming style where you specify *what* components should be linked together rather than manually writing procedural code to pass data between functions.
- **Fixed Control Flow:** A workflow where the execution order is hardcoded and deterministic (Step A always leads to Step B), contrasting with dynamic, LLM-driven loops.

---



### Q7. What are tools?

Example: [q07_what_are_tools.py](examples/langchain/q07_what_are_tools.py)

**Answer.**

1. **What is a Tool?**
  - A tool is a standard Python function wrapped with metadata and a decorator (like `@tool`) that makes it readable and invokable by a Large Language Model.
2. **The Docstring is Actually Part of the Prompt:**
  - This is a critical interview talking point: **the docstring is not optional documentation for humans.** The LLM reads the function name, parameter types, and docstring directly at runtime to understand *what* the tool does, *when* to use it, and *what* arguments to pass. Vague docstrings lead to tool-calling errors.
3. **From Text Generator to Agent:**
  - Without tools, an LLM is passive—it can only talk about doing things or hallucinate data. Tools bridge the gap, giving the model hands to interact with databases, external APIs, and internal software systems.
4. **The Security Boundary (The Model Never Executes Code):**
  - An LLM cannot run code or touch your database directly. When a model "calls" a tool, it merely outputs a structured JSON request containing the tool name and arguments. **Your application code** intercepts that request, verifies it, and decides whether to safely execute it.



### 2. Real-World Interview Example

> *"If an interviewer asks how we bridge LLMs with external systems, I explain that we use **LangChain tools** wrapped with the* `@tool` *decorator. I always emphasize that the docstring functions as a core part of the system prompt—if you write a weak description, the model won't know when to trigger the tool. Crucially, I note that the model never executes code directly; it simply returns a structured function call request, and our application execution layer securely runs the code and feeds the result back into the loop."*



### 3. Keywords Explained

- **Tool Decorator (**`@tool`**):** A helper utility in LangChain that inspects a standard Python function's signature and docstring, automatically formatting it into a schema that the LLM provider's API understands.
- **Function Calling Schema:** A structured JSON definition sent to the LLM detailing available tools, expected arguments, and descriptions so the model can generate valid invocation payloads.
- **Execution Boundary:** The strict security separation between the LLM's text output and your application's actual runtime environment, ensuring the model cannot perform unauthorized actions.

```python
from langchain_core.tools import tool
@tool
def get_stock_price(ticker: str) -> float:
"""Return the current price for a stock ticker."""
return fetch_price(ticker)
```

---



### Q8. What are agents?

Example: [q08_what_are_agents.py](examples/langchain/q08_what_are_agents.py)

**Answer.**

1. **The Autonomous Loop:**
  - Unlike a static chain, an agent is a **dynamic execution loop**. The model analyzes the current state, picks a tool, inspects the tool's output, and uses that new information to decide whether to run another tool or stop and provide a final answer.
2. **Modern LangChain 1.x Architecture (**`create_agent`**):**
  - In modern LangChain, agents are instantiated using helper functions like `create_agent`. Crucially, `create_agent` **runs directly on the LangGraph runtime**. This means you get production-grade state persistence and human-in-the-loop capabilities out of the box without having to manually code custom graph nodes and edges.
3. **The Trade-Off: Agents vs. Chains:**
  - Use **chains** when the workflow path is fixed and predictable.
  - Use **agents** when you *cannot* predict the path in advance (e.g., troubleshooting an open-ended IT ticket where the next step depends entirely on what the previous log search uncovers).
  - *The cost:* That flexibility introduces non-determinism, increases latency, and consumes significantly more API tokens.



### 2. Real-World Interview Example

> *"If an interviewer asks when to use an agent versus a standard chain, I explain that it comes down to predictability. If our workflow follows a strict, known sequence—like formatting a prompt, calling an LLM, and parsing the JSON—we use a lightweight **chain**. But if the task requires exploratory problem-solving, such as an IT triage agent that must dynamically search logs, query databases, and decide its own next steps based on error outputs, we deploy an **agent** using LangChain's* `create_agent` *on top of the LangGraph runtime for built-in persistence."*



### 3. Keywords Explained

- **Dynamic Loop:** An iterative program structure where the number of execution steps is not hardcoded, but instead adapts dynamically based on intermediate runtime data and model decisions.
- **LangGraph Runtime:** The underlying state machine engine that powers modern LangChain 1.x agents, handling checkpoints, execution threads, and state transitions automatically.
- **Determinism:** A property of software where running the exact same input code always yields the exact same execution path and output, which agents sacrifice in exchange for adaptive flexibility.

---



### Q9. What are Runnables?

Example: [q09_what_are_runnables.py](examples/langchain/q09_what_are_runnables.py)

**Answer.**

1. **The Universal Interface Contract:**
  - A **Runnable** is the foundational abstraction pattern in LangChain. Whether it is a prompt template, a chat model, an output parser, a retriever, or a complete chain, every major component implements the Runnable interface.
2. **The Four Core Methods:**
  - `.invoke(input)`: Runs the component once with a single input payload.
  - `.batch(inputs)`: Executes the component concurrently across a list of multiple inputs, drastically speeding up batch data processing.
  - `.stream(input)`: Yields output data progressively in chunks as it is generated (essential for responsive web UI text streaming).
  - `.astream(input)`: The asynchronous counterpart for high-concurrency event loops.
3. **Why the Pipe Operator (**`|`**) Works:**
  - Because every individual piece adheres to the Runnable protocol, they speak the same language. When you write `prompt | model`, you are composing two Runnables into a **new, larger Runnable**, which inherits all the same execution methods.
4. **Built-In Capabilities:**
  - You never have to write custom threading logic for batching or streaming data. By inheriting the Runnable protocol, any composed pipeline instantly gets batching and streaming capabilities out of the box.



### 2. Real-World Interview Example

> *"If an interviewer asks how LangChain components seamlessly snap together, I explain that **Runnables** are the universal design pattern powering the entire framework. Because prompts, models, and output parsers all implement the exact same execution contract—featuring* `.invoke()`*,* `.batch()`*, and* `.stream()`*—they can be piped together using LCEL. For instance, when we build a user-facing chatbot API, wrapping our pipeline in a Runnable means we can instantly support real-time token streaming via* `.stream()` *or high-throughput evaluations using* `.batch()` *without writing any custom concurrency code."*



### 3. Keywords Explained

- **Interface Contract:** An architectural agreement defining a strict set of methods and behaviors that any implementing class or component must support.
- **Asynchronous Streaming (**`.astream()`**):** A non-blocking execution method that yields data chunks incrementally over an asynchronous event loop, preventing server thread starvation.
- **Component Composition:** The software design principle of combining small, independent functional units to construct complex, modular systems.

---



### Q10. What are output parsers?

Example: [q10_output_parsers.py](examples/langchain/q10_output_parsers.py)

**Answer.**

1. **What is an Output Parser?**
  - Large language models fundamentally output raw strings of text. However, production applications require structured, typed data (like a Python dictionary or a validated Pydantic object). An output parser bridges that gap by converting, cleaning, and validating the model's text response.
2. **The Pydantic Parser Example:**
  - Using classes like `PydanticOutputParser` combined with a Pydantic `BaseModel` (e.g., `Review` with `severity: str` and `issues: list[str]`) ensures that if the model's output fails to match your schema, the parser raises a validation error instead of passing bad data downstream.
3. **The Historical Context (Why Parsers Existed):**
  - In older LLM paradigms, models frequently outputted malformed JSON or conversational filler alongside their data. Output parsers were mandatory to strip out markdown text blocks and clean up messy responses.
4. **Modern LangChain 1.x (Native Structured Output):**
  - Modern LLM APIs (and LangChain 1.x integrations) now support **native structured output** via parameters like `response_format`. The model's underlying API guarantees valid JSON matching your schema natively at the token level, eliminating the need for an extra text-parsing step or a secondary cleanup call.



### 2. Real-World Interview Example

> *"If an interviewer asks how we enforce data shapes from LLM responses, I explain that while traditional LangChain workflows relied heavily on **output parsers** (like* `PydanticOutputParser`*) to clean and validate raw string outputs, modern architectures have evolved. With LangChain 1.x and modern LLM providers supporting native structured output via* `response_format`*, we let the provider enforce the JSON schema at the token level. I still use traditional parsers when working with local open-source models or providers that lack robust native structured generation, but native generation is our production standard for safety and performance."*



### 3. Keywords Explained

- **Output Parser:** A utility component in LangChain responsible for taking raw LLM text strings and parsing them into structured types or validated objects.
- **Pydantic Model:** A Python data-validation library that uses type annotations to enforce runtime data structures, serving as the schema standard across modern LLM applications.
- **Native Structured Output (**`response_format`**):** A provider-level feature that constrains the LLM's decoding process so that every generated token strictly conforms to a specified JSON schema.

```python
from pydantic import BaseModel
from langchain_core.output_parsers import PydanticOutputParser
class Review(BaseModel):
severity: str
issues: list[str]
```

---



## RAG



### Q11. How does LangChain implement RAG?

Example: [q11_how_rag_works.py](examples/langchain/q11_how_rag_works.py)

**Answer.**

### 1. Core Technical Explanation (Breaking Down the Paragraph)

Here is what this answer means, broken down into four key engineering concepts:

1. **What is RAG?**
  - **Retrieval-Augmented Generation (RAG)** is an architectural pattern where you fetch relevant documents at query time and inject them directly into the prompt. This forces the LLM to answer using your private company data rather than relying solely on its public training memory.
2. **The Two Separate Pipelines:**
  - **The Indexing Pipeline (Offline):** Runs in the background ahead of time. You load documents, split them into manageable text chunks, pass them through an embedding model, and save the resulting vectors into a vector database. This only runs once (or incrementally when source data updates).
  - **The Retrieval Pipeline (Online/Per-Query):** Runs in real time whenever a user asks a question. The system embeds the query, searches the vector store for the top $k$ most semantically similar chunks, and passes them into the prompt.
3. **The LCEL Retrieval Chain Example:**
  - In LangChain, this is wired up cleanly using the pipe operator: `{"context": retriever, "question": RunnablePassthrough()} | prompt | model`. The retriever automatically fetches the documents, while `RunnablePassthrough` carries the raw user question straight through to the prompt template.
4. **The Engineering Truth (Garbage In, Garbage Out):**
  - A crucial interview takeaway: **Most RAG engineering happens in the indexing pipeline.** Document parsing quality, chunk sizes, overlap settings, and metadata tags completely determine your retrieval ceiling. If your chunking strategy is flawed, no amount of prompt tuning or fancy LLM selection will save your answers.



### 2. Real-World Interview Example

> *"If an interviewer asks how we implement RAG in production, I explain that we split the architecture into an **offline indexing pipeline** and an **online retrieval pipeline**. The indexing pipeline handles parsing, chunking, and vector embedding into our database ahead of time. The retrieval pipeline executes at query time, using LangChain's vector store retriever piped into an LCEL chain to inject context dynamically. I always emphasize that RAG success isn't about prompt engineering; it's heavily dictated by the indexing pipeline—specifically how clean your text parsing, chunk sizes, and metadata filters are."*



### 3. Keywords Explained

- **Chunking:** The process of breaking large documents into smaller, uniform segments of text so they fit within LLM context windows and match vector embedding granularities.
- **Embedding Model:** A machine learning model that converts text fragments into multi-dimensional numerical vectors to represent semantic meaning.
- `RunnablePassthrough`**:** A special LangChain utility runnable that takes the input it receives and passes it through unchanged, often used in parallel dictionary mappings inside LCEL chains.

```python
retriever = vector_store.as_retriever(search_kwargs={"k": 4})
```

```python
chain = (
{"context": retriever, "question": RunnablePassthrough()}
| prompt
| model)
```

---



### Q12. What is a retriever?

Example: [q12_what_is_a_retriever.py](examples/langchain/q12_what_is_a_retriever.py)

**Answer.**

### 1. Core Technical Explanation (Breaking Down the Paragraph)

Here is what this answer means, broken down into three key engineering concepts:

1. **The Definition of a Retriever:**
  - A retriever is simply an interface component defined by its signature: it **takes a query string and returns a list of documents**.
  - While vector search over embedding databases is the most common implementation, a retriever is not limited to vectors—it can query a traditional SQL database, an enterprise search API, or a file system.
2. **The Power of Hybrid Retrieval:**
  - This decoupled definition allows for advanced production patterns like **hybrid search**.
  - Semantic vector search is incredible at capturing abstract, conceptual meanings, but it frequently fails to match exact, rigid strings like product SKUs, specific serial numbers, or error codes (e.g., `ERR_CONNECTION_RESET`). Keyword search (like BM25) excels at exact matches.
3. **Combining Strategies:**
  - In a production-grade system, you wrap a keyword searcher and a vector searcher into separate retrievers, execute both in parallel, and merge/rerank their results before passing them to the LLM.



### 2. Real-World Interview Example

> *"If an interviewer asks what a retriever is, I explain that while developers usually think of vector databases, a retriever is fundamentally any component that maps a query string to documents. This abstraction is powerful because it enables **hybrid retrieval architectures**. In our technical documentation search system, pure vector search struggled to find specific error codes because embeddings blur exact alphanumeric terms. By combining a dense vector retriever with a sparse keyword (BM25) retriever and reranking the results, we dramatically improved precision for exact system error lookups."*



### 3. Keywords Explained

- **Hybrid Retrieval:** An advanced search strategy that combines semantic (vector-based) search with lexical (keyword-based) search to capture both conceptual meaning and exact term matching.
- **Semantic Search:** A search method that looks at the underlying intent and conceptual meaning of a query rather than performing a strict character-by-character text match.
- **Lexical / Keyword Search:** Traditional text matching (such as BM25 or SQL `LIKE` queries) that looks for exact string occurrences of search terms in a document.

---



### Q13. How do vector databases fit into LangChain?

Example: [q13_vector_databases.py](examples/langchain/q13_vector_databases.py)

**Answer.**

1. **What Vector Databases Do:**
  - A vector database stores high-dimensional embedding vectors and uses mathematical distance metrics (like cosine similarity) to quickly find the nearest neighbors matching a query vector.
2. **The LangChain** `VectorStore` **Abstraction:**
  - LangChain wraps disparate vector databases (Chroma, Pinecone, Qdrant, pgvector) behind a single, unified `VectorStore` interface.
3. **The Prototype-to-Production Workflow:**
  - Because the methods are identical, you can spin up a local prototype using lightweight tools and seamlessly scale to a managed cloud database later without rewriting your retrieval application code.
4. **How to Choose a Vector Database in an Interview:**
  - **Chroma / FAISS:** Perfect for local development, small datasets, or offline prototyping.
  - `pgvector`**:** The ideal choice if your enterprise stack already runs PostgreSQL and you want to avoid spinning up and managing a separate, dedicated database service.
  - **Pinecone / Qdrant:** The go-to managed cloud solutions when you require enterprise scale, high query volume, and advanced metadata filtering.



### 2. Real-World Interview Example

> *"If an interviewer asks how we select a vector database for an enterprise project, I explain that while LangChain's unified* `VectorStore` *interface makes vendor swapping trivial, the actual infrastructure choice depends heavily on operational overhead and scale. For local prototyping, we use **Chroma**. If the client's stack is already anchored in PostgreSQL, we leverage* `pgvector` *to avoid introducing a new database to our infrastructure. But for massive public-facing systems requiring high-throughput filtering and managed scalability, we deploy dedicated managed services like **Pinecone** or **Qdrant**."*



### 3. Keywords Explained

- **VectorStore Interface:** LangChain's standardized programming contract that normalizes database interactions (like `.add_texts()` and `.similarity_search()`) across different vector providers.
- **Embedding Distance Metric:** The mathematical formula (such as Cosine Similarity, Euclidean Distance, or Dot Product) used by vector databases to calculate how closely related two text vectors are.
- **Metadata Filtering:** The ability to restrict a vector search query using structured tags or attributes (e.g., searching only documents where `department == "engineering"`), dramatically increasing retrieval precision.

---



### Q14. What embedding models can LangChain use?

Example: [q14_embedding_models.py](examples/langchain/q14_embedding_models.py)

**Answer.**

### 1. Core Technical Explanation (Breaking Down the Paragraph)

Here is what this answer means, broken down into four key engineering concepts:

1. **Supported Embedding Providers:**
  - LangChain integrates with both hosted cloud APIs (like OpenAI, Cohere, and Voyage AI) and local open-source options (such as Hugging Face's `sentence-transformers` models for air-gapped or privacy-strict environments).
2. **The Two Decision Drivers (Dimension Count & Domain Fit):**
  - **Dimension Count:** The size of the output vector (e.g., 384 vs. 1536 dimensions) directly impacts storage space, memory usage, and vector search query speed.
  - **Domain Fit:** General-purpose models work well for standard web text, but specialized fields like legal contracts, medical literature, or codebases require **domain-tuned embedding models** to capture industry-specific semantic nuances.
3. **The Golden Rule (Never Mix Embedding Models):**
  - This is a classic interview trick question: **The exact same embedding model used to index your documents must be used to embed the incoming user query.**
  - If you switch models, the underlying vector math changes entirely, rendering all previously stored vectors completely useless. Changing your embedding model means you must **reindex your entire database from scratch**.



### 2. Real-World Interview Example

> *"If an interviewer asks how we choose and manage embedding models in a production RAG pipeline, I explain that we select providers like OpenAI or open-source Hugging Face models based on privacy constraints, dimension size, and domain specificity. For instance, when building a medical diagnostic search tool, a standard general-purpose model underperformed, so we switched to a domain-tuned biomedical embedding model. Crucially, I always emphasize the number one rule of vector infrastructure: you can never hot-swap your embedding model mid-project without triggering a complete database reindexing, because old and new vectors live in entirely different mathematical spaces."*



### 3. Keywords Explained

- **Embedding Dimension:** The length of the numerical array produced by an embedding model (e.g., 1536 floats for OpenAI's `text-embedding-3-small`), representing the complexity and coordinate depth of the semantic space.
- **Domain-Tuned Model:** An embedding model specifically trained or fine-tuned on specialized corporate data (like legal text or code repositories) to vastly improve retrieval accuracy in niche industries.
- **Reindexing:** The resource-intensive offline process of pulling all source documents, breaking them into chunks, and running them through a new embedding model to regenerate an entirely fresh vector database.

---



### Q15. How would you improve retrieval quality?

Example: [q15_improve_retrieval.py](examples/langchain/q15_improve_retrieval.py)

**Answer.**

When optimizing retrieval quality, you should never guess. Bad LLM answers almost always stem from poor retrieval rather than a weak model. Improving retrieval requires working systematically through strategies sorted by complexity and impact:

1. **Inspect Before Changing (The Golden Rule):**
  - Before tweaking parameters, actually pull the retrieved chunks for a failing query and read them. Identify whether the relevant information was missing entirely, buried too deep, or lost due to noise.
2. **Chunking Strategy:**
  - Avoid naive, fixed-character counts. Chunks that are too small lose crucial context; chunks that are too large dilute the embedding with irrelevant data. Instead, split documents on structural semantic boundaries like headings, sections, or paragraphs.
3. **Metadata Filtering:**
  - Attach critical tags during indexing (e.g., `document_type`, `date`, `department`). At query time, filter your database based on these tags before running similarity ranking, drastically reducing the search space.
4. **Hybrid Search:**
  - Combine dense vector search (for semantic meaning) with sparse keyword search (like BM25 for exact codes, names, or serial numbers) so exact terms are never missed.
5. **Reranking:**
  - Cast a wide net by retrieving 20 to 50 candidate chunks cheaply using vector search, then pass them through a powerful cross-encoder model to rescore and filter down to the top 4 most precise chunks.
6. **Query Rewriting / Expansion:**
  - Users naturally write short, ambiguous questions. Automatically expanding a single user query into multiple semantic variations and merging the retrieved results catches context that a single phrase would miss.



### 2. Real-World Interview Example

> *"If an interviewer asks how I systematically debug and improve a low-performing RAG pipeline, I explain that I start by inspecting the retrieved chunks rather than tuning the prompt. From there, I optimize iteratively: I move from fixed-size chunking to structure-aware semantic splitting, add metadata filters to narrow down namespaces, and implement a **hybrid search** strategy to catch exact keyword matches. Finally, if precision is still lacking, I introduce a **reranking** step—pulling a wide candidate set via vector search and using a cross-encoder to bubble the best context straight to the top."*



### 3. Keywords Explained

- **Cross-Encoder Reranker:** A high-precision machine learning model that evaluates a query and a document chunk *together simultaneously* to score their precise relevance, unlike embedding models that evaluate them independently (bi-encoders).
- **Sparse Keyword Search:** Traditional lexical search algorithms (like BM25) that find exact string token overlaps rather than semantic vector distances.
- **Semantic Splitting:** The technique of breaking documents apart based on topical changes or paragraph structures rather than arbitrary, rigid character lengths.

---



### Q16. How do you reduce hallucinations in a RAG application?

Example: [q16_reduce_hallucinations.py](examples/langchain/q16_reduce_hallucinations.py)

**Answer.**

1. **RAG Grounds, It Doesn't Erase:**
  - A common misconception is that adding RAG completely eliminates hallucinations. In reality, RAG **mitigates** hallucinations by supplying external facts, but if the retrieval step pulls irrelevant documents or the context is ambiguous, the model can still fabricate plausible-sounding answers.
2. **Four Practical Levers to Cut Hallucinations:**
  - **Context Highlighting (Strict Prompting):** Explicitly instruct the model in the system prompt to use *only* the provided context and to refuse to answer (e.g., *"If the context does not contain the answer, say 'I don't know'")* when it lacks information.
  - **Require Citations:** Force the model to attach a source reference or quote for every claim it makes, turning hidden unsupported statements into visible failures.
  - **Similarity Thresholds (Empty Retrieval is Valid):** If retrieved chunks fall below a specific mathematical similarity score, drop them entirely rather than forcing the model to read irrelevant text. **Returning an empty context and refusing to answer is a valid, preferred outcome.**
  - **Grounding Check / Verification Pass:** Run a secondary verification step (either via an extra model call or an automated evaluation framework) to cross-reference every claim in the draft answer against the raw source text.
3. **The Production Reality:**
  - While validation passes and extra LLM calls increase latency and token costs, they are crucial for high-stakes enterprise applications (like medical, legal, or financial systems) where factual accuracy is non-negotiable.



### 2. Real-World Interview Example

> *"If an interviewer asks how we actively minimize hallucinations in a production RAG system, I explain that RAG reduces hallucinations by grounding the model in retrieved facts, but it isn't a silver bullet. To lock down reliability, we implement a multi-layered defense: we enforce strict system prompts telling the model to refuse queries when context is missing, we require inline citations for every assertion, and crucially, we set a **similarity threshold** on our retriever. If a user query matches nothing in our vector database, returning 'I don't know' is always better than passing low-scoring noise that forces the LLM to hallucinate."*



### 3. Keywords Explained

- **Faithfulness / Groundedness:** A metric tracking how strictly an LLM's generated response adheres exclusively to the provided context without introducing outside assumptions.
- **Similarity Threshold:** A numerical score cutoff applied to vector search results to filter out weak or irrelevant chunks before they ever reach the LLM context window.
- **Refusal Behavior:** Programmed or prompted instructions that force an AI agent or chatbot to decline answering when source data is insufficient, preventing speculative guesswork.

---



## Agents



### Q17. What is an AI agent?

Example: [q17_what_is_an_ai_agent.py](examples/langchain/q17_what_is_an_ai_agent.py)

**Answer.**

An agent is a system where the model decides the control flow. You give it a goal and a set of tools, and it chooses which tool to call, reads the result, and decides whether to continue.
You can compare that to a normal program, where you write the branching logic yourself. In an agent, the model writes it at runtime.
The loop is short:
Model receives the goal and available tools
Model returns a tool call
Your code runs the tool and returns the result
Model sees the result and either calls another tool or answers
Everything else, such as memory and human approval, is built around that loop.

---



### Q18. How do LangChain agents choose tools?

Example: [q18_how_agents_choose_tools.py](examples/langchain/q18_how_agents_choose_tools.py)

**Answer.**

Through the model's tool-calling ability, not through anything LangChain does. This confuses candidates who assume the framework has selection logic in it.
**LangChain converts your tool definitions into the schema format the provider expects and sends them with the prompt. The model reads the tool names, descriptions, and parameter types, then returns a structured request naming one tool and its** arguments. LangChain executes it and feeds the result back.
So tool selection quality depends on three inputs you control:

- **Descriptions:** Write what the tool does and when to use it, not what it returns
- **Names:** search_internal_docs tells the model more than search_v2
- **Count:** Selection accuracy drops as the tool list grows

That last point comes up often. If you're adding thirty tools at one agent, you should consider splitting the work across subagents, each with a small focused set.

---



### Q19. What is the difference between a chain and an agent?

Example: [q19_chain_vs_agent.py](examples/langchain/q19_chain_vs_agent.py)

**Answer.**

Who decides what happens next.
In a chain, you decide at build time. The steps run in the order you wrote them, every single execution takes the same path, and the number of model calls is known before you start.
In an agent, the model decides at runtime. The path changes per input, the number of model calls is unknown, and two identical questions can produce different execution traces.
The interview answer is that a chain is a program and an agent is a program that writes itself as it runs.

---



### Q20. When should you avoid agents?

Example: [q20_when_to_avoid_agents.py](examples/langchain/q20_when_to_avoid_agents.py)

**Answer.**

Whenever a chain will do. That's the short answer, and it's the one most candidates skip past.
Don't go with the agent when the workflow is known. If your application always retrieves, then summarizes, then formats, hand-writing those three steps gives you lower cost and predictable errors.
Also, skip it when you need cost ceilings. An agent that loops five times on a hard question costs five times what you budgeted, and a bad prompt can turn that into fifteen.
Finally, skip it when errors are expensive. Agents that can send emails or do something sensitive need approval, and at that point you've rebuilt a workflow with extra steps.
The general idea is that if you can draw the flowchart, build the flowchart.

---



### Q21. How do you evaluate an agent?

Example: [q21_evaluate_an_agent.py](examples/langchain/q21_evaluate_an_agent.py)

**Answer.**

Checking the final answer isn't enough. An agent can reach a correct conclusion through four wasted tool calls, and that gets expensive with repetition.
Evaluate at two levels.
Trajectory covers the path. Did the agent call the right tools, in a sensible order, without redundant steps? You build this from traced runs, comparing the actual tool sequence against what a correct run should look like.
Outcome covers the result. Is the final answer correct, grounded in what the tools returned, and in the format your application expects?
Track the operational numbers alongside both. Steps per run and tokens per run will tell you whether an agent that works in testing can handle production traffic.
Build the evaluation set from failures. Every time an agent takes a wrong path, that input becomes a test case, and the set gets more useful over time than anything you'd write upfront.

---



## Memory



### Q22. What is memory in LangChain?

Example: [q22_what_is_memory.py](examples/langchain/q22_what_is_memory.py)

**Answer.**

LLM APIs are stateless. Every request arrives with no knowledge of what came before, so anything the model should remember has to be sent again in the prompt.
Memory is the layer that decides what gets sent. It stores conversation state between turns and selects what goes back into context on the next call.
LangChain splits this into two kinds:

- **Short-term memory:** History within a single conversation, scoped to a thread_id
- **Long-term memory:** Facts that persist across conversations, scoped to a user_id

If you've ever noticed how ChatGPT or Claude point to stuff from past conversations, that's a long-term memory.

---



### Q23. What memory types are available?

Example: [q23_memory_types.py](examples/langchain/q23_memory_types.py)

**Answer.**

This is where the version matters. The classic memory classes - buffer, buffer window, summary, and entity - are deprecated and scheduled for removal in 2.0. They are located in langchain-classic now.
The current approach uses LangGraph persistence.
Checkpointers handle short-term memory. They save graph state after each step, keyed by thread. InMemorySaver is good enough for development, but something like PostgresSaver is worth looking into for anything further.

```python
from langgraph.checkpoint.memory import InMemorySaverfrom langchain.agents import create_agent
```

```python
agent = create_agent(model="openai:gpt-4o", tools=tools, checkpointer=InMemorySaver())
agent.invoke({"messages": [...]}, config={"configurable": {"thread_id": "user-42"}})
```

Stores handle long-term memory. A BaseStore keeps user-scoped facts outside any single thread, so an agent can recall a preference from a conversation three weeks ago.
Memory has been rewritten more than once as the framework evolved, and saying so in an interview is a point in your favor. It shows you've tracked the migrations and been around long enough to notice.

---



### Q24. How would you summarize conversation history?

Example: [q24_summarize_history.py](examples/langchain/q24_summarize_history.py)

**Answer.**

You replace old messages with a model-generated summary and keep recent messages unchanged. Recent messages carry detail the user expects you to remember. Older messages usually only need the gist.
LangChain 1.x has summarization middleware for this, so you don't build the trigger logic yourself. It watches token count against the model's context window and compresses history when you cross the threshold.

```python
from langchain.agents.middleware import SummarizationMiddleware
```

```python
agent = create_agent(
model="openai:gpt-4o",
tools=tools,
middleware=[SummarizationMiddleware(model="openai:gpt-4o-mini")])
```

Two design decisions come with it. Summarizing costs an extra model call, so using a cheaper model for the summary is generally a good idea. And summaries are lossy by definition, which means anything the application needs exactly (think order number, a deadline) belongs in structured storage, not in a summary.

---



### Q25. What problems arise with long conversations?

Example: [q25_long_conversation_problems.py](examples/langchain/q25_long_conversation_problems.py)

**Answer.**

Four, and you usually have to deal with a combination of them.

- **Cost:** You resend the full history on every turn. A conversation that grows to 50 turns means the early messages have been paid for 50 times
- **Latency:** Longer prompts take longer to process. Users can feel the conversation slowing down as it goes
- **Context overflow:** Eventually parts of the history become irrelevant for the current conversation, and without a compression strategy, the application can behave unexpectedly
- **Attention dilution:** Models handle information in the middle of a long context worse than information at the start or end. A detail from turn 12 can be present in the prompt and still get ignored.

That last one is the answer that separates candidates. Fitting inside the context window isn't the same as the model using what's in it.

---



### Q26. How would you reduce context growth?

Example: [q26_reduce_context_growth.py](examples/langchain/q26_reduce_context_growth.py)

**Answer.**

This is really common interview question, and you have four good strategies, starting with the cheapest one:

- **Trimming:** Keep the last N messages and delete the rest
- **Summarization:** Compress old turns into a summary and keep recent ones unchanged

Retrieval over history: Store past messages in a vector store and pull back only what's relevant to the current question

- **Structured extraction:** Pull facts out of the conversation into a store, then add only the facts that matter

Most production systems combine trimming with summarization, because it's simple and predictable. Retrieval over history is good for assistants with months of conversation behind them, where a summary would flatten too much.
Tool outputs also deserve a mention. A single API response can dump thousands of tokens into state, and truncating tool results before they hit the message list often saves more context than anything you do to the conversation itself.

---



## Architecture



### Q27. Design a chatbot using LangChain

Example: [q27_design_a_chatbot.py](examples/langchain/q27_design_a_chatbot.py)

**Answer.**

Start by asking what the chatbot knows. A chatbot over your own documents is a different system than one that answers from the model's training data, and interviewers often leave this ambiguous on purpose.
For a document-grounded chatbot, the components are:

- **Ingestion:** Loaders, a splitter, an embedding model, and a vector store, run offline
- **Retrieval:** A retriever that pulls relevant chunks per query
- **Generation:** A prompt template combining history, retrieved context, and the current question
- **Persistence:** A checkpointer keyed by thread_id so each user's conversation stays separate

The design questions here are history and retrieval interacting. A follow-up like "what about that one?" has no meaning as a standalone query, so you rewrite it against the conversation history before embedding it. If you skip that step, the retrieval will fail on every follow-up question.

---



### Q28. Design a document question-answering system

Example: [q28_document_qa.py](examples/langchain/q28_document_qa.py)

**Answer.**

Same skeleton as the chatbot, but different priorities. Here the answer has to be traceable, because users need to know which document a claim came from.
You need to make three decisions:

- **Chunk with metadata:** Store the source file, page number, and section with every chunk so citations are possible

Retrieve wider, then rerank: Fetch twenty candidates, rescore with a cross-encoder, pass the top four

- **Return sources with the answer:** Cite the chunks used, so a wrong answer can be traced to a wrong retrieval

Say out loud that empty retrieval is a valid result. A system that always returns four chunks will hand the model useless data when the question isn't covered, and the model will use it.

---



### Q29. Design a customer support assistant

Example: [q29_customer_support_assistant.py](examples/langchain/q29_customer_support_assistant.py)

**Answer.**

This one has an action layer, which changes the architecture. A chatbot returns text, but a support assistant looks up orders, checks company policy, issues refunds, and escalates to a human.
There are three layers you need to mention:

- **Knowledge:** RAG over help articles and policy documents
- **Actions:** Tools that work with your order system or CRM
- **Control:** Approval gates on anything with financial or account impact

The interesting design question is agent versus routing. A full agent handles open-ended requests but costs more and behaves less predictably. Routing a classified intent to a fixed chain per category costs less and fails in ways you can predict. Most production support systems run a router in front of a small set of chains and reserve the agent for what doesn't classify well with chains.
Also, escalation to a human is part of the design. Define what triggers it - low confidence or repeated failures - and how conversation state transfers.

---



### Q30. How would you build a multi-step workflow?

Example: [q30_multi_step_workflow.py](examples/langchain/q30_multi_step_workflow.py)

**Answer.**

Pick the primitive that matches how much you know upfront.
If the sequence is fixed, compose Runnables with LCEL. This is both cheap and predictable, and it's also easy to test.
If the sequence has branches and loops but you know the shape, use LangGraph. You define nodes and edges yourself, so control flow stays deterministic while allowing conditional paths. LangGraph 1.x adds per-node timeouts and node-level error handlers, which matter when one slow step shouldn't hang the whole run.
If the sequence depends on what the model finds, use an agent. This is the expensive option and belongs last.
State design is where these interviews go next. Every step reads and writes a shared state, so decide early what lives there and what gets passed along. Dumping every intermediate result into state is never optimal, so keep that in mind.

---



### Q31. How would you add human approval?

Example: [q31_human_approval.py](examples/langchain/q31_human_approval.py)

**Answer.**

LangGraph interrupts. The graph pauses before a sensitive step, persists its state through the checkpointer, and waits. Your application shows the pending action, a human approves or rejects, and the run resumes from the checkpoint.
LangChain 1.x has human-in-the-loop middleware that configures this up for agents, so you mark which tools need approval instead of building the pause logic yourself.
The design detail worth raising is that this only works with a persistent checkpointer. Approval can take minutes or days, and in-memory state doesn't survive that. PostgresSaver or something equivalent is a requirement.
Then decide what needs a gate. Reads usually don't, but writes to production systems or anything moving money do.

---



### Q32. How would you debug complex chains?

Example: [q32_debug_complex_chains.py](examples/langchain/q32_debug_complex_chains.py)

**Answer.**

Tracing, and the answer should name LangSmith. Every model call, tool call, and intermediate output gets logged with its inputs, outputs, latency, and token count, so you can see which step produced the wrong value instead of guessing from a bad final answer.
Without a trace you're more or less debugging a black box. A wrong answer from a five-step chain has five possible causes, and printing the output tells you nothing about which one it was.
There are two habits that help before you get to tracing:
Test components in isolation: Every Runnable has .invoke(), so run the retriever alone with a known query and check what comes back
Stream intermediate steps: Watching output arrive step by step shows you where a run stalls or goes off track
For agents specifically, read the trajectory rather than the answer. The tool sequence tells you whether the model misunderstood the goal or just got a bad result from a tool that worked correctly.

---



## Production



### Q33. How do you deploy LangChain applications?

Example: [q33_deploy.py](examples/langchain/q33_deploy.py)

**Answer.**

A LangChain application is a Python application, so the deployment starts out familiar. You wrap it in a web framework like FastAPI, containerize it, and run it wherever you run your other services.
Then the differences show up. Here are a couple you should note if asked this question.
Long request times. A single agent run can take 30 seconds or more. Default gateway timeouts will kill it, so you either raise them or move the work to a background queue and stream results back.
Statefulness. Conversation state can't live in process memory if you run more than one instance. It goes in a shared checkpointer backed by Postgres or Redis, so any instance can pick up any thread.
Secrets and rate limits. API keys go in a secrets manager, and provider rate limits apply across your whole app rather than per instance.
LangSmith Deployment is the managed option built for LangGraph applications, and it handles persistence, streaming, and long-running tasks for you. Self-hosting works fine too, as long as you plan for the three points above.

---



### Q34. How do you monitor LLM applications?

Example: [q34_monitor.py](examples/langchain/q34_monitor.py)

**Answer.**

Standard application monitoring tells you the service is up, but it won't tell you the quality of the answers degraded in the last couple of days.
There are two additional layers you need:

- **Infrastructure metrics:** Latency, error rate, and throughput, the same as any service
- **LLM-specific metrics:** Tokens per request, cost per request, tool call counts, and output quality

Tracing makes the second layer possible. LangSmith logs every step of a run with inputs, outputs, timing, and token counts, so a slow request can be traced to the exact model call or tool that caused it.
Then there's the part nobody thinks about until you need to justify why your app got worse. Model quality drifts even when your code doesn't change, because providers update models behind the same endpoint name. You should pin model versions where the provider allows it, and run a small evaluation set on a schedule so you find out if there are any recent drastic changes you need to account for.

---



### Q35. How do you cache responses?

Example: [q35_cache_responses.py](examples/langchain/q35_cache_responses.py)

**Answer.**

LangChain has a caching layer that sits in front of the model, and turning it on takes a few lines of code:

```python
from langchain_core.globals import set_llm_cachefrom langchain_core.caches import InMemoryCache
```

set_llm_cache(InMemoryCache())

Exact-match caching keys on the full prompt, so it only helps when the same prompt repeats word for word. That's more common than it sounds.
Semantic caching goes further. It embeds the query and returns a cached response when a new query is close enough, which catches "what's your refund policy" and "how do refunds work" as the same question. But it introduces false hits, so the similarity threshold becomes something you need to tune.
Two other options are worth naming:

- **Provider-side prompt caching:** Anthropic and OpenAI cache long shared prefixes, which reduces cost on large system prompts and retrieved context
- **Embedding caching:** Cache embeddings for repeated documents so reindexing doesn't repay for vectors you already have

---



### Q36. How do you handle API failures?

Example: [q36_api_failures.py](examples/langchain/q36_api_failures.py)

**Answer.**

Assume providers fail, because they do. Rate limits and outages are normal operating conditions rather you should account for.
Retries handle the transient stuff. LangChain 1.x has model retry middleware with configurable exponential backoff, so a 429 or a dropped connection gets retried without you writing the loop.
Fallbacks handle the rest. Every Runnable has .with_fallbacks(), which swaps in an alternative when the primary fails:

```python
model = primary_model.with_fallbacks([backup_model])
```

The design decision is what the fallback should be. A different provider protects you from one vendor going down. A smaller model from the same provider protects you from capacity limits but not from an outage. Pick based on which failure you're actually worried about.
And decide what happens when everything fails. A cached stale answer or an honest error message are all valid, but returning nothing or a 500 status isn't.

---



### Q37. How do you control costs?

Example: [q37_control_costs.py](examples/langchain/q37_control_costs.py)

**Answer.**

Costs run away because token usage grows in places nobody watches.
Start with model routing. Not every step needs your most expensive model - summarization, and query rewriting run fine on a small one, and reserving the large model for final generation reduces the bill without degrading quality where users notice.
Then work on what you send:
Trim context: Summarize old turns and truncate long tool outputs before they reach the prompt

- **Retrieve less:** Four good chunks are better than twenty mediocre ones, and reranking gets you there

Cache aggressively: Think repeated prompts and shared prefixes
Cap agent steps: A step limit will make sure you don't get into unbounded loop scenarios
Track cost per request rather than total spend. Total spend tells you the bill went up, but cost per request tells you whether that's growth or a regression.

---



### Q38. How do you evaluate application quality?

Example: [q38_evaluate_quality.py](examples/langchain/q38_evaluate_quality.py)

**Answer.**

Manual review doesn't scale past a handful of examples, so you need a dataset and a way to score against it.
Build the dataset from actual inputs. Production traces give you actual user questions, and every failure you find becomes a permanent test case. A set built this way stays useful in a way that invented examples never do.
Then pick scoring methods that match what you're checking:

- **Deterministic checks:** Format validity, schema conformance, and required fields, all cheap and exact
- **Reference comparison:** Similarity against a known-good answer, when one exists
- **LLM-as-judge:** A model scoring the output against criteria like groundedness and relevance
- **Human review:** A small sample, used to check that your automated scores match human judgment

That last one matters more than you might expect. LLM-as-judge scales well but can drift, so you calibrate it against human labels rather than trusting it outright.
For RAG specifically, evaluate retrieval separately from generation. If you only score the final answer, you can't tell whether the retriever missed the document or the model ignored it.

---



## Advanced



### Q39. How would you build a multi-agent system?

Example: [q39_multi_agent.py](examples/langchain/q39_multi_agent.py)

**Answer.**

First, argue against it. Multi-agent systems add coordination overhead and more failure modes, so a single agent with a well-chosen tool set is the better answer more often than candidates assume.
The case for splitting is tool count and context. Selection accuracy reduces as the tool list grows, and one agent holding context for four unrelated jobs wastes tokens on every call. Splitting by domain fixes both.
There are two patterns worth naming:

- **Supervisor:** A coordinator agent routes work to specialist subagents and assembles the results
- **Handoff:** Agents pass control directly to each other, with no central coordinator

Supervisor is easier to reason about and easier to trace, so it's the default. Handoff suits workflows where the path is genuinely sequential.
The hard part here is state. Decide what each subagent sees, because passing full conversation history to every one of them recreates the context problem you split to avoid. Passing a scoped summary usually works better.

---



### Q40. How would you implement retries and fallbacks?

Example: [q40_retries_and_fallbacks.py](examples/langchain/q40_retries_and_fallbacks.py)

**Answer.**

Retries and fallbacks solve different failures, and treating them as one thing is a common mistake.
Retry when the same call might work on a second attempt. For example, when you run into rate limits or timeouts. Use exponential backoff with jitter so your retries don't arrive in a synchronized burst, and limit the attempts so a failing provider doesn't multiply your latency.
Fall back when retrying won't help. A provider outage or a model that can't produce valid structured output all need a different path rather than another attempt.
The subtle part is idempotency. Retrying a model call is safe, but retrying a tool call that charges a credit card isn't. Tools with side effects need idempotency keys, or you need retry logic that stops at the tool boundary.
Also decide where the retry lives. A retry at the model level re-runs one call, but a retry at the graph node level re-runs everything in that node. LangGraph 1.x gives you node-level error handlers and per-node timeouts for exactly this reason.

---



### Q41. How would you manage context efficiently?

Example: [q41_manage_context.py](examples/langchain/q41_manage_context.py)

**Answer.**

Treat the context window as a budget you allocate.
There are four things you need to account for in this budget:
System prompt and tool definitions: Fixed cost on every call
Conversation history: Grows every turn
Retrieved documents: Grows with how many chunks you pass
Tool outputs: Grows unpredictably
Tool outputs deserve attention because they're the one people forget. A single API response can dump thousands of tokens into state, and truncating or summarizing tool results before they hit the message list often saves more than anything you do to history.
Then there's the quality argument. Models handle information in the middle of a long context worse than information at the start or end. So a full context window isn't just expensive but it also produces worse answers than a smaller well-chosen one.

---



### Q42. How would you optimize latency?

Example: [q42_optimize_latency.py](examples/langchain/q42_optimize_latency.py)

**Answer.**

Measure before you change anything. Traces will break a request into parts like model calls and retrieval, so you get a better picture of what's the bottleneck.
Once you know where the time goes:

- **Stream:** Streaming doesn't reduce total time, but time to first token is what users actually feel. This is the cheapest win available
- **Parallelize:** Independent calls (model, tool) should run concurrently. .batch() and LangGraph's concurrent node execution both handle this

Use smaller models where they fit: Classification and rewriting steps don't need your largest model, and each change will reduce the overall runtime
Reduce model calls: Native structured output removes a parsing call, and skipping unnecessary query rewriting removes another

- **Cache:** Exact and semantic caching can turn a repeated question into a lookup.

For agents specifically, latency is a function of step count. An agent that takes six steps to answer what a chain answers in two is an architecture problem, not a latency one.

---



### Q43. How would you evaluate tool selection?

Example: [q43_evaluate_tool_selection.py](examples/langchain/q43_evaluate_tool_selection.py)

**Answer.**

Score the trajectory. An agent can reach the right conclusion after calling three wrong tools.
Build a dataset where each input has a known correct tool sequence. Then measure how often the agent picks the right tool, how many redundant calls it makes, and how often it stops at the right point instead of looping.
Failures usually trace back to the tool definitions rather than the model. Overlapping descriptions make two tools look interchangeable and vague descriptions leave the model guessing. Also, a long tool list dilutes attention across all of them.
So when tool selection is bad, rewrite the descriptions before you change models. It's cheaper and it works more often.

---



### Q44. How would you build an enterprise RAG pipeline?

Example: [q44_enterprise_rag.py](examples/langchain/q44_enterprise_rag.py)

**Answer.**

Enterprise changes the requirements more than the architecture. There are four constraints you'll usually run into.
Access control. Different users can see different documents, so permissions have to apply at retrieval time. Store access metadata with every chunk and filter before similarity ranking, because filtering after retrieval leaks the existence of documents users shouldn't know about.
Incremental indexing. Full reindexing doesn't work at scale. You track document versions and update only what changed, which means content hashing and a deletion path for removed documents.
Multiple sources. Wikis, ticketing systems, file shares, and databases all contain relevant content, and each needs its own loader and refresh schedule.
Auditability. Every answer needs to be traceable to its sources, and every query needs to be logged for compliance review.
The design question interviewers focus on is freshness versus cost. Real-time indexing is expensive, scheduled batch indexing is cheap but stale, and the right answer depends on how fast the underlying documents change.

---



### Q45. What are common production bottlenecks?

Example: [q45_production_bottlenecks.py](examples/langchain/q45_production_bottlenecks.py)

**Answer.**

Four show up again and again.

- **Context growth:** This is the most common one. History, retrieval, and tool outputs expand until requests get slow and expensive.
- **Retrieval quality:** Bad chunks produce bad answers no matter which model reads them, and teams often tune prompts for weeks before checking what retrieval step actually returned.
- **Unbounded agent loops:** Without a step limit, one hard question can cost twenty times what you budgeted.
- **Provider rate limits:** Limits apply across your whole application, so traffic growth hits a ceiling that has nothing to do with your infrastructure.

Notice that none of these problems are related to model quality. Production LangChain issues are almost always about what surrounds the model call, and saying that in an interview shows you have real-world experience.

---



## Comparisons



### Q46. LangChain vs. LlamaIndex

Example: [q46_langchain_vs_llamaindex.py](examples/langchain/q46_langchain_vs_llamaindex.py)

**Answer.**

LlamaIndex started as a retrieval framework. It has more depth in document parsing and indexing strategies than LangChain does, so a search-heavy application over complex documents is where it's strongest.
LangChain is broader. Agents and multi-step orchestration are the center of the framework, with retrieval as one component among many.
The interview-ready version is that LlamaIndex is retrieval-first and LangChain is orchestration-first. If your application is mostly search over documents, LlamaIndex gives you more out of the box. If retrieval is one step inside a larger workflow, LangChain fits better. Plenty of teams use both.

---



### Q47. LangChain vs. Semantic Kernel

Example: [q47_langchain_vs_semantic_kernel.py](examples/langchain/q47_langchain_vs_semantic_kernel.py)

**Answer.**

Semantic Kernel is Microsoft's framework, and its strongest case is a .NET or Azure scenarios. It's built with enterprise integration in mind, and the C# support is first-class.
LangChain is Python-first with a TypeScript port, and it has a much larger ecosystem of integrations and a faster release cadence.
So the honest comparison is about the environment. If your organization runs on Microsoft infrastructure and writes C#, Semantic Kernel makes sense. Outside that, LangChain's ecosystem is hard to match.

---



### Q48. LangChain vs. Haystack

Example: [q48_langchain_vs_haystack.py](examples/langchain/q48_langchain_vs_haystack.py)

**Answer.**

Haystack comes from the search world and is built around production NLP pipelines. Its pipeline model is explicit and declarative, which makes it easy to reason about and easy to deploy as a service.
LangChain covers more ground, especially around agents, and has more provider integrations.
Choose Haystack when your application is a search or question-answering service and you want a stable, well-defined pipeline. Go with LangChain when the workflow involves agents, tools, or branching logic that a linear pipeline doesn't express well.

---



### Q49. LangGraph vs. LangChain

Example: [q49_langgraph_vs_langchain.py](examples/langchain/q49_langgraph_vs_langchain.py)

**Answer.**

This one comes up most often, and the framing changed with the 1.0 release, so make sure you don't give an outdated answer.
They aren't competitors. LangGraph is the low-level runtime for stateful, graph-based workflows, and LangChain's create_agent is built on top of it. When you use an agent, you're already using LangGraph, whether you wrote graph code or not.
The choice is about how much control you need:
Use create_agent when a standard agent loop with middleware covers your workflow
Use LangGraph directly when you need custom nodes, explicit branching, cycles you define yourself, or multi-agent coordination
Starting with create_agent and dropping to LangGraph when you hit its limits is the recommended path, and both use the same persistence and streaming underneath.

---



## From addonQA (LangChain extras)

*Source: [addonQA.md](addonQA.md) — original answers.*

### Q50. How have "Chains" evolved in LangChain from legacy versions to modern architecture?

Example: [q50_chains_evolved.py](examples/langchain/q50_chains_evolved.py)

**Answer.**

How have "Chains" evolved in LangChain from legacy versions to modern architecture?

- **Answer:** In early LangChain, chains were rigid, pre-packaged Python classes like `LLMChain` or `SequentialChain` that acted as black boxes—bundling prompts, models, and parsers internally, which made debugging and custom logic insertion difficult. Today, we build chains declaratively using **LCEL (LangChain Expression Language)** and **Runnables**. Because every component implements a unified Runnable interface, we can chain prompts, models, and custom logic together using the native pipe operator (`|`). This provides full transparency, native async support, and built-in streaming without sacrificing flexibility.

---



### Q51. What problems did LCEL and the Runnable interface solve?

Example: [q51_lcel_problems_solved.py](examples/langchain/q51_lcel_problems_solved.py)

**Answer.**

What problems did LCEL and the Runnable interface solve?

- **Answer:** Previously, every LangChain component had a different execution syntax (`.format()` for prompts, `.predict()` for models, `.run()` for chains). LCEL standardized everything under the **Runnable** interface, offering a unified protocol:
- `.invoke()` for single execution.
- `.batch()` for parallel execution over multiple inputs.
- `.stream()` for token-by-token or event output.
- `.ainvoke()` for async operations.

---



### Q52. What are the primary Runnable primitives used in LCEL pipelines?

Example: [q52_runnable_primitives.py](examples/langchain/q52_runnable_primitives.py)

**Answer.**

What are the primary Runnable primitives used in LCEL pipelines?

- **Answer:**
- `RunnableSequence` (The pipe operator `|`): Routes output from step A directly into step B.
- `RunnableParallel`: Executes multiple runnables simultaneously against the same input (e.g., fetching context and user questions concurrently).
- `RunnablePassthrough`: Passes data straight through or injects auxiliary variables using `.assign()`.
- `RunnableLambda`: Wraps custom Python functions so they plug seamlessly into a declarative chain.
- `RunnableBranch`: Implements conditional if/else routing logic inside a pipe.

---



### Q53. What are "Deep Agents" and what specific problems do they solve?

Example: [q53_deep_agents.py](examples/langchain/q53_deep_agents.py)

**Answer.**

What are "Deep Agents" and what specific problems do they solve?

- **Answer:** Deep Agents are advanced, long-horizon, autonomous AI agent systems designed for complex multi-step workflows. Traditional agents fail at scale due to **Token Bloat** (chat history growing too large), **Lack of Macro-Planning** (acting blindly turn-by-turn), and **Polluted Working Memory**. Deep Agents solve these through a dedicated harness layer.

---



### Q54. What are the 4 core pillars of Deep Agent architecture?

Example: [q54_deep_agent_pillars.py](examples/langchain/q54_deep_agent_pillars.py)

**Answer.**

What are the 4 core pillars of Deep Agent architecture?

- **Answer:**

1. **Explicit Planning (**`todo_write`**):** Generates and updates an externalized markdown checklist to track progress across long horizons.
2. **Persistent Virtual Filesystem (Scratchpads):** Uses `read_file` and `write_file` tools to store notes and drafts outside the main chat history.
3. **Sub-Agent Context Quarantine:** The main orchestrator delegates heavy investigative tasks to specialized sub-agents working in isolation, returning *only* a clean, distilled summary.
4. **Advanced Context Middleware:** Automated wrappers handling historical summarization, token trimming, and tool-call patching.

---



### Q55. How do Claude Skills integrate with Deep Agents compared to static instructions?

Example: [q55_skills_vs_static_instructions.py](examples/langchain/q55_skills_vs_static_instructions.py)

**Answer.**

How do Claude Skills integrate with Deep Agents compared to static instructions?

- **Answer:** Claude Skills are not passive, static instruction text files; they are modular, dynamic functional units integrated into the agentic harness. They provide structured operational behaviors that the agent can invoke dynamically based on context, bridging raw model reasoning with specialized execution.

---



### Q56. What is `RecursiveCharacterTextSplitter` and why is it the industry standard?

Example: [q56_recursive_character_splitter.py](examples/langchain/q56_recursive_character_splitter.py)

**Answer.**

What is `RecursiveCharacterTextSplitter` and why is it the industry standard?

- **Answer:** It is LangChain's most popular text splitter for RAG pipelines. Instead of splitting text blindly at fixed character lengths (which ruins sentences), it uses a hierarchical list of separators (paragraphs `\n\n`, lines `\n`, spaces  ``0, and characters `""`). It keeps semantic units together as long as possible before resorting to smaller cuts.

---



### Q57. What are the major text splitting strategy types?

Example: [q57_text_splitting_strategies.py](examples/langchain/q57_text_splitting_strategies.py)

**Answer.**

What are the major text splitting strategy types?

- **Answer:**
- **Character-Based:** Naive character splits or recursive hierarchical splits.
- **Token-Based:** Splits strictly by model tokenizer limits (`tiktoken`) to protect context windows.
- **Structure-Aware:** Markdown header splitters or programming-language syntax splitters (preserving function/class blocks).
- **Semantic & Advanced:** `SemanticChunker` (splitting based on embedding distance shifts) and Parent-Child hierarchical chunking.

---



### Q58. Outline the architecture for Project 1: Conversational FAQ & Ticket Escalation Agent.?

Example: [q58_faq_ticket_escalation.py](examples/langchain/q58_faq_ticket_escalation.py)

**Answer.**

Outline the architecture for Project 1: Conversational FAQ & Ticket Escalation Agent.

- **Answer:**
- **Tech Stack:** FastAPI, LangGraph, ChromaDB, Mock ServiceNow API wrapper.
- **Workflow:** User submits query $\rightarrow$ `ingest_query` node $\rightarrow$ vector store retrieval node $\rightarrow$ conditional evaluation router. If sufficient info is found, it synthesizes a direct response; if information is missing or out of scope, it routes to the `ticket_escalation_node` to format user metadata and generate a mock ServiceNow Ticket ID. State is persisted via checkpointers.

---



### Q59. Outline the architecture for Project 2: Interactive Resume Document Writing Agent.?

Example: [q59_resume_writing_agent.py](examples/langchain/q59_resume_writing_agent.py)

**Answer.**

Outline the architecture for Project 2: Interactive Resume Document Writing Agent.

- **Answer:**
- **Tech Stack:** FastAPI, LangGraph, `python-docx`, Pydantic validation.
- **Workflow:** Backend loads a baseline professional `.docx` template containing structural tags (`{{FULL_NAME}}`, `{{EXPERIENCE}}`). The agent conducts a multi-turn conversational interview to gather missing profile data iteratively, validates inputs using Pydantic, programmatically injects data into the Word template using `python-docx` while preserving styling, and exposes a secure FastAPI download endpoint.

---



### Q60. Explain the 5 core components and data flow of a LangChain RAG pipeline.?

Example: [q60_rag_five_components.py](examples/langchain/q60_rag_five_components.py)

**Answer.**

Explain the 5 core components and data flow of a LangChain RAG pipeline.

### Core Technical Explanation

1. **Document Loader:** Connects to data sources (PDFs, Notion, SQL, web pages) and converts raw data into standard LangChain `Document` objects containing raw text (`page_content`) and metadata (`source`, `page_number`).
2. **Text Splitter:** Breaks large documents into smaller chunks (e.g., `RecursiveCharacterTextSplitter`). Large language models have finite context windows, and smaller, focused chunks improve search accuracy.
3. **Embedding Model:** Takes each text chunk and converts it into a high-dimensional vector (a list of numbers) that represents its semantic meaning.
4. **Vector Store:** A specialized database (Chroma, FAISS, Pinecone) that indexes and stores these vectors along with their text chunks for fast similarity searches.
5. **Retriever:** A component that accepts a user query as a string, converts that query into an embedding, performs a mathematical search (like cosine similarity) against the Vector Store, and returns the top-$k$ most relevant text chunks.



### Real-World Interview Example

> *"To build an internal HR policy bot: First,* `PyPDFLoader` *reads the employee handbook. Second,* `RecursiveCharacterTextSplitter` *breaks the text into 500-character chunks with a 50-character overlap so sentences aren't cut in half. Third, an OpenAI embeddings model converts chunks into vectors. Fourth, the vectors are stored in Chroma DB. Fifth, when an employee asks 'How many sick days do I get?', the retriever pulls the top 3 relevant chunks, injects them into the prompt, and the LLM writes an answer grounded only in those chunks."*



### Keywords Explained

- **Semantic Meaning:** The conceptual meaning of text rather than exact keyword matches (e.g., "automobile" and "car" have high semantic similarity).
- **Cosine Similarity:** A mathematical formula that measures the angle between two vectors to determine how similar two pieces of text are.
- **Chunk Overlap:** Duplicating a small amount of text between consecutive chunks to ensure context is not lost at the boundary cut.
- **Top-k:** The specific number ($k$) of best-matching document chunks returned from the search.

---



### Q61. What is the Runnable Interface, and how does it work under the hood?

Example: [q61_runnable_interface.py](examples/langchain/q61_runnable_interface.py)

**Answer.**

What is the Runnable Interface, and how does it work under the hood?

### Core Technical Explanation

1. **Standardized Method Contract:** Any LangChain object that inherits from `Runnable` guarantees the implementation of four core execution methods:

- `.invoke(input)`: Runs the component on a single input synchronously.
- `.batch([inputs])`: Runs the component on multiple inputs in parallel using thread pools.
- `.stream(input)`: Yields output chunks in real-time as they are computed.
- `.ainvoke()`, `.abatch()`, `.astream()`: Asynchronous versions running natively on Python's `asyncio` event loop.

1. **Composition Mechanics:**

- When you write `a | b`, Python calls `a.__or__(b)`. LangChain catches this and creates a `RunnableSequence` object.
- When you supply a dictionary `{ "context": retriever, "question": RunnablePassthrough() }`, LangChain wraps it into a `RunnableParallel` object, running both branches simultaneously.

1. **Input/Output Type Safety:** Each Runnable specifies its expected input type and output type, allowing LangChain to validate whether two components can safely pipe into each other before runtime.



### Real-World Interview Example

> *"Because both* `ChatOpenAI` *and* `StrOutputParser` *implement the Runnable interface, I do not have to write custom streaming logic for web sockets. When building a chat UI in FastAPI, I simply call* `await chain.astream(user_message)`*, and it yields tokens one by one as they arrive from the model provider."*



### Keywords Explained

- **Interface / Protocol:** A contract in programming that requires classes to provide specific methods with consistent names and arguments.
- **Event Loop:** The core mechanism in Python `asyncio` that manages and executes non-blocking tasks concurrently on a single thread.
- **Type Safety:** Catching data type errors (like passing an integer where a string is required) before or during execution.

---



### Q62. How does Memory work in LangChain, and what are the architectural trade-offs between Buffer, Summary, and VectorStore memory?

Example: [q62_memory_tradeoffs.py](examples/langchain/q62_memory_tradeoffs.py)

**Answer.**

How does Memory work in LangChain, and what are the architectural trade-offs between Buffer, Summary, and VectorStore memory?

### Core Technical Explanation

1. **Stateless Model Reality:** LLMs have no internal memory across separate API calls. LangChain simulates memory by storing past messages and injecting them into the prompt context on every new turn.
2. **ConversationBufferMemory:**

- *Mechanism:* Appends raw messages (`HumanMessage`, `AIMessage`) to a list and injects the complete history into the prompt every time.
- *Trade-off:* High accuracy and zero context loss early on. However, it quickly consumes the token budget, increases API latency and costs, and eventually crashes when the model's context limit is reached.

1. **ConversationSummaryMemory:**

- *Mechanism:* When new messages arrive, a secondary LLM call condenses the existing conversation into a rolling text summary.
- *Trade-off:* Token consumption stays flat over long conversations. However, each turn requires an extra LLM call (adding cost and delay), and specific granular details (like numbers, IDs, or exact quotes) can be lost during summarization.

1. **VectorStore-backed Memory:**

- *Mechanism:* Stores every past message as an embedding in a vector database. On a new turn, it queries the database and injects only the past messages that are semantically relevant to the current user query.
- *Trade-off:* Allows conversations to scale indefinitely and only uses tokens for relevant context. However, it can fail to retrieve important past details if the current question uses different phrasing.



### Real-World Interview Example

> *"In a technical support chatbot: If we use **Buffer Memory**, after 40 messages the app will hit the token limit or become too expensive. If we use **Summary Memory**, the model might summarize 'The error code was 0x80070005' into 'The user encountered an access error', losing the exact code needed for troubleshooting. With **VectorStore Memory**, when the user later asks 'What was that error code I saw earlier?', the system retrieves only that specific past exchange without needing the other 39 messages."*



### Keywords Explained

- **Stateless:** A system that retains no memory of past interactions; every new request is processed completely from scratch.
- **Token Budget / Context Window:** The maximum number of tokens an LLM can read and generate in a single request.
- **Granular Details:** Small, specific pieces of information such as dates, IDs, names, or numbers that are easily lost in broad summaries.

---



### Q63. What is a `SKILL.md` file, and how is it used in agentic architectures?

Example: [q63_skill_md.py](examples/langchain/q63_skill_md.py)

**Answer.**

What is a `SKILL.md` file, and how is it used in agentic architectures?

### Core Technical Explanation

1. **Modular Workflow Standard:** A `SKILL.md` file is a portable Markdown file that contains a specialized step-by-step playbook or workflow for an AI agent.
2. **Dynamic Progressive Disclosure:** Instead of bloating the main system prompt with endless instructions for every possible task, an agent's system prompt only lists the metadata (names and descriptions) of available `SKILL.md` files. The agent loads the heavy Markdown instructions into its context window *only* when that specific skill is requested.
3. **Structure:**

- *YAML Frontmatter:* Contains the `name` and `description` used for tool discovery.
- *Markdown Body:* Contains the exact procedural instructions, rules, code structures, and error-handling steps the agent must execute.



### Real-World Interview Example

> *"If I am building an AI coding agent, I don't want to cram instructions for database migrations, unit testing, and git commits into the main system prompt all at once. Instead, I create a* `commit-workflow.skill.md` *file. When the user asks to push changes, the agent reads the skill metadata, dynamically loads the markdown steps for writing conventional commits into memory, and executes the exact playbook."*



### Keywords Explained

- **Progressive Disclosure:** An interface design pattern where information is revealed only as it is needed to save cognitive load or context window limits.
- **Frontmatter:** A block of YAML metadata placed at the very beginning of a Markdown file.

---



### Q64. How do you approach Testing and Evaluation for non-deterministic AI agents and graphs?

Example: [q64_testing_nondeterministic.py](examples/langchain/q64_testing_nondeterministic.py)

**Answer.**

How do you approach Testing and Evaluation for non-deterministic AI agents and graphs?

### Core Technical Explanation

1. **Why Traditional Unit Tests Fail:** Traditional software unit tests expect exact outputs (e.g., `assert result == 5`). Because LLMs generate natural language variations, exact string matching fails for AI applications.
2. **Dataset-Driven Evaluation (Evals):** Instead of single hardcoded tests, developers create a dataset of test cases containing inputs and expected ground-truth behavior or goals.
3. **LLM-as-a-Judge:** You use a stronger, specialized LLM (like GPT-4) to evaluate the output of your agent graph against criteria like factual accuracy, helpfulness, and safety.
4. **CI/CD Integration:** Running these evaluation datasets automatically whenever code or prompts change ensures that an update to a prompt template doesn't accidentally break the agent's core capabilities.



### Real-World Interview Example

> *"To test our legal document summarization graph, we don't write assertions like* `assert output == 'contract valid'`*. Instead, we set up a LangSmith evaluation dataset with 50 sample contracts and expected summaries. We run our graph against that dataset, and use an **LLM-as-a-judge** prompt to score whether the agent successfully extracted the key liability clauses without hallucinating."*



### Keywords Explained

- **LLM-as-a-Judge:** Using a powerful language model to automatically review, grade, and score the output of another AI application based on specific rules.
- **Ground Truth:** The verified, correct, and trusted reference data used to check whether an AI model's answer is accurate.

---



### Q65. How do you handle Observability and Tracing using LangSmith?

Example: [q65_langsmith_tracing.py](examples/langchain/q65_langsmith_tracing.py)

**Answer.**

How do you handle Observability and Tracing using LangSmith?

### Core Technical Explanation

1. **LangSmith Tracing:** Setting `LANGCHAIN_TRACING_V2=true` automatically intercepts every prompt generation, token count, tool invocation, and state transition.
2. **Execution Tree Visualization:** Logs every run as a hierarchical tree. You can inspect the exact prompt sent, the raw response, token consumption, and latency.
3. **Playgrounds:** Export a trace into a playground environment to tweak the prompt and re-test instantly.



### Real-World Interview Example

> *"If a user reports an agent failed on step 5, I open LangSmith, look up the thread ID, and trace the execution tree. I can see the exact prompt generated, check token usage, and measure the exact latency of the tool call that caused the error."*



### Keywords Explained

- **Trace:** A recorded log showing the step-by-step path of a request.

---



### Q66. How do you approach Testing and Evaluation for AI agents?

Example: [q66_testing_agents.py](examples/langchain/q66_testing_agents.py)

**Answer.**

How do you approach Testing and Evaluation for AI agents?

### Core Technical Explanation

1. **Dataset-Driven Evaluation:** Instead of hardcoded assertions, developers create datasets of inputs and expected ground-truth behavior.
2. **LLM-as-a-Judge:** Use a stronger LLM (like GPT-4) to evaluate the agent's output against criteria like factual accuracy and helpfulness.
3. **CI/CD Integration:** Run evaluation datasets automatically whenever code changes to prevent breaking core capabilities.



### Real-World Interview Example

> *"To test our legal summarization graph, we set up an evaluation dataset with 50 sample contracts. We run our graph against it and use an **LLM-as-a-judge** prompt to score whether the agent extracted key clauses without hallucinating."*



### Keywords Explained

- **Ground Truth:** The verified, correct reference data used to check an AI model's accuracy.

---

