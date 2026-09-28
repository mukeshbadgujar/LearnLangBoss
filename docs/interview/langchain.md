# LangChain interview answers

For the interview Q&A bank (answers from the sample file, with index), open [langchain-qa.md](langchain-qa.md). This file is the chapter-by-chapter bank.

Speak the **Answer** first: objects, fields, and what fails. If they push, name the **Trap** — the reply that sounds fine and is wrong. Each item links to the mechanism chapter so you can reopen the failure table.

## Contents

- [00 — Environment, providers, and keys](#00--environment-providers-and-keys)
- [01 — Models and messages](#01--models-and-messages)
- [02 — Prompt templates](#02--prompt-templates)
- [03 — Chat prompts](#03--chat-prompts)
- [04 — Few-shot](#04--few-shot)
- [05 — Output parsers](#05--output-parsers)
- [06 — Memory](#06--memory)
- [07 — LCEL](#07--lcel)
- [08 — Chains](#08--chains)
- [09 — Document loaders](#09--document-loaders)
- [10 — Text splitters](#10--text-splitters)
- [11 — Embeddings](#11--embeddings)
- [12 — Vector stores](#12--vector-stores)
- [13 — Retrievers](#13--retrievers)
- [14 — Retrieval chains](#14--retrieval-chains)
- [15 — Advanced RAG](#15--advanced-rag)
- [16 — Tools](#16--tools)
- [16a — Code execution](#16a--code-execution)
- [16b — MCP](#16b--mcp)
- [17 — Agents](#17--agents)
- [17a — Agent loop](#17a--agent-loop)
- [18 — Graphs over agents](#18--graphs-over-agents)
- [22 — Multimodal](#22--multimodal)
- [23 — Caching](#23--caching)
- [24 — Structured outputs](#24--structured-outputs)
- [25 — Routing](#25--routing)
- [28 — Hub](#28--hub)

## 00 — Environment, providers, and keys

### How does this curriculum pick a chat provider?

**Answer.** The course does not hard-code one vendor. A shared factory picks the first working key. Use `get_chat_model()`. It reads `.env`, then `active_provider`. If `LLM_PROVIDER` is set and its key exists, that pin wins. Otherwise Groq, then OpenRouter, then OpenAI, then Anthropic. Failure: rename `GROQ_API_KEY` to `GROQ_KEY` and you get `NoProviderConfigured` after restart.

**Trap.** People often say just set `OPENAI_API_KEY` and call `ChatOpenAI`. That misses the shared factory, Groq-first order, and the OpenRouter special case.

**Chapter.** [00](../theory/00-environment.md)

### What does `invoke` return, and where do cost and truncation live?

**Answer.** You do not get a plain string back from chat. You get a full message object. Call `model.invoke(...)` and you receive an `AIMessage`. Read `.content` for text. Read `.response_metadata["finish_reason"]` for stop vs cut-off. Read `.usage_metadata` for tokens. Failure: parse JSON from `.content` after a `length` finish and `json.loads` blows up far from the real cause.

**Trap.** People often say `invoke` returns a string. That misses truncation and the bill.

**Chapter.** [00](../theory/00-environment.md)

### Why is dimension count alone not proof that embeddings work?

**Answer.** A fake embedder can still spit out 384 floats. Length is not quality. Check related text scores higher than unrelated by a clear margin before any RAG lab. Use `get_embeddings()` / `EMBEDDING_PROVIDER` separately from chat. Failure: chat works, embeddings report `fake`, and retrieval neighbors are random.

**Trap.** People often say if `len(vector) == 384`, embeddings are fine. That misses semantic sanity checks.

**Chapter.** [00](../theory/00-environment.md)

### What fails when the import works but the notebook still cannot call a model?

**Answer.** Import success is not a healthy stack. Packages may sit in a different Python than the kernel. Run the bootstrap health checks, then `python -m pip install -r requirements.txt` against the GenAI Mastery kernel. Keep keys only in repo-root `.env`. Failure: `ModuleNotFoundError` after you already "installed" into another interpreter.

**Trap.** People often say environment setup is optional if the import works. That misses kernel vs install mismatches.

**Chapter.** [00](../theory/00-environment.md)

## 01 — Models and messages

### What is a chat model call, really?

**Answer.** You send a typed message list. You get an assistant message back. Use a chat model from `get_chat_model()`. The text is `.content`. Stop reason and cost live on the same `AIMessage`. Failure: show only `.content` in a UI and truncated answers look "fine."

**Trap.** People often say it returns a string. That misses truncation and token accounting.

**Chapter.** [01](../theory/01-models-messages.md)

### Does the model remember the conversation?

**Answer.** No. Each call is fresh. You must resend the full list, or reload state from a checkpointer by `thread_id`. Append the prior `AIMessage` before the next `HumanMessage`. Failure: ask "And what happens if I exceed it?" alone and the model invents nonsense with no plan context.

**Trap.** People often say the model remembers the conversation. That misses that you own the message list.

**Chapter.** [01](../theory/01-models-messages.md)

### What does `finish_reason` mean for JSON parsing?

**Answer.** Check why generation stopped before you parse. Read `response_metadata["finish_reason"]`. `stop` means finished. `length` means truncated. `tool_calls` means run tools. Failure: half-JSON after `max_tokens=16` fails `json.loads` three layers away from the real cause.

**Trap.** People often say if parsing fails, the prompt was wrong. That misses truncation first.

**Chapter.** [01](../theory/01-models-messages.md)

### Is `temperature=0` deterministic enough for exact string tests?

**Answer.** Near-deterministic is not identical. Providers can still drift. Assert structure or categories, not exact model strings. Prefer chat models like `ChatOpenAI` / `get_chat_model()` over legacy completion APIs. Failure: a flaky unit test that expects one exact sentence.

**Trap.** People often say `temperature=0` makes output identical every time. That misses provider and float noise.

**Chapter.** [01](../theory/01-models-messages.md)

### How is `batch` different from a for-loop of `invoke`?

**Answer.** Batch runs many inputs at once on the Runnable interface. Wall-clock drops. Token cost per item stays the same. Use `model.batch(...)` and cap with `config={"max_concurrency": N}` when rate limits matter. Failure: fire a huge batch and trip provider 429s.

**Trap.** People often say `batch` is just a for-loop helper. That misses concurrency and rate limits.

**Chapter.** [01](../theory/01-models-messages.md)

### What belongs on every production chat path besides `.content`?

**Answer.** Ops need stop reason and usage on every call. Log `finish_reason` and `usage_metadata` before you trust or parse the reply. Failure: tiny `max_tokens` and a correct prompt still yield broken JSON because the reply was cut off.

**Trap.** People often say only the assistant text matters for ops. That misses billing and truncation signals.

**Chapter.** [01](../theory/01-models-messages.md)

## 02 — Prompt templates

### Why not build the prompt with an f-string?

**Answer.** Templates declare required inputs and fail early when one is missing. An f-string will silently bake in `None`. Prefer `PromptTemplate` / `ChatPromptTemplate`. Failure: a missing system line becomes the prompt the model actually answers from.

**Trap.** People often say templates are just convenience. That misses validation at format time.

**Chapter.** [02](../theory/02-prompt-templates.md)

### Why prefer `.invoke` / pipe over `.format()` then `model.invoke(string)`?

**Answer.** Format is for eyeballing. Production keeps types in a Runnable pipe so batch, stream, and callbacks still work. Use `.invoke` or `|` so you stay on `PromptValue` → model → parser. Failure: a 60-line anonymous f-string that cannot A/B or publish to Hub.

**Trap.** People often say always `.format()` then `model.invoke(string)`. That misses the Runnable contract.

**Chapter.** [02](../theory/02-prompt-templates.md)

### How do you put literal JSON braces in a template?

**Answer.** Single braces mean variables. Literal JSON needs doubled braces. Write `{{"status": "ok"}}` so `{question}` still substitutes. Failure: paste `{"status": "ok"}` and get `KeyError` / `ValueError` at format time.

**Trap.** People often say just paste the JSON example into the template string. That misses the brace-escaping rule.

**Chapter.** [02](../theory/02-prompt-templates.md)

### Why is `.partial(date=date.today().isoformat())` a footgun?

**Answer.** That freezes the date at bind time, often at import. Yesterday sticks after deploy. Pass a zero-arg callable so each render re-evaluates. Use `.partial(...)` for stable boot values like tier or SLA. Failure: a "today" line that is still last week's date in production.

**Trap.** People often say partial with `date.today()` is fine. That misses bind-time freeze.

**Chapter.** [02](../theory/02-prompt-templates.md)

### What is the point of composing identity + rules + task templates?

**Answer.** Each piece can version alone. Colleagues reuse named modules instead of copy-paste. Compose with `|` into `template | model | StrOutputParser()`. Failure: omit a declared variable and you want a loud `KeyError` before a paid call, not a silent bad prompt.

**Trap.** People often say one giant prompt string is easier to maintain. That misses reuse and loud failure.

**Chapter.** [02](../theory/02-prompt-templates.md)

### What happens if you omit a declared input variable?

**Answer.** Format fails before the model runs. That saves a wasted paid call. `invoke` / `format` raises `KeyError` (or validation error). Check callers against `input_variables`. Failure: expect the model to paper over a missing slot and still get a good answer.

**Trap.** People often say the model will ignore missing slots. That misses fail-at-format-time design.

**Chapter.** [02](../theory/02-prompt-templates.md)

## 03 — Chat prompts

### What does `ChatPromptTemplate` actually render?

**Answer.** It builds a list of role-tagged messages, not one big string. That is what agents and RAG use. Prefer `ChatPromptTemplate.from_messages(...)` over hand-splicing lists. Failure: wrong order of system, history, and question after fragile list surgery.

**Trap.** People often say `ChatPromptTemplate` is `PromptTemplate` with chat in the name. That misses the message-list shape.

**Chapter.** [03](../theory/03-chat-prompts.md)

### What is `MessagesPlaceholder` for?

**Answer.** It is a variable-length message slot filled at invoke time. Use it for history, few-shot blocks, or context messages. Cold start: pass `history=[]` or set `optional=True`. Failure: omit `history` and get `KeyError` on the first turn.

**Trap.** People often say placeholders are only for chat history. That misses few-shot and RAG context slots.

**Chapter.** [03](../theory/03-chat-prompts.md)

### Why is naive `history[-k:]` dangerous with tools?

**Answer.** A raw slice can orphan tool calls and break provider validity rules. Prefer `trim_messages` with a real token counter and tool-call pairing. Failure: keep last k strings forever and the token bill explodes after about ten turns.

**Trap.** People often say just keep the last k strings. That misses tool-call pairing and token growth.

**Chapter.** [03](../theory/03-chat-prompts.md)

### Is in-process `self.history` production memory?

**Answer.** No. It dies with the process and does not branch or audit well. Move to a LangGraph checkpointer plus `thread_id`. Failure: restart the worker and every chat is gone.

**Trap.** People often say in-memory `self.history` is production memory. That misses durability and multi-process reality.

**Chapter.** [03](../theory/03-chat-prompts.md)

### Does placeholder order in the template matter?

**Answer.** Yes. Author order is render order. Put system, then examples, then history, then the human question in `from_messages`. Swapping placeholders changes what sits next to the question. Failure: weak system text without "never invent" and the model invents policy.

**Trap.** People often say message order is whatever the invoke dict iteration order is. That misses template author order.

**Chapter.** [03](../theory/03-chat-prompts.md)

### How do you prime an answer shape inside a chat template?

**Answer.** Show an assistant turn before the real question so the format is demonstrated. Add an `("ai", ...)` priming message in `from_messages`, plus a final `("human", "{question}")`. Failure: rely only on the system message and still get free-form replies.

**Trap.** People often say only the system message can shape the reply format. That misses template-level priming.

**Chapter.** [03](../theory/03-chat-prompts.md)

## 04 — Few-shot

### When do few-shot examples help triage?

**Answer.** Zero-shot invents a new line format each time. Show a few examples with the exact target shape. Put those examples in the prompt bank the selector uses. Failure: one poisoned label and the model copies the bad category forever.

**Trap.** People often say fine-tuning is required to teach a triage format. That misses few-shot for format lock-in.

**Chapter.** [04](../theory/04-few-shot.md)

### Why not ship dozens of examples on every request?

**Answer.** Every example costs tokens every request. Gains flatten fast. Prefer three to five relevant examples via a selector. Length selectors only fit a budget. Semantic selectors pick neighbors. Failure: a long ticket eats `max_length` and starves the example bank.

**Trap.** People often say more examples always help. That misses token cost and flattening returns.

**Chapter.** [04](../theory/04-few-shot.md)

### Do few-shot examples guarantee a schema?

**Answer.** No. They make a format likely. Enforcement needs `.with_structured_output()` or a parser. Combine few-shot for style with structured output for fields. Failure: treat examples as a contract and still get missing fields under load.

**Trap.** People often say few-shot guarantees the schema. That misses enforcement vs likelihood.

**Chapter.** [04](../theory/04-few-shot.md)

### How do you debug a selector that picks the wrong neighbours?

**Answer.** Print what it actually selected for the failing input. Inspect `to_messages()` / `select_examples`. Length and semantic selectors are not the same tool. Failure: no real embeddings, so a semantic feedback loop cannot work.

**Trap.** People often say length-based and semantic selectors are interchangeable. That misses budget vs similarity.

**Chapter.** [04](../theory/04-few-shot.md)

## 05 — Output parsers

### Why not `json.loads` the model output?

**Answer.** Models wrap JSON in preambles, fences, or trailing prose. Raw loads are unreliable. Prefer `JsonOutputParser` / Pydantic parsers, or better, native structured output. Failure: fence markers break a naive `json.loads` on an otherwise valid object.

**Trap.** People often say just `json.loads` the model output. That misses wrappers and drift.

**Chapter.** [05](../theory/05-output-parsers.md)

### How do you get JSON you can trust? (parser side)

**Answer.** Prefer the provider path when it exists. A parser on free text is a fallback and needs repair. Catch `OutputParserException` for bad fields. Failure: incomplete JSON from truncation while you blame the prompt.

**Trap.** People often say tell it to reply in JSON. That misses hope versus schema.

**Chapter.** [05](../theory/05-output-parsers.md), [24](../theory/24-structured-outputs.md)

### What is the difference between `OutputFixingParser` and `RetryOutputParser`?

**Answer.** Fixing repairs format with a second LLM call and may invent missing values. Retry re-asks with the prompt when content was omitted. Prefer `RetryOutputParser` / `parse_with_prompt` for content gaps. Failure: trust invented fields from a fixer on a high-cost decision.

**Trap.** People often say `OutputFixingParser` always makes the answer correct. That misses invented content.

**Chapter.** [05](../theory/05-output-parsers.md)

### Are parsers obsolete because of structured output?

**Answer.** No. Prefer structured output when the provider supports it. Keep parsers for unsupported models, repair, streaming partial JSON, and non-JSON shapes. On LangChain v1 some parsers live under `langchain_classic.output_parsers`. Failure: `.with_structured_output` raises and you have no fallback path.

**Trap.** People often say parsers are obsolete because of structured output. That misses compatibility and repair.

**Chapter.** [05](../theory/05-output-parsers.md)

### What production resilience pattern wraps a strict parse chain?

**Answer.** Fail toward a human queue, not a silent wrong category. Wire `prompt | model | parser`, then `with_fallbacks` to a fixer, then a `SAFE_DEFAULT` with `needs_human=True`. Failure: catch bare `Exception` and retry the same prompt forever.

**Trap.** People often say catch `Exception` and retry the same prompt. That misses safe defaults and fallbacks.

**Chapter.** [05](../theory/05-output-parsers.md)

### When does a fixing parser invent dangerous fields?

**Answer.** When required fields were missing and the repair LLM fills them to satisfy the schema. Use fixing for format only. Use retry for content gaps. Use safe defaults for irreversible actions. Failure: invented `needs_human` or category that ships as truth.

**Trap.** People often say if it parses, the business fields are trustworthy. That misses repair hallucinations.

**Chapter.** [05](../theory/05-output-parsers.md)

## 06 — Memory

### Where does conversation memory live now?

**Answer.** In the message list you pass, or in a LangGraph checkpoint keyed by `thread_id`. Old buffer memory lived inside the chain object. Failure: two users or two processes and the old in-chain memory collapses.

**Trap.** People often say LangChain has a memory module you attach. That misses the legacy API era.

**Chapter.** [06](../theory/06-memory.md)

### Why is `ConversationBufferMemory` the wrong interview answer?

**Answer.** It is deprecated / moved to classic. Modern code uses a checkpointer plus `thread_id` with inspectable state via `get_state`. Failure: copy-paste classic memory into v1 and think you shipped production memory.

**Trap.** People often say use `ConversationBufferMemory`. That misses checkpointer plus thread id.

**Chapter.** [06](../theory/06-memory.md)

### What goes wrong if two users share a `thread_id`?

**Answer.** They see each other's context. One thread id per session or user channel. Failure: turn-two amnesia because you neither resent history nor attached a stable checkpointer thread. Also: `InMemorySaver` loses chats on restart.

**Trap.** People often say thread id is just a log label. That misses session isolation.

**Chapter.** [06](../theory/06-memory.md)

### Does window memory keep the last k facts forever?

**Answer.** No. It drops older messages on purpose. A name said early vanishes once outside the window. Widen `k`, summarise, or add vector recall when "What is my name?" fails after several turns. Failure: treat the window as permanent fact storage.

**Trap.** People often say window memory keeps the last k facts forever. That misses deliberate dropping.

**Chapter.** [06](../theory/06-memory.md)

### Is summarisation free compression?

**Answer.** No. It costs an extra LLM call and loses exact wording. Keep recent turns with `keep=`. Tune `trigger` / `keep` when summaries drop critical details. Persist the raw transcript for regulated domains. Failure: summarise away a key constraint and the next answer invents policy.

**Trap.** People often say summarisation is free compression. That misses cost and loss of wording.

**Chapter.** [06](../theory/06-memory.md)

### How do you attach memory to `create_agent`?

**Answer.** Pass a checkpointer into the agent, then send the same `thread_id` every turn. Example: `InMemorySaver` plus `config={"configurable": {"thread_id": "..."}}`. A new id starts cold. Failure: pass a legacy Memory object into a chain constructor and expect durable sessions.

**Trap.** People often say pass a Memory object into the chain constructor. That misses checkpointer config.

**Chapter.** [06](../theory/06-memory.md)

## 07 — LCEL

### What does the pipe operator do?

**Answer.** It wires runnables into a sequence. Left output becomes right input. It is not string concat. Use `|` to build a `RunnableSequence`. Batch, retry, and fallback still apply. Failure: types do not line up and you get cryptic mid-chain errors.

**Trap.** People often say it is syntactic sugar for a function call. That misses streaming, config, and callbacks through the sequence.

**Chapter.** [07](../theory/07-lcel.md)

### What is the correct type flow for prompt | model | parser?

**Answer.** Dict in, prompt value, assistant message, then parsed value. Order is `prompt | model | parser`. Putting the model first feeds the wrong type into the prompt. Failure: `model | prompt` and a baffling type error mid-chain.

**Trap.** People often say put the model first, then the prompt. That misses the type pipeline.

**Chapter.** [07](../theory/07-lcel.md)

### How do you keep the original question after a retrieval step?

**Answer.** Do not forward only documents. Enrich a dict so the question survives. Use `RunnablePassthrough`, `RunnableParallel`, or `.assign`. Wrap Python like PII redaction with `RunnableLambda`. Failure: next stage only sees docs and forgets the user question.

**Trap.** People often say the next stage only needs the documents. That misses keeping keys for the prompt.

**Chapter.** [07](../theory/07-lcel.md)

### How do retry and fallback differ?

**Answer.** Retry repeats the same runnable for transient blips. Fallback swaps to another runnable when a provider is down. Use `.with_retry(...)` for connection hiccups. Use `.with_fallbacks([...])` across providers. Cap `batch` with `max_concurrency`. Failure: treat them as one knob and still hammer a dead primary.

**Trap.** People often say retry and fallback are the same resilience knob. That misses same-runnable vs swap-runnable.

**Chapter.** [07](../theory/07-lcel.md)

### Can LCEL replace LangGraph?

**Answer.** No. LCEL is for DAGs. Cycles, human pause, and durable step persistence need LangGraph. If you bury a Python while-loop and call it LCEL, you still lack graph guarantees. Failure: need interrupt-and-resume and LCEL cannot express it cleanly.

**Trap.** People often say LCEL can replace LangGraph. That misses cycles and durable pauses.

**Chapter.** [07](../theory/07-lcel.md)

### What do `input_schema` and `get_graph` give you on a composed chain?

**Answer.** They let you inspect shape before shipping. You can verify pipe order and parallel branches. Prefer async entry points like `ainvoke` in services. Failure: treat the chain as opaque and only see the final string in production incidents.

**Trap.** People often say composed chains are opaque; you only see the final string. That misses schema and graph inspectability.

**Chapter.** [07](../theory/07-lcel.md)

## 08 — Chains

### Is `SequentialChain` how you build multi-step apps today?

**Answer.** No. It is legacy string wiring with weak streaming and parallelism. Rewrite in LCEL with `|` and `.assign` for typed objects and clearer traces. Failure: assume a string bridge between steps when the second prompt needs several fields.

**Trap.** People often say SequentialChain is how you build multi-step LangChain apps. That misses modern LCEL.

**Chapter.** [08](../theory/08-chains.md)

### When should independent LLM fields share one `.assign`?

**Answer.** When keys do not depend on each other, run them together to save round-trips. Put category and severity in the same `.assign`. Keep dependent stages sequential. Failure: always chain one LLM after another "for clarity" and pay double latency.

**Trap.** People often say always chain one LLM call after another for clarity. That misses concurrent assign.

**Chapter.** [08](../theory/08-chains.md)

### Is a Python loop inside `@chain` as good as a graph?

**Answer.** Fine for prototypes. Not for mid-loop inspection, human approval, or crash resume. Cap impossible QA loops and escalate with a max-attempts verdict. Failure: spin forever with no graph visibility in traces.

**Trap.** People often say any Python loop inside `@chain` is as good as a graph. That misses inspectability and resume.

**Chapter.** [08](../theory/08-chains.md)

### When should a chain skip the LLM entirely?

**Answer.** When a deterministic rule is enough. Short tickets under a word threshold can early-return from `@chain` with no model call. At high volume that is a cost line. Failure: "always call the LLM; prompts are cheap" at ten thousand tickets a day.

**Trap.** People often say always call the LLM; prompts are cheap. That misses volume cost.

**Chapter.** [08](../theory/08-chains.md)

## 09 — Document loaders

### What does a document loader actually produce?

**Answer.** Documents with text plus metadata, not bare strings. Plan `source`, `domain`, and `sensitivity` at ingest. `TextLoader`, `CSVLoader`, and `PyPDFLoader` differ in cardinality. Cost here is CPU and IO only. Failure: skip metadata now and you cannot filter or cite later.

**Trap.** People often say a document loader just reads files into strings. That misses the metadata contract.

**Chapter.** [09](../theory/09-document-loaders.md)

### Why can a PDF "load successfully" and still poison RAG?

**Answer.** Load success means no exception, not good text. Multi-column mess, empty scans, and repeated headers all index as junk. Audit lengths with `audit_extraction` before embedding. Failure: silent empty extraction and "my RAG returns nothing."

**Trap.** People often say if the PDF loads without an exception, the text is fine. That misses layout and OCR failures.

**Chapter.** [09](../theory/09-document-loaders.md)

### How do you enforce access control on documents?

**Answer.** Put clearance in metadata and filter server-side at retrieval. If a restricted chunk reaches the prompt, it already leaked. A system line saying "ignore restricted docs" is not an ACL. Failure: filter only in the prompt and confidential text still enters context.

**Trap.** People often say access control can be a system prompt that says ignore restricted docs. That misses real ACL filters.

**Chapter.** [09](../theory/09-document-loaders.md)

### What encoding and batch-ingest footguns show up on Windows?

**Answer.** Encoding and one bad file can kill the whole job. Set `encoding="utf-8"` on text loaders. Isolate per-file try/except on nightly walks. Failure: one corrupt PDF aborts the ingest and nothing else indexes that night.

**Trap.** People often say loaders are fire-and-forget; failures are rare. That misses encoding and batch isolation.

**Chapter.** [09](../theory/09-document-loaders.md)

### When do you hand-build `Document` lists instead of convenience loaders?

**Answer.** When default loaders give bad cardinality or chrome-heavy text. Build Documents by hand for Markdown sections or soup-strained web articles. Mechanism at load time is cardinality and metadata quality. Failure: expect a better embedding model to fix empty pages or missing `source`.

**Trap.** People often say always use the highest-level loader available. That misses content quality at ingest.

**Chapter.** [09](../theory/09-document-loaders.md)

### What should you verify before the first `embed_documents` call?

**Answer.** Audit before you pay. Run `corpus_report` / `audit_extraction` for non-empty text, present `source`, and expected sensitivity counts. Failure: index first, then discover empty pages after the embedding bill.

**Trap.** People often say index first, debug retrieval quality later. That misses pre-embed audits.

**Chapter.** [09](../theory/09-document-loaders.md)

## 10 — Text splitters

### Why not split every 500 characters with no overlap?

**Answer.** Character cuts ignore structure and can still blow size limits when a separator never fires. Prefer `RecursiveCharacterTextSplitter` with a separator ladder and about 10–20% overlap. Failure: wrong chunking forces a full corpus re-embed later.

**Trap.** People often say just split every 500 characters with no overlap. That misses structure and boundary facts.

**Chapter.** [10](../theory/10-text-splitters.md)

### Should `SemanticChunker` be the default?

**Answer.** No. It costs an embed per sentence and is slow. Use it when recursive splitting fails on your eval set. Ordinary splitters cost zero LLM calls. Failure: default to semantic chunking and burn index time for little gain.

**Trap.** People often say SemanticChunker should be the default because it understands meaning. That misses cost as the escape hatch.

**Chapter.** [10](../theory/10-text-splitters.md)

### Are larger chunks always safer?

**Answer.** No. Huge chunks bury the answer. Tiny chunks miss multi-key facts and add noise. Sweep sizes and overlaps on a fixed question set. Failure: pick one extreme and never measure self-containment.

**Trap.** People often say larger chunks are always safer. That misses buried answers and mixed topics.

**Chapter.** [10](../theory/10-text-splitters.md)

### How do you keep citations and structure after splitting?

**Answer.** Split Documents so metadata like `page` survives. Use `split_documents`. For Markdown, run `MarkdownHeaderTextSplitter` first so headers become metadata. Failure: rebuild bare strings and lose citation paths.

**Trap.** People often say rebuild Documents manually after splitting; metadata is optional. That misses citation survival.

**Chapter.** [10](../theory/10-text-splitters.md)

### When should you not split at all?

**Answer.** When records are already atomic. CSV rows should not be blindly re-chunked. Use language-aware splitters for code. Measure self-containment before changing production sizes. Failure: one global `chunk_size` for every corpus shape.

**Trap.** People often say every corpus needs the same chunk_size. That misses atomic records and domain shape.

**Chapter.** [10](../theory/10-text-splitters.md)

### What fails when overlap is zero on policy text?

**Answer.** Facts get cut from their conditions. Answers sound confident and wrong. Raise overlap and re-run self-containment. Failure: treat overlap as pure storage waste and ship severed policy clauses.

**Trap.** People often say overlap only wastes storage. That misses boundary safety on policy text.

**Chapter.** [10](../theory/10-text-splitters.md)

## 11 — Embeddings

### How does RAG retrieve?

**Answer.** Chunks become vectors. The question becomes a vector with the same model. The store returns nearby chunks, optionally filtered. The model only sees what you put in the prompt. Failure: the right chunk never retrieves, so the answer cannot be grounded.

**Trap.** People often say the vector database understands the document. That misses similarity over stored vectors.

**Chapter.** [11](../theory/11-embeddings.md), [12](../theory/12-vector-stores.md)

### Why must `embed_query` and `embed_documents` stay role-correct?

**Answer.** Some models treat short questions and long passages differently. Wrong method shifts geometry and hurts recall. Always match API to role. Failure: swap methods and neighbors look random even with a good store.

**Trap.** People often say `embed_query` and `embed_documents` are interchangeable. That misses asymmetric training.

**Chapter.** [11](../theory/11-embeddings.md)

### Does similarity above 0.8 mean the answer is correct?

**Answer.** No. Negations and nearby numbers often score high. Similarity only finds candidates. Read the chunk text with the model or a reranker. Failure: "may / may not work remotely" or 60 vs 90 days look "close enough" by cosine.

**Trap.** People often say similarity above 0.8 means the answer is correct. That misses negation and numeric near-misses.

**Chapter.** [11](../theory/11-embeddings.md)

### How should embedding caches be namespaced?

**Answer.** Key by text hash so unchanged text is safe to reuse. Never reuse a namespace across models. Leave query cache off for open-ended chat. Use `CacheBackedEmbeddings.from_bytes_store(..., namespace=...)`. Failure: stale vectors from another model with no warning.

**Trap.** People often say caching embeddings risks wrong answers because the text might change. That misses hash miss on change; the real footgun is cross-model namespaces.

**Chapter.** [11](../theory/11-embeddings.md)

### What happens if you mix embedding dimensions in one index?

**Answer.** Distance geometry breaks and neighbors go nonsense. Never mix models in one index. Probe dimension at create time. `EMBEDDING_PROVIDER` is independent of chat. Failure: assume the store will reconcile different vector sizes.

**Trap.** People often say the store will reconcile different vector sizes. That misses fixed index dimension.

**Chapter.** [11](../theory/11-embeddings.md)

### What is the cost shape of embeddings in RAG?

**Answer.** Offline document embedding dominates. Query cost is one embed per question unless cache hits. Re-embed without cache means paying again every deploy. Failure: treat embedding cost as negligible next to chat and blow the index budget.

**Trap.** People often say embedding cost is negligible next to the chat model. That misses bulk index cost.

**Chapter.** [11](../theory/11-embeddings.md)

### Why do opposite claims still retrieve together?

**Answer.** Embeddings capture topic, not logic. High cosine for opposite claims is expected. Never trust similarity alone for facts, numbers, or negation. Failure: assume a "good" model understands contradiction.

**Trap.** People often say a good embedding model understands contradiction. That misses topic geometry.

**Chapter.** [11](../theory/11-embeddings.md)

### What API surface should you name in an embeddings interview?

**Answer.** Name the role methods and the cache wrapper. Say `embed_query` / `embed_documents`, plus `CacheBackedEmbeddings` with a byte store and namespace. Provider choice locks dimension. Failure: wave at `model.encode` with no caching or re-index story.

**Trap.** People often say embeddings are just `model.encode` with no caching concerns. That misses roles, cache, and rebuild.

**Chapter.** [11](../theory/11-embeddings.md)

## 12 — Vector stores

### Is FAISS a database server?

**Answer.** No. FAISS is a library inside your process. You save and load files. There is no network service unless you wrap it. Prefer Chroma, pgvector, or Pinecone when you need a database shape. Failure: treat FAISS like Postgres and expect remote multi-writer ops.

**Trap.** People often say FAISS is a database server like Postgres. That misses in-process file-backed reality.

**Chapter.** [12](../theory/12-vector-stores.md)

### Why does `allow_dangerous_deserialization=True` exist on FAISS load?

**Answer.** The docstore is pickle. Unpickling untrusted data can run code. Pass `True` only for indexes you built or trust. After load, smoke-test `index.ntotal` and a known query. Failure: always set the flag and load a random pickle from the internet.

**Trap.** People often say `allow_dangerous_deserialization=True` is just an annoying flag — always set it. That misses real RCE risk.

**Chapter.** [12](../theory/12-vector-stores.md)

### Why are metadata filters not the same as prompt instructions for ACL?

**Answer.** Prompts are requests. Filters remove forbidden chunks before context. Only filters are an ACL. Map clearances to allowed sensitivity sets server-side. Failure: filter only in the prompt and restricted docs still leak into the window.

**Trap.** People often say metadata filters and system prompts are equivalent for access control. That misses pre-context removal.

**Chapter.** [12](../theory/12-vector-stores.md)

### When do you use MMR instead of pure similarity?

**Answer.** Pure similarity returns near-duplicates. MMR diversifies. Call `max_marginal_relevance_search` with `fetch_k` larger than `k`. Failure: top-k shows the same leave rule three times and nothing else.

**Trap.** People often say top-k similarity is always enough. That misses redundancy.

**Chapter.** [12](../theory/12-vector-stores.md)

### What must match when creating a managed vector index?

**Answer.** Index dimension must equal embedding dimension. Probe dim from the embedding model at create time. Model change still means full re-embed. Failure: wrong Pinecone dim returns nonsense or create fails.

**Trap.** People often say the store resizes vectors automatically. That misses fixed dimension contracts.

**Chapter.** [12](../theory/12-vector-stores.md)

### Which search APIs should you be ready to name?

**Answer.** Name the usual search and CRUD surface. Say `similarity_search`, score variants, MMR, `add_documents` / `delete`, and `.as_retriever`. Chroma filters use `$eq`, `$and`, `$in`. Failure: claim stores only expose a single `search` method.

**Trap.** People often say vector stores only expose a single `search` method. That misses the real API surface.

**Chapter.** [12](../theory/12-vector-stores.md)

### Is a hallucinated answer a vector-store bug?

**Answer.** Often no. Weak grounding or empty retrieval handed to the model is more common. Fix refusal prompts and empty handling before swapping FAISS for Chroma. Failure: blame the database when the model answered from junk context.

**Trap.** People often say wrong answers mean the vector database is broken. That misses grounding and retrieval quality.

**Chapter.** [12](../theory/12-vector-stores.md)

### What files does FAISS `save_local` write, and how do you validate reload?

**Answer.** Several files under the artifact directory, including a pickle docstore. After `load_local(..., allow_dangerous_deserialization=True)`, check `index.ntotal` and a smoke query. Treat that as part of deploy. Failure: assume reload always works and ship a broken index.

**Trap.** People often say saving FAISS is a single opaque blob; reload always works. That misses multi-file artifacts and smoke tests.

**Chapter.** [12](../theory/12-vector-stores.md)

## 13 — Retrievers

### Why not just increase `k` until answers improve?

**Answer.** Larger k raises recall and also dumps noise. The model then latches onto the wrong sentence. Prefer hybrid plus compression or rerank, and measure `hit@k` plus answer quality. Failure: same paragraph thrice after raising k — try MMR.

**Trap.** People often say just increase `k` until answers improve. That misses noise in the prompt.

**Chapter.** [13](../theory/13-retrievers.md)

### Why did embeddings not make keyword search obsolete?

**Answer.** Dense misses exact ids that BM25 nails. BM25 misses paraphrases dense nails. That is why `EnsembleRetriever` with RRF weights exists. Sweep weights on a labelled set. Failure: drop keyword search and miss `SOC 2` or ticket ids.

**Trap.** People often say embeddings made keyword search obsolete. That misses exact-token failure modes.

**Chapter.** [13](../theory/13-retrievers.md)

### When is empty retrieval the correct outcome?

**Answer.** When a score threshold says nothing is on-topic. Empty lets the chain refuse instead of hallucinating from junk. Treating empty as a bug forces weak hits into context. Failure: stuff low-score docs and invent an answer.

**Trap.** People often say an empty retrieval is a bug. That misses refusal as the safe outcome.

**Chapter.** [13](../theory/13-retrievers.md)

### What do multi-query and compression add?

**Answer.** Multi-query expands the ask into several retrievals. Compression shrinks noisy context before generation. Use `MultiQueryRetriever` and `ContextualCompressionRetriever`. You pay extra LLM or embed calls to save prompt tokens. Failure: stack every transform without measuring tokens or quality.

**Trap.** People often say always stack every retriever transform. That misses cost versus lift.

**Chapter.** [13](../theory/13-retrievers.md)

### What is `ParentDocumentRetriever` for?

**Answer.** Small children match. Large parents answer. You need a vectorstore, a docstore, and both splitters. Put business ACL in a custom `BaseRetriever`, not only in prompts. Failure: one chunk size that is bad at both matching and answering.

**Trap.** People often say one chunk size serves both matching and answering. That misses child/parent roles.

**Chapter.** [13](../theory/13-retrievers.md)

### How should you evaluate retriever changes?

**Answer.** Freeze an eval set and compare methods on `hit@k`. Include similarity, MMR, BM25, and hybrid. Hybrid usually wins on mixed exact-plus-conceptual sets. Failure: change the retriever and eyeball one question.

**Trap.** People often say change the retriever and eyeball one question. That misses labelled metrics.

**Chapter.** [13](../theory/13-retrievers.md)

### How do you expose a vector store as a retriever?

**Answer.** Wrap the store so chains can compose it. Call `.as_retriever(search_type=..., search_kwargs=...)`. Then add ensemble, multi-query, or compression as needed. Failure: call `similarity_search` from the chain and treat retrievers as optional wrappers.

**Trap.** People often say call `similarity_search` from the chain; retrievers are optional wrappers. That misses composition.

**Chapter.** [13](../theory/13-retrievers.md)

### What APIs should you list for hybrid retrieval?

**Answer.** Name BM25, ensemble, and the vector retriever. Say `BM25Retriever.from_documents`, `EnsembleRetriever(weights=...)`, and vector `as_retriever`. RRF-style fusion is why ensemble beats either channel alone. Failure: concatenate two lists and call it hybrid.

**Trap.** People often say hybrid means calling two retrievers and concatenating lists. That misses rank fusion.

**Chapter.** [13](../theory/13-retrievers.md)

## 14 — Retrieval chains

### Why do follow-up questions fail?

**Answer.** Short follow-ups are not searchable as written. "What about contractors?" needs a rewrite into a standalone question first. A history-aware step uses chat history, then embeds that. Failure: pass the whole history into the retriever, which embeds a string, not a conversation.

**Trap.** People often say just pass the whole history into the retriever. That misses standalone rewrite.

**Chapter.** [14](../theory/14-retrieval-chains.md)

### Why is chat memory alone not enough for RAG follow-ups?

**Answer.** Memory helps the answer prompt. Retrieval still embeds the raw follow-up unless you rewrite first. Use `create_history_aware_retriever`. The classic bug is the retriever input. Failure: topic-change follow-ups cling to the old topic.

**Trap.** People often say chat memory is enough for RAG follow-ups. That misses retriever rewrite.

**Chapter.** [14](../theory/14-retrieval-chains.md)

### Can `create_retrieval_chain` retry if the answer looks wrong?

**Answer.** No. It is retrieve once, answer once. No cycles, no re-query, no human pause. Those need a graph. Failure: keep wrapping the helper hoping it becomes self-reflective RAG.

**Trap.** People often say `create_retrieval_chain` can retry if the answer looks wrong. That misses the fixed pipeline.

**Chapter.** [14](../theory/14-retrieval-chains.md)

### Does a strong system prompt guarantee no outside knowledge?

**Answer.** No. Prompt rules are requests. Verify with groundedness checks. Treat retrieved context as DATA. Failure: off-corpus questions still get fluent invented policy.

**Trap.** People often say a strong system prompt guarantees the model never uses outside knowledge. That misses verification.

**Chapter.** [14](../theory/14-retrieval-chains.md)

### What does a good history-aware follow-up transcript look like?

**Answer.** Turn one asks sick leave days. Turn two asks for the certificate. Turn three asks about staying out longer. Without rewrite, turn two retrieves unrelated certificate docs. With history-aware retrieval, sources stay on leave policy. Failure: raise k on the same raw follow-up and hope.

**Trap.** People often say follow-ups only need the same retriever with a larger k. That misses rewrite.

**Chapter.** [14](../theory/14-retrieval-chains.md)

### How should RAG UIs avoid a blank screen during retrieval?

**Answer.** Do not block on one silent `invoke`. Stream tokens. Show sources when `context` arrives. Streaming improves UX without changing call counts. Failure: users stare at a blank page until the full chain finishes.

**Trap.** People often say users can wait for the full chain to finish. That misses streaming UX.

**Chapter.** [14](../theory/14-retrieval-chains.md)

### What is the injection defense for retrieved documents?

**Answer.** Context is DATA, not commands. Put that in the system rule. Probe with a doc that says ignore previous instructions. Combine with refusal when context lacks the answer. Failure: trust retrieved text because it came from "our" corpus.

**Trap.** People often say retrieved text is trusted because it came from our corpus. That misses injection in chunks.

**Chapter.** [14](../theory/14-retrieval-chains.md)

### When do you stop extending retrieval-chain helpers?

**Answer.** When you need re-retrieval, routing, human pause, or reflective loops. Helpers are fixed pipelines. Move those needs to LangGraph. Failure: nest more LCEL around `create_retrieval_chain` until it "acts like an agent."

**Trap.** People often say keep wrapping `create_retrieval_chain` until it behaves like an agent. That misses graph boundaries.

**Chapter.** [14](../theory/14-retrieval-chains.md)

## 15 — Advanced RAG

### What is HyDE, and when does it hurt?

**Answer.** HyDE invents a hypothetical passage, embeds that, and retrieves real docs nearby. The invented text can be wrong. It only needs to look like a document. Skip it for alien domains or exact ids like `TCK-1006` or `SOC 2`. Failure: assume HyDE is better because the LLM "knows" the answer.

**Trap.** People often say HyDE is better because the LLM knows the answer. That misses shape-match versus facts.

**Chapter.** [15](../theory/15-advanced-rag.md)

### What is hybrid RRF doing in advanced RAG?

**Answer.** It merges dense and sparse rankings so rare tokens and paraphrases both surface. Adopt hybrid first on eval before heavier transforms. Failure: always run the full advanced stack when dense already ties on a clean corpus.

**Trap.** People often say always run the full advanced pipeline in production. That misses measure-and-stop.

**Chapter.** [15](../theory/15-advanced-rag.md)

### When do you rerank?

**Answer.** When the right chunk sits low under a bi-encoder. Rerank top N down to a few for generation. Right chunk at rank twelve is a rank problem, not always a missing-doc problem. Failure: assume similarity always puts the answer in top three.

**Trap.** People often say if it is in the index, similarity search will put it in top-3. That misses ranking failures.

**Chapter.** [15](../theory/15-advanced-rag.md)

### Are CRAG / Adaptive RAG / Self-RAG LangChain import classes here?

**Answer.** No. Here they are named compositions: grader plus fallback plus graph control. Speak the names. Implement the durable loop later with LangGraph. Failure: hunt for a special import class called CRAG.

**Trap.** People often say CRAG is a special LangChain class you import. That misses composition versus product class.

**Chapter.** [15](../theory/15-advanced-rag.md)

### What does query decomposition fix?

**Answer.** Compound questions that one retrieval only half-answers. Split into sub-queries, retrieve per part, then synthesize. Multi-query and HyDE fix vocabulary mismatch. Decomposition fixes multi-facet asks. Failure: one embedding of the full compound question.

**Trap.** People often say one embedding of the full compound question is enough. That misses multi-facet retrieval.

**Chapter.** [15](../theory/15-advanced-rag.md)

### How do you adopt advanced techniques without cargo-culting?

**Answer.** Compare dense, BM25, hybrid, HyDE, and advanced on the same eval set. Print `hit@k` and seconds per query. Adoption order: hybrid, then rerank, then transforms. Failure: claim advanced always beats dense on every corpus.

**Trap.** People often say advanced RAG always beats dense on every corpus. That misses measured adoption.

**Chapter.** [15](../theory/15-advanced-rag.md)

### What failure maps to which advanced fix?

**Answer.** Match the symptom to one fix. Vocab mismatch → multi-query or HyDE. Exact terms → hybrid RRF. Right doc ranked low → rerank. Compound question → decomposition. Named CRAG-style loops → later graphs. Failure: turn on every advanced flag at once.

**Trap.** People often say turn on every advanced flag at once. That misses the symptom map.

**Chapter.** [15](../theory/15-advanced-rag.md)

### Why measure seconds per query for advanced pipelines?

**Answer.** Stacked LLM transforms can explode latency while `hit@k` barely moves. Drop losers on both quality and time. On clean small data, dense often matches advanced cheaper. Failure: treat latency as a deploy concern and ignore it in quality experiments.

**Trap.** People often say latency is a deploy concern; quality experiments ignore it. That misses seconds-per-query trade-offs.

**Chapter.** [15](../theory/15-advanced-rag.md)

## 16 — Tools

### What happens when you `bind_tools` — does the LLM run your Python?

**Answer.** No. The model only emits tool calls. Your runtime runs the functions and returns matching `ToolMessage`s. That gap is where authz and audits live. Break `tool_call_id` pairing and the next provider request is invalid. Failure: assume bind means Python ran inside the model.

**Trap.** People often say when you bind tools, the LLM runs your Python. That misses host execution and id pairing.

**Chapter.** [16](../theory/16-tools.md)

### Why are tool docstrings part of the product?

**Answer.** They are the prompt the model uses to choose tools. Vague docs cause non-selection and bad arguments. Verb-phrase names, type hints, and Pydantic `args_schema` matter. Failure: five near-duplicate tools and the model picks the wrong one.

**Trap.** People often say tool docstrings are just for developers. That misses model selection.

**Chapter.** [16](../theory/16-tools.md)

### Why is `SQLDatabaseToolkit` not safe by default?

**Answer.** An LLM-generated `DELETE` can execute on a writable connection. Use a read-only DB user first. Keyword filters are a second layer only. Failure: trust the model to be "helpful" and wipe a table.

**Trap.** People often say SQLDatabaseToolkit is safe because the model is helpful. That misses write blast radius.

**Chapter.** [16](../theory/16-tools.md)

### How should tools fail back to the model?

**Answer.** Do not raise into the agent loop and crash. Return model-readable guidance strings. Normalize ambiguous ids. Use a calculator tool for math, never raw `eval`. Failure: let exceptions bubble and kill the whole run.

**Trap.** People often say let exceptions bubble; the agent framework will handle them. That misses guidance returns.

**Chapter.** [16](../theory/16-tools.md)

### What is the manual tool-call message contract?

**Answer.** After `bind_tools`, inspect `AIMessage.tool_calls`, run each tool, append `ToolMessage(..., tool_call_id=call["id"])`, then invoke again. Do not paste the result as a `HumanMessage`. Failure: wrong message type and the provider rejects the transcript.

**Trap.** People often say append a HumanMessage with the tool result. That misses `ToolMessage` and id matching.

**Chapter.** [16](../theory/16-tools.md)

### When do agents ignore your tool?

**Answer.** Vague docstring, bad name, or tool overload. Rewrite descriptions, cut the surface, specialize. Pydantic constraints stop nonsense args before execution. Failure: assume the model will discover the right tool if it exists.

**Trap.** People often say the model will discover the right tool if it exists. That misses naming and docs as product.

**Chapter.** [16](../theory/16-tools.md)

### How do tools relate to agents, graphs, and MCP?

**Answer.** Tools are schemas plus host execution. Agents automate the message loop. Graphs add durable state. MCP moves tools out of process. Notebook 16 rules still apply after MCP. Failure: treat tools, agents, and MCP as interchangeable buzzwords.

**Trap.** People often say tools, agents, and MCP are interchangeable buzzwords. That misses layer roles.

**Chapter.** [16](../theory/16-tools.md)

### What belongs on a tool-design checklist?

**Answer.** Clear name, when-to-use description, types, constraints, compact returns, guidance errors, small surface. Prefer typed tools over a free REPL. Failure: one mega-tool with a mode enum that confuses selection.

**Trap.** People often say expose one mega-tool with a mode enum. That misses focused tool surfaces.

**Chapter.** [16](../theory/16-tools.md)

## 16a — Code execution

### Why is a system prompt not enough to secure a Python REPL?

**Answer.** Prompts are requests. Sandboxing is enforcement. Isolate filesystem, network, imports, and time. If you cannot isolate, ship a calculator or typed tool instead. Failure: a REPL that shares process FS reads `.env` and SSH keys.

**Trap.** People often say a system prompt telling the REPL not to import `os` is enough. That misses hard controls.

**Chapter.** [16a](../theory/16a-code-execution.md)

### Should new work use `AgentExecutor` for code interpreters?

**Answer.** No. That path is historical. Use `create_agent` with middleware limits like `ModelCallLimitMiddleware`. Cap unbounded loops whether or not the sandbox holds. Failure: follow old tutorials into `AgentExecutor`.

**Trap.** People often say use AgentExecutor for code interpreters like the old tutorials. That misses modern create_agent.

**Chapter.** [16a](../theory/16a-code-execution.md)

### Why prefer typed DataFrame tools over `create_csv_agent`?

**Answer.** CSV agents often need `allow_dangerous_code=True` and run model-written pandas. Prefer a typed aggregation tool unless you have a real sandbox. Router tools keep the model off a REPL on the happy path. Failure: treat `create_csv_agent` as the standard table path.

**Trap.** People often say `create_csv_agent` is the standard way to analyze tables. That misses dangerous code flags.

**Chapter.** [16a](../theory/16a-code-execution.md)

### How do you harden a narrow `calculate` tool?

**Answer.** Empty builtins, allow-listed characters or AST, no import escapes. Wide allow-lists recreate the REPL. Still add call-limit middleware. Write files only under `ctx.artifact(...)`. Failure: `eval` plus a prompt ban on imports.

**Trap.** People often say a calculator with `eval` is fine if the prompt forbids imports. That misses sandbox holes.

**Chapter.** [16a](../theory/16a-code-execution.md)

### What is the sandbox checklist before shipping REPL access?

**Answer.** Allow-list imports. No host FS. No network. Resource limits. Audit logs. Prefer calculator and typed DataFrame tools. Deliberately try to read `.env`. Failure: ship demo REPL settings to production.

**Trap.** People often say demo REPL settings are close enough for production. That misses blast radius.

**Chapter.** [16a](../theory/16a-code-execution.md)

### Where should agent-generated files land?

**Answer.** Under an artifacts path from context, never the repo root. Use `ctx.artifact(...)`. Pair with call-limit middleware. Failure: save plots next to the notebook and overwrite project files.

**Trap.** People often say saving next to the notebook is fine for generated plots. That misses deploy incidents.

**Chapter.** [16a](../theory/16a-code-execution.md)

## 16b — MCP

### Is MCP a LangChain feature?

**Answer.** No. MCP is a protocol. Adapters turn server tools into LangChain `BaseTool`s. Use `langchain-mcp-adapters` and `get_tools()`. After that, notebook 16 and 17 rules still apply. Failure: treat MCP as a LangChain-only API.

**Trap.** People often say MCP is a LangChain feature. That misses the open protocol.

**Chapter.** [16b](../theory/16b-mcp.md)

### When should you keep `@tool` in-process instead of MCP?

**Answer.** MCP adds process and network overhead. Keep private helpers in-process. Use MCP for shared or isolated capabilities owned elsewhere. One concern per server. Failure: replace every `@tool` with MCP for "cleanliness."

**Trap.** People often say replace every `@tool` with MCP for cleanliness. That misses overhead and ownership.

**Chapter.** [16b](../theory/16b-mcp.md)

### Do new server tools appear automatically mid-run?

**Answer.** No. Discovery happens when you list tools. After adding a server tool, restart or reconnect so the client re-lists. Call `await client.get_tools()`. Failure: expect new tools mid-run with no refresh.

**Trap.** People often say once the client is connected, new server tools appear automatically mid-run. That misses rediscovery.

**Chapter.** [16b](../theory/16b-mcp.md)

### When do you choose stdio vs HTTP transport?

**Answer.** Prefer stdio locally for demos. Use HTTP for shared services once you accept network ops. Server side: `FastMCP` plus `mcp.run(transport=...)`. Client side: `MultiServerMCPClient` with command/args or url. Failure: expose HTTP from day one for a laptop demo.

**Trap.** People often say always expose MCP over HTTP from day one. That misses local stdio simplicity.

**Chapter.** [16b](../theory/16b-mcp.md)

### How do multiple MCP servers show up to one agent?

**Answer.** Point one client at several server entries. Math over stdio and weather over HTTP can both appear once enabled. Adding `divide` on the math server needs a client refresh, not agent edits. Failure: assume one agent process can only speak to one MCP server.

**Trap.** People often say one agent process can only speak to one MCP server. That misses multi-server clients.

**Chapter.** [16b](../theory/16b-mcp.md)

### What operational failures are specific to MCP?

**Answer.** Client started before server update misses tools. Commented HTTP entries hide capabilities. Unversioned schema changes break clients overnight. Secrets leak if tools return keys. Version servers and coordinate clients. Failure: treat MCP as plug-and-play with no ops surface.

**Trap.** People often say MCP is plug-and-play with no ops surface. That misses discovery and versioning.

**Chapter.** [16b](../theory/16b-mcp.md)

## 17 — Agents

### What is an agent loop?

**Answer.** The model asks for tools. You run them. You send results back with matching ids. Then you call the model again until it answers normally. `create_agent` is that loop with middleware. Failure: break `tool_call_id` pairing and the next request is invalid.

**Trap.** People often say the agent decides and Python magically runs tools. That misses the id pairing contract.

**Chapter.** [17](../theory/17-agents.md), [16](../theory/16-tools.md), [17a](../theory/17a-agent-loop.md)

### Are agents always better than chains because they can use tools?

**Answer.** No. Agents are slower, costlier, and less predictable. If you can draw the flowchart, build the flowchart. Use agents when the step order is unknown ahead of time. Failure: default to agents on a fixed retrieve-then-answer path.

**Trap.** People often say agents are always better than chains because they can use tools. That misses cost and predictability.

**Chapter.** [17](../theory/17-agents.md)

### Is `create_agent` unrelated to LangGraph?

**Answer.** No. It returns a compiled graph with model and tools nodes. Checkpointing, streaming, and HITL lessons apply. Inspect with `get_graph()`. Reach for a custom `StateGraph` when you need guarantees beyond the tool loop. Failure: treat create_agent as a separate universe from graphs.

**Trap.** People often say `create_agent` is unrelated to LangGraph. That misses the compiled graph under it.

**Chapter.** [17](../theory/17-agents.md)

### Why did native tool calling beat ReAct text agents?

**Answer.** Text plans drift and break parsers in production. Native tool calling returns structured JSON from the provider. Keep ReAct mainly for small local models without tool calling. Failure: careful prompting still fails under load on Action/Action Input formats.

**Trap.** People often say ReAct text agents are fine if the prompt is careful. That misses format drift.

**Chapter.** [17](../theory/17-agents.md)

### How do you stop infinite agent loops?

**Answer.** Cap model calls with middleware and a clear exit behavior. Also fix root causes: weak tool returns, vague tools, prompts that answer without tools. Use `ModelCallLimitMiddleware`. Failure: raise temperature and hope it stops.

**Trap.** People often say raise temperature and hope it stops. That misses hard call limits.

**Chapter.** [17](../theory/17-agents.md)

### What control plane does middleware give you?

**Answer.** Limits, retries, errors, fallbacks, structured output, and checkpointers. System text is also a control: loose prompts skip tools; strict prompts force them. Failure: treat middleware as optional sugar around `create_agent`.

**Trap.** People often say middleware is optional sugar around `create_agent`. That misses the real control plane.

**Chapter.** [17](../theory/17-agents.md)

### How do agents return typed objects?

**Answer.** Set a structured response strategy on the agent, then read the typed field after the tool loop. Use `response_format=ToolStrategy(Schema)` and `structured_response`. Failure: parse free-text finals and call that "typed."

**Trap.** People often say agents can only return free-text final answers. That misses structured_response.

**Chapter.** [17](../theory/17-agents.md), [24](../theory/24-structured-outputs.md)

### When do you graduate from `create_agent` to an explicit `StateGraph`?

**Answer.** When you need guaranteed steps, interrupts, arbitrary cycles, or compliance edges prompts cannot enforce. Remember create_agent already is a small graph for the tool loop. Failure: rewrite every agent as a hand-rolled StateGraph for no new guarantee.

**Trap.** People often say replace every agent with a hand-rolled StateGraph. That misses when guarantees actually require it.

**Chapter.** [17](../theory/17-agents.md)

## 17a — Agent loop

### Is `create_agent` a different algorithm than hand-rolled `bind_tools`?

**Answer.** No. It automates the same message-level loop and adds middleware. If you cannot hand-write Level 1, you cannot debug Level 4. Keep a hand-rolled loop in your notes. Ship `create_agent`. Failure: treat them as different algorithms.

**Trap.** People often say `create_agent` uses a totally different algorithm than bind_tools. That misses the same message contract.

**Chapter.** [17a](../theory/17a-agent-loop.md)

### Why is ReAct prompting less reliable than native tool calling?

**Answer.** Text plans break on formatting typos. Native tool calling returns structured JSON from the provider. Count Level 3 parse failures rather than pretend careful prompts fix them. Failure: ship ReAct under load and eat `OutputParserException`-class outages.

**Trap.** People often say ReAct prompting is just as reliable as tool calling if you are careful. That misses format fragility.

**Chapter.** [17a](../theory/17a-agent-loop.md)

### What should you debug first when an agent misbehaves?

**Answer.** Log the messages list before each model call. Most failures are missing `ToolMessage`s, weak system rules, or id mismatches. Copy `call["id"]` into `ToolMessage(tool_call_id=...)`. Failure: tweak temperature first.

**Trap.** People often say when an agent misbehaves, tweak temperature first. That misses the transcript contract.

**Chapter.** [17a](../theory/17a-agent-loop.md)

### What do the four loop levels teach?

**Answer.** Level 1 is manual bind_tools. Level 2 is schema-shaped function calling. Level 3 is ReAct text. Level 4 is `create_agent` plus call limits. Level 4 should match Level 1 answers with less boilerplate. Failure: skip earlier levels and cannot debug Level 4.

**Trap.** People often say only Level 4 matters; the earlier levels are history. That misses debug skill.

**Chapter.** [17a](../theory/17a-agent-loop.md)

### What failure looks like when `ToolMessage`s are omitted?

**Answer.** The next model call lacks tool outputs. The model invents values or acts confused. Cap iterations when the loop never reaches a normal stop. Failure: assume the model remembers the last tool return implicitly.

**Trap.** People often say the model implicitly remembers the last tool return. That misses explicit ToolMessages.

**Chapter.** [17a](../theory/17a-agent-loop.md)

### What is the debugging order for agent loops?

**Answer.** Print messages. Verify `tool_call_id`. Check system rules. Only then tune sampling. Prefer native tool calling over ReAct for production. Failure: start with temperature and top-p sweeps.

**Trap.** People often say start with temperature and top-p sweeps. That misses transcript-first debugging.

**Chapter.** [17a](../theory/17a-agent-loop.md)

## 18 — Graphs over agents

### When do you refuse an agent?

**Answer.** When the steps are known in advance. An agent spends extra calls choosing tools and can loop. A chain or a graph with explicit edges does the same work with a cost you can count. Failure: use agents as the modern default for everything.

**Trap.** People often say agents are the modern way to do everything. That misses known-path economics.

**Chapter.** [18](../theory/18-graphs-over-agents.md)

### Are agents and graphs interchangeable ways to call tools?

**Answer.** No. Agents hide control flow in prompts. Graphs make edges explicit and enforceable. Persistence, interrupts, and guaranteed QA need graph state. Failure: skip QA under agent flexibility on a compliance path.

**Trap.** People often say agents and graphs are interchangeable ways to call tools. That misses enforceable edges.

**Chapter.** [18](../theory/18-graphs-over-agents.md)

### Does LangGraph replace `create_agent`?

**Answer.** No. `create_agent` already is a graph. Reach for custom `StateGraph` when you need cycles and guarantees beyond the standard tool loop. Failure: rewrite every agent "because LangGraph."

**Trap.** People often say LangGraph replaces create_agent. That misses create_agent as a graph already.

**Chapter.** [18](../theory/18-graphs-over-agents.md)

### Why is a Python while-loop plus DB write not a checkpointer?

**Answer.** Checkpointers integrate with graph steps, interrupts, update_state, and history APIs. Ad-hoc DB writes do not give `invoke(None, thread)` resume or time travel. Failure: process crash loses attempt two when state lives only in locals.

**Trap.** People often say a Python while-loop with a database write is the same as a checkpointer. That misses graph integration.

**Chapter.** [18](../theory/18-graphs-over-agents.md)

### How do human-in-the-loop pauses work on a graph?

**Answer.** Compile with `interrupt_before`. Inspect or edit with `update_state`. Resume with `invoke(None, thread)`. Do not re-send the full original input or you restart from scratch. Failure: edit a local variable and re-call invoke with the same input.

**Trap.** People often say re-call invoke with the same input after the human edits a local variable. That misses resume semantics.

**Chapter.** [18](../theory/18-graphs-over-agents.md)

### What does time travel give you that agents lack?

**Answer.** Every step becomes addressable. Use `get_state_history` to recover, branch, or debug. Agents without durable external state cannot offer that. Failure: rely on stdout logs to replay a multi-step run.

**Trap.** People often say logging stdout is enough to replay a multi-step agent. That misses addressable history.

**Chapter.** [18](../theory/18-graphs-over-agents.md)

## 22 — Multimodal

### How do you pass an image to a chat vision model?

**Answer.** Providers need URL or base64 content blocks inside the message, not a filesystem path. Private URLs often cannot be fetched. Inline data URLs. Map MIME correctly. Failure: pass a file path and the model never sees pixels.

**Trap.** People often say pass the file path to the model. That misses content-block requirements.

**Chapter.** [22](../theory/22-multimodal.md)

### Who owns approve/reject policy on a receipt or expense image?

**Answer.** The model reads pixels. Python owns arithmetic and policy for audit. Always verify `subtotal + tax ≈ total`. Digit misreads are common and silent. Failure: let the vision model decide approve or reject alone.

**Trap.** People often say let the vision model decide approve/reject. That misses deterministic policy in code.

**Chapter.** [22](../theory/22-multimodal.md)

### Is higher resolution always better?

**Answer.** No. Tokens scale with image size. About 1024px often matches accuracy at far lower cost. Resize before encode. Failure: send huge images, overflow context, and spike the bill.

**Trap.** People often say higher resolution is always better. That misses token cost.

**Chapter.** [22](../theory/22-multimodal.md)

### Is multimodal the same as image generation?

**Answer.** No. Chat vision is input understanding. Image output is a separate generation API. Treat OCR'd image text as untrusted. Failure: call multimodal "image generation" and miss injection in pixels.

**Trap.** People often say multimodal equals image generation. That misses input understanding versus generation APIs.

**Chapter.** [22](../theory/22-multimodal.md)

## 23 — Caching

### What kinds of cache exist around LLM calls?

**Answer.** Exact and semantic caches reuse answers. Prompt caching only discounts repeated prefixes while still generating new output. Normalize inputs or exact keys miss. Failure: treat all three as one idea called "cheaper completions."

**Trap.** People often say caching equals cheaper completions as one idea. That misses three different layers.

**Chapter.** [23](../theory/23-caching.md)

### When is semantic cache dangerous?

**Answer.** Near-synonym and negation collisions can serve the wrong answer. Avoid it for pricing, legal, and medical. Raise thresholds and log both questions when paraphrases collide. Failure: serve "not covered" as "covered."

**Trap.** People often say semantic cache is always better. That misses negation collisions.

**Chapter.** [23](../theory/23-caching.md)

### Why is `InMemoryCache` not production?

**Answer.** It dies on restart. Use SQLite or Redis and isolate per tenant. Version keys or set TTL so stale policy does not survive a doc change. Failure: shared caches leak answers across tenants.

**Trap.** People often say InMemoryCache is fine for production. That misses durability and tenant isolation.

**Chapter.** [23](../theory/23-caching.md)

### Exact hit vs prompt-cache hit — what is the difference?

**Answer.** Exact hits replay a stored generation and skip the model. Prompt-cache hits still generate new tokens but discount the repeated prefix. Semantic hits reuse an answer for a nearby question. Failure: read a latency dashboard and assume the provider "got faster."

**Trap.** People often say a cache hit means the provider regenerated faster. That misses replay versus prefix discount.

**Chapter.** [23](../theory/23-caching.md)

### How do you invalidate cached policy answers?

**Answer.** Version the cache key with prompt or policy version, or use TTL. Without invalidation, yesterday's policy is served forever. Normalize tiny input changes so expected misses are not "broken cache." Failure: assume caches invalidate themselves when the corpus changes.

**Trap.** People often say caches invalidate themselves when the corpus changes. That misses versioned keys and TTL.

**Chapter.** [23](../theory/23-caching.md)

### Which direction is "stricter" for Redis semantic distance?

**Answer.** Lower distance is stricter. Teams who treat 0.1 as looser open negation collisions. Cosine similarity thresholds run the opposite way. Say which metric you mean. Failure: confuse distance with similarity direction.

**Trap.** People often say distance threshold 0.1 is looser. That misses lower-distance-is-stricter.

**Chapter.** [23](../theory/23-caching.md)

## 24 — Structured outputs

### How do you get JSON you can trust?

**Answer.** Prefer the provider path that enforces a schema. Use `.with_structured_output` or agent `response_format`. A parser on free text is a fallback and needs repair. Failure: tell it to reply in JSON and call that a schema.

**Trap.** People often say tell it to reply in JSON. That misses hope versus enforcement.

**Chapter.** [24](../theory/24-structured-outputs.md), [05](../theory/05-output-parsers.md)

### Is structured output just JSON mode in the prompt?

**Answer.** No. Tool calling or constrained decoding binds tokens to the schema. Prompting for JSON is hope. `.with_structured_output` is the guarantee path when supported. Failure: treat prompt JSON mode as structured output.

**Trap.** People often say structured output is just JSON mode in the prompt. That misses constrained decoding.

**Chapter.** [24](../theory/24-structured-outputs.md)

### Why keep Pydantic validators if the provider validates?

**Answer.** The provider validates shape. Your validators encode product rules the model forgets. Put reasoning fields before decision fields. Split fat schemas. Failure: security tickets that skip escalation because only JSON shape was checked.

**Trap.** People often say Pydantic validators are redundant if the provider validates. That misses product rules.

**Chapter.** [24](../theory/24-structured-outputs.md)

### Should you always use `json_schema` method?

**Answer.** No. It is not universal. Unsupported providers raise. Prefer portable `function_calling` / `ToolStrategy` unless you need a specific method. Failure: hard-require `json_schema` everywhere.

**Trap.** People often say always use `json_schema`. That misses provider support gaps.

**Chapter.** [24](../theory/24-structured-outputs.md)

### How do agents return structured objects?

**Answer.** Set `response_format=ToolStrategy(Schema)` and read `structured_response` after the tool loop. Without it, agents return prose. Use Pydantic when you need attributes. Failure: claim agents cannot return typed objects.

**Trap.** People often say agents cannot return typed objects. That misses ToolStrategy and structured_response.

**Chapter.** [24](../theory/24-structured-outputs.md)

### How do you handle parse failures without crashing?

**Answer.** Include the raw payload and check parse errors. Route to manual review instead of raising. Business-rule validation can fail after shape OK. Failure: assume structured output means nothing can fail.

**Trap.** People often say if structured output is on, nothing can fail. That misses raw include and review paths.

**Chapter.** [24](../theory/24-structured-outputs.md)

### Why put reasoning fields before the decision field?

**Answer.** Models decide better when they write rationale first. Decision-first schemas weaken judgments. Combine with product validators for irreversible routes. Failure: demand twenty nested fields in one constrained decode.

**Trap.** People often say field order in the schema does not matter. That misses reasoning-before-decision.

**Chapter.** [24](../theory/24-structured-outputs.md)

### When do you fall back from structured output to parsers?

**Answer.** When the provider cannot constrain decode, when you need repair, streaming partial JSON, or non-JSON shapes. Prefer structured output when available. Keep parsers as compatibility and repair. Failure: pick one forever and refuse the other.

**Trap.** People often say pick one forever: either structured output or parsers. That misses layered use.

**Chapter.** [24](../theory/24-structured-outputs.md), [05](../theory/05-output-parsers.md)

## 25 — Routing

### Is routing just if/else on intents?

**Answer.** Production stacks cascade keyword, then embedding, then LLM. Confidence is a first-class control. Most traffic can be semantic. LLM is the expensive uncertain tier. Failure: treat routing as one if/else on labels.

**Trap.** People often say routing is just an if/else on intents. That misses cascaded cost tiers.

**Chapter.** [25](../theory/25-routing.md)

### Why not always use the LLM router?

**Answer.** Cost explodes. Keyword catches clear phrases. Semantic covers paraphrase. LLM handles residue. Require a cosine floor so weather does not route to billing. Failure: always-LLM on every greeting.

**Trap.** People often say always use the LLM router. That misses cascade economics.

**Chapter.** [25](../theory/25-routing.md)

### How do handoffs differ from routing?

**Answer.** Routing is once at entry. Handoffs reclassify mid-flight with shared notes and hop limits. Without `max_hops`, specialists ping-pong forever. Ambiguous "it's broken" needs a clarifier, not a forced wrong desk. Failure: treat handoffs as the same as entry routing.

**Trap.** People often say handoffs are the same as routing. That misses mid-flight reclassification.

**Chapter.** [25](../theory/25-routing.md)

### Is `RunnableBranch` a multi-agent system?

**Answer.** No. It picks one path to completion. Shared-state handoffs need LangGraph conditional edges. Category bleed means overlapping route descriptions. Failure: call RunnableBranch multi-agent because several specialists exist as branches.

**Trap.** People often say `RunnableBranch` is a multi-agent system. That misses one-shot path selection.

**Chapter.** [25](../theory/25-routing.md)

### What failure modes belong in a routing interview answer?

**Answer.** Name the usual misses: missing keywords, no semantic floor, no hop cap, forced routes on ambiguity, always-LLM cost, overlapping categories. Fixes are cascade, cosine floor, `max_hops`, clarifier, and sharper descriptions. Failure: claim a good classifier makes routing bugs disappear.

**Trap.** People often say if the classifier is a good model, routing bugs disappear. That misses control-plane failures.

**Chapter.** [25](../theory/25-routing.md)

### When must routing move from LCEL to LangGraph?

**Answer.** When a branch must re-route later with shared state. `RunnableBranch` is one-shot. Mid-flight handoffs, hop counting, and inspectable notes need graph edges. Keep LCEL cascades for entry classification when one path finishes the request. Failure: nest RunnableBranches until handoffs "work."

**Trap.** People often say nest RunnableBranches until handoffs work. That misses graph state.

**Chapter.** [25](../theory/25-routing.md)

## 28 — Hub

### Should production pull the latest Hub prompt unpinned?

**Answer.** No. Unpinned pull makes behaviour depend on someone else's edit with no deploy. Pin `owner/name:hash` and upgrade on purpose. Log prompt name and version in traces. Failure: quality drops weeks later with no attributeable change.

**Trap.** People often say pull the latest Hub prompt in production so you always get improvements. That misses pin-and-upgrade discipline.

**Chapter.** [28](../theory/28-hub.md)

### Does Hub replace git for prompts?

**Answer.** No. Files keep prompts and code in sync with PR review and offline use. Hub wins when non-engineers iterate in the UI. Many teams use a file registry. Failure: missing write-capable `LANGSMITH_API_KEY` and push fails.

**Trap.** People often say Hub replaces git for prompts. That misses code-synced file workflows.

**Chapter.** [28](../theory/28-hub.md)

### Is "sounds better" enough to ship a prompt change?

**Answer.** No. A/B with fact recall and refusal accuracy. Explicit refusal instructions often matter more than tone. Isolate the prompt variable. Failure: change model and prompt together and call the A/B conclusive.

**Trap.** People often say if the new prompt sounds better, ship it. That misses measured refusals and facts.

**Chapter.** [28](../theory/28-hub.md)

### Why put prompt version in trace metadata?

**Answer.** Without version you cannot attribute regressions to a prompt change. Pair pinned Hub hashes or file versions with logged metadata. Behaviour that changed with no deploy is the unpinned-pull failure mode. Failure: leave prompt version out of traces.

**Trap.** People often say prompt version does not belong in traces. That misses regression attribution.

**Chapter.** [28](../theory/28-hub.md)
