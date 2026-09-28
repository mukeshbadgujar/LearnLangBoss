# LangSmith and production interview answers

Say the short answer. If they push, use the trap correction. Each item links to the mechanism chapter. This bank covers observability, evaluation, streaming, cost, gateways, Studio, deploy, and the FastAPI isolation checks that keep production safe. Broader service design stays in [system-design.md](system-design.md).

## Contents

- [19 — LangSmith](#19--langsmith)
- [20 — Callbacks](#20--callbacks)
- [20a — Guardrails](#20a--guardrails)
- [21 — Streaming](#21--streaming)
- [26 — Evaluation](#26--evaluation)
- [26a — Unit testing](#26a--unit-testing)
- [27 — Usage tracking](#27--usage-tracking)
- [27a — Gateways](#27a--gateways)
- [45 — LangSmith Studio](#45--langsmith-studio)
- [46 — Deployment](#46--deployment)
- [53 — Thread isolation](#53--thread-isolation)

## 19 — LangSmith

### What does LangSmith store?

**Answer.** It keeps a tree of what happened in one run so you can replay it later. Each step is a span: the prompt that was actually sent, model in and out, tool in and out, token counts, latency, errors, plus tags and metadata you attached. It is not your app database and it is not the checkpointer. If you look for chat history there, you will not find a durable thread.

**Trap.** People often say "it logs print statements." That misses that prints are not spans — undecorated Python never shows up.

**Chapter.** [19](../theory/19-langsmith.md)

### How do you debug a wrong answer?

**Answer.** Open that user's run. First open the prompt that was actually sent. Then check why the model stopped (`finish_reason`), then what chunks or tool arguments went in, then which node or path ran, then tokens and latency, then how many loop rounds fired. Most "the model is dumb" tickets are a bad prompt or a missing chunk.

**Trap.** People often say "change the temperature and retry." That misses looking at the input the model actually saw.

**Chapter.** [19](../theory/19-langsmith.md)

### How do you attach a user to a trace?

**Answer.** Put identity on the runnable config so support can find one run fast. Use `metadata` for `user_id`, `tenant`, `prompt_version`; tags for environment; `thread_id` under `configurable` for the conversation. Then a ticket is a filter, not an afternoon of scrolling. Without those fields you search text and still miss the run.

**Trap.** People often say "we have one project and we search the text." That misses that you will not find the run.

**Chapter.** [19](../theory/19-langsmith.md)

### What turns tracing on?

**Answer.** You turn it on with process env, not a special invoke wrapper. Set `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`, and usually `LANGSMITH_PROJECT` — prefer the `LANGSMITH_*` names. Once that switch is on, LangChain and LangGraph runnables in-process emit runs. Failures still look blank if you skip `run_name` / tags / metadata, or if plain helpers never get `@traceable`.

**Trap.** People often say "turn on the API key and you're done." That misses that without names and tags the trees are unfilterable, and without `@traceable` plain Python never appears.

**Chapter.** [19](../theory/19-langsmith.md)

### What is a run versus a span?

**Answer.** A run is the whole tree you open in the UI — one top-level call plus everything nested under it. A span is one timed step inside that tree: prompt, LLM, tool, retriever, parser, or a `@traceable` function. If you treat them as the same word, you will talk past the interviewer about what the UI actually shows.

**Trap.** People often say "LangSmith replaces LangChain." That misses that it observes and evaluates; it does not replace models, stores, or orchestration.

**Chapter.** [19](../theory/19-langsmith.md)

### Why set `run_name`, tags, and metadata?

**Answer.** So you can find and slice runs later. Unnamed trees show up as generic `RunnableSequence` or `LangGraph` labels. `run_name` is the human verb label; tags are filter chips like `env:prod`; metadata holds searchable keys like `user_id`, `tenant`, `prompt_version`. Without them a customer report does not map to their run.

**Trap.** People often say "names are cosmetic." That misses that without them you cannot slice by version, tenant, or environment.

**Chapter.** [19](../theory/19-langsmith.md)

### When do you need `@traceable`?

**Answer.** LangChain pieces already trace themselves; your own business logic does not. Ranking, filters, and custom lookups stay invisible unless you decorate them — for example `@traceable(run_type="retriever", name="policy_lookup")` (or `tool` / `chain`). `run_type` only changes how the UI draws the span. Undecorated helpers never appear under the chain invoke.

**Trap.** People often say "everything under the chain invoke is already traced." That misses that undecorated helpers stay invisible.

**Chapter.** [19](../theory/19-langsmith.md)

### How do you attach user feedback to the right call?

**Answer.** Bind feedback to one concrete run id, not to a vague dashboard number. Wrap the exact invoke in `collect_runs()`, take `runs.traced_runs[0].id`, then call `client.create_feedback(run_id, key=..., score=..., comment=...)`. If you pick the wrong nested id, you annotate the wrong tree.

**Trap.** People often say "feedback is a dashboard metric." That misses that a wrong nested id annotates the wrong tree.

**Chapter.** [19](../theory/19-langsmith.md)

## 20 — Callbacks

### Why keep custom callback handlers if you have LangSmith?

**Answer.** LangSmith is one callback path; your product may still need others. Use handlers for audit sinks, WebSocket fans, or cost trackers that LangSmith does not own. Push side effects through callbacks; pull typed UI events with `astream_events`. Dropping handlers because "we have LangSmith" leaves those sinks with nowhere to plug in.

**Trap.** People often say "callbacks are legacy; use LangSmith only." That misses that custom sinks still need handlers.

**Chapter.** [20](../theory/20-callbacks.md)

### Where do you attach a token-count handler?

**Answer.** Attach it per request so one user's tracker cannot see another user's tokens. Pass `config={"callbacks": [tracker]}` on the call — not once on the shared model constructor. Constructor scope shares mutable state across users and becomes a billing bug and a data leak.

**Trap.** People often say "attach the cost tracker once on the model." That misses that it mixes tenants.

**Chapter.** [20](../theory/20-callbacks.md)

### Why did `on_llm_new_token` never fire?

**Answer.** That hook only fires when the call actually streams. An `invoke` path will not emit token events. For LCEL UIs, prefer `astream_events` with version `"v2"` so you see typed internals. If you waited on `on_llm_new_token` after `invoke`, silence is expected.

**Trap.** People often say "`on_llm_new_token` always fires." That misses that it only fires on a streaming call.

**Chapter.** [20](../theory/20-callbacks.md)

### Should a handler raise when logging fails?

**Answer.** No — a raise from a handler aborts the whole run. Swallow metrics and audit errors; only raise for a deliberate circuit breaker. Prefer `AsyncCallbackHandler` for I/O so sync DB writes do not stall the event loop. Key tool audit rows by `run_id`, and implement `on_tool_error` as well as `on_tool_end`. Treating logging as a gate kills healthy requests.

**Trap.** People often say "handlers can validate and fail the request freely." That misses that logging is not a gate.

**Chapter.** [20](../theory/20-callbacks.md)

### Where do token counts live in a handler?

**Answer.** Read them at the end of the LLM call, not from a guess. In `on_llm_end`, take `response.generations[0][0].message.usage_metadata` (guard `AttributeError` / `IndexError`), then sum `input_tokens` and `output_tokens` across nested calls on one per-request tracker. A cost report of zero usually means you missed `usage_metadata` or attached the handler wrong.

**Trap.** People often say "cost report shows zero so usage is free." That misses that you likely never read `usage_metadata` or attached the handler wrong.

**Chapter.** [20](../theory/20-callbacks.md)

## 20a — Guardrails

### Where do guardrails sit?

**Answer.** They sit as middleware around the agent, cheap checks first. Filter injection and redact PII before the model, run the model and tools, then check output again (PII, topic fence). Human approval is for irreversible tools, not for every token. A disclaimer in the system prompt is not a control.

**Trap.** People often say "we put a disclaimer in the system prompt." That misses that a disclaimer is not a control.

**Chapter.** [20a](../theory/20a-guardrails.md)

### What does `PIIMiddleware` do?

**Answer.** It applies built-in strategies on known PII types: `redact`, `mask`, `block`, or hash. Turn it on with `apply_to_input=True` and/or `apply_to_output=True`. Input redaction keeps raw email out of provider logs and model context; output catches echoes. Redacting only on the way out is too late — the provider already saw it.

**Trap.** People often say "redact on output only." That misses that the raw email already reached the provider.

**Chapter.** [20a](../theory/20a-guardrails.md)

### When is `before_agent` better than `after_agent`?

**Answer.** Use `before_agent` when you want to stop early and spend no tokens. It can short-circuit with `jump_to` and return a final `AIMessage` for injection phrases. Use `after_agent` to validate or replace the final answer after the run. Keep both when you need early refuse and post-answer fences. One fat middleware that always calls an LLM judge wastes money on cheap keyword blocks.

**Trap.** People often say "one fat middleware does everything." That misses that cheap keyword blocks should run before expensive LLM judges.

**Chapter.** [20a](../theory/20a-guardrails.md)

### Do guardrails replace human-in-the-loop?

**Answer.** No. Guardrails own automatic refuse and redact. HITL still owns irreversible tool calls. Cap attack loops with `ModelCallLimitMiddleware(run_limit=..., exit_behavior="end")`. For PII, `block` refuses the request; `redact` / `mask` let the workflow continue without raw PII — pick by product and compliance, not habit.

**Trap.** People often say "`block` is always best for PII." That misses that `redact` and `mask` let the workflow continue without raw PII when that is what compliance allows.

**Chapter.** [20a](../theory/20a-guardrails.md)

## 21 — Streaming

### Streaming: why did the client get nothing until the end?

**Answer.** The server never emitted tokens as they were produced. Something called `invoke` instead of `stream` / `astream`, or a proxy buffered the SSE response. A blocking post-step that needs the whole input collapses the stream to one late chunk. Graph handlers must `ainvoke` or `astream` so one user does not block the event loop. Emit a terminal event so the client can tell finished from died.

**Trap.** People often say "streaming is a frontend setting." That misses that the server has to emit tokens.

**Chapter.** [21](../theory/21-streaming.md)

### What is the difference between `stream`, `astream`, and `astream_events`?

**Answer.** `.stream` and `.astream` yield pieces of the final answer (`AIMessageChunk` pieces you can add with `+`). `astream_events(version="v2")` exposes typed internal events like `on_retriever_end` and `on_chat_model_stream` for sources-then-answer UIs. They are not the same API. Using one where you meant the other leaves the UI without sources or without tokens.

**Trap.** People often say "`stream` and `astream_events` are the same." That misses final output versus internals.

**Chapter.** [21](../theory/21-streaming.md)

### What kills streaming in production?

**Answer.** The stream dies when something buffers or blocks it end to end. Common killers: `invoke` instead of stream, a step that needs full input, nginx or proxy buffering without `X-Accel-Buffering: no`, no SSE heartbeats so the proxy closes the connection, mid-stream exceptions without a partial/discard policy, and continuing generation after the client disconnects. Any LCEL chain that hides a blocking step collapses to one late chunk.

**Trap.** People often say "any LCEL chain streams." That misses that blocking steps collapse it to one chunk.

**Chapter.** [21](../theory/21-streaming.md)

### Does streaming make the model faster?

**Answer.** No — total generation time is about the same. What changes is time to first token and progressive paint. Retrieval, rewriting, and blocking pre-steps all count against first token. Optimising the wrong stage can make first paint worse even if the LLM is fine.

**Trap.** People often say "TTFT only measures the LLM." That misses that pre-steps own first paint too.

**Chapter.** [21](../theory/21-streaming.md)

### Which LangGraph `stream_mode` for a chat UI?

**Answer.** Chat tokens want `messages` — filter by `langgraph_node` so tool thoughts do not paint as answer tokens. Use `updates` for status like "calling tool X…". Avoid relying on `values` alone for token-by-token chat; that mode is whole-state snapshots, not token grain.

**Trap.** People often say "use `values` for chat UIs." That misses that those are whole-state snapshots, not token grain.

**Chapter.** [21](../theory/21-streaming.md)

## 26 — Evaluation

### How do you know a prompt change helped?

**Answer.** You score the new prompt on a frozen set of inputs and reference outputs, not on vibes. Use an evaluator (exact match or a judge) and an experiment — LangSmith `evaluate` or a local harness — and ship only if the score holds. Add production failures into that dataset (chapter 19) so the bug cannot return quietly. A/B without a frozen set cannot explain a drop later.

**Trap.** People often say "we A/B tested in production without a frozen set." That misses that you cannot explain a drop later.

**Chapter.** [26](../theory/26-evaluation.md), [19](../theory/19-langsmith.md)

### What is the RAG triad?

**Answer.** Three separate checks so you know which layer failed. Context relevance is question versus retrieved context; faithfulness is whether the answer stays grounded in that context; answer relevance is answer versus question. Low context relevance → fix retrieval. Low faithfulness → fix prompt or grounding. Low answer relevance → the model answered a different question. One blended accuracy number hides which layer broke.

**Trap.** People often say "one accuracy number is enough." That misses that combined scores hide which layer failed.

**Chapter.** [26](../theory/26-evaluation.md)

### When do you use LLM-as-judge?

**Answer.** After code can already check the easy parts. Run deterministic retrieval (`hit@k`, MRR) and generation checks (fact recall, refusal accuracy) first. Judges cover qualities code cannot check, with structured `Judgement` output and human calibration. Prefer pairwise preference with both orderings to cancel position bias. Asking "is this good?" with no rubric drifts.

**Trap.** People often say "just ask an LLM if the answer is good." That misses that uncalibrated absolute scores drift — start with code checks.

**Chapter.** [26](../theory/26-evaluation.md)

### Why track refusal accuracy?

**Answer.** It catches confident invention on questions the docs cannot answer — the failure mode production regrets most. Wider `k` can raise fact recall while lowering refusal accuracy, so measure both. Treating refusal as optional lets hallucinations pass as wins.

**Trap.** People often say "refusal accuracy is optional." That misses that it is the metric that catches confident invention.

**Chapter.** [26](../theory/26-evaluation.md)

### Why is higher `hit@k` not automatically better?

**Answer.** Wider context often raises recall while giving the model weakly related chunks to latch onto. That can hurt refusal accuracy and cost. Treat the trade as a product decision and watch refusal and ops metrics in the same harness — not as a free win.

**Trap.** People often say "higher hit@k means the system is better." That misses refusal and operational metrics in the same harness.

**Chapter.** [26](../theory/26-evaluation.md)

### How do you make pairwise A/B fair?

**Answer.** Absolute judge scores drift between runs, so do not trust them for A/B. Ask which of two answers is better, run both orderings, and only count a win when the judge agrees with itself — that cancels position bias. Calibrate the rubric on labelled examples before you trust scores.

**Trap.** People often say "absolute judge scores are stable enough for A/B." That misses position bias — prefer pairwise with dual ordering.

**Chapter.** [26](../theory/26-evaluation.md)

### What belongs in the same eval harness as quality?

**Answer.** Operational metrics sit next to quality: tokens, p50/p95 latency, call count, cost. Quality up with bill up is still a regression for many products. If eval is "accuracy only," you will ship an expensive improvement.

**Trap.** People often say "eval is accuracy only." That misses that you will ship an expensive "improvement."

**Chapter.** [26](../theory/26-evaluation.md)

## 26a — Unit testing

### Do evaluation notebooks replace unit tests?

**Answer.** No. Evaluation measures aggregate quality on a dataset. Unit tests lock exact grader, router, and graph properties on fixtures and run every PR. You need both — notebook 26 on a schedule plus package-local pytest. Aggregate scores do not tell you which grader broke.

**Trap.** People often say "evaluation notebooks replace unit tests." That misses that aggregate scores do not localise which grader broke.

**Chapter.** [26a](../theory/26a-unit-testing.md)

### Can you test LangGraph without calling an LLM?

**Answer.** Yes. Stub LLM nodes as lambdas and assert on `Command` destinations, reducer merges, and revision-budget paths. That costs zero tokens and stays deterministic. Reserve live models for a thin smoke set. Saying you cannot test without an LLM is how routing bugs escape into CI.

**Trap.** People often say "you cannot test LangGraph without calling an LLM." That misses that stubbed nodes are free and deterministic.

**Chapter.** [26a](../theory/26a-unit-testing.md)

### What should grader fixtures cover beyond the happy path?

**Answer.** Cover the paths that fail quietly in production. Assert jailbreak / refuse routes (`datasource == "refuse"`), hallucination-negative documents (grounded is False), and revision budget caps that force publish or end. Happy-path-only suites miss silent rot.

**Trap.** People often say "if the happy-path grader passes, the system is safe." That misses that refuse and grounding negatives are the asserts that matter.

**Chapter.** [26a](../theory/26a-unit-testing.md)

### Why not one end-to-end agent test?

**Answer.** End-to-end tests are slow, flaky, and do not tell you which grader or edge broke. Isolate components first; use eval harnesses for aggregate quality. One mega agent test will not name the failing fixture.

**Trap.** People often say "one end-to-end agent test is enough." That misses that it will not tell you which fixture failed.

**Chapter.** [26a](../theory/26a-unit-testing.md)

## 27 — Usage tracking

### What do you bill against?

**Answer.** Bill against what the provider reported, not a character guess. Use `usage_metadata` on each response (`input_tokens`, `output_tokens`, cache/reasoning details). Estimate size before send with `get_num_tokens` or tiktoken. Characters÷4 is a rough English heuristic only — JSON, code, and non-Latin scripts break it.

**Trap.** People often say "estimate tokens with characters divided by four." That misses that billing should use `usage_metadata`; the heuristic is only a guess.

**Chapter.** [27](../theory/27-usage-tracking.md)

### Why does agent cost explode with tool rounds?

**Answer.** Full history is resent every tool turn, so cost grows roughly with rounds squared, not linearly. Cap iterations with `BudgetGuard` or `ModelCallLimitMiddleware`, trim history, and route easy traffic to cheaper paths. Treating cost as linear in tool calls underprices the bill.

**Trap.** People often say "agent cost is linear in the number of tool calls." That misses that caps are cost controls because history is resent every round.

**Chapter.** [27](../theory/27-usage-tracking.md)

### How do you attribute cost per user or feature?

**Answer.** Tag every call so you can charge back and find the expensive path. Use invocation-scoped trackers with tenant/feature tags, or LangSmith `tags` / `metadata` (`user_id`, `tenant_id`, `feature`). Enforce ceilings with raise-from-callback `BudgetGuard` or graceful `ModelCallLimitMiddleware`. A single total spend number cannot chargeback.

**Trap.** People often say "a single total spend number is enough." That misses that without attribution nobody owns the bill.

**Chapter.** [27](../theory/27-usage-tracking.md)

### When should a usage callback raise?

**Answer.** Only for a deliberate circuit breaker — for example `BudgetGuard` after a USD or call ceiling. Callback exceptions abort the run and are not isolated, so do not raise from ordinary logging. Prefer middleware for graceful agent stops that still return partial work. Raising from every callback kills healthy requests on metric bugs.

**Trap.** People often say "raise from every callback to be safe." That misses that this kills healthy requests on metric bugs.

**Chapter.** [27](../theory/27-usage-tracking.md)

## 27a — Gateways

### What is wrong with `shared/llm.py` picking one provider at import?

**Answer.** Import-time selection dies with that provider for the whole process. Mid-flight outages take every caller down. Prefer `.with_fallbacks([...])` or a gateway so failover is per call. Pointing everything at one key and restarting on failure is not resilience.

**Trap.** People often say "point everything at one provider API key and restart if it fails." That misses that this is not resilience.

**Chapter.** [27a](../theory/27a-gateways.md)

### When do you add LiteLLM (or another gateway)?

**Answer.** Graduate into it; do not start there. Helper first, then `.with_fallbacks` after the first outage, then a gateway when a second service needs the same routing, pooling, and metering. Talk to an OpenAI-compatible proxy with `ChatOpenAI(base_url=...)`. Day-one LiteLLM is premature infra before shared rules exist.

**Trap.** People often say "always deploy LiteLLM on day one." That misses that this is premature infra before shared rules exist.

**Chapter.** [27a](../theory/27a-gateways.md)

### Are fallbacks the same as task-based routing?

**Answer.** No. Fallbacks are failure-driven: primary errors, then secondary. Task routing is cost and quality on healthy calls — classify on small models, reason or write on larger ones. You usually want both. A `ROUTING_TABLE` maps task → model id through the gateway. Treating them as one knob confuses outage handling with cost control.

**Trap.** People often say "fallbacks and task routing are the same thing." That misses that they are different knobs.

**Chapter.** [27a](../theory/27a-gateways.md)

### Why pool API keys?

**Answer.** One key hits rate limits long before you "scale users." Key pooling or gateway deployments turn 429 storms into a fleet problem. Central metering at the gateway beats twelve home-grown counters. Continuous production traffic already needs this; waiting for "scale" is late.

**Trap.** People often say "key pooling is optional until you scale users." That misses that continuous production traffic already needs it.

**Chapter.** [27a](../theory/27a-gateways.md)

## 45 — LangSmith Studio

### What is Studio for?

**Answer.** It is a local UI on `langgraph dev` that runs the compiled graph from `langgraph.json`, highlights nodes (including loops), shows state, and lets you edit a checkpoint and continue. The trace link jumps into LangSmith. That is how you test "what if the router had said billing?" without a code change. It is not only a hosted cloud IDE — `langgraph dev` is local and in-memory.

**Trap.** People often say "Studio is a hosted IDE in the cloud only." That misses that `langgraph dev` is local and in-memory.

**Chapter.** [45](../theory/45-langsmith-studio.md)

### What goes in `langgraph.json`?

**Answer.** Three things the Studio process needs to boot your graph. `dependencies` lists packages or paths; `graphs` maps a UI name to `module.py:compiled_graph`; `env` points at the env file loaded before import. Export compiled graphs usually without your own checkpointer — Studio or Platform supplies one. A bad path or an uncompiled export leaves Studio empty.

**Trap.** People often say "any Python file path is fine." That misses that it must point at a compiled graph export.

**Chapter.** [45](../theory/45-langsmith-studio.md)

### How do you edit state in Studio?

**Answer.** Pause or interrupt mid-run, inspect the checkpoint, edit fields (for example force a route), and continue. That is the counterfactual debugger — fork without a code change. Deep-link the same run into LangSmith to open the prompt that was actually sent and the nested LLM spans. The ASCII graph is static topology; the trace is the path and loop count that actually ran.

**Trap.** People often say "the ASCII graph is the same as the trace." That misses that ASCII is static topology; the trace is the dynamic path and loop count.

**Chapter.** [45](../theory/45-langsmith-studio.md)

### Should you trace 100% of production?

**Answer.** Always set identity fields (`run_name`, tags, metadata including `prompt_version`, `thread_id`). Sample in prod (for example 10%); clear callbacks when not sampled; separate projects per environment. Never put secrets in tags or metadata. Trace all of dev. Tracing every prod call grows bill and noise.

**Trap.** People often say "trace 100% of production." That misses that bill and noise both grow — sample prod.

**Chapter.** [45](../theory/45-langsmith-studio.md)

### How does Studio relate to LangSmith?

**Answer.** Studio is the visual IDE for LangGraph; LangSmith observes and evaluates. Studio deep-links into LangSmith traces for the same run. You can run graphs without LangSmith; ship with the same discipline offline via `debug_run` (`stream_mode="updates"`). Studio is not a replacement app server.

**Trap.** People often say "Studio is a separate product from LangSmith." That misses that they share the run — Studio is not a replacement app server.

**Chapter.** [45](../theory/45-langsmith-studio.md)

### What debugging order do you use on a graph trace?

**Answer.** Open the prompt that was actually sent first. Then `finish_reason`, then the node path, then tool or retriever I/O, then the token and latency waterfall, then loop iteration count. Graph traces nest node children and LLM spans; loop cost often hides in iteration count, not one fat call. If latency is bad, open the waterfall before blaming the model.

**Trap.** People often say "if latency is bad, the model is slow." That misses opening the waterfall — retrieval, rerank, or a hidden extra LLM call usually owns wait or cost.

**Chapter.** [45](../theory/45-langsmith-studio.md)

### Why always store `prompt_version` in metadata?

**Answer.** So you can slice runs and dataset scores by version after a change. Without it you cannot compare mean score per prompt version or prove a regression gate. Version living only in git does not let you filter production by commit message alone.

**Trap.** People often say "version is in git; we do not need it on traces." That misses that you cannot filter production by commit message alone.

**Chapter.** [45](../theory/45-langsmith-studio.md)

## 46 — Deployment

### What breaks on deploy?

**Answer.** Things that worked in one process fail when state and workers become real. In-memory checkpoints vanish. A renamed node breaks threads paused on the old name. A new required state key needs an explicit migration or old threads will not resume. SQLite plus several workers will lock. Trace 100 percent of a large prod project and the bill and the noise both grow — sample prod, trace all of dev. Restart alone does not continue users unless the saver is shared and the graph shape is compatible.

**Trap.** People often say "we restart and users continue." That misses that this only works if the saver is shared and the graph shape is compatible.

**Chapter.** [46](../theory/46-deployment.md)

### Why stamp `graph_version` and write `migrate_state`?

**Answer.** A paused thread resumes against today's code, so old checkpoints need a migration path. First node stamps `GRAPH_VERSION`; on mismatch, `migrate_state(values, from_version)` renames fields, `setdefault`s required keys, and stamps the current version — like a DB migration for checkpoints. Deploying a new graph is not like deploying a stateless API when HITL threads resume against new topology.

**Trap.** People often say "deploying a new graph is like deploying a stateless API." That misses that HITL threads resume against new topology.

**Chapter.** [46](../theory/46-deployment.md)

### Why does renaming a node break paused threads?

**Answer.** The checkpointer records the next node name. Threads paused on the old name cannot resume after a rename or removal. Safe changes: optional state keys, new nodes off existing paths, prompt edits. Breaking changes need dual-version drain or finish/expire in-flight threads first. Rolling back the container does not roll back conversations that already carry v2 keys.

**Trap.** People often say "rolling back the container rolls back all conversations." That misses that threads created under v2 may contain keys v1 does not understand — use optional keys or abandon those threads.

**Chapter.** [46](../theory/46-deployment.md)

### SQLite with multiple workers — safe?

**Answer.** No. SQLite needs a single writer process. Multi-worker or multi-replica writers get lock errors. Postgres plus multi-worker is the production pattern. `InMemorySaver` loses threads across processes and restarts. Running SQLite with `--workers 4` fails for that reason.

**Trap.** People often say "SQLite is fine with `--workers 4`." That misses that validation flags that combo for a reason.

**Chapter.** [46](../theory/46-deployment.md)

### Is LangServe the modern way to deploy LangGraph?

**Answer.** No for graphs with checkpoints, interrupts, and stream modes. LangServe was historical for LCEL chains (`add_routes`). Prefer FastAPI calling `ainvoke` / `astream`, or LangGraph Platform via `langgraph.json`. CLI: `langgraph dev` (local Studio), `langgraph build`, `langgraph up`.

**Trap.** People often say "LangServe is the modern way to deploy LangGraph." That misses that FastAPI or Platform is the better fit for graphs.

**Chapter.** [46](../theory/46-deployment.md)

### When do you compile the graph in a server?

**Answer.** Once at startup, not per request. Compile in lifespan (or equivalent) with an `lru_cache` checkpointer singleton. Compiling per request burns latency and can exhaust pools. Split `/health` vs `/ready`; stay async throughout; treat interrupts as normal awaiting-approval responses plus resume.

**Trap.** People often say "compile the graph inside the request handler so each request is fresh." That misses that this is an expensive common mistake.

**Chapter.** [46](../theory/46-deployment.md)

### What LangGraph-specific signals should you monitor?

**Answer.** Watch queue and checkpoint health, not only CPU. Track approval queue depth, threads paused over 24h, `GraphRecursionError` count, checkpoint table growth without retention, checkpoint write latency, tokens per request vs baseline. Standard APM alone will not tell you forty customers wait on an approval nobody is looking at.

**Trap.** People often say "CPU and HTTP error rate are enough." That misses that stale HITL threads are invisible there.

**Chapter.** [46](../theory/46-deployment.md)

## 53 — Thread isolation

Production interview questions for the FastAPI surface. Broader service design stays in [system-design.md](system-design.md).

### How do you keep one user out of another's chat?

**Answer.** Put the user id inside the checkpoint key, not only a conversation id. Build `thread_id_for(user_id, conversation_id)` → `{user_id}::{conversation_id}`, load by that key, and return 404 on mismatch so you do not confirm the other conversation exists. Auth that only checks a header after loading the thread is too late — a guessed global id still loads.

**Trap.** People often say "we check the user id in the handler but the thread id is global." That misses that a guessed id still loads.

**Chapter.** [53](../theory/53-fastapi.md)

### Why `ainvoke` / `astream` in handlers?

**Answer.** Sync `invoke` blocks the event loop, so concurrent users line up. Async graph calls keep wall clock near the slowest request, not the sum. Build the graph once in lifespan; do not compile per request. Compiling inside the handler for "freshness" is the expensive mistake.

**Trap.** People often say "compile the graph inside the request handler so each request is fresh." That misses that lifespan builds once.

**Chapter.** [53](../theory/53-fastapi.md)

### What must an SSE chat stream emit?

**Answer.** A real stream has a start, tokens, and a clear end. Emit events such as `start`, `status`, `token`, then a terminal `done` (or `error`) with citations or escalated. Clients cannot tell finished from died without a terminal event. Check `request.is_disconnected()` to stop billing; set `Cache-Control: no-cache` and `X-Accel-Buffering: no`. Silence when tokens stop is not a protocol.

**Trap.** People often say "SSE is done when tokens stop." That misses that silence is not a protocol.

**Chapter.** [53](../theory/53-fastapi.md)

### What belongs in FastAPI lifespan for a LangGraph service?

**Answer.** Build shared heavy objects once at startup. On startup: model, retriever, checkpointer, compiled graph on `app.state`, set `ready`. Yield while serving. Shutdown clears readiness. Production swaps `InMemorySaver` for `PostgresSaver`. Do not rebuild the index at import or per request. `/health` must never call the model — cost and provider flakes are not liveness.

**Trap.** People often say "health checks should ping the model." That misses that `/health` must never call the model — cost and provider flakes are not liveness.

**Chapter.** [53](../theory/53-fastapi.md)

### Why 404 instead of 403 on another user’s conversation?

**Answer.** With user-embedded thread keys, a foreign id finds empty state. Returning 403 confirms the resource exists to an attacker. 404 treats missing and unauthorised the same. Prefer that over an existence leak.

**Trap.** People often say "return 403 when another user guesses a conversation id." That misses the existence leak.

**Chapter.** [53](../theory/53-fastapi.md)
