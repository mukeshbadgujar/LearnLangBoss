# LangGraph interview answers

For the interview Q&A bank (answers from the sample file, with index), open [langgraph-qa.md](langgraph-qa.md). This file is the chapter-by-chapter bank.

Say the short answer. If they push, use the correction. Each item links to the mechanism chapter. Prefer reducers, interrupts, and stream modes over vague “the graph runs.”

## Contents

- [29 — Graphs vs agents](#29--graphs-vs-agents)
- [30 — State schemas](#30--state-schemas)
- [31 — StateGraph](#31--stategraph)
- [32 — Conditional routing](#32--conditional-routing)
- [33 — Compile, invoke, stream](#33--compile-invoke-stream)
- [34 — Checkpointing](#34--checkpointing)
- [35 — SQLite and Postgres](#35--sqlite-and-postgres)
- [36 — Human-in-the-loop](#36--human-in-the-loop)
- [37 — Message history](#37--message-history)
- [38 — Fan-out and fan-in](#38--fan-out-and-fan-in)
- [39 — Time travel](#39--time-travel)
- [40 — Subgraphs](#40--subgraphs)
- [41 — Multi-agent](#41--multi-agent)
- [42 — Custom streams](#42--custom-streams)
- [43 — Hybrid](#43--hybrid)
- [43a — Workflows](#43a--workflows)
- [44 — Retries](#44--retries)
- [47 — Context](#47--context)
- [48 — Self-reflective RAG](#48--self-reflective-rag)
- [48a — Reflection](#48a--reflection)
- [49 — Memory store](#49--memory-store)
- [50 — Deep agents](#50--deep-agents)
- [50a — Backends and skills](#50a--backends-and-skills)

## 29 — Graphs vs agents

### When do you use a graph instead of a chain or `create_agent`?

**Answer.** A chain is a fixed pipe. Same steps, same order, no pause mid-run. `create_agent` is a model that calls tools until it stops. You cannot veto before something like `issue_refund` fires. A `StateGraph` is named steps over shared state. With a checkpointer you can pause, edit, resume, and rewind. Agents are graphs under the hood. The real choice is prebuilt loop versus your own control. Upgrade when you need pause, durable threads, or clear branches—not for one prompt then parse.

**Trap.** People often say graphs are always better than chains and agents. That misses when a fixed pipe or a simple tool loop is enough.

**Chapter.** [29](../theory/29-graphs-vs-agents.md)

### Are agents and graphs different products?

**Answer.** No. `create_agent` builds a graph: model node, tools, and a loop edge. Print `create_agent(...).get_graph().draw_ascii()` and you see that shape. Custom graphs give you interrupt, edit, and resume mid-run that plain agent `invoke` does not expose. Debugging an agent is debugging that diagram.

**Trap.** People often say agents and graphs are different products. That misses that the agent is a prebuilt graph.

**Chapter.** [29](../theory/29-graphs-vs-agents.md)

### Does `interrupt_before` alone make a run durable?

**Answer.** No. Durability needs a checkpointer plus a stable `thread_id`—think conversation id. Interrupt without a saver is only an in-process pause. Reviewers act from another process hours later. Without a saver there is nothing to resume. Same `thread_id` keeps tickets apart: ticket-9001 does not touch ticket-4821.

**Trap.** People often say `interrupt_before` alone makes the run durable. That misses the saver and the conversation id.

**Chapter.** [29](../theory/29-graphs-vs-agents.md)

### Do nodes mutate shared state in place?

**Answer.** No. Nodes return partial updates—changed keys only, or `None` for side effects. LangGraph merges those into channels. In-place mutation would break checkpointing and rewind. The runtime would not know what changed. Undeclared keys are dropped. Align returns with the schema or the write vanishes.

**Trap.** People often say nodes mutate a shared state object. That misses the merge contract and why checkpoints need diffs.

**Chapter.** [29](../theory/29-graphs-vs-agents.md)

## 30 — State schemas

### Why did my node's write disappear?

**Answer.** Two different bugs. If the key is not on the schema, the return is dropped. If the key has no reducer, a later write overwrites it. Two nodes writing it in the same step raise. Returning `{"notes": [mine]}` on a plain list replaces the list. `Annotated[..., operator.add]` appends. Reducer means append versus replace.

**Trap.** People often say LangGraph merges all dicts deeply. That misses that the default is overwrite per channel.

**Chapter.** [30](../theory/30-state-schemas.md)

### What is a partial update?

**Answer.** A node sees full state and returns only the keys it changed. LangGraph merges those keys with each channel's rule. You do not rebuild the whole dict. Returning `None` is fine for side-effect-only nodes. That is what makes checkpoint diffs and parallel writers workable.

**Trap.** People often say a node returns the full new state every time. That misses partial updates and channel merge.

**Chapter.** [30](../theory/30-state-schemas.md)

### Reducers vs overwrite — what is the default?

**Answer.** Default per channel is overwrite: last write wins. Attach merge with `Annotated[T, reducer]`. `operator.add` concatenates lists and strings and sums numbers. Chat needs `add_messages` for append, replace-by-id, and `RemoveMessage`. Blind `operator.add` on messages breaks edits and streaming id updates. Without a reducer, concurrent writes to the same key in one step are an error.

**Trap.** People often say fan-out always merges writes somehow. That misses that you must declare the reducer.

**Chapter.** [30](../theory/30-state-schemas.md)

### What happens to undeclared / dropped keys?

**Answer.** Return keys that are not on the schema are silently dropped on merge. A typo looks like a no-op node. Treat the schema as the only writable surface. Optional `input_schema` / `output_schema` hide internal scratch from callers without inventing extra channels at the API edge.

**Trap.** People often say wrong keys raise immediately at compile time. That misses silent drop at merge.

**Chapter.** [30](../theory/30-state-schemas.md)

### When does Pydantic validation run?

**Answer.** With a Pydantic state model, validation runs on input to each node. A bad write from the last node is never checked on the way out. `invoke` still returns a plain dict. TypedDict is hints only—it will accept `"confidence": "not a number"`. Put trust-boundary checks in Pydantic or at the API layer after the graph.

**Trap.** People often say Pydantic state means every write is validated immediately. That misses that validation is on node input, not on every outbound write.

**Chapter.** [30](../theory/30-state-schemas.md)

### Why flatten channels instead of nesting a big `data: dict`?

**Answer.** Reducers cannot merge nested fields cleanly. Flat, typed channels with explicit merge rules keep parallel writers and checkpoints honest. Unbounded append-only lists need a bound—`last_n` or summarisation. Nested blobs fight both reducers and checkpoint size.

**Trap.** People often say one nested dict channel is simpler and fine. That misses reducers and fat checkpoints.

**Chapter.** [30](../theory/30-state-schemas.md)

## 31 — StateGraph

### What is a superstep?

**Answer.** One wave of nodes that are ready to run. They read state from the start of the step. They return partial updates. Reducers merge those updates. Only then does the runtime look at edges for the next wave. A checkpoint—a save game—is written from that merged state if you have a saver.

**Trap.** People often say nodes run and state updates immediately, like assigning a dict. That misses the barrier. Parallel nodes would race. They do not see each other's writes until the merge.

**Chapter.** [31](../theory/31-stategraph.md)

### What is the node contract?

**Answer.** A node is `(state) -> partial dict | None`. Register with `add_node`. Wire with `add_edge` / `add_sequence`. Terminate at `END` from `START`. Prefer explicit names so stream labels and edges stay stable when you rename functions. `compile()` fails on dangling edges. Orphans can still compile and never run.

**Trap.** People often say a node returns the full new state. That misses the partial-update contract.

**Chapter.** [31](../theory/31-stategraph.md)

### What does `recursion_limit` count?

**Answer.** Supersteps, not individual node executions. One fan-out step with three parallel nodes is still one superstep toward the limit. Design loops with quality and budget exits. Treat `GraphRecursionError` / `recursion_limit` as a safety net, not the exit condition.

**Trap.** People often say `recursion_limit` is how many times a node can run. That misses that it counts waves, not node calls.

**Chapter.** [31](../theory/31-stategraph.md)

### How is `create_agent` rebuilt from parts?

**Answer.** `MessagesState`, a model node with `bind_tools`, `ToolNode` for tool calls, `tools_condition` to route to tools or `END`, and an edge `tools → model` for the loop. That is the prebuilt agent diagram. Missing the no-tool-call path to `END` leaves a loop that never ends.

**Trap.** People often say `create_agent` is magic and graphs are different. That misses that you can rebuild the same diagram from parts.

**Chapter.** [31](../theory/31-stategraph.md)

## 32 — Conditional routing

### How do you stop a loop?

**Answer.** Put a counter in state and route to `END` when it trips. Set `recursion_limit` so a bug cannot spin until the process dies. A conditional edge that always returns the same node is an infinite loop. The limit is the backstop, not the design.

**Trap.** People often say the framework stops when the model is done. That misses that only an agent node has that stop. Your edges run until you say `END`.

**Chapter.** [32](../theory/32-conditional-routing.md)

### What does a path function return?

**Answer.** A routing key (or list of keys), not a state merge. It is not a node. No channel writes. Side effects belong in real nodes. Cover every key in `path_map` or you get `KeyError`. Annotate with `Literal[...]` so diagrams show destinations even without a map.

**Trap.** People often say the path function is just another node. That misses that it only chooses the next step.

**Chapter.** [32](../theory/32-conditional-routing.md)

### Why classify-then-route instead of putting the LLM in the router?

**Answer.** Model fills a field in state. Plain Python returns that key to `path_map`. The decision is loggable, testable, and overridable. A model inside every path function makes production routing hard to audit. Prefer `if` for thresholds and compliance. Reserve the model for free-text judgement captured earlier.

**Trap.** People often say put the LLM in the router for maximum flexibility. That misses auditability and stable production routing.

**Chapter.** [32](../theory/32-conditional-routing.md)

### What is `Command` with `goto` and `update`?

**Answer.** A node can return `Command(update={...}, goto="next")` so one decision both writes state and chooses the next node. Use it when update and route are one decision. Keep `add_conditional_edges` when routing should stay separately testable. Destinations on `Command` keep diagrams honest.

**Trap.** People often say `Command` replaces conditional edges everywhere. That misses when you want routing tested on its own.

**Chapter.** [32](../theory/32-conditional-routing.md)

### Why dual exits on every loop?

**Answer.** Quality met and budget exhausted. Quality-only loops burn tokens until `recursion_limit` when the model never reaches the bar. Early exit to `END` for a spam guard is the same idea: stop when state already says stop. Path functions must read keys already committed before the path runs—not keys written later in the same superstep.

**Trap.** People often say loops exit when the answer is good enough. That misses the budget exit when quality never arrives.

**Chapter.** [32](../theory/32-conditional-routing.md)

### Can a path return multiple destinations?

**Answer.** Yes. Returning a list of keys fans out to those nodes in parallel. Fan-in waits for whichever set fired. That is how optional grammar, financial, and legal checks run only when needed. Shared list channels still need reducers—append versus replace.

**Trap.** People often say conditional edges always pick exactly one next node. That misses list returns for fan-out.

**Chapter.** [32](../theory/32-conditional-routing.md)

## 33 — Compile, invoke, stream

### What does `compile()` do?

**Answer.** It attaches infrastructure—checkpointer, interrupts, cache—and freezes a runnable graph. The builder stays reusable for another compile. Prefer `ainvoke` / `astream` under async servers. Sync `invoke` inside FastAPI blocks the event loop.

**Trap.** People often say compile is just a syntax check and invoke does the real wiring. That misses that compile freezes the runnable with its infrastructure.

**Chapter.** [33](../theory/33-compile-invoke-stream.md)

### `updates` vs `values` vs `messages` vs `custom`?

**Answer.** `updates` yields per-node partial updates for progress UI. `values` yields full state snapshots and includes the initial state first, so you get one more chunk than `updates`. `messages` is the token stream with metadata—always filter by `langgraph_node` or tags. `custom` carries events from `get_stream_writer()` inside nodes. Modes answer different questions. Combine with a mode list when the UI needs more than one.

**Trap.** People often say just use `stream`—there is one stream mode. That misses that modes answer different questions.

**Chapter.** [33](../theory/33-compile-invoke-stream.md)

### Why is `get_stream_writer` silent under `invoke`?

**Answer.** Outside streaming the writer is intentionally a no-op so the same node stays dual-use. Custom progress appears only when you stream with `stream_mode="custom"` (or a list including it). Do not treat silence under `invoke` as a broken API.

**Trap.** People often say `get_stream_writer` is broken because invoke returns nothing from it. That misses that the writer is a no-op outside streaming.

**Chapter.** [33](../theory/33-compile-invoke-stream.md)

### How do you filter token streams when one node has two model calls?

**Answer.** Filtering only by node name is not enough. Tag models with `with_config(tags=...)` and gate on those tags in the `messages` consumer. Unfiltered `messages` streams will show outline and draft tokens the chat UI should not display.

**Trap.** People often say filter by `langgraph_node` alone and you always know which call streamed. That misses two model calls inside one node.

**Chapter.** [33](../theory/33-compile-invoke-stream.md)

## 34 — Checkpointing

### What is `thread_id`?

**Answer.** The conversation id for a checkpoint chain—think save-game slot. Same id, same conversation state. A new id is a new conversation. The saver loads the latest checkpoint for that id and the graph continues from the next node. Without a checkpointer the config field does nothing useful.

**Trap.** People often say it is just a label for logs. That misses that it is the primary key of persisted state.

**Chapter.** [34](../theory/34-checkpointing.md)

### How do you resume after an interrupt?

**Answer.** Same `thread_id`, and `invoke(None, config)` (or `Command(resume=...)` for dynamic interrupts). Passing fresh input instead of `None` looks like a restart. Resume continues from the paused `next`, not from the beginning of the thread. Stability of the id across HTTP requests is mandatory.

**Trap.** People often say pass the original user payload again to continue. That misses `invoke(None, config)` on the same conversation id.

**Chapter.** [34](../theory/34-checkpointing.md)

### Is checkpointing only chat memory?

**Answer.** No. The same per-thread save games unlock interrupt and resume, crash recovery, and time travel. Memory is one consequence. `get_state` shows values, `next`, checkpoint id, and metadata. `get_state_history` lists snapshots with `source` of `input`, `loop`, or `update`.

**Trap.** People often say checkpointing is just chat memory. That misses pause, recovery, and rewind.

**Chapter.** [34](../theory/34-checkpointing.md)

### Why does `update_state` need `as_node`?

**Answer.** Without `as_node`, an external write may change values but leave scheduling wrong. `as_node` attributes the write as if that node produced it so `next` stays correct. After `interrupt_before=["handle"]`, updating as `triage` keeps you paused before handle with corrected fields, then `invoke(None, config)` finishes.

**Trap.** People often say `update_state` only changes values. That misses that `as_node` keeps scheduling honest.

**Chapter.** [34](../theory/34-checkpointing.md)

### Why keep large documents out of state?

**Answer.** Full state is serialised every superstep. Huge blobs make each step slow and checkpoints fat. Store documents by id and keep `document_ids` or collection names in state. Bound messages the same way.

**Trap.** People often say checkpoints are free—put the whole corpus in state. That misses serialisation cost every step.

**Chapter.** [34](../theory/34-checkpointing.md)

### Is `InMemorySaver` production-ready?

**Answer.** No. It is a process-local dict (alias `MemorySaver`). Restarts wipe threads. It is not multi-process safe. Use Sqlite or Postgres for durable apps. Threads also do not expire alone—call `delete_thread` for retention and privacy.

**Trap.** People often say `InMemorySaver` is fine for production. That misses restarts and multiple workers.

**Chapter.** [34](../theory/34-checkpointing.md)

## 35 — SQLite and Postgres

### SQLite or Postgres?

**Answer.** SQLite is one writer. Two uvicorn workers will lock or corrupt the story of who owns the file. Postgres is the saver when more than one process must read and write the same threads. In-memory savers die with the process, including on deploy.

**Trap.** People often say SQLite is fine in production if we use a volume. That misses multiple workers sharing one file.

**Chapter.** [35](../theory/35-sqlite-postgres.md)

### Does WAL make SQLite safe for multiple replicas?

**Answer.** WAL helps readers alongside one writer on one host. Multiple app replicas sharing a file still hit locking and filesystem issues. Use Postgres for multi-replica. Even with WAL, set `busy_timeout` to reduce lock failures. It does not become multi-writer.

**Trap.** People often say SQLite with WAL handles multiple app replicas. That misses that WAL is still one writer on one host.

**Chapter.** [35](../theory/35-sqlite-postgres.md)

### Why does `from_conn_string` fail after the `with` block?

**Answer.** It is a context manager. Leaving the block closes the connection. Servers should keep an explicit long-lived connection or pool on the app object. Call `setup()` once per deploy for Postgres tables. Match sync savers to `invoke` and async savers to `ainvoke`—mismatches block or raise.

**Trap.** People often say `from_conn_string` returns a long-lived saver you can stash freely. That misses that the connection closes when the `with` ends.

**Chapter.** [35](../theory/35-sqlite-postgres.md)

### Why does checkpoint disk grow forever?

**Answer.** Every superstep persists state and history accumulates. Schedule `delete_thread` for inactive threads and on user request. Keep documents out of state so each write stays small.

**Trap.** People often say the database cleans old threads automatically. That misses that you must delete threads yourself.

**Chapter.** [35](../theory/35-sqlite-postgres.md)

## 36 — Human-in-the-loop

### How does human-in-the-loop work?

**Answer.** HITL is pause for a human click. You interrupt before or after a node, or call `interrupt()` inside one. The runtime persists the checkpoint—the save game—and returns. Resume uses the same `thread_id` and supplies the human value. The graph does not block a server thread waiting. The HTTP request ends, and a later request continues.

**Trap.** People often say the node sleeps until the user clicks. That misses production resume from the checkpoint in another process.

**Chapter.** [36](../theory/36-human-in-the-loop.md)

### Static `interrupt_before` vs `interrupt(payload)`?

**Answer.** Static pauses every time at named nodes. Resume with `invoke(None, config)`. `interrupt({...})` is conditional, carries a reviewer payload—action, args, context—and resumes with `Command(resume=...)`. Detect pause via `__interrupt__` in the result or enumerate `snapshot.tasks` / `task.interrupts` for approval inboxes.

**Trap.** People often say `interrupt_before` and `interrupt()` are the same thing. That misses static every-time pause versus conditional payload interrupt.

**Chapter.** [36](../theory/36-human-in-the-loop.md)

### Why do side effects double-execute on resume?

**Answer.** On resume the node re-runs from the top. The resume value is injected when `interrupt()` is hit again. Anything before that call—emails, writes—runs twice. Put the interrupt first in a gate node and run effects after the edge from the gate. Gate on irreversible, visible, or expensive actions, not model confidence.

**Trap.** People often say on resume, execution continues from the line after `interrupt()` without re-running earlier code. That misses re-entry from the top of the node.

**Chapter.** [36](../theory/36-human-in-the-loop.md)

### What belongs in the interrupt payload?

**Answer.** Enough for a UI: kind, amount, question, tool args to edit. Empty payloads leave reviewers with nothing to show. Four patterns: approve, edit args, clarify, rewrite. `HumanInTheLoopMiddleware` declares tool approval for agents the same way.

**Trap.** People often say HITL is just logging the model's decision for a human to read later. That misses that the human must act and the graph must resume with a value.

**Chapter.** [36](../theory/36-human-in-the-loop.md)

### Can you HITL without a checkpointer?

**Answer.** Not in production. Reviewers act from another process hours later. Without a durable saver there is no pause to resume. In-memory savers also lose pauses on restart.

**Trap.** People often say you can HITL without a checkpointer if you keep the Python process alive. That misses restarts and multi-process review.

**Chapter.** [36](../theory/36-human-in-the-loop.md)

### How do you reject a tool path cleanly?

**Answer.** Resume with `"reject"` (or equivalent) and return `ToolMessage`s that close the tool calls, or abandon the thread. Approving by mistake or skipping the reject path still runs tools. Conditional `interrupt()` with a threshold avoids waking humans for tiny refunds.

**Trap.** People often say rejection is just not clicking approve so tools never run. That misses that you must resume with a reject path that closes the tool calls.

**Chapter.** [36](../theory/36-human-in-the-loop.md)

## 37 — Message history

### What breaks if you delete messages carelessly?

**Answer.** An `AIMessage` with `tool_calls` must keep its `ToolMessage` partners. Delete one side and the next model call is an invalid history. `add_messages` plus `RemoveMessage` is the supported way to drop by id. Trimming from the front without respecting pairs is how chats start erroring after a long thread.

**Trap.** People often say trim to the last N messages. That misses that N can split a tool-call pair.

**Chapter.** [37](../theory/37-message-history.md)

### Trim-for-model vs trim-state?

**Answer.** Trim-for-model bounds tokens per call while keeping the full transcript in the checkpointer for audit and time travel. Trim-state permanently drops from the live tip. Prefer `trim_messages` with structure-aware options (`start_on="human"`, `allow_partial=False`) over naive slices.

**Trap.** People often say trimming state and trimming what the model sees are the same. That misses audit and rewind needs for the full transcript.

**Chapter.** [37](../theory/37-message-history.md)

### Does `RemoveMessage` erase past checkpoints?

**Answer.** No. It updates current state through the reducer. Prior checkpoints in `get_state_history` can still hold old messages until retention deletes them. For a full wipe of the live channel use `REMOVE_ALL_MESSAGES`. PII redaction of the tip does not purge history—use `delete_thread` or checkpointer purge when required.

**Trap.** People often say `RemoveMessage` erases history everywhere including past checkpoints. That misses that old save games still hold the messages.

**Chapter.** [37](../theory/37-message-history.md)

### When is summarisation worth it?

**Answer.** When users notice forgetting after trim. It costs an extra model call each time the threshold fires. Prefer trim-for-model first. Summarise older turns into a `summary` field and inject into system text. Wire the summarise node so the threshold actually fires.

**Trap.** People often say summarisation is always cheaper than sending full history. That misses the extra model call and lossy compression.

**Chapter.** [37](../theory/37-message-history.md)

## 38 — Fan-out and fan-in

### What is `Send`?

**Answer.** A way to start N copies of a node in one superstep, each with its own input, when N is only known at runtime. Their outputs must land in a channel with a reducer—append versus replace. Without the reducer the parallel writes are an error. Fan-out is not a for-loop inside one node.

**Trap.** People often say fan-out is just a for-loop inside one node. That misses that a loop is sequential and hides partial failure. `Send` is the runtime's fan-out.

**Chapter.** [38](../theory/38-fanout-fanin.md)

### Why must `Send` pair with a reducer?

**Answer.** Parallel workers writing one list key without `Annotated[..., operator.add]` (or a custom reducer) raise `InvalidUpdateError`. The worker receives exactly the `arg` dict in `Send(node, arg)`, not the full parent state. Returns still merge into parent channels through reducers.

**Trap.** People often say `Send` passes the full parent state into each worker. That misses that each worker gets only its `arg` dict.

**Chapter.** [38](../theory/38-fanout-fanin.md)

### Can parallel nodes see each other's writes in the same superstep?

**Answer.** No. Same-superstep nodes see state from before the step. Peer writes appear only after the barrier. Put readers in the next superstep. Fan-in waits for all inbound paths, not “first edge ready.” `defer=True` postpones until nothing else is pending.

**Trap.** People often say parallel nodes can read each other's outputs in the same step. That misses the barrier merge.

**Chapter.** [38](../theory/38-fanout-fanin.md)

### What if one parallel branch fails?

**Answer.** By default the superstep fails and sibling results are lost. Isolate with in-node `try/except` when partial success is acceptable. `RetryPolicy` retries that node for transient errors before the step accepts failure. Cap pressure with `max_concurrency`—provider rate limits usually bind before CPU.

**Trap.** People often say if one parallel branch fails, the others' results are kept by default. That misses that the whole superstep fails.

**Chapter.** [38](../theory/38-fanout-fanin.md)

## 39 — Time travel

### What is time travel?

**Answer.** Checkpoints are immutable save games. You invoke again with an older checkpoint id. That forks a new branch. It does not edit the past in place. Studio's “edit state and continue” is the same idea with a UI.

**Trap.** People often say you rewind the database row. That misses that you start a new run from a copy of an old snapshot.

**Chapter.** [39](../theory/39-time-travel.md)

### Does replay use a cache?

**Answer.** No. `invoke(None, past_config)` re-executes forward from that checkpoint. Model calls and side effects run again. Outputs can change. Effectful nodes must be idempotent—guard with state keys or an outbox. Same discipline as HITL double-execution.

**Trap.** People often say time travel loads a cached result without re-running nodes. That misses full re-execution from the snapshot.

**Chapter.** [39](../theory/39-time-travel.md)

### How do you find the moment before a node runs?

**Answer.** Walk `get_state_history` and filter `snapshot.next == ("outline",)` (or the target node). That is before the node. After it runs, `next` has moved on. `update_state` on that past config forks. The original lineage remains under the same `thread_id` with a new `checkpoint_id`.

**Trap.** People often say find the moment after `outline` by reading `values['outline']` only. That misses that `next` tells you before versus after.

**Chapter.** [39](../theory/39-time-travel.md)

### Why isn't `thread_id` enough after forks?

**Answer.** After the first fork, siblings share a conversation id but differ by `checkpoint_id`. Product UIs must persist which tip is current. Walk `parent_config` to find parents with more than one child—those are fork points. Updating the latest config instead of a historical one overwrites the wrong tip.

**Trap.** People often say track only `thread_id`; the latest tip is always what you want. That misses sibling tips after forks.

**Chapter.** [39](../theory/39-time-travel.md)

## 40 — Subgraphs

### Shared state vs wrapper for subgraphs?

**Answer.** Direct `add_node("quality", quality_graph)` works when keys overlap. Shared reducers concatenate across the boundary. Different vocabularies need a wrapper that maps parent fields into the child's invoke and maps outputs back—or use the child's `input_schema` / `output_schema` as the public contract.

**Trap.** People often say always add the compiled subgraph as a node; schemas will align. That misses mismatched vocabularies that need a wrapper.

**Chapter.** [40](../theory/40-subgraphs.md)

### Who owns the checkpointer for a child graph?

**Answer.** Children inherit the parent's checkpointer. Compile the child without its own saver. A separate child `InMemorySaver` fights nested checkpoints and loses work on restart. `get_state(config, subgraphs=True)` exposes nested task state under the parent thread.

**Trap.** People often say give every subgraph its own checkpointer for isolation. That misses nested checkpointing under the parent saver.

**Chapter.** [40](../theory/40-subgraphs.md)

### How do child interrupts resume?

**Answer.** Child `interrupt()` propagates to the parent result (`__interrupt__`). Resume with `Command(resume=...)` on the parent `thread_id`. That is what makes approval-gate subgraphs composable. Parents do not need child-internal configs.

**Trap.** People often say interrupts inside a subgraph must be resumed with the child's config. That misses parent-thread resume.

**Chapter.** [40](../theory/40-subgraphs.md)

### Why `subgraphs=True` on stream?

**Answer.** By default the parent collapses the child to one update. `subgraphs=True` yields namespace tuples so you can attribute nested nodes (`quality.strip` vs parent). Use `xray=True` on diagrams the same way. Parallel double-use of a shared-state child needs wrappers with isolated invoke payloads.

**Trap.** People often say `stream` always shows every nested node update. That misses the default collapse without `subgraphs=True`.

**Chapter.** [40](../theory/40-subgraphs.md)

## 41 — Multi-agent

### Supervisor or network?

**Answer.** A supervisor is a central node that chooses the next agent. The path is auditable and you pay one extra model call per hop. A network lets specialists hand off with `Command`. It is cheaper and easier to ping-pong, so you need a hop cap and a visited set. Only the child agent's final message should return to the parent, not its whole tool trace.

**Trap.** People often say multi-agent is always more accurate. That misses that it is often 3–5x the cost and fails by looping.

**Chapter.** [41](../theory/41-multi-agent.md)

### Why hop caps and `visited`?

**Answer.** Networks ping-pong. Supervisors thrash without a limit. `MAX_HOPS` and a visited or completed set are mandatory safety, not polish. Hierarchical supervisors help when specialists exceed about six. Measure with a call counter before committing to the multiplier.

**Trap.** People often say network is free because there is no supervisor call. That misses ping-pong cost without hop caps.

**Chapter.** [41](../theory/41-multi-agent.md)

### What is the subagent return contract?

**Answer.** Only the final message (or `response_format=` / shared artifact) returns to the parent. Scraping every inner tool call violates isolation and pollutes synthesis. Handoff tools use `InjectedToolCallId`—the model never invents it—and may use `Command.PARENT` to bubble up.

**Trap.** People often say the parent can inspect every tool call inside a subagent. That misses isolation and a clean return contract.

**Chapter.** [41](../theory/41-multi-agent.md)

### When should you split into multi-agent at all?

**Answer.** When tool selection fails or the system prompt bloated—not for diagram aesthetics. Cost is typically 3–5× a single agent. Put remits in structured routing field descriptions so routers stay specific. Shared merged scratchpads beat re-deriving facts from prose.

**Trap.** People often say multi-agent always beats one agent. That misses cost and when a single agent still works.

**Chapter.** [41](../theory/41-multi-agent.md)

### How do handoffs use `Command`?

**Answer.** A specialist returns `Command(update={...}, goto=peer_or_END)` after a structured handoff decision. Set `destinations=` so diagrams stay honest. Cap hops before the structured call so the limit path can still write a closing message and go to `END`.

**Trap.** People often say handoff is just another tool call with no graph effect. That misses `Command` writing state and choosing the next peer.

**Chapter.** [41](../theory/41-multi-agent.md)

### Supervisor vs hierarchical?

**Answer.** Supervisor: one coordinator, predictable, one extra call per hop—default choice. Hierarchical: layered supervisors when you have many specialists (support vs sales under a top coordinator). Network: distributed, cheapest per hop, highest ping-pong risk.

**Trap.** People often say always start with a network because it is more “agentic.” That misses auditability and ping-pong risk.

**Chapter.** [41](../theory/41-multi-agent.md)

## 42 — Custom streams

### What is `StreamWriter` for, and how does it map to SSE?

**Answer.** `get_stream_writer()` emits to the custom channel for progress that `updates` and `messages` modes do not express. Production feeds often combine `["custom", "messages", "updates"]` and map them to SSE JSON (`type: progress|token|node`). Under nginx, disable buffering (`X-Accel-Buffering: no`). The writer is a no-op under `invoke`.

**Trap.** People often say custom streams replace `updates` and `messages`. That misses that custom fills the gaps those modes do not cover.

**Chapter.** [42](../theory/42-custom-streams.md)

### Why whitelist custom event fields?

**Answer.** Custom channels are the least protected surface—treat them like public HTTP. Emit ids, titles, scores. Never `api_key` or raw internal text. Give events a `kind` and a dataclass. Throttle per-item floods with something like `throttled_progress`.

**Trap.** People often say emit every document to the browser for transparency. That misses leaking secrets and flooding the client.

**Chapter.** [42](../theory/42-custom-streams.md)

### Do subgraph custom events reach the parent?

**Answer.** Yes. They surface on the parent custom stream. Pass `subgraphs=True` so the namespace tuple names which child emitted them. Without it you still get events but lose attribution.

**Trap.** People often say subgraph events are invisible to the parent. That misses that they surface; you only lose which child without `subgraphs=True`.

**Chapter.** [42](../theory/42-custom-streams.md)

### How do you consume combined stream modes?

**Answer.** When `stream_mode` is a list, the iterator yields `(mode, payload)`. Branch on `custom` vs `messages` vs `updates`. For tokens unpack `(token, metadata)` and gate on `langgraph_node`. For updates, `list(chunk)[0]` is the node name for SSE `type: node` events.

**Trap.** People often say combined modes still yield one flat dict per chunk. That misses the `(mode, payload)` tuple shape.

**Chapter.** [42](../theory/42-custom-streams.md)

## 43 — Hybrid

### What does "model interprets, code decides" mean?

**Answer.** The model extracts structured fields (`with_structured_output`) without arithmetic. A pure Python policy node applies caps, thresholds, and routing `if`s so the same claim yields the same total every run. The explain node must use computed figures only—recalculation reintroduces non-determinism. Line items are the audit trail.

**Trap.** People often say structured output means the model can safely compute money. That misses that code must own the arithmetic.

**Chapter.** [43](../theory/43-hybrid.md)

### Why keep escalation policy in code?

**Answer.** Labels come from the model. Four `if`s decide legal, high-value churn, and so on. Legal can read it. Unit tests call `apply_policy` without an API key. Prompt-only policy cannot be asserted exactly and drifts across runs.

**Trap.** People often say put the escalation policy in the system prompt so product can change it. That misses testability and drift.

**Chapter.** [43](../theory/43-hybrid.md)

### Why validate at the extraction boundary?

**Answer.** Hybrid moves risk there: missing nights, unknown grades, implausible amounts. Catch them before policy. Route bad extractions to `human_review`. Schema strictness alone does not prove the claim is coherent.

**Trap.** People often say validation is optional if the schema is strict. That misses coherent claims beyond field types.

**Chapter.** [43](../theory/43-hybrid.md)

### What is a cheap gate?

**Answer.** Deterministic filters—empty, oversized, injection patterns—before any LLM call. On public endpoints they typically remove a double-digit percentage of traffic. Verify-and-retry for SQL: model proposes, code checks SELECT-only and allowlisted tables, then routes back with the error while attempts remain.

**Trap.** People often say cheap gates are optional on internal tools. That misses free filtering before you spend model calls.

**Chapter.** [43](../theory/43-hybrid.md)

## 43a — Workflows

### What are the five workflow patterns?

**Answer.** Chaining: fixed edges. Parallelization: fixed `Send` list you wrote in code. Routing: one of N fixed specialists. Orchestrator-worker: model plans sections of unknown length, then `Send` workers. Evaluator-optimizer: generate–critique loop with `MAX_REVISIONS` / `recursion_limit`. Name the pattern in design reviews so the team shares vocabulary.

**Trap.** People often say all dynamic graphs are the same “agent workflow.” That misses five distinct patterns with different control surfaces.

**Chapter.** [43a](../theory/43a-workflows.md)

### Orchestrator-worker vs parallelization?

**Answer.** Parallelization fans out a list hard-coded in code. Orchestrator-worker fans out a list the model planned at runtime (`Plan.sections` → `assign_workers`). Fixed Send lists are parallelization. Model-planned Send lists are orchestrator-worker. Do not use orch-worker for three fixed rubrics.

**Trap.** People often say orchestrator-worker is just parallelization with a fancy name. That misses who decides the worker list—code versus model.

**Chapter.** [43a](../theory/43a-workflows.md)

### Why cap evaluator-optimizer?

**Answer.** Without `MAX_REVISIONS` a never-satisfied critic burns the budget. Stopping conditions are passable or cap—never until perfect. Pair with hard checks in code (word count) so passable cannot ignore a numeric budget the prompt only requested. Worker list channels still need `operator.add`.

**Trap.** People often say if the critic is strict enough, you do not need a revision cap. That misses budget burn from a never-satisfied critic.

**Chapter.** [43a](../theory/43a-workflows.md)

### Routing vs orchestrator-worker?

**Answer.** Routing picks one of N fixed specialists from a classify schema. Orchestrator-worker spawns N workers from a plan of unknown length. Evaluator-optimizer targets draft quality. Notebook 48 reflects on retrieval—different bottleneck.

**Trap.** People often say routing and orchestrator-worker both pick specialists dynamically the same way. That misses one-of-N versus spawn-N-from-plan.

**Chapter.** [43a](../theory/43a-workflows.md)

## 44 — Retries

### `RetryPolicy` vs tool error as `ToolMessage`?

**Answer.** `RetryPolicy` on a node retries transient failures with `retry_on`, backoff, and jitter before the superstep fails. Permanent errors should fail once. Uncaught tool raises kill the agent loop. `ToolException` through `ToolNode` becomes a `ToolMessage` the model can act on. Error text is a prompt—make it actionable.

**Trap.** People often say if a tool raises, the agent will see it and recover. That misses that uncaught raises kill the loop; use `ToolMessage` for recovery.

**Chapter.** [44](../theory/44-retries.md)

### Retries vs circuit breakers?

**Answer.** Retries help brief blips. Breakers stop paying timeout cost on every request after repeated dependency failures. Timeouts (`with_timeout` / node `timeout=`) bound hung calls. Provider failure uses `with_fallbacks` or middleware. Crash mid-run uses checkpointer plus `invoke(None, config)`.

**Trap.** People often say circuit breakers are the same as retries. That misses stop-paying-after-repeated-failure versus try-again-on-blips.

**Chapter.** [44](../theory/44-retries.md)

### What should `error_handler` do?

**Answer.** Recover into degraded state and flag it so the answer can caveat missing CRM context. Hiding failure produces a silently incomplete reply users trust. Always classify transient, permanent, and degraded before choosing the layer.

**Trap.** People often say error_handler means the failure is hidden. That misses flagging degraded answers for the user.

**Chapter.** [44](../theory/44-retries.md)

### Should you always retry three times?

**Answer.** No. Retry only transient classes (408/429/5xx-style predicates). Permanent 400s waste latency. Pair with hop caps / `recursion_limit` / call-limit middleware so runaway loops cannot retry forever.

**Trap.** People often say always retry three times. That misses permanent errors and runaway loops.

**Chapter.** [44](../theory/44-retries.md)

## 47 — Context

### Stuff vs map-reduce vs refine — call counts?

**Answer.** Stuff: one shot when content fits the usable budget. Map-reduce: about n+1 calls (per-chunk then reduce), weak cross-chunk reasoning unless hierarchical. Refine: about n sequential calls, preserves narrative order. Retrieve+stuff: roughly one answer call plus embedding work—usually wins for specific questions. Meter before locking a strategy.

**Trap.** People often say map-reduce is the default for anything long. That misses retrieve+stuff for specific questions and stuff when it fits.

**Chapter.** [47](../theory/47-context.md)

### What is usable context budget?

**Answer.** Window minus system prompt, tools, output reserve, and margin. Usable content can be far below the advertised 8k. Prefer stuff when it fits. For specific questions retrieve then stuff. Map-reduce for whole-corpus summaries. Refine when order matters (transcripts).

**Trap.** People often say 8k context means 8k for the document. That misses system, tools, output reserve, and margin.

**Chapter.** [47](../theory/47-context.md)

### Why `NOTHING RELEVANT` in map-reduce?

**Answer.** Irrelevant chunks should return empty summaries so the reduce prompt never sees noise. Without that filter the reduce step is noisy. If reduce itself overflows, use hierarchical map-reduce or retrieval—not another blind map.

**Trap.** People often say always map-reduce, then stuff the summaries. That misses filtering noise and overflow at reduce.

**Chapter.** [47](../theory/47-context.md)

### Token-triggered compression vs message-count?

**Answer.** Trigger on token count (`TRIGGER_TOKENS`), keep recent messages, summarise older into a rolling `summary`, and `RemoveMessage` older ids. Message-count triggers waste budget when message lengths vary. Compression is lossy—demand numbers you need in the prompt.

**Trap.** People often say compress when message count hits N. That misses variable message length and token budget.

**Chapter.** [47](../theory/47-context.md)

## 48 — Self-reflective RAG

### CRAG vs self-RAG graders?

**Answer.** Both use structured graders, but the distinctive pieces differ. Self-reflective RAG emphasises relevance, grounding, and usefulness with rewrite / regenerate / refuse paths. CRAG’s distinctive piece is corpus confidence → external supplement with an explicit caveat when `supplemented` is true. Which graders exist is how you tell the patterns apart.

**Trap.** People often say CRAG and self-RAG are the same. That misses corpus-confidence supplement versus relevance-grounding-usefulness loops.

**Chapter.** [48](../theory/48-self-reflective-rag.md)

### Is self-reflective RAG just a better prompt?

**Answer.** No. Relevance, grounding, and usefulness are separate structured nodes with routed repairs—not one mega-prompt. Cap every loop (`MAX_RETRIES`, `MAX_GENERATIONS`). Parallel grading via `Send` collapses n sequential relevance round trips into one fan-out step.

**Trap.** People often say self-reflective RAG is just a better prompt. That misses separate grader nodes, routes, and caps.

**Chapter.** [48](../theory/48-self-reflective-rag.md)

### Why are correct refusals a success metric?

**Answer.** Unanswerable questions should `give_up`, not invent policy. Tuning graders until the bot never refuses reintroduces hallucinations. Protect HALLUCINATED / correct-refusal cases in eval. Audit keep-rate: under ~20% too strict, over ~90% too lenient.

**Trap.** People often say tune until the bot never refuses. That misses that correct refusal is a success, not a failure.

**Chapter.** [48](../theory/48-self-reflective-rag.md)

### Why adaptive triage before full reflection?

**Answer.** Full reflection typically costs 4–8× plain RAG. Use a cheap retrieval score first. Reflect only when weak. Always-on graders multiply cost for little gain. Use a small grader model. Log every decision in `trace`.

**Trap.** People often say always run all graders for quality. That misses the cost multiplier when retrieval is already strong.

**Chapter.** [48](../theory/48-self-reflective-rag.md)

### What happens after grading finds no documents?

**Answer.** Route to rewrite while retries remain, else `give_up`. After generation, ungrounded answers regenerate. Useless answers rewrite if budget remains, else END. Cap generations so a never-satisfied grader cannot spend the budget.

**Trap.** People often say keep rewriting until something looks grounded. That misses the give-up path and generation caps.

**Chapter.** [48](../theory/48-self-reflective-rag.md)

### Should CRAG silently blend web results into policy answers?

**Answer.** No. Append an explicit caveat when supplemented. Otherwise a policy lookup becomes a hidden web guess. Corpus-wide fallback decisions are separate from per-document relevance grades.

**Trap.** People often say CRAG should silently blend web results into policy answers. That misses the explicit supplement caveat.

**Chapter.** [48](../theory/48-self-reflective-rag.md)

## 48a — Reflection

### Reflection vs Reflexion vs notebook 48?

**Answer.** Notebook 48 critiques retrieved docs and answer faithfulness—retrieval bottleneck. Reflection (48a) critiques the draft itself and revises copy. Reflexion adds structured critique (`missing` / `superfluous` / `search_queries`), tool research, then revise with citations. Pick by bottleneck: draft quality, factual gaps, or retrieval quality.

**Trap.** People often say reflection and notebook 48 are the same idea. That misses draft critique versus retrieval faithfulness.

**Chapter.** [48a](../theory/48a-reflection.md)

### Why feed critique as `HumanMessage`?

**Answer.** If critique arrives as `AIMessage`, the writer defends the draft. Source feeds critique as `HumanMessage` so the writer revises. Cap with message count (`len(messages) > 6 → END`) or `MAX_REVISIONS`—uncapped loops never end.

**Trap.** People often say feed the critic’s output as another AIMessage so the transcript stays consistent. That misses that the writer then defends instead of revising.

**Chapter.** [48a](../theory/48a-reflection.md)

### Why does Reflexion need structured output?

**Answer.** The schema is the control surface. `search_queries` drive research. Without it the loop wanders. `ReviseAnswer.references` keeps citations honest. Hard word budgets stop revise bloat. Reflection is evaluator-optimizer with a fixed critic and a message-count stop.

**Trap.** People often say Reflexion does not need structured output if the prompt says to search. That misses that the schema drives research and citations.

**Chapter.** [48a](../theory/48a-reflection.md)

### Do more revision passes always help?

**Answer.** No. Reflexion tends to bloat without a length check. Cap revisions and measure under the word budget. Vague “search more” critiques without `AnswerQuestion` schema produce unfocused research.

**Trap.** People often say more revision passes always improve quality. That misses bloat and unfocused research.

**Chapter.** [48a](../theory/48a-reflection.md)

## 49 — Memory store

### Store vs checkpointer?

**Answer.** Checkpointer is per-thread automatic state—messages, scratch. Think save game for one conversation. The store holds deliberate cross-thread facts with namespaces—preferences, org knowledge, procedural instructions. Use both: Monday’s thread writes memories; Thursday’s new thread recalls via the store with no shared checkpoint.

**Trap.** People often say checkpointing is long-term memory. That misses that the store is for cross-thread facts.

**Chapter.** [49](../theory/49-memory-store.md)

### InMemoryStore TTL behaviour?

**Answer.** `InMemoryStore` TTL raises `NotImplementedError`. Durable Sqlite/Postgres stores implement TTL, and expiry needs an explicit `sweep_ttl()` (or a sweeper at boot). Expired items can remain readable until swept.

**Trap.** People often say put TTL on InMemoryStore for session facts, or that expiry is lazy on read. That misses NotImplementedError and the need to sweep.

**Chapter.** [49](../theory/49-memory-store.md)

### How do you namespace for tenants?

**Answer.** Put org/tenant id from authentication at the front of the namespace tuple—never from the model. Bad order leaks cross-tenant. Extraction should default to empty lists. Saving every turn pollutes semantic search.

**Trap.** People often say let the model choose the namespace string. That misses auth-owned tenant isolation.

**Chapter.** [49](../theory/49-memory-store.md)

### How do tools access the store?

**Answer.** `get_store` inside nodes or `InjectedStore` on tools. Compile with `checkpointer=...` and `store=...`. Reconciliation chooses insert / update / skip against nearest neighbours. Hard-code safety gates so credentials never persist because the model said they were useful.

**Trap.** People often say semantic search always ranks correctly with fake embeddings in the notebook. That misses real ranking and credential safety gates.

**Chapter.** [49](../theory/49-memory-store.md)

## 50 — Deep agents

### Deep agent vs a graph you wrote — harness side?

**Answer.** A deep agent adds a plan list (`TodoListMiddleware`), a file workspace with a merging reducer, and the ability to spawn subagents, with a hard call cap (`ModelCallLimitMiddleware`). The packaged harness already includes those tools. The primitives version is what you want when you must test the middleware alone. It is a harness around the same model—not a smarter model. Most tasks stay chain, agent, or small graph. Deep agents cost 5–20× and are harder to debug.

**Trap.** People often say deep agent means a smarter model, or always use deepagents for agents. That misses harness and cost—not IQ.

**Chapter.** [50](../theory/50-deep-agents.md)

### Why only the final subagent summary returns?

**Answer.** Isolation: returning the whole tool trace pollutes parent synthesis. Cap delegation budget and depth so subagents cannot recurse until bills explode. Score with a rubric—cites figures, covers competitors. Length is not the metric.

**Trap.** People often say return the whole subagent transcript to the parent for transparency. That misses isolation and polluted synthesis.

**Chapter.** [50](../theory/50-deep-agents.md)

### How do workspace file writes work?

**Answer.** Writes that update state need `Command` with `InjectedToolCallId`, not a plain string return. `files` needs a merging reducer. Default overwrite leaves one file when two writes share a superstep. Gate HITL on irreversible tools only—gating reads teaches approve-without-reading.

**Trap.** People often say file writes can return a string like other tools. That misses `Command` plus a merging reducer for state updates.

**Chapter.** [50](../theory/50-deep-agents.md)

### What are the four deep-agent capabilities?

**Answer.** Plan as state, filesystem/workspace, subagents, and harness (procedure + middleware + limits). Prompt alone is insufficient. Pair with skills and backends in 50a when shipping the packaged API.

**Trap.** People often say deep agent means a bigger system prompt. That misses plan, workspace, subagents, and harness limits.

**Chapter.** [50](../theory/50-deep-agents.md)

## 50a — Backends and skills

### Deep agent vs a graph you wrote — backends and skills side?

**Answer.** Backends decide where files live. **StateBackend**: bytes in graph state / `result["files"]`, gone without checkpointer. **FilesystemBackend**: real disk under a sandbox root. **StoreBackend**: LangGraph store—durable cross-thread. Skills use progressive disclosure via `SKILL.md` then deeper files. `AGENTS.md` is standing workspace context. Package to ship. Primitives (50) to learn and customise.

**Trap.** People often say deep agent means a smarter model. That misses harness and storage, not IQ.

**Chapter.** [50a](../theory/50a-backends-and-skills.md)

### StateBackend vs FilesystemBackend vs StoreBackend?

**Answer.** StateBackend: files in result/state; need a checkpointer for durability across process death. FilesystemBackend: survives exit under `root_dir` sandbox only. StoreBackend: file bytes in the long-term store across threads. Checkpointer still holds per-thread graph state—use both when you need both scopes.

**Trap.** People often say StoreBackend and checkpointer are interchangeable, or that StateBackend files are durable because they appear in the result. That misses scopes—thread state versus cross-thread store versus process death.

**Chapter.** [50a](../theory/50a-backends-and-skills.md)

### Why never point FilesystemBackend at the repo root?

**Answer.** Path traversal and accidental reads/writes of secrets and source are real. Use a scratch sandbox under artifacts only. Deny `..`. Pointing at `.` is how repos get corrupted.

**Trap.** People often say FilesystemBackend should mount the project so the agent can edit code. That misses sandbox safety and path traversal.

**Chapter.** [50a](../theory/50a-backends-and-skills.md)

### What is SKILL.md progressive disclosure?

**Answer.** Short `SKILL.md` (when-to-use + core workflow) first. Deeper `instructions.md` only when needed. That keeps the base system prompt small so the model follows short instructions better than a wall of procedures. `AGENTS.md` is standing context the agent re-reads—you still set a system prompt that tells it to follow AGENTS.md.

**Trap.** People often say skills are just more system prompt text, or that AGENTS.md replaces the system prompt entirely. That misses progressive disclosure and the standing-context role of AGENTS.md.

**Chapter.** [50a](../theory/50a-backends-and-skills.md)
