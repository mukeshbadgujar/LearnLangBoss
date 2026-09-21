# %% [markdown]
# # 20 - Callbacks
#
# | | |
# |---|---|
# | **Level** | Intermediate |
# | **Time** | 35 minutes |
# | **Prerequisites** | `19_langsmith` |
# | **Checklist ID** | `20_callbacks` |
#
# ## Why this matters
#
# Callbacks are the hook system underneath LangChain. Every time a chain starts, a
# model emits a token, a tool runs, or something errors, an event fires. LangSmith
# tracing is itself just a callback handler.
#
# You write your own when you need to push events somewhere LangChain does not
# know about: your metrics system, an audit log, a WebSocket to the browser, or a
# per-request cost counter.

# %%
from pathlib import Path
import sys

_root = Path.cwd()
while not (_root / "shared").is_dir() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from shared.notebook_setup import setup, require  # noqa: E402

ctx = setup("20_callbacks")

# %%
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from shared.llm import get_chat_model

model = get_chat_model()

chain = (
    ChatPromptTemplate.from_template("Answer in one sentence: {question}")
    | model
    | StrOutputParser()
).with_config(run_name="one_liner")

# %% [markdown]
# ## 1. `StdOutCallbackHandler`: see the events

# %%
from langchain_core.callbacks import StdOutCallbackHandler

chain.invoke({"question": "What is a vector database?"}, config={"callbacks": [StdOutCallbackHandler()]})

# %% [markdown]
# ## 2. The event lifecycle
#
# | Event | Fires when | Useful for |
# |---|---|---|
# | `on_chain_start` / `on_chain_end` | a chain or runnable begins/ends | timing, request context |
# | `on_llm_start` | just before the provider call | logging the exact prompt |
# | `on_llm_new_token` | each streamed token | pushing to a UI |
# | `on_llm_end` | provider responded | token counts, cost |
# | `on_tool_start` / `on_tool_end` | a tool runs | audit logging |
# | `on_retriever_start` / `on_retriever_end` | retrieval | recording which docs were used |
# | `on_*_error` | anything raises | alerting |
#
# Subclass `BaseCallbackHandler` and implement only what you need.

# %%
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler


class EventLogger(BaseCallbackHandler):
    """Print a readable trace of everything that happens."""

    def __init__(self):
        self.depth = 0

    def _line(self, text: str) -> None:
        print("  " * self.depth + text)

    def on_chain_start(self, serialized: dict, inputs: dict, **kwargs: Any) -> None:
        name = kwargs.get("name") or (serialized or {}).get("name") or "chain"
        self._line(f"> chain start: {name}  inputs={list(inputs) if isinstance(inputs, dict) else type(inputs).__name__}")
        self.depth += 1

    def on_chain_end(self, outputs: Any, **kwargs: Any) -> None:
        self.depth = max(0, self.depth - 1)
        self._line("< chain end")

    def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs: Any) -> None:
        self._line(f"  llm start: {len(prompts)} prompt(s), first is {len(prompts[0])} chars")

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        usage = {}
        try:
            usage = response.generations[0][0].message.usage_metadata or {}
        except (AttributeError, IndexError):
            pass
        self._line(f"  llm end  : tokens={usage.get('total_tokens', '?')}")

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs: Any) -> None:
        self._line(f"  tool start: {(serialized or {}).get('name')}({input_str[:60]})")

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        self._line(f"  tool end  : {str(output)[:60]}")

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        self._line(f"  LLM ERROR: {type(error).__name__}: {error}")


chain.invoke({"question": "Why is chunk overlap useful?"}, config={"callbacks": [EventLogger()]})

# %% [markdown]
# ## 3. A cost-tracking handler
#
# The most common real use. Accumulate token usage per request so you can bill,
# budget or alert.

# %%
class CostTracker(BaseCallbackHandler):
    """Accumulate token usage and estimated cost across a whole request."""

    # USD per 1M tokens - adjust for your provider and model.
    PRICES = {
        "default": {"input": 0.15, "output": 0.60},
        "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 2.50, "output": 10.00},
    }

    def __init__(self):
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.model_name = "default"

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        self.calls += 1
        try:
            message = response.generations[0][0].message
        except (AttributeError, IndexError):
            return
        usage = getattr(message, "usage_metadata", None) or {}
        self.input_tokens += usage.get("input_tokens", 0)
        self.output_tokens += usage.get("output_tokens", 0)
        meta = getattr(message, "response_metadata", {}) or {}
        self.model_name = meta.get("model_name") or meta.get("model") or self.model_name

    @property
    def cost_usd(self) -> float:
        price = next(
            (p for name, p in self.PRICES.items() if name != "default" and name in self.model_name),
            self.PRICES["default"],
        )
        return (self.input_tokens * price["input"] + self.output_tokens * price["output"]) / 1_000_000

    def report(self) -> str:
        return (
            f"{self.calls} call(s) | in={self.input_tokens} out={self.output_tokens} "
            f"total={self.input_tokens + self.output_tokens} | ~${self.cost_usd:.6f} | {self.model_name}"
        )


tracker = CostTracker()
questions = ["What is RAG?", "What is a checkpointer?", "What is MMR?"]
chain.batch([{"question": q} for q in questions], config={"callbacks": [tracker]})
print(tracker.report())

# %% [markdown]
# ## 4. Constructor vs invocation scope
#
# Where you attach a handler decides what it sees.

# %%
lifetime_tracker = CostTracker()
model_with_handler = get_chat_model(callbacks=[lifetime_tracker])   # every call, forever

request_tracker = CostTracker()                                     # this call only

pinned_chain = ChatPromptTemplate.from_template("{question}") | model_with_handler | StrOutputParser()

pinned_chain.invoke({"question": "One word: what is latency?"})
pinned_chain.invoke({"question": "One word: what is throughput?"}, config={"callbacks": [request_tracker]})

print("constructor-scoped (2 calls):", lifetime_tracker.report())
print("request-scoped    (1 call) :", request_tracker.report())

# %% [markdown]
# **Use request scope for anything per-user.** A handler attached in the
# constructor is shared by every request in the process, which means one user's
# tokens land in another user's counter - and in a web server that is a data leak
# as well as a billing bug.

# %% [markdown]
# ## 5. Streaming tokens to a UI

# %%
class TokenStreamHandler(BaseCallbackHandler):
    """What a WebSocket handler looks like - here it just prints."""

    def __init__(self):
        self.tokens: list[str] = []

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.tokens.append(token)
        print(token, end="", flush=True)   # replace with websocket.send(token)

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        print(f"\n[done: {len(self.tokens)} tokens]")


streamer = TokenStreamHandler()
list(chain.stream({"question": "Why do agents need loop limits?"}, config={"callbacks": [streamer]}))

# %% [markdown]
# > `on_llm_new_token` only fires when the call actually streams. If you use
# > `invoke` rather than `stream`, you get one `on_llm_end` and no token events.
# > For LCEL, `astream_events` (notebook 21) is usually the better modern API.

# %% [markdown]
# ## 6. An audit log for tool calls
#
# In a regulated environment you must be able to prove what the system did.

# %%
import json
from datetime import datetime, timezone

from langchain.tools import tool


class AuditLog(BaseCallbackHandler):
    """Append every tool invocation to a JSONL file."""

    def __init__(self, path: Path, actor: str):
        self.path = path
        self.actor = actor
        self._pending: dict[UUID, dict] = {}

    def on_tool_start(self, serialized: dict, input_str: str, *, run_id: UUID, **kwargs: Any) -> None:
        self._pending[run_id] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "actor": self.actor,
            "tool": (serialized or {}).get("name", "unknown"),
            "input": input_str[:300],
        }

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        record = self._pending.pop(run_id, {})
        record.update({"status": "ok", "output": str(output)[:300]})
        self._write(record)

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        record = self._pending.pop(run_id, {})
        record.update({"status": "error", "error": f"{type(error).__name__}: {error}"})
        self._write(record)

    def _write(self, record: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")


@tool
def get_leave_balance(employee_id: str) -> str:
    """Look up an employee's remaining annual leave balance."""
    balances = {"E-101": 18, "E-103": 7, "E-104": 21}
    if employee_id.upper() not in balances:
        raise ValueError(f"unknown employee {employee_id}")
    return f"{employee_id.upper()} has {balances[employee_id.upper()]} days remaining."


audit_path = ctx.artifact("tool_audit.jsonl")
audit_path.write_text("", encoding="utf-8")
audit = AuditLog(audit_path, actor="mukesh@northwind.example")

from langchain.agents import create_agent

agent = create_agent(
    model=model,
    tools=[get_leave_balance],
    system_prompt="Use the tool to answer leave balance questions.",
)

agent.invoke(
    {"messages": [{"role": "user", "content": "What is the leave balance for E-104 and for E-999?"}]},
    config={"callbacks": [audit]},
)

print("audit log:")
for line in audit_path.read_text(encoding="utf-8").strip().splitlines():
    print(" ", line)

# %% [markdown]
# Both the successful call and the failure were recorded, with actor and
# timestamp. That file is what you hand to an auditor.

# %% [markdown]
# ## 7. Recording which documents were used

# %%
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from shared.llm import get_embeddings

policy = Path(ctx.data("leave_policy.txt")).read_text(encoding="utf-8")
store = FAISS.from_documents(
    [Document(c, metadata={"source": "leave_policy.txt"})
     for c in RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60).split_text(policy)],
    get_embeddings(),
)


class RetrievalRecorder(BaseCallbackHandler):
    """Capture what retrieval returned - essential for explaining an answer later."""

    def __init__(self):
        self.retrievals: list[dict] = []

    def on_retriever_end(self, documents, **kwargs: Any) -> None:
        self.retrievals.append({
            "count": len(documents),
            "sources": [d.metadata.get("source") for d in documents],
            "previews": [" ".join(d.page_content.split())[:60] for d in documents],
        })


recorder = RetrievalRecorder()
retriever = store.as_retriever(search_kwargs={"k": 2})

rag = (
    {"context": retriever, "question": lambda x: x["question"]}
    | ChatPromptTemplate.from_template("Answer from context.\n\n{context}\n\nQ: {question}")
    | model
    | StrOutputParser()
)

answer = rag.invoke({"question": "How many bereavement leave days are there?"}, config={"callbacks": [recorder]})
print(answer.strip())
print("\nretrieval record:")
for record in recorder.retrievals:
    for preview in record["previews"]:
        print("   -", preview)

# %% [markdown]
# ## 8. Async handlers
#
# In an async service, use `AsyncCallbackHandler` so your hook does not block the
# event loop. A synchronous handler that writes to a database will stall every
# concurrent request.

# %%
from langchain_core.callbacks import AsyncCallbackHandler


class AsyncMetrics(AsyncCallbackHandler):
    """Push metrics without blocking - here simulated with asyncio.sleep."""

    def __init__(self):
        self.events: list[str] = []

    async def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs: Any) -> None:
        import asyncio

        await asyncio.sleep(0)   # stand-in for `await metrics_client.increment(...)`
        self.events.append("llm_start")

    async def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        self.events.append("llm_end")


metrics = AsyncMetrics()
await chain.ainvoke({"question": "What is backpressure?"}, config={"callbacks": [metrics]})
print("async events:", metrics.events)

# %% [markdown]
# ## 9. Callbacks vs `astream_events` vs LangSmith
#
# | | Use when |
# |---|---|
# | **Callbacks** | You need to *push* events out: metrics, audit log, WebSocket, cost meter |
# | **`astream_events`** | You need to *pull* events in your own async loop - usually nicer for UI streaming |
# | **LangSmith** | You need a searchable history with a UI, comparisons and datasets |
#
# They are not alternatives. A production system typically runs LangSmith for
# observability, one custom callback for cost/audit, and `astream_events` for the
# user-facing stream.

# %% [markdown]
# ## 10. Pitfalls
#
# 1. **Never raise inside a handler.** An exception in `on_llm_end` can break the
#    run. Wrap handler bodies in `try/except`.
# 2. **Do not block.** A slow synchronous handler adds its latency to every call.
# 3. **Handler state is shared** when attached in a constructor - use
#    request-scoped handlers for per-user data.
# 4. **Do not log full prompts blindly.** They often contain PII or customer data;
#    redact first.

# %%
class SafeHandler(BaseCallbackHandler):
    """Defensive shape for a production handler."""

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        try:
            usage = response.generations[0][0].message.usage_metadata or {}
            print(f"  [safe] tokens={usage.get('total_tokens')}")
        except Exception as exc:  # never let a metrics bug break the request
            print(f"  [safe] handler failed quietly: {type(exc).__name__}")


chain.invoke({"question": "Why must callbacks not raise?"}, config={"callbacks": [SafeHandler()]})

# %% [markdown]
# ## Try it yourself
#
# 1. **Budget guard.** Extend `CostTracker` to raise a custom `BudgetExceeded`
#    exception once estimated spend passes a threshold, and catch it around a
#    batch run.
# 2. **Latency histogram.** Write a handler that records duration per `run_name`
#    using `on_chain_start`/`on_chain_end` and prints a summary table.
# 3. **Redacting logger.** Write a handler that logs prompts with emails and
#    phone numbers masked.
# 4. **Compare approaches.** Implement token streaming twice - once with
#    `on_llm_new_token`, once with `astream_events` - and decide which you would
#    ship.

# %% [markdown]
# ## Recap
#
# | Concept | Takeaway |
# |---|---|
# | `BaseCallbackHandler` | Subclass and implement only the events you need |
# | `StdOutCallbackHandler` | Quick visibility into the event sequence |
# | Constructor vs `config={"callbacks": [...]}` | Process-wide vs per-request scope |
# | `on_llm_end` | Where token usage and cost live |
# | `on_llm_new_token` | Only fires when streaming |
# | `on_tool_start/end/error` | Audit logging with `run_id` to pair them |
# | `on_retriever_end` | Record which documents produced an answer |
# | `AsyncCallbackHandler` | Required in async services; never block the loop |
# | Handlers must not raise | Wrap bodies in `try/except` |
#
# ## Next
#
# -> [20a_guardrails_pii_and_safety.ipynb](20a_guardrails_pii_and_safety.ipynb)
