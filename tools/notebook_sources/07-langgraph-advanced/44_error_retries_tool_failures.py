# %% [markdown]
# # 44 - Errors, Retries and Tool Failures
#
# | | |
# |---|---|
# | **Level** | Advanced (LangGraph) |
# | **Time** | 50 minutes |
# | **Prerequisites** | `43_hybrid_deterministic_llm` |
# | **Checklist ID** | `44_error_retries_tool_failures` |
#
# ## Why this matters
#
# Everything in an LLM system fails: providers rate-limit you, tools time out,
# APIs return 500s, models emit invalid JSON, and networks drop. A demo ignores
# all of this. A production system has an answer for each.
#
# The important distinction is between failures worth **retrying** (transient),
# failures worth **routing around** (a dead dependency), and failures worth
# **stopping for** (bad input). Treating them all the same is how you turn a
# 2-second blip into a 40-second timeout and a $12 bill.

# %%
from pathlib import Path
import sys

_root = Path.cwd()
while not (_root / "shared").is_dir() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from shared.notebook_setup import setup, require  # noqa: E402

ctx = setup("44_error_retries_tool_failures")

# %%
import operator
import random
import time
from typing import Annotated, Literal, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from shared.llm import get_chat_model

model = get_chat_model()
parser = StrOutputParser()

# %% [markdown]
# ## 1. Classify the failure first

# %%
class TransientError(Exception):
    """Might succeed on a retry: timeout, 429, 503, connection reset."""


class PermanentError(Exception):
    """Will never succeed: bad input, 404, unauthorised, malformed request."""


class DegradedError(Exception):
    """The dependency is down; route around it rather than retrying forever."""


CLASSIFICATION = {
    "429 Too Many Requests": (TransientError, "retry with backoff"),
    "503 Service Unavailable": (TransientError, "retry with backoff"),
    "Connection timed out": (TransientError, "retry with backoff"),
    "400 Bad Request": (PermanentError, "fail fast, fix the input"),
    "401 Unauthorized": (PermanentError, "fail fast, fix credentials"),
    "404 Not Found": (PermanentError, "fail fast, or treat as an empty result"),
    "Circuit open": (DegradedError, "route to a fallback"),
}

print(f"{'failure':28} {'class':17} response")
for failure, (kind, response) in CLASSIFICATION.items():
    print(f"{failure:28} {kind.__name__:17} {response}")

# %% [markdown]
# **Retrying a permanent error is pure waste**: three attempts, three times the
# latency, same failure. Getting this classification right is most of the work.

# %% [markdown]
# ## 2. `RetryPolicy` on a node

# %%
from langgraph.types import RetryPolicy


class FetchState(TypedDict):
    source: str
    data: str
    attempts: Annotated[int, operator.add]


attempt_log: list[str] = []


def flaky_fetch(state: FetchState) -> dict:
    attempt_log.append(state["source"])
    count = attempt_log.count(state["source"])
    if count < 3:
        raise TransientError(f"attempt {count}: connection reset")
    return {"data": f"payload from {state['source']}", "attempts": count}


builder = StateGraph(FetchState)
builder.add_node(
    "fetch",
    flaky_fetch,
    retry_policy=RetryPolicy(
        max_attempts=5,
        initial_interval=0.05,
        backoff_factor=2.0,
        max_interval=1.0,
        jitter=True,
        retry_on=TransientError,       # <- only this class is retried
    ),
)
builder.add_edge(START, "fetch")
builder.add_edge("fetch", END)

start = time.perf_counter()
outcome = builder.compile().invoke({"source": "crm", "data": "", "attempts": 0})
print(f"succeeded after {outcome['attempts']} attempts in {time.perf_counter() - start:.2f}s")
print(f"result: {outcome['data']}")

# %% [markdown]
# | Parameter | Meaning |
# |---|---|
# | `max_attempts` | Total tries, including the first |
# | `initial_interval` | Delay before the second attempt |
# | `backoff_factor` | Multiplier per attempt (2.0 = exponential) |
# | `max_interval` | Ceiling on the delay |
# | `jitter` | Randomise delays - **essential** to avoid thundering herds |
# | `retry_on` | Exception class, tuple, or a predicate |
#
# ### `retry_on` as a predicate
#
# Most real APIs signal retryability in a status code, not an exception type.

# %%
class HTTPError(Exception):
    def __init__(self, status: int):
        self.status = status
        super().__init__(f"HTTP {status}")


def is_retryable(exc: Exception) -> bool:
    return isinstance(exc, HTTPError) and exc.status in {408, 429, 500, 502, 503, 504}


calls = {"n": 0}


def http_node(state: FetchState) -> dict:
    calls["n"] += 1
    if calls["n"] == 1:
        raise HTTPError(503)        # retryable
    if calls["n"] == 2:
        raise HTTPError(429)        # retryable
    return {"data": "ok", "attempts": calls["n"]}


builder2 = StateGraph(FetchState)
builder2.add_node("http", http_node,
                  retry_policy=RetryPolicy(max_attempts=4, initial_interval=0.02, retry_on=is_retryable))
builder2.add_edge(START, "http")
builder2.add_edge("http", END)
print(builder2.compile().invoke({"source": "api", "data": "", "attempts": 0}))

# %%
calls["n"] = 0


def bad_request_node(state: FetchState) -> dict:
    calls["n"] += 1
    raise HTTPError(400)            # NOT retryable


builder3 = StateGraph(FetchState)
builder3.add_node("http", bad_request_node,
                  retry_policy=RetryPolicy(max_attempts=4, initial_interval=0.02, retry_on=is_retryable))
builder3.add_edge(START, "http")
builder3.add_edge("http", END)
try:
    builder3.compile().invoke({"source": "api", "data": "", "attempts": 0})
except HTTPError as exc:
    print(f"{exc} raised after {calls['n']} call(s) - correctly not retried")

# %% [markdown]
# ## 3. `error_handler` - recover instead of raising
#
# When a node fails past its retries, `error_handler` lets a replacement node
# produce a usable state instead of crashing the run.

# %%
class EnrichState(TypedDict):
    customer: str
    profile: str
    degraded: bool
    log: Annotated[list[str], operator.add]


def fetch_profile(state: EnrichState) -> dict:
    raise DegradedError("the CRM is unreachable")


def profile_fallback(state: EnrichState) -> dict:
    return {"profile": "(profile unavailable - proceeding with limited context)",
            "degraded": True, "log": ["fell back to a degraded profile"]}


builder4 = StateGraph(EnrichState)
builder4.add_node("profile", fetch_profile, error_handler=profile_fallback)
builder4.add_node("respond", lambda s: {"log": [f"answered with degraded={s['degraded']}"]})
builder4.add_edge(START, "profile")
builder4.add_edge("profile", "respond")
builder4.add_edge("respond", END)

print(builder4.compile().invoke({"customer": "Globex", "profile": "", "degraded": False, "log": []}))

# %% [markdown]
# The run completed with a clearly flagged degradation rather than failing.
# **Always set the flag** - an answer produced without the customer's profile is
# not the same answer, and downstream code (and humans) need to know.

# %% [markdown]
# ## 4. Tool failures
#
# A tool that raises inside an agent kills the loop. A tool that returns an error
# *message* lets the model react - try a different tool, ask for clarification,
# or tell the user honestly.

# %%
from langchain_core.tools import ToolException, tool


@tool
def get_ticket(ticket_id: str) -> str:
    """Look up a support ticket by ID. IDs look like T-123."""
    if not ticket_id.startswith("T-"):
        raise ToolException(f"Invalid ticket id {ticket_id!r}. IDs start with 'T-', for example T-101.")
    if ticket_id == "T-999":
        raise ToolException("Ticket T-999 does not exist. Use search_tickets to find the right id.")
    return f"{ticket_id}: open, team=billing, subject='duplicate charge'"


@tool
def search_tickets(query: str) -> str:
    """Search tickets by keyword when you do not know the exact id."""
    return "Matches: T-101 (duplicate charge), T-088 (refund request)"


print("raising directly:")
try:
    get_ticket.invoke({"ticket_id": "999"})
except ToolException as exc:
    print(f"  ToolException: {exc}")

# %% [markdown]
# ### `ToolNode` converts exceptions into messages
#
# By default `ToolNode` catches the exception and returns its text as a
# `ToolMessage`, so the model sees the error and can recover.

# %%
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.prebuilt import ToolNode, tools_condition

tools = [get_ticket, search_tickets]
tool_node = ToolNode(tools)

failing_call = AIMessage(
    content="",
    tool_calls=[{"name": "get_ticket", "id": "tc1", "args": {"ticket_id": "999"}}],
)
result = tool_node.invoke({"messages": [HumanMessage("look up 999"), failing_call]})
print("ToolMessage the model receives:")
print(" ", result["messages"][-1].content)

# %% [markdown]
# **Error messages are prompts.** Write them for the model: say what was wrong,
# and say what to do instead. Compare:
#
# | Bad | Good |
# |---|---|
# | `KeyError: 'ticket_id'` | `Missing required argument 'ticket_id'. Provide an id like T-101.` |
# | `Invalid input` | `Invalid ticket id '999'. IDs start with 'T-'. Use search_tickets if unsure.` |
# | `Request failed` | `The ticket service timed out. Retry once, then tell the user it is unavailable.` |

# %%
model_with_tools = model.bind_tools(tools)


def agent(state: MessagesState) -> dict:
    return {"messages": [model_with_tools.invoke(state["messages"])]}


builder5 = StateGraph(MessagesState)
builder5.add_node("agent", agent)
builder5.add_node("tools", tool_node)
builder5.add_edge(START, "agent")
builder5.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
builder5.add_edge("tools", "agent")
recovering_agent = builder5.compile()

outcome = recovering_agent.invoke({"messages": [HumanMessage("What is the status of ticket 999?")]})
for message in outcome["messages"]:
    kind = message.__class__.__name__.replace("Message", "")
    calls = getattr(message, "tool_calls", None)
    suffix = f"  -> {[c['name'] for c in calls]}" if calls else ""
    print(f"{kind:9} {(message.content or '')[:100]}{suffix}")

# %% [markdown]
# The agent hit the error, read the guidance, and recovered by itself. That only
# works because the error message told it what to do.

# %% [markdown]
# ## 5. Tool timeouts
#
# A tool with no timeout can hang a request forever.

# %%
import concurrent.futures


def with_timeout(seconds: float):
    """Decorator that turns a slow call into a readable ToolException."""
    def wrap(fn):
        def wrapped(*args, **kwargs):
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(fn, *args, **kwargs)
                try:
                    return future.result(timeout=seconds)
                except concurrent.futures.TimeoutError:
                    raise ToolException(
                        f"{fn.__name__} timed out after {seconds}s. "
                        "Tell the user this data source is slow right now; do not retry."
                    )
        wrapped.__name__ = fn.__name__
        wrapped.__doc__ = fn.__doc__
        return wrapped
    return wrap


@tool
@with_timeout(0.3)
def slow_report(region: str) -> str:
    """Generate a regional report. Can be slow."""
    time.sleep(2.0)
    return f"report for {region}"


try:
    slow_report.invoke({"region": "ap-south-1"})
except ToolException as exc:
    print(f"ToolException: {exc}")

# %% [markdown]
# LangGraph also supports a node-level `timeout=` on `add_node`, which is the
# right tool when the whole step should be bounded rather than one call.

# %% [markdown]
# ## 6. Circuit breakers
#
# Retrying a dependency that has been down for ten minutes wastes time on every
# request. A circuit breaker gives up quickly after repeated failures and
# periodically checks whether it has recovered.

# %%
from dataclasses import dataclass, field


@dataclass
class CircuitBreaker:
    """closed = normal, open = failing fast, half_open = probing recovery."""

    failure_threshold: int = 3
    recovery_seconds: float = 0.5
    failures: int = 0
    opened_at: float = 0.0
    state: Literal["closed", "open", "half_open"] = "closed"

    def before_call(self) -> None:
        if self.state == "open":
            if time.perf_counter() - self.opened_at >= self.recovery_seconds:
                self.state = "half_open"
            else:
                raise DegradedError("circuit open - not attempting the call")

    def record_success(self) -> None:
        self.failures = 0
        self.state = "closed"

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.state = "open"
            self.opened_at = time.perf_counter()

    def call(self, fn, *args, **kwargs):
        self.before_call()
        try:
            result = fn(*args, **kwargs)
        except DegradedError:
            raise
        except Exception:
            self.record_failure()
            raise
        self.record_success()
        return result


breaker = CircuitBreaker(failure_threshold=3, recovery_seconds=0.4)
service_up = {"value": False}


def unreliable_service() -> str:
    if not service_up["value"]:
        raise TransientError("service down")
    return "service response"


print(f"{'call':>5} {'state':10} outcome")
for i in range(1, 9):
    if i == 6:
        service_up["value"] = True
        time.sleep(0.45)
    try:
        result = breaker.call(unreliable_service)
        print(f"{i:>5} {breaker.state:10} {result}")
    except Exception as exc:
        print(f"{i:>5} {breaker.state:10} {type(exc).__name__}: {exc}")

# %% [markdown]
# Calls 4 and 5 failed **instantly** instead of waiting for a timeout. That is
# the point: under a sustained outage your latency stays flat instead of
# collapsing.

# %% [markdown]
# ## 7. Graceful degradation in a graph
#
# Combine everything: retries for transient failures, a fallback path for
# degraded dependencies, and an honest answer either way.

# %%
class SupportState(TypedDict):
    question: str
    account: str
    docs: str
    answer: str
    degraded: Annotated[list[str], operator.add]
    log: Annotated[list[str], operator.add]


crm_breaker = CircuitBreaker(failure_threshold=2, recovery_seconds=10)
crm_healthy = {"value": False}


def fetch_account(state: SupportState) -> dict:
    def call():
        if not crm_healthy["value"]:
            raise TransientError("CRM timeout")
        return "Globex: enterprise, 240 seats, 2 open disputes"

    return {"account": crm_breaker.call(call), "log": ["crm ok"]}


def account_fallback(state: SupportState) -> dict:
    return {"account": "", "degraded": ["account context unavailable"], "log": ["crm fallback"]}


def fetch_docs(state: SupportState) -> dict:
    return {"docs": "Duplicate charges are refunded within 3 business days of confirmation.",
            "log": ["docs ok"]}


def compose(state: SupportState) -> dict:
    caveat = ("\nNote: we could not load full account context, so this answer is general."
              if state["degraded"] else "")
    answer = (ChatPromptTemplate.from_template(
        "Account: {account}\nPolicy: {docs}\n\nAnswer in 2 sentences: {question}"
    ) | model | parser).invoke({"account": state["account"] or "(unavailable)",
                                "docs": state["docs"], "question": state["question"]})
    return {"answer": answer.strip() + caveat, "log": ["composed"]}


builder6 = StateGraph(SupportState)
builder6.add_node("account", fetch_account,
                  retry_policy=RetryPolicy(max_attempts=2, initial_interval=0.02, retry_on=TransientError),
                  error_handler=account_fallback)
builder6.add_node("docs", fetch_docs)
builder6.add_node("compose", compose)
builder6.add_edge(START, "account")
builder6.add_edge(START, "docs")
builder6.add_edge("account", "compose")
builder6.add_edge("docs", "compose")
builder6.add_edge("compose", END)
resilient = builder6.compile()

# %%
for label, healthy in [("CRM down", False), ("CRM healthy", True)]:
    crm_healthy["value"] = healthy
    crm_breaker.state, crm_breaker.failures = "closed", 0
    outcome = resilient.invoke({"question": "We were charged twice - when will the refund arrive?",
                                "account": "", "docs": "", "answer": "", "degraded": [], "log": []})
    print(f"\n--- {label} --- degraded={outcome['degraded']} log={outcome['log']}")
    print(f"  {outcome['answer'][:220]}")

# %% [markdown]
# Same graph, both outcomes useful, and the degraded case says so.

# %% [markdown]
# ## 8. Model-level resilience
#
# Providers fail too. Fall back to another one.

# %%
from shared.llm import available_providers, get_chat_model

providers = available_providers()
print("configured providers:", providers or "none")

if len(providers) > 1:
    primary = get_chat_model(provider=providers[0])
    backup = get_chat_model(provider=providers[1])
    resilient_model = primary.with_fallbacks([backup])
    print(f"fallback chain: {providers[0]} -> {providers[1]}")
else:
    resilient_model = model.with_retry(stop_after_attempt=3)
    print("single provider: using with_retry instead of a cross-provider fallback")

print(resilient_model.invoke("Reply with the single word: ready").content.strip())

# %% [markdown]
# For agents, `ModelFallbackMiddleware` does the same thing declaratively:
#
# ```python
# from langchain.agents.middleware import ModelFallbackMiddleware
#
# agent = create_agent(primary, tools, middleware=[ModelFallbackMiddleware(backup, third_choice)])
# ```

# %% [markdown]
# ## 9. Caching as failure mitigation
#
# A cached result is also a result you cannot fail to produce. Node-level caching
# skips re-execution entirely when the input is unchanged.

# %%
from langgraph.cache.memory import InMemoryCache
from langgraph.types import CachePolicy

executions = []


class CacheState(TypedDict):
    query: str
    result: str


def expensive(state: CacheState) -> dict:
    executions.append(state["query"])
    time.sleep(0.2)
    return {"result": f"computed for {state['query']}"}


builder7 = StateGraph(CacheState)
builder7.add_node("expensive", expensive, cache_policy=CachePolicy(ttl=60))
builder7.add_edge(START, "expensive")
builder7.add_edge("expensive", END)
cached_graph = builder7.compile(cache=InMemoryCache())

for i in range(3):
    start = time.perf_counter()
    cached_graph.invoke({"query": "monthly report", "result": ""})
    print(f"  run {i + 1}: {time.perf_counter() - start:.3f}s")
print(f"node executed {len(executions)} time(s) for 3 runs")

# %% [markdown]
# `langgraph.cache.sqlite.SqliteCache(path=...)` gives the same thing across
# process restarts.

# %% [markdown]
# ## 10. Recovering from a crash
#
# With a checkpointer, an unhandled failure is recoverable: completed steps are
# saved, so a rerun resumes rather than restarting.

# %%
class PipelineState(TypedDict):
    stage: Annotated[list[str], operator.add]
    payload: str


should_fail = {"value": True}


def step_one(state: PipelineState) -> dict:
    return {"stage": ["one"], "payload": "gathered"}


def step_two(state: PipelineState) -> dict:
    if should_fail["value"]:
        raise RuntimeError("downstream service exploded")
    return {"stage": ["two"], "payload": state["payload"] + " -> processed"}


def step_three(state: PipelineState) -> dict:
    return {"stage": ["three"], "payload": state["payload"] + " -> delivered"}


builder8 = StateGraph(PipelineState)
builder8.add_sequence([("one", step_one), ("two", step_two), ("three", step_three)])
builder8.add_edge(START, "one")
builder8.add_edge("three", END)
pipeline = builder8.compile(checkpointer=InMemorySaver())

cfg = {"configurable": {"thread_id": "crash-1"}}
try:
    pipeline.invoke({"stage": [], "payload": ""}, cfg)
except RuntimeError as exc:
    print(f"crashed: {exc}")

snapshot = pipeline.get_state(cfg)
print(f"saved state: stage={snapshot.values['stage']} next={snapshot.next}")

should_fail["value"] = False
print("after the fix, resuming:", pipeline.invoke(None, cfg))

# %% [markdown]
# `step_one` did not re-run. For a pipeline whose first stage is an expensive
# retrieval or a large model call, that is the difference between a cheap retry
# and an expensive one.

# %% [markdown]
# ## 11. A checklist
#
# | Layer | Mechanism |
# |---|---|
# | Transient node failure | `RetryPolicy` with `retry_on` and `jitter` |
# | Permanent node failure | Do **not** retry; fail fast or route to a human |
# | Unrecoverable node | `error_handler` returning a degraded state |
# | Slow call | `with_timeout` on the tool, or `timeout=` on the node |
# | Repeated dependency failure | Circuit breaker |
# | Tool error | `ToolException` with **actionable** text for the model |
# | Provider failure | `with_fallbacks` or `ModelFallbackMiddleware` |
# | Repeat work | `cache_policy` + `compile(cache=...)` |
# | Crash mid-run | Checkpointer; resume with `invoke(None, config)` |
# | Runaway loop | `recursion_limit`, hop caps, `ModelCallLimitMiddleware` |
# | Always | Flag degradation in state so the answer can say so |

# %% [markdown]
# ## Try it yourself
#
# 1. **Retry budget.** Track total retry time in state and stop retrying once a
#    per-request budget is exhausted.
# 2. **Half-open probing.** Extend the circuit breaker so `half_open` allows only
#    one probe at a time and reopens immediately on failure.
# 3. **Rewrite five error messages.** Take five unhelpful tool errors and rewrite
#    them as instructions, then measure whether the agent recovers more often.
# 4. **Chaos test.** Make each of three parallel sources fail 30% of the time and
#    confirm the graph still answers, correctly flagged, on every run.

# %% [markdown]
# ## Recap
#
# | Concept | Takeaway |
# |---|---|
# | Classify first | Transient / permanent / degraded - different responses |
# | `RetryPolicy` | `max_attempts`, `backoff_factor`, `jitter`, `retry_on` |
# | `retry_on` predicate | Match on status codes, not just exception types |
# | `error_handler=` | Recover into a degraded state instead of crashing |
# | `ToolException` | Becomes a `ToolMessage` the model can act on |
# | Error messages are prompts | Say what was wrong **and** what to do instead |
# | Timeouts | Bound every external call; unbounded calls hang requests |
# | Circuit breaker | Fail fast under sustained outage; keeps latency flat |
# | `with_fallbacks` | Cross-provider resilience for the model itself |
# | `CachePolicy` + `cache=` | Skip re-execution; also a failure mitigation |
# | Checkpointed recovery | Resume after a crash without repeating completed steps |
# | Flag degradation | An answer built on missing data must say so |
#
# ## Next
#
# -> [45_langsmith_and_studio.ipynb](45_langsmith_and_studio.ipynb)
