# Errors, Retries, and Tool Failures

## 30-second answer

Classify failures first: **transient** (retry with backoff), **permanent** (fail fast), **degraded** (route around). Attach `RetryPolicy` on nodes with `retry_on` and jitter. Prefer `ToolException` text the model can act on — `ToolNode` turns it into a `ToolMessage` — over raising and killing the loop. Bound slow calls with timeouts. Use a circuit breaker under sustained outage. Flag degradation in state so answers can say so.

## Tiny example

Support graph fetches CRM profile. CRM returns 503 → `RetryPolicy` backs off. Still failing → `error_handler` sets `degraded=True` and a limited profile. Compose appends "CRM unavailable; answer may be incomplete." Docs fetch still succeeded in parallel.

## Why it exists

Providers rate-limit, tools time out, APIs return 500s. Treating every failure as "retry three times" turns a 2-second blip into a 40-second bill. Production needs a different answer per class of failure.

## Runtime

```mermaid
flowchart TD
  call[Node_or_tool] -->|TransientError| retry[RetryPolicy]
  retry -->|success| ok[Continue]
  retry -->|exhausted| handler[error_handler]
  call -->|PermanentError| fail[Fail_fast]
  call -->|DegradedError| breaker[Circuit_open]
  breaker --> fallback[Fallback_state]
  toolExc[ToolException] --> toolMsg[ToolMessage]
  toolMsg --> agent[Model_recovers]
```

*Picture:* retry only transient; permanent fails fast; tools speak to the model via ToolMessage.

| Example | Class | Response |
|---|---|---|
| 429 / 503 / connection reset | Transient | Retry with backoff |
| 400 / 401 / 404 | Permanent | Fail fast |
| Circuit open | Degraded | Fallback |

**RetryPolicy:** `max_attempts`, `initial_interval`, `backoff_factor`, `jitter=True`, `retry_on=` class or predicate.

**Tool failures:** raise `ToolException` with actionable text. Bad: `KeyError: 'ticket_id'`. Good: `Invalid ticket id '999'. IDs start with 'T-'.`

**Circuit breaker:** closed → open after threshold → half_open probe.

Also in source: model `with_fallbacks` / `with_retry`, `CachePolicy`, checkpointer resume, `recursion_limit`, `ModelCallLimitMiddleware`.

## Objects, fields, and merge rules

| Mechanism | Key fields |
|---|---|
| `RetryPolicy` | max_attempts, backoff, jitter, retry_on |
| `error_handler=` | Fallback node → usable state |
| `ToolException` | → ToolMessage content |
| `degraded` | Append list so answers can caveat |
| `CachePolicy` + `compile(cache=...)` | Skip re-execution |

## Control surface

- Classify before choosing retry vs fail vs fallback.
- Write tool errors as instructions for the model.
- Always flag degradation when answering without a dependency.
- Bound every external call.

## Failure anatomy

| Failure | Fix |
|---|---|
| Retrying 400 four times | Narrow `retry_on` |
| Tool raise kills agent loop | `ToolException` → ToolMessage |
| Timeout stacks under outage | Circuit breaker |
| Silent incomplete answer | Set `degraded` flag |

## Keywords

- **transient / permanent / degraded** — classify first
- **`RetryPolicy`** — node-level backoff
- **`ToolException`** — model-recoverable tool errors
- **circuit breaker** — fail fast under sustained outage
- **`error_handler`** — usable fallback state

## Minimal fragment

```python
from langgraph.types import RetryPolicy

builder.add_node(
    "fetch_crm",
    fetch_crm,
    retry_policy=RetryPolicy(
        max_attempts=4,
        initial_interval=0.5,
        backoff_factor=2.0,
        jitter=True,
        retry_on=TransientError,
    ),
    error_handler=profile_fallback,  # sets degraded=True
)
# tools: raise ToolException("Invalid ticket id '999'. IDs start with 'T-'.")
```

## Interview traps

**Shallow:** "Always retry three times."

**Correction:** Retry transient only. Permanent errors waste latency and money.

**Shallow:** "Let tools raise raw exceptions inside agents."

**Correction:** `ToolException` becomes a ToolMessage the model can act on.

## Lab

[44_error_retries_tool_failures.ipynb](../../07-langgraph-advanced/44_error_retries_tool_failures.ipynb)
