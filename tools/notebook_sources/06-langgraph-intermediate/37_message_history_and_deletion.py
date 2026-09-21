# %% [markdown]
# # 37 - Message History and Deletion
#
# | | |
# |---|---|
# | **Level** | Intermediate (LangGraph) |
# | **Time** | 45 minutes |
# | **Prerequisites** | `36_human_in_the_loop` |
# | **Checklist ID** | `37_message_history_and_deletion` |
#
# ## Why this matters
#
# A checkpointed conversation grows forever. Turn 50 resends turns 1 through 49,
# which means cost grows quadratically, latency climbs, and eventually you hit
# the context window and the whole thing stops working.
#
# You have to actively manage history. The complication is that message lists
# have structure - a tool call and its result are a pair, and naive trimming
# splits them and produces a provider error that is genuinely confusing the
# first time you see it.

# %%
from pathlib import Path
import sys

_root = Path.cwd()
while not (_root / "shared").is_dir() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from shared.notebook_setup import setup, require  # noqa: E402

ctx = setup("37_message_history_and_deletion")

# %%
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from shared.llm import get_chat_model

model = get_chat_model()

# %% [markdown]
# ## 1. Watching a conversation grow

# %%
def chat_node(state: MessagesState) -> dict:
    return {"messages": [model.invoke([("system", "Answer in one short sentence.")] + state["messages"])]}


builder = StateGraph(MessagesState)
builder.add_node("chat", chat_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)
growing = builder.compile(checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": "growth"}}
QUESTIONS = [
    "What is a vector database?",
    "How is it different from a normal database?",
    "When would I not use one?",
    "What about hybrid search?",
    "Does that need a reranker?",
]

print(f"{'turn':>4} {'messages':>9} {'tokens':>8}")
for turn, question in enumerate(QUESTIONS, start=1):
    growing.invoke({"messages": [HumanMessage(question)]}, config)
    history = growing.get_state(config).values["messages"]
    tokens = model.get_num_tokens("\n".join(m.content for m in history if isinstance(m.content, str)))
    print(f"{turn:>4} {len(history):>9} {tokens:>8}")

# %% [markdown]
# Five turns, ten messages. At turn 50 you would resend all 100 - every time.
# The input token count is what grows; that is what you pay for.

# %% [markdown]
# ## 2. `RemoveMessage` - deleting from state
#
# The `add_messages` reducer treats a `RemoveMessage` with an existing id as a
# deletion instruction.

# %%
def keep_last_n(n: int):
    """Node factory: delete everything except the most recent n messages."""
    def node(state: MessagesState) -> dict:
        excess = state["messages"][:-n] if len(state["messages"]) > n else []
        return {"messages": [RemoveMessage(id=m.id) for m in excess]}
    return node


builder2 = StateGraph(MessagesState)
builder2.add_node("chat", chat_node)
builder2.add_node("trim", keep_last_n(4))
builder2.add_edge(START, "chat")
builder2.add_edge("chat", "trim")
builder2.add_edge("trim", END)
trimming = builder2.compile(checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": "trimmed"}}
print(f"{'turn':>4} {'kept':>5}")
for turn, question in enumerate(QUESTIONS, start=1):
    trimming.invoke({"messages": [HumanMessage(question)]}, config)
    print(f"{turn:>4} {len(trimming.get_state(config).values['messages']):>5}")

print("\nsurviving messages:")
for message in trimming.get_state(config).values["messages"]:
    print(f"   {message.__class__.__name__:13} {(message.content or '')[:62]}")

# %% [markdown]
# State stays bounded no matter how long the conversation runs. Note that
# deletion is permanent for *future* turns but the old checkpoints still hold
# the full history - notebook 39 can still travel back to them.

# %%
history = list(trimming.get_state_history(config))
print(f"checkpoint message counts (newest first): {[len(s.values.get('messages', [])) for s in history[:8]]}")

# %% [markdown]
# ### Clearing everything

# %%
from langgraph.graph.message import REMOVE_ALL_MESSAGES


def clear_history(state: MessagesState) -> dict:
    """Wipe the conversation - 'start a new chat' in your UI."""
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}


builder3 = StateGraph(MessagesState)
builder3.add_node("clear", clear_history)
builder3.add_edge(START, "clear")
builder3.add_edge("clear", END)

messages = [HumanMessage(f"m{i}", id=f"id{i}") for i in range(6)]
print("before:", len(messages), "-> after:", len(builder3.compile().invoke({"messages": messages})["messages"]))

# %% [markdown]
# ## 3. The tool-pair trap
#
# This is the bug this notebook exists for. Providers require that every
# `AIMessage` with `tool_calls` is immediately followed by a matching
# `ToolMessage`. Trim blindly and you break that invariant.

# %%
conversation = [
    SystemMessage("You are a support assistant.", id="s0"),
    HumanMessage("How many tickets are open for billing?", id="h1"),
    AIMessage(content="", id="a1", tool_calls=[{"name": "count_tickets", "id": "tc1", "args": {"team": "billing"}}]),
    ToolMessage(content="14", tool_call_id="tc1", id="t1"),
    AIMessage("There are 14 open billing tickets.", id="a2"),
    HumanMessage("And for platform?", id="h2"),
    AIMessage(content="", id="a3", tool_calls=[{"name": "count_tickets", "id": "tc2", "args": {"team": "platform"}}]),
    ToolMessage(content="37", tool_call_id="tc2", id="t2"),
    AIMessage("There are 37 open platform tickets.", id="a4"),
]


def describe(messages, label: str) -> None:
    print(f"{label}:")
    for m in messages:
        marker = ""
        if getattr(m, "tool_calls", None):
            marker = f"  calls={[c['id'] for c in m.tool_calls]}"
        elif isinstance(m, ToolMessage):
            marker = f"  responds_to={m.tool_call_id}"
        print(f"   {m.id:4} {m.__class__.__name__:13} {(m.content or '')[:38]:40}{marker}")


describe(conversation, "full conversation")

# %%
naive = conversation[-4:]
describe(naive, "\nnaive: last 4 messages")

orphans = [m.tool_call_id for m in naive if isinstance(m, ToolMessage)]
callers = [c["id"] for m in naive if getattr(m, "tool_calls", None) for c in m.tool_calls]
print(f"\ntool results present : {orphans}")
print(f"tool calls present   : {callers}")
print(f"orphaned results     : {set(orphans) - set(callers)}  <- most providers reject this")

# %% [markdown]
# The error you get reads something like *"messages with role 'tool' must be a
# response to a preceding message with 'tool_calls'"*, and it is easy to blame on
# the model rather than on your trimming.
#
# ### `trim_messages` understands the structure

# %%
from langchain_core.messages import trim_messages

safe = trim_messages(
    conversation,
    max_tokens=60,
    token_counter=model,
    strategy="last",          # keep the most recent
    include_system=True,      # never drop the system prompt
    start_on="human",         # a valid window starts at a human turn
    allow_partial=False,      # never split a message
)
describe(safe, "trim_messages result")

orphans = {m.tool_call_id for m in safe if isinstance(m, ToolMessage)}
callers = {c["id"] for m in safe if getattr(m, "tool_calls", None) for c in m.tool_calls}
print(f"\norphaned tool results: {orphans - callers or 'none'}")

# %% [markdown]
# | Parameter | What it does |
# |---|---|
# | `strategy="last"` | Keep the newest messages (almost always what you want) |
# | `strategy="first"` | Keep the oldest - for summarising the start of a session |
# | `include_system=True` | Pin the system prompt regardless of budget |
# | `start_on="human"` | Ensures the window begins at a valid boundary |
# | `end_on=` | Ensures it *ends* on a given type |
# | `allow_partial=False` | Never truncate a single message's content |
# | `token_counter=model` | Use the model's real tokeniser (pass `len` to count messages) |

# %% [markdown]
# ### Counting messages instead of tokens
#
# Sometimes "keep 6 messages" is the requirement. Pass `len` as the counter.

# %%
by_count = trim_messages(conversation, max_tokens=5, token_counter=len,
                         strategy="last", include_system=True, start_on="human")
print(f"kept {len(by_count)} messages: {[m.id for m in by_count]}")

# %% [markdown]
# ## 4. Trim for the model, keep the state
#
# An important distinction. There are two different things you can trim:
#
# 1. **What you send the model** - cheap, reversible, keeps full history on disk
# 2. **What lives in state** - saves storage, but the history is gone
#
# Most of the time you want the first.

# %%
def trim_before_model(state: MessagesState) -> dict:
    """State keeps everything; the model sees a window."""
    window = trim_messages(
        state["messages"],
        max_tokens=300,
        token_counter=model,
        strategy="last",
        include_system=True,
        start_on="human",
    )
    return {"messages": [model.invoke([("system", "Answer in one short sentence.")] + window)]}


builder4 = StateGraph(MessagesState)
builder4.add_node("chat", trim_before_model)
builder4.add_edge(START, "chat")
builder4.add_edge("chat", END)
windowed = builder4.compile(checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": "windowed"}}
for question in QUESTIONS:
    windowed.invoke({"messages": [HumanMessage(question)]}, config)

stored = windowed.get_state(config).values["messages"]
sent = trim_messages(stored, max_tokens=300, token_counter=model, strategy="last",
                     include_system=True, start_on="human")
print(f"stored in state : {len(stored)} messages")
print(f"sent to model   : {len(sent)} messages")

# %% [markdown]
# You keep the full transcript for audit, analytics and time travel, while
# paying only for the window. This is the default you should reach for.

# %% [markdown]
# ## 5. Summarisation - trimming without amnesia
#
# Trimming forgets. Summarising compresses. The cost is one extra model call
# whenever the threshold is crossed.

# %%
from typing import Annotated

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class SummaryState(TypedDict):
    messages: Annotated[list, add_messages]
    summary: str


KEEP_RECENT = 4
SUMMARISE_AFTER = 8


def summarising_chat(state: SummaryState) -> dict:
    system = "Answer in one short sentence."
    if state.get("summary"):
        system += f"\n\nEarlier in this conversation: {state['summary']}"
    return {"messages": [model.invoke([("system", system)] + state["messages"])]}


def maybe_summarise(state: SummaryState) -> dict:
    messages = state["messages"]
    if len(messages) <= SUMMARISE_AFTER:
        return {}

    older, recent = messages[:-KEEP_RECENT], messages[-KEEP_RECENT:]
    transcript = "\n".join(f"{m.__class__.__name__}: {m.content}" for m in older if isinstance(m.content, str))
    instruction = (
        f"Existing summary:\n{state['summary']}\n\nExtend it with this transcript:\n{transcript}"
        if state.get("summary")
        else f"Summarise this conversation in 2 sentences, keeping any facts about the user:\n{transcript}"
    )
    summary = model.invoke(instruction).content

    return {
        "summary": summary,
        "messages": [RemoveMessage(id=m.id) for m in older],
    }


builder5 = StateGraph(SummaryState)
builder5.add_node("chat", summarising_chat)
builder5.add_node("summarise", maybe_summarise)
builder5.add_edge(START, "chat")
builder5.add_edge("chat", "summarise")
builder5.add_edge("summarise", END)
summarising = builder5.compile(checkpointer=InMemorySaver())

# %%
config = {"configurable": {"thread_id": "summarised"}}
LONG = QUESTIONS + [
    "My team is called Kestrel, by the way.",
    "What would you recommend we start with?",
    "How long would that take?",
    "What is the main risk?",
]

print(f"{'turn':>4} {'messages':>9} summary")
for turn, question in enumerate(LONG, start=1):
    summarising.invoke({"messages": [HumanMessage(question)], "summary": ""}, config)
    state = summarising.get_state(config).values
    print(f"{turn:>4} {len(state['messages']):>9} {state['summary'][:62]}")

# %%
answer = summarising.invoke({"messages": [HumanMessage("What is my team called?")]}, config)
print("only 4-5 recent messages remain, but:")
print(" ", answer["messages"][-1].content.strip()[:140])

# %% [markdown]
# The team name was mentioned nine turns ago and has been deleted from state -
# the summary carried it. That is the whole value of summarisation over trimming.
#
# ### `SummarizationMiddleware` - the same thing for agents

# %%
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.tools import tool


@tool
def count_tickets(team: str) -> int:
    """Count open tickets for a team."""
    return {"billing": 14, "platform": 37}.get(team.lower(), 0)


summarising_agent = create_agent(
    model,
    [count_tickets],
    middleware=[SummarizationMiddleware(model=model, trigger=("tokens", 600), keep=("messages", 4))],
    checkpointer=InMemorySaver(),
)

cfg = {"configurable": {"thread_id": "agent-summary"}}
for question in ["How many billing tickets?", "And platform?", "Which is worse?"]:
    summarising_agent.invoke({"messages": [HumanMessage(question)]}, cfg)
print("agent messages in state:", len(summarising_agent.get_state(cfg).values["messages"]))

# %% [markdown]
# ## 6. Comparing the strategies

# %%
import time


def measure(graph, config_id: str, questions: list[str], extra: dict | None = None) -> dict:
    cfg = {"configurable": {"thread_id": config_id}}
    start = time.perf_counter()
    for question in questions:
        graph.invoke({"messages": [HumanMessage(question)], **(extra or {})}, cfg)
    state = graph.get_state(cfg).values
    text = "\n".join(m.content for m in state["messages"] if isinstance(m.content, str))
    return {
        "seconds": round(time.perf_counter() - start, 1),
        "messages": len(state["messages"]),
        "state_tokens": model.get_num_tokens(text),
    }


rows = {
    "no management": measure(growing, "cmp-none", LONG),
    "trim state": measure(trimming, "cmp-trim", LONG),
    "trim for model": measure(windowed, "cmp-window", LONG),
    "summarise": measure(summarising, "cmp-sum", LONG, {"summary": ""}),
}

print(f"{'strategy':16} {'messages':>9} {'state tokens':>13} {'seconds':>8}")
for label, stats in rows.items():
    print(f"{label:16} {stats['messages']:>9} {stats['state_tokens']:>13} {stats['seconds']:>8}")

# %% [markdown]
# | Strategy | Keeps history | Cost | Loses information |
# |---|---|---|---|
# | None | Yes | Grows without bound | No (until the context breaks) |
# | Trim state | No | Bounded | **Yes, permanently** |
# | Trim for model | Yes | Bounded per call | Model forgets, records remain |
# | Summarise | Compressed | Bounded + summary calls | Detail, but keeps key facts |
#
# **Default to "trim for model".** Add summarisation when users notice the model
# forgetting. Trim state only when storage or privacy demands it.

# %% [markdown]
# ## 7. Selective deletion
#
# Sometimes you need to remove specific content, not the oldest content -
# a user asking you to forget something, or PII that should never have been
# stored.

# %%
class RedactState(MessagesState):
    redactions: int


SENSITIVE = ["password", "api key", "ssn", "credit card"]


def redact(state: RedactState) -> dict:
    """Delete any message containing a sensitive marker, keeping a placeholder."""
    updates, removed = [], 0
    for message in state["messages"]:
        text = (message.content or "").lower() if isinstance(message.content, str) else ""
        if any(marker in text for marker in SENSITIVE):
            updates.append(RemoveMessage(id=message.id))
            removed += 1
    if removed:
        updates.append(HumanMessage(f"[{removed} message(s) redacted by policy]"))
    return {"messages": updates, "redactions": removed}


builder6 = StateGraph(RedactState)
builder6.add_node("redact", redact)
builder6.add_edge(START, "redact")
builder6.add_edge("redact", END)

dirty = [
    HumanMessage("How do I reset my password?", id="d1"),
    AIMessage("Use the reset link on the login page.", id="d2"),
    HumanMessage("My api key is sk-abc123, is that the problem?", id="d3"),
    AIMessage("Please never share keys in chat.", id="d4"),
]
cleaned = builder6.compile().invoke({"messages": dirty, "redactions": 0})
print(f"removed {cleaned['redactions']} message(s)\n")
for message in cleaned["messages"]:
    print(f"   {message.__class__.__name__:13} {(message.content or '')[:60]}")

# %% [markdown]
# Remember this only cleans **current** state. Old checkpoints still contain the
# key, so a real redaction also needs `delete_thread()` or a checkpointer-level
# purge. Design for this before you need it.

# %% [markdown]
# ## 8. Rules of thumb
#
# 1. **Never trim naively when tools are involved** - use `trim_messages`.
# 2. **Pin the system prompt** with `include_system=True`.
# 3. **Trim for the model first**; only trim state when you must.
# 4. **Summarise when users complain about forgetting**, not before - it costs a
#    call every time it fires.
# 5. **Budget in tokens, not messages** - one message can be 4 tokens or 4,000.
# 6. **Test at turn 50**, not turn 3. Every one of these bugs appears late.

# %%
def history_budget(model, system_prompt: str, context_window: int = 8192, reserve_output: int = 1000) -> int:
    """How many tokens of history can you actually afford?"""
    system_tokens = model.get_num_tokens(system_prompt)
    return context_window - reserve_output - system_tokens - 200      # 200 for tool schemas etc.


system = "You are Northwind's support assistant. Be concise, never promise refunds, cite policy."
print(f"usable history budget: {history_budget(model, system):,} tokens")
print(f"at ~80 tokens per message, that is roughly {history_budget(model, system) // 80} messages")

# %% [markdown]
# ## Try it yourself
#
# 1. **Break it deliberately.** Trim a tool-using conversation with a plain slice
#    and send it to the model. Read the provider error in full.
# 2. **Hybrid strategy.** Summarise everything older than 10 messages *and* trim
#    the summary itself once it exceeds 200 tokens.
# 3. **Measure the crossover.** Find the turn count at which summarisation
#    becomes cheaper than resending full history.
# 4. **Implement "forget that".** A node that finds the last user message
#    matching a phrase, deletes it and the reply, and confirms.

# %% [markdown]
# ## Recap
#
# | Concept | Takeaway |
# |---|---|
# | Unbounded history | Cost grows quadratically; context eventually breaks |
# | `RemoveMessage(id=...)` | Deletes from state via the `add_messages` reducer |
# | `REMOVE_ALL_MESSAGES` | Clear the conversation entirely |
# | **Tool-pair trap** | Naive slicing orphans `ToolMessage`s and providers reject it |
# | `trim_messages` | Structure-aware; `include_system`, `start_on`, `allow_partial` |
# | `token_counter=len` | Count messages instead of tokens when that is the requirement |
# | Trim for the model | **Default**: bounded cost, full history retained |
# | Summarisation | Compresses instead of forgetting; costs a call when it fires |
# | `SummarizationMiddleware` | The same thing declaratively for `create_agent` |
# | Redaction | Cleans current state only - old checkpoints still hold the data |
#
# ## Next
#
# -> [38_parallel_fanout_fanin.ipynb](38_parallel_fanout_fanin.ipynb)
