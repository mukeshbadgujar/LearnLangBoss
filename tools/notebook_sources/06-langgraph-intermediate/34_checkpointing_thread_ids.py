# %% [markdown]
# # 34 - Checkpointing and Thread IDs
#
# | | |
# |---|---|
# | **Level** | Intermediate (LangGraph) |
# | **Time** | 45 minutes |
# | **Prerequisites** | `33_compile_invoke_stream` |
# | **Checklist ID** | `34_checkpointing_thread_ids` |
#
# ## Why this matters
#
# Notebook 06 covered LangChain's memory classes and said the modern answer is a
# checkpointer. This is that answer.
#
# A checkpointer saves the full graph state after **every step**, keyed by a
# `thread_id`. That one mechanism gives you conversation memory, crash recovery,
# human-in-the-loop, and time travel - they are not four features, they are four
# consequences of the same thing.

# %%
from pathlib import Path
import sys

_root = Path.cwd()
while not (_root / "shared").is_dir() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from shared.notebook_setup import setup, require  # noqa: E402

ctx = setup("34_checkpointing_thread_ids")

# %%
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from shared.llm import get_chat_model

model = get_chat_model()

# %% [markdown]
# ## 1. Without a checkpointer, nothing is remembered

# %%
def chat_node(state: MessagesState) -> dict:
    return {"messages": [model.invoke(state["messages"])]}


builder = StateGraph(MessagesState)
builder.add_node("chat", chat_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

forgetful = builder.compile()

forgetful.invoke({"messages": [HumanMessage("My name is Mukesh and I work in platform engineering.")]})
second = forgetful.invoke({"messages": [HumanMessage("What is my name?")]})
print(second["messages"][-1].content.strip()[:140])

# %% [markdown]
# Each `invoke` starts from nothing. Now add three words.

# %%
remembering = builder.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "mukesh-1"}}

remembering.invoke({"messages": [HumanMessage("My name is Mukesh and I work in platform engineering.")]}, config)
second = remembering.invoke({"messages": [HumanMessage("What is my name and what do I do?")]}, config)
print(second["messages"][-1].content.strip()[:160])

print(f"\nmessages now in thread: {len(remembering.get_state(config).values['messages'])}")

# %% [markdown]
# Notice what you did **not** write: no history list, no append logic, no memory
# object. The `add_messages` reducer on `MessagesState` plus a checkpointer is the
# whole implementation.

# %% [markdown]
# ## 2. `thread_id` is the unit of isolation
#
# One `thread_id` = one conversation. Different ids never see each other.

# %%
for name, thread in [("Mukesh", "mukesh-1"), ("Priya", "priya-1")]:
    cfg = {"configurable": {"thread_id": thread}}
    if thread == "priya-1":
        remembering.invoke({"messages": [HumanMessage("My name is Priya and I work in finance.")]}, cfg)
    state = remembering.get_state(cfg)
    print(f"{thread:10} {len(state.values['messages'])} messages")

print()
for thread in ["mukesh-1", "priya-1"]:
    cfg = {"configurable": {"thread_id": thread}}
    answer = remembering.invoke({"messages": [HumanMessage("What is my name?")]}, cfg)
    print(f"{thread:10} -> {answer['messages'][-1].content.strip()[:70]}")

# %% [markdown]
# ### Choosing a `thread_id`
#
# | Product | Sensible `thread_id` |
# |---|---|
# | Chat app | The conversation/session id |
# | Support tool | The ticket id - the thread outlives any one agent |
# | Document workflow | The document id |
# | Per-user assistant | The user id (one long-running thread) |
#
# **It must be unique per isolated conversation and stable across requests.**
# A random id per request silently disables memory - the single most common
# checkpointing bug.

# %%
import uuid

for _ in range(2):
    bad_config = {"configurable": {"thread_id": str(uuid.uuid4())}}   # new id every call
    remembering.invoke({"messages": [HumanMessage("My name is Sam.")]}, bad_config)
    reply = remembering.invoke({"messages": [HumanMessage("What is my name?")]}, bad_config)
print("With a fresh id each request, the second turn still works within the request,")
print("but nothing survives to the next one -", reply["messages"][-1].content.strip()[:60])

# %% [markdown]
# ## 3. Inspecting state
#
# `get_state()` returns a `StateSnapshot` - far more than just the values.

# %%
config = {"configurable": {"thread_id": "mukesh-1"}}
snapshot = remembering.get_state(config)

print(f"values keys   : {sorted(snapshot.values)}")
print(f"messages      : {len(snapshot.values['messages'])}")
print(f"next          : {snapshot.next}          # empty tuple = finished")
print(f"checkpoint id : {snapshot.config['configurable']['checkpoint_id']}")
print(f"created_at    : {snapshot.created_at}")
print(f"step          : {snapshot.metadata.get('step')}")
print(f"source        : {snapshot.metadata.get('source')}")
print(f"tasks         : {snapshot.tasks}")

# %% [markdown]
# | Field | Meaning |
# |---|---|
# | `values` | The state right now |
# | `next` | Nodes about to run; empty when complete |
# | `config` | Includes `checkpoint_id` - the address of this exact moment |
# | `metadata` | `step`, `source` (`input`/`loop`/`update`), writes |
# | `tasks` | Pending work, including any pending interrupts |
# | `parent_config` | The checkpoint this one followed |

# %% [markdown]
# ## 4. Editing state from outside the graph
#
# `update_state()` writes as if a node had done it - reducers apply.

# %%
class TicketState(MessagesState):
    priority: str
    assignee: str


def handle(state: TicketState) -> dict:
    reply = model.invoke(
        [("system", f"You are handling a {state['priority']} ticket. Reply in 2 sentences.")] + state["messages"]
    )
    return {"messages": [reply]}


builder2 = StateGraph(TicketState)
builder2.add_node("handle", handle)
builder2.add_edge(START, "handle")
builder2.add_edge("handle", END)
ticket_graph = builder2.compile(checkpointer=InMemorySaver())

cfg = {"configurable": {"thread_id": "ticket-77"}}
ticket_graph.invoke(
    {"messages": [HumanMessage("Reports have been failing for two days.")], "priority": "low", "assignee": ""},
    cfg,
)
print("before:", ticket_graph.get_state(cfg).values["priority"])

ticket_graph.update_state(cfg, {"priority": "critical", "assignee": "oncall"})
print("after :", ticket_graph.get_state(cfg).values["priority"], "/", ticket_graph.get_state(cfg).values["assignee"])

follow_up = ticket_graph.invoke({"messages": [HumanMessage("Any update?")]}, cfg)
print("\nnext reply now reflects the new priority:")
print(" ", follow_up["messages"][-1].content.strip()[:150])

# %% [markdown]
# ### `as_node` - pretend a specific node wrote it
#
# This matters because it determines **what runs next**. Without it, the update
# is recorded but the graph does not know where it is in the flow.

# %%
builder3 = StateGraph(TicketState)
builder3.add_node("triage", lambda s: {"priority": "low"})
builder3.add_node("handle", handle)
builder3.add_edge(START, "triage")
builder3.add_edge("triage", "handle")
builder3.add_edge("handle", END)
flow = builder3.compile(checkpointer=InMemorySaver(), interrupt_before=["handle"])

cfg = {"configurable": {"thread_id": "asnode-1"}}
flow.invoke({"messages": [HumanMessage("Exports are broken.")], "priority": "", "assignee": ""}, cfg)
print("paused before:", flow.get_state(cfg).next)

flow.update_state(cfg, {"priority": "critical"}, as_node="triage")
print("after update as_node='triage', next is still:", flow.get_state(cfg).next)

result = flow.invoke(None, cfg)
print("resumed and finished:", flow.get_state(cfg).next, "| priority used:", result["priority"])

# %% [markdown]
# ## 5. History and the shape of a thread

# %%
config = {"configurable": {"thread_id": "mukesh-1"}}
history = list(remembering.get_state_history(config))

print(f"{len(history)} checkpoints, newest first:\n")
print(f"{'step':>5} {'source':10} {'next':14} {'msgs':>5}  checkpoint_id")
for snap in history:
    print(f"{str(snap.metadata.get('step')):>5} {str(snap.metadata.get('source')):10} "
          f"{str(snap.next):14} {len(snap.values.get('messages', [])):>5}  "
          f"{snap.config['configurable']['checkpoint_id'][:8]}")

# %% [markdown]
# `source` tells you how each checkpoint came about:
#
# - `input` - a new `invoke` supplied fresh input
# - `loop` - the graph ran a superstep
# - `update` - `update_state()` was called
#
# Notebook 39 uses `checkpoint_id` to resume from any of these.

# %% [markdown]
# ## 6. Resuming from a specific checkpoint

# %%
target = history[2]
replay_config = {"configurable": {"thread_id": "mukesh-1",
                                  "checkpoint_id": target.config["configurable"]["checkpoint_id"]}}

past = remembering.get_state(replay_config)
print(f"that checkpoint had {len(past.values['messages'])} messages")
print(f"current thread has  {len(remembering.get_state(config).values['messages'])}")

# %% [markdown]
# Reading the past is safe. Writing from the past creates a **fork** - a new
# branch of history, which is what makes "what if we had answered differently"
# possible. Notebook 39 covers it properly.

# %% [markdown]
# ## 7. Checkpointing is not free
#
# Every step serialises the entire state. Large state means slow steps and a
# large database.

# %%
import time


class HeavyState(MessagesState):
    documents: list[str]


def heavy_node(state: HeavyState) -> dict:
    return {"messages": [AIMessage("done")]}


builder4 = StateGraph(HeavyState)
builder4.add_node("work", heavy_node)
builder4.add_edge(START, "work")
builder4.add_edge("work", END)
heavy = builder4.compile(checkpointer=InMemorySaver())

for size_kb in (1, 200, 2000):
    payload = ["x" * 1024] * size_kb
    cfg = {"configurable": {"thread_id": f"heavy-{size_kb}"}}
    start = time.perf_counter()
    heavy.invoke({"messages": [HumanMessage("go")], "documents": payload}, cfg)
    print(f"  state ~{size_kb:>5} KB -> {time.perf_counter() - start:.3f}s per run")

# %% [markdown]
# **Keep large payloads out of state.** Put the document in object storage or a
# vector store and keep an id or path in state:

# %%
# Don't:  {"documents": [<40 MB of text>]}
# Do:     {"document_ids": ["doc-4821", "doc-4822"], "collection": "handbook"}

class LeanState(MessagesState):
    document_ids: list[str]
    collection: str


print("lean state fields:", list(LeanState.__annotations__))

# %% [markdown]
# ## 8. `InMemorySaver` and its limits
#
# `InMemorySaver` (also exported as `MemorySaver`) is a dict. It is perfect for
# notebooks and tests, and wrong for anything else.

# %%
from langgraph.checkpoint.memory import MemorySaver

print("MemorySaver is InMemorySaver:", MemorySaver is InMemorySaver)

# %% [markdown]
# | | `InMemorySaver` | `SqliteSaver` | `PostgresSaver` |
# |---|---|---|---|
# | Survives restart | No | Yes | Yes |
# | Multi-process | No | Risky (file locks) | Yes |
# | Setup | None | A file path | A running database |
# | Use for | Notebooks, tests | Local apps, single-process tools | **Production** |
#
# Notebook 35 sets up the other two.

# %% [markdown]
# ## 9. Thread hygiene
#
# Threads accumulate forever unless you delete them. Checkpointers expose
# deletion, and you should use it - both for storage and for privacy.

# %%
saver = InMemorySaver()
housekeeping = builder.compile(checkpointer=saver)

for i in range(3):
    housekeeping.invoke({"messages": [HumanMessage(f"message {i}")]}, {"configurable": {"thread_id": f"tmp-{i}"}})

cfg = {"configurable": {"thread_id": "tmp-1"}}
print("before delete:", len(housekeeping.get_state(cfg).values.get("messages", [])), "messages")

saver.delete_thread("tmp-1")
print("after delete :", housekeeping.get_state(cfg).values, "(empty - the thread is gone)")
print("tmp-0 untouched:", len(housekeeping.get_state({'configurable': {'thread_id': 'tmp-0'}}).values["messages"]))

# %% [markdown]
# A realistic retention policy: delete threads inactive for 30 days, delete
# immediately on user request, and never checkpoint a thread that contains
# secrets you are not entitled to keep.

# %% [markdown]
# ## 10. A complete stateful assistant
#
# Conversation memory, per-thread settings, and a summary field that survives
# across turns.

# %%
class AssistantState(MessagesState):
    user_name: str
    topics_discussed: list[str]
    turn_count: int


def assistant(state: AssistantState) -> dict:
    system = "You are Northwind's internal assistant. Answer in 2 sentences."
    if state.get("user_name"):
        system += f" The user's name is {state['user_name']}."
    if state.get("topics_discussed"):
        system += f" Previously discussed: {', '.join(state['topics_discussed'][-3:])}."

    reply = model.invoke([("system", system)] + state["messages"])
    return {"messages": [reply], "turn_count": state.get("turn_count", 0) + 1}


builder5 = StateGraph(AssistantState)
builder5.add_node("assistant", assistant)
builder5.add_edge(START, "assistant")
builder5.add_edge("assistant", END)
assistant_graph = builder5.compile(checkpointer=InMemorySaver())

cfg = {"configurable": {"thread_id": "user-4821"}}
assistant_graph.update_state(cfg, {"user_name": "Mukesh", "topics_discussed": ["leave policy"], "turn_count": 0})

for question in ["What did we talk about before?", "And how many days was it?"]:
    outcome = assistant_graph.invoke({"messages": [HumanMessage(question)]}, cfg)
    print(f"Q: {question}")
    print(f"A: {outcome['messages'][-1].content.strip()[:150]}\n")

state = assistant_graph.get_state(cfg).values
print(f"turns={state['turn_count']} messages={len(state['messages'])} user={state['user_name']}")

# %% [markdown]
# ## Try it yourself
#
# 1. **Prove isolation.** Run the same question on three `thread_id`s with
#    different stored names and confirm no leakage.
# 2. **Simulate a crash.** Interrupt a graph, create a *new* compiled graph from
#    the same saver, and resume with the same `thread_id`.
# 3. **Measure the cost of state size.** Extend the timing loop to 5 MB and plot
#    per-step latency against state size.
# 4. **Write a retention job** that lists threads, finds those whose newest
#    checkpoint is older than N days, and deletes them.

# %% [markdown]
# ## Recap
#
# | Concept | Takeaway |
# |---|---|
# | Checkpointer | Saves full state after every step; enables memory, HITL, time travel |
# | `thread_id` | Unit of isolation; **must be stable across requests** |
# | Random id per request | The most common memory bug - silently disables it |
# | `get_state()` | Returns values, `next`, `checkpoint_id`, metadata, pending tasks |
# | `update_state()` | External write; reducers apply |
# | `as_node=` | Controls what runs next after an external update |
# | `get_state_history()` | Newest first; `source` is `input` / `loop` / `update` |
# | State size | Serialised every step - keep blobs out, store ids |
# | `InMemorySaver` | Notebooks and tests only; lost on restart |
# | `delete_thread()` | Retention and privacy; threads do not expire on their own |
#
# ## Next
#
# -> [35_sqlite_postgres_savers.ipynb](35_sqlite_postgres_savers.ipynb)
