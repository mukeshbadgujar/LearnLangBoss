# %% [markdown]
# # 49 - Long-Term Semantic Memory
#
# | | |
# |---|---|
# | **Level** | Advanced (LangGraph) |
# | **Time** | 55 minutes |
# | **Prerequisites** | `34_checkpointing_thread_ids`, `48_self_reflective_rag` |
# | **Checklist ID** | `49_long_term_semantic_memory` |
#
# ## Why this matters
#
# A checkpointer gives your graph memory **within a thread**. Start a new
# conversation and everything is gone - the user has to say "I'm on the Kestrel
# team, we deploy to ap-south-1" every single time.
#
# Long-term memory is a **store**: a separate, thread-independent key-value
# space, optionally with semantic search over the values. It is how an
# assistant remembers your name next week, remembers the decision your team
# made last month, and stops asking questions it already knows the answer to.
#
# | | Checkpointer | Store |
# |---|---|---|
# | Scope | One `thread_id` | Any namespace you choose (user, org, agent) |
# | Holds | The whole graph state at each step | Facts you deliberately saved |
# | Written by | The framework, automatically | Your code, deliberately |
# | Survives new conversation | No | Yes |
# | Searchable | By thread, by checkpoint | By namespace, filter, and meaning |
#
# You almost always want **both**.

# %%
from pathlib import Path
import sys

_root = Path.cwd()
while not (_root / "shared").is_dir() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from shared.notebook_setup import setup, require  # noqa: E402

ctx = setup("49_long_term_semantic_memory")

# %%
import operator
import uuid
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_store
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field

from shared.llm import get_chat_model, get_embeddings

model = get_chat_model()

# %% [markdown]
# ## 1. The store API
#
# Five methods. `put`, `get`, `search`, `delete`, `list_namespaces`.
# A namespace is a **tuple of strings** - treat it like a folder path.

# %%
store = InMemoryStore()

NAMESPACE = ("memories", "user_mukesh")

store.put(NAMESPACE, "fact-1", {"text": "Leads the Kestrel platform team", "kind": "profile"})
store.put(NAMESPACE, "fact-2", {"text": "Prefers concise answers with code examples", "kind": "preference"})
store.put(NAMESPACE, "fact-3", {"text": "Deploys to AWS ap-south-1", "kind": "profile"})

item = store.get(NAMESPACE, "fact-1")
print(f"key        {item.key}")
print(f"namespace  {item.namespace}")
print(f"value      {item.value}")
print(f"created    {item.created_at:%Y-%m-%d %H:%M}")

print(f"\nall in namespace: {len(store.search(NAMESPACE))}")
print(f"preferences only: {[i.value['text'] for i in store.search(NAMESPACE, filter={'kind': 'preference'})]}")
print(f"namespaces:       {store.list_namespaces()}")

# %% [markdown]
# ### Namespace design
#
# The namespace is your isolation boundary. Get it wrong and one tenant reads
# another's memories.

# %%
def memory_namespace(org_id: str, user_id: str, category: str = "facts") -> tuple[str, ...]:
    """Org first, so a search prefix can never leak across tenants."""
    return ("org", org_id, "user", user_id, category)


for org, user in [("northwind", "mukesh"), ("northwind", "priya"), ("acme", "mukesh")]:
    ns = memory_namespace(org, user)
    store.put(ns, "n1", {"text": f"{user} at {org} note"})

print("prefix search is the isolation boundary:")
print(f"  all of northwind:        {len(store.search(('org', 'northwind')))}")
print(f"  only mukesh@northwind:   {len(store.search(memory_namespace('northwind', 'mukesh')))}")
print(f"  only mukesh@acme:        {len(store.search(memory_namespace('acme', 'mukesh')))}")

# %% [markdown]
# **Rule:** the tenant identifier goes at the *front* of the namespace, and it
# comes from your authentication layer - never from user input or model output.

# %% [markdown]
# ## 2. Semantic search over memories
#
# Pass an `index` config and the store embeds the named fields on write, so
# `search(..., query=...)` becomes a vector search.

# %%
embeddings = get_embeddings()

semantic_store = InMemoryStore(index={
    "embed": embeddings,
    "dims": len(embeddings.embed_query("dimension probe")),
    "fields": ["text"],          # which value fields to embed
})

MEMORIES = [
    ("m1", "Leads the Kestrel platform team of 6 engineers", "profile"),
    ("m2", "Prefers short answers with runnable code", "preference"),
    ("m3", "Production runs on AWS in ap-south-1", "infrastructure"),
    ("m4", "Decided in March to standardise on Postgres over MySQL", "decision"),
    ("m5", "Dislikes being asked to repeat context", "preference"),
    ("m6", "Vector corpus is roughly 2 million chunks", "infrastructure"),
    ("m7", "Budget approval needed above INR 5 lakh", "constraint"),
]
for key, text, kind in MEMORIES:
    semantic_store.put(("memories", "mukesh"), key, {"text": text, "kind": kind})

for query in ["where does their stuff run?", "how should I write to them?"]:
    hits = semantic_store.search(("memories", "mukesh"), query=query, limit=2)
    print(f"\n{query!r}")
    for hit in hits:
        print(f"   {hit.score:.3f}  {hit.value['text']}")

# %% [markdown]
# > Scores are meaningless if `shared.llm` fell back to `DeterministicFakeEmbedding`.
# > Install `langchain-huggingface[full]` or set `OPENAI_API_KEY` for real ones -
# > the plumbing is identical either way.

# %%
# Combine semantic search with a metadata filter - narrow first, then rank.
hits = semantic_store.search(("memories", "mukesh"), query="what do they like?",
                             filter={"kind": "preference"}, limit=3)
print("preferences ranked by relevance:")
for hit in hits:
    print(f"   {hit.score:.3f}  {hit.value['text']}")

# %% [markdown]
# ## 3. Using a store inside a graph
#
# Two ways to reach the store from a node. Both work; the injected parameter is
# more explicit, `get_store()` is easier in helper functions.

# %%
class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    recalled: list[str]


def recall(state: ChatState, *, store) -> dict:
    """Store arrives as a keyword argument when the node declares it."""
    question = state["messages"][-1].content
    hits = store.search(("memories", state["user_id"]), query=question, limit=3)
    return {"recalled": [h.value["text"] for h in hits]}


def respond(state: ChatState) -> dict:
    """get_store() reaches the same store without changing the signature."""
    _ = get_store()
    context = "\n".join(f"- {m}" for m in state["recalled"]) or "(nothing remembered yet)"
    system = ("You are a helpful colleague. Use what you remember about this person, "
              f"and never ask them to repeat it.\n\nWhat you remember:\n{context}")
    return {"messages": [model.invoke([("system", system)] + state["messages"])]}


builder = StateGraph(ChatState)
builder.add_node("recall", recall)
builder.add_node("respond", respond)
builder.add_edge(START, "recall")
builder.add_edge("recall", "respond")
builder.add_edge("respond", END)

graph = builder.compile(checkpointer=InMemorySaver(), store=semantic_store)
print(graph.get_graph().draw_ascii())

# %%
result = graph.invoke(
    {"messages": [HumanMessage("Should we move our index to a managed service?")],
     "user_id": "mukesh", "recalled": []},
    {"configurable": {"thread_id": "brand-new-conversation"}},
)
print(f"recalled: {result['recalled']}\n")
print(result["messages"][-1].content.strip()[:420])

# %% [markdown]
# That was a **brand new thread**. The checkpointer had nothing; the store
# supplied the context.

# %% [markdown]
# ## 4. Writing memories automatically
#
# The interesting question is not how to save a memory - it is deciding *what*
# is worth saving. Use structured extraction with a strict definition.

# %%
class Memory(BaseModel):
    """One durable fact worth remembering about the user."""

    text: str = Field(description="A single self-contained fact, written in the third person")
    kind: Literal["profile", "preference", "decision", "infrastructure", "constraint"]


class MemoryExtraction(BaseModel):
    """Memories worth persisting from this exchange, if any."""

    memories: list[Memory] = Field(default_factory=list)


extractor = ChatPromptTemplate.from_messages([
    ("system",
     "Extract only facts that will still be true and still be useful in a month.\n\n"
     "SAVE: roles, team names, stack and infrastructure, stated preferences, decisions "
     "and their reasons, hard constraints.\n"
     "DO NOT SAVE: anything specific to this conversation, questions the user asked, "
     "your own suggestions, transient state, or anything you inferred rather than were told.\n\n"
     "Return an empty list if nothing qualifies. An empty list is the correct answer most of the time."),
    ("human", "Conversation:\n{exchange}"),
]) | model.with_structured_output(MemoryExtraction)

SAMPLES = [
    "User: I'm Mukesh, I run the Kestrel platform team.\nAssistant: Nice to meet you.",
    "User: What's the capital of France?\nAssistant: Paris.",
    "User: We went with Postgres because the team already knows it.\nAssistant: Sensible.",
    "User: Can you rewrite that shorter?\nAssistant: Sure, here it is.",
]
for exchange in SAMPLES:
    found = extractor.invoke({"exchange": exchange}).memories
    label = ", ".join(f"[{m.kind}] {m.text}" for m in found) or "(nothing)"
    print(f"{exchange.splitlines()[0][:52]:54} -> {label[:70]}")

# %% [markdown]
# The "empty list is usually correct" instruction matters. Without it, models
# happily save "the user asked about France", and within a week the store is
# full of noise that pollutes every semantic search.

# %% [markdown]
# ## 5. Avoiding duplicates and stale facts
#
# Naive appending gives you "Deploys to ap-south-1" seven times, plus a
# contradiction after they migrate. Search before writing.

# %%
class Reconciliation(BaseModel):
    """How a new fact relates to an existing one."""

    action: Literal["insert", "update", "skip"]
    reason: str = Field(description="Under 12 words")


reconciler = ChatPromptTemplate.from_messages([
    ("system",
     "Decide how a new fact relates to the closest existing memory.\n"
     "- 'skip' if it says the same thing\n"
     "- 'update' if it contradicts or supersedes the existing one\n"
     "- 'insert' if it is genuinely new information"),
    ("human", "Existing: {existing}\n\nNew: {new}"),
]) | model.with_structured_output(Reconciliation)


def save_memory(store, user_id: str, memory: Memory, verbose: bool = True) -> str:
    namespace = ("memories", user_id)
    closest = store.search(namespace, query=memory.text, limit=1)

    if not closest:
        store.put(namespace, str(uuid.uuid4()), memory.model_dump())
        action = "insert"
    else:
        existing = closest[0]
        verdict = reconciler.invoke({"existing": existing.value["text"], "new": memory.text})
        action = verdict.action
        if action == "insert":
            store.put(namespace, str(uuid.uuid4()), memory.model_dump())
        elif action == "update":
            store.put(namespace, existing.key, memory.model_dump())   # same key = overwrite

    if verbose:
        print(f"  {action:6} {memory.text[:56]}")
    return action


scratch = InMemoryStore(index={"embed": embeddings,
                               "dims": len(embeddings.embed_query("x")),
                               "fields": ["text"]})

print("writing a stream of facts, some repeated, one contradicting:")
for text, kind in [
    ("Production runs on AWS in ap-south-1", "infrastructure"),
    ("Production runs on AWS in ap-south-1", "infrastructure"),     # exact duplicate
    ("Leads the Kestrel platform team", "profile"),
    ("Production migrated to AWS eu-west-1 in June", "infrastructure"),   # supersedes
]:
    save_memory(scratch, "demo", Memory(text=text, kind=kind))

print(f"\nstore now holds {len(scratch.search(('memories', 'demo')))} memories:")
for item in scratch.search(("memories", "demo")):
    print(f"   {item.value['text']}")

# %% [markdown]
# > With fake embeddings the nearest-neighbour lookup is random, so the
# > reconciler may make odd calls here. With real embeddings the duplicate is
# > skipped and the migration overwrites the old region.

# %% [markdown]
# ## 6. A graph that remembers as it talks
#
# Recall before answering, extract after. The extraction runs on a cheap model
# and does not block the reply.

# %%
class MemoryChatState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    recalled: list[str]
    saved: Annotated[int, operator.add]


def recall_node(state: MemoryChatState) -> dict:
    store = get_store()
    hits = store.search(("memories", state["user_id"]),
                        query=state["messages"][-1].content, limit=4)
    return {"recalled": [h.value["text"] for h in hits]}


def respond_node(state: MemoryChatState) -> dict:
    context = "\n".join(f"- {m}" for m in state["recalled"]) or "(nothing yet)"
    system = ("You are a helpful colleague. Be concise. Use what you remember and "
              f"never ask the person to repeat it.\n\nWhat you remember:\n{context}")
    return {"messages": [model.invoke([("system", system)] + state["messages"])]}


def remember_node(state: MemoryChatState) -> dict:
    exchange = "\n".join(
        f"{'User' if m.type == 'human' else 'Assistant'}: {m.content}"
        for m in state["messages"][-2:] if isinstance(m.content, str)
    )
    store = get_store()
    saved = 0
    for memory in extractor.invoke({"exchange": exchange}).memories:
        if save_memory(store, state["user_id"], memory, verbose=False) != "skip":
            saved += 1
    return {"saved": saved}


builder2 = StateGraph(MemoryChatState)
builder2.add_sequence([("recall", recall_node), ("respond", respond_node), ("remember", remember_node)])
builder2.add_edge(START, "recall")
builder2.add_edge("remember", END)

live_store = InMemoryStore(index={"embed": embeddings,
                                  "dims": len(embeddings.embed_query("x")),
                                  "fields": ["text"]})
memory_chat = builder2.compile(checkpointer=InMemorySaver(), store=live_store)

# %%
def say(text: str, thread: str, user: str = "mukesh") -> dict:
    return memory_chat.invoke(
        {"messages": [HumanMessage(text)], "user_id": user, "recalled": [], "saved": 0},
        {"configurable": {"thread_id": thread}},
    )


print("--- Monday's conversation (thread A) ---")
for text in [
    "Hi, I'm Mukesh and I lead the Kestrel platform team.",
    "We run everything on AWS ap-south-1 and latency matters more than cost to us.",
]:
    outcome = say(text, "monday")
    print(f"  saved {outcome['saved']}  |  {outcome['messages'][-1].content.strip()[:90]}")

print(f"\nstore now holds {len(live_store.search(('memories', 'mukesh')))} memories:")
for item in live_store.search(("memories", "mukesh")):
    print(f"   [{item.value['kind']:14}] {item.value['text']}")

# %%
print("--- Thursday, brand new thread, no shared checkpoint ---")
outcome = say("Remind me which region we're in and what we care about most?", "thursday")
print(f"recalled: {outcome['recalled']}\n")
print(outcome["messages"][-1].content.strip()[:300])

# %% [markdown]
# ## 7. Letting the agent manage its own memory
#
# Instead of an extraction node, give the agent tools. `InjectedStore` hides the
# store parameter from the model - it sees only `fact` and `kind`.

# %%
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedStore
from langgraph.store.base import BaseStore


@tool
def remember_fact(fact: str, kind: str, store: Annotated[BaseStore, InjectedStore()]) -> str:
    """Save a durable fact about the user. Only for facts still useful in a month."""
    store.put(("memories", "mukesh"), str(uuid.uuid4()), {"text": fact, "kind": kind})
    return f"Remembered: {fact}"


@tool
def search_memory(query: str, store: Annotated[BaseStore, InjectedStore()]) -> str:
    """Search what you remember about the user."""
    hits = store.search(("memories", "mukesh"), query=query, limit=4)
    return "\n".join(f"- {h.value['text']}" for h in hits) or "Nothing remembered about that."


@tool
def forget_fact(fact_substring: str, store: Annotated[BaseStore, InjectedStore()]) -> str:
    """Delete a remembered fact the user says is wrong or out of date."""
    for item in store.search(("memories", "mukesh")):
        if fact_substring.lower() in item.value["text"].lower():
            store.delete(("memories", "mukesh"), item.key)
            return f"Forgot: {item.value['text']}"
    return "No matching memory found."


print("model sees these arguments (store is hidden):")
for tool_fn in (remember_fact, search_memory, forget_fact):
    print(f"  {tool_fn.name:16} {list(tool_fn.args.keys())}")

# %%
memory_agent = create_agent(
    model,
    tools=[remember_fact, search_memory, forget_fact],
    system_prompt=(
        "You are a colleague with long-term memory. Search your memory before asking the "
        "user anything about themselves. Save durable facts as you learn them. "
        "If the user corrects you, forget the old fact and save the new one."
    ),
    checkpointer=InMemorySaver(),
    store=live_store,
)

for text in ["Actually we moved to eu-west-1 last month.",
             "What's my team and where do we deploy?"]:
    reply = memory_agent.invoke({"messages": [HumanMessage(text)]},
                                {"configurable": {"thread_id": "agent-thread"}})
    used = [tc["name"] for m in reply["messages"] for tc in getattr(m, "tool_calls", [])]
    print(f"\n> {text}")
    print(f"  tools: {used}")
    print(f"  {reply['messages'][-1].content.strip()[:220]}")

# %% [markdown]
# **Trade-off.** Tool-managed memory is flexible and self-correcting, but the
# model decides what to keep, so quality varies and every turn costs extra
# calls. Extraction nodes are predictable and cheaper to reason about. Many
# production systems run extraction for writes and expose only a search tool.

# %% [markdown]
# ## 8. Durable stores
#
# `InMemoryStore` dies with the process. `SqliteStore` for a single box,
# `PostgresStore` for anything real.

# %%
from langgraph.store.sqlite import SqliteStore

store_path = ctx.artifact("memories.sqlite")
with SqliteStore.from_conn_string(str(store_path)) as durable:
    durable.setup()                               # creates tables on first use
    durable.put(("memories", "mukesh"), "persisted-1",
                {"text": "Prefers Postgres over MySQL", "kind": "preference"})
    print(f"wrote to {store_path.name}")

with SqliteStore.from_conn_string(str(store_path)) as reopened:
    items = reopened.search(("memories", "mukesh"))
    print(f"reopened process, still there: {[i.value['text'] for i in items]}")

# %% [markdown]
# ```python
# # Production: Postgres, with semantic search, shared across every replica.
# from langgraph.store.postgres import PostgresStore
#
# with PostgresStore.from_conn_string(os.environ["DATABASE_URL"], index={
#     "embed": get_embeddings(), "dims": 1536, "fields": ["text"],
# }) as store:
#     store.setup()
#     graph = builder.compile(checkpointer=checkpointer, store=store)
# ```
#
# Semantic indexing needs `pgvector` in the database. Without the index config
# the store still works as plain key-value - `search(query=...)` just falls back
# to no ranking.

# %% [markdown]
# ## 9. Expiry and forgetting
#
# Memory that only grows becomes memory that is wrong. Three mechanisms:

# %%
# 1. TTL - the store expires the item for you.
#    InMemoryStore does NOT support TTL; a durable store does.
import time

try:
    InMemoryStore().put(("session",), "temp", {"text": "x"}, ttl=1.0)
except NotImplementedError as exc:
    print(f"InMemoryStore: {exc}")

ttl_config = {"default_ttl": 0.02,            # minutes (1.2 seconds) so the demo finishes
              "refresh_on_read": False,       # True keeps actively used memories alive
              "sweep_interval_minutes": 60}

with SqliteStore.from_conn_string(":memory:", ttl=ttl_config) as ttl_store:
    ttl_store.setup()
    ttl_store.put(("session", "u1"), "temp", {"text": "currently debugging the ingest job"})
    print(f"\nimmediately:     {ttl_store.get(('session', 'u1'), 'temp') is not None}")

    time.sleep(3.0)
    print(f"past expiry:     {ttl_store.get(('session', 'u1'), 'temp') is not None}  "
          f"<- still there; expiry is not lazy")
    print(f"swept:           {ttl_store.sweep_ttl()} item(s)")
    print(f"after sweep:     {ttl_store.get(('session', 'u1'), 'temp') is not None}")

# %% [markdown]
# Three things to internalise:
#
# - **TTL is in minutes**, not seconds.
# - **Expiry happens on a sweep, not on read.** An expired item you never swept
#   is still readable. In a service, call `start_ttl_sweeper()` once at
#   startup; in a batch job, call `sweep_ttl()` yourself.
# - **The sweep compares against SQLite's `CURRENT_TIMESTAMP`, which has
#   one-second granularity.** Sub-second TTLs behave unpredictably, which is
#   why the demo above sleeps three seconds for a 1.2-second TTL. This matters
#   only for tests; real TTLs are hours or days.

# %%
# 2. Explicit deletion when the user corrects you (the forget_fact tool above).

# 3. Periodic consolidation - merge and prune on a schedule.
class Consolidated(BaseModel):
    """A deduplicated, current set of memories."""

    memories: list[Memory]
    dropped: list[str] = Field(default_factory=list, description="Facts removed, and why")


consolidator = ChatPromptTemplate.from_messages([
    ("system", "Merge duplicates, drop superseded facts, and keep the current truth. "
               "Preserve every distinct piece of information. List what you dropped."),
    ("human", "Memories:\n{memories}"),
]) | model.with_structured_output(Consolidated)


def consolidate(store, user_id: str) -> None:
    namespace = ("memories", user_id)
    existing = store.search(namespace, limit=100)
    if len(existing) < 2:
        print("nothing to consolidate")
        return

    listing = "\n".join(f"- [{i.value['kind']}] {i.value['text']}" for i in existing)
    result = consolidator.invoke({"memories": listing})

    for item in existing:
        store.delete(namespace, item.key)
    for memory in result.memories:
        store.put(namespace, str(uuid.uuid4()), memory.model_dump())

    print(f"{len(existing)} -> {len(result.memories)} memories")
    for note in result.dropped:
        print(f"   dropped: {note}")


consolidate(live_store, "mukesh")
print("\nafter consolidation:")
for item in live_store.search(("memories", "mukesh")):
    print(f"   [{item.value['kind']:14}] {item.value['text']}")

# %% [markdown]
# Consolidation is destructive. Run it on a copy first, and keep an audit trail
# of deletions - "the assistant forgot something important" is a hard bug to
# diagnose without one.

# %% [markdown]
# ## 10. Memory that is not about a user
#
# The store is generic. Three other shapes worth knowing:

# %%
shared_store = InMemoryStore(index={"embed": embeddings,
                                    "dims": len(embeddings.embed_query("x")),
                                    "fields": ["text"]})

# Episodic: past runs an agent can learn from.
shared_store.put(("episodes", "support-agent"), "ep-1", {
    "text": "Refund request over INR 50,000 - escalated to finance, resolved in 2 days",
    "outcome": "success",
})

# Procedural: instructions the agent refines about how to do its job.
shared_store.put(("instructions", "support-agent"), "tone", {
    "text": "Always acknowledge the delay before explaining the policy. Customers react badly "
            "to policy-first replies.",
})

# Organisational: shared across every user in a tenant.
shared_store.put(("org", "northwind", "knowledge"), "oncall", {
    "text": "P1 escalations go to the platform on-call, not the team lead.",
})

for namespace in shared_store.list_namespaces():
    item = shared_store.search(namespace)[0]
    print(f"{'/'.join(namespace):34} {item.value['text'][:58]}")

# %% [markdown]
# Procedural memory is the interesting one: after a bad interaction, have the
# agent write an instruction for itself, and prepend those instructions to the
# system prompt on the next run. That is a feedback loop you can inspect and
# edit, unlike fine-tuning.

# %% [markdown]
# ## 11. Privacy is not optional
#
# Long-term memory means storing personal data indefinitely. Three things you
# owe your users:

# %%
def export_memories(store, user_id: str) -> list[dict]:
    """Right to access - show them everything you hold."""
    return [{"text": i.value["text"], "kind": i.value.get("kind"), "saved": str(i.created_at)}
            for i in store.search(("memories", user_id), limit=1000)]


def delete_all_memories(store, user_id: str) -> int:
    """Right to erasure - and it must actually delete, not just hide."""
    items = store.search(("memories", user_id), limit=1000)
    for item in items:
        store.delete(("memories", user_id), item.key)
    return len(items)


SENSITIVE = ("password", "credit card", "aadhaar", "ssn", "api key", "secret")


def is_safe_to_remember(text: str) -> bool:
    """Never persist a secret, whatever the model decides is 'useful'."""
    return not any(marker in text.lower() for marker in SENSITIVE)


print("export:", export_memories(live_store, "mukesh")[:2])
for candidate in ["Leads the Kestrel team", "Their API key is sk-abc123"]:
    print(f"  safe={is_safe_to_remember(candidate)!s:5} {candidate}")

# %% [markdown]
# Wire `is_safe_to_remember` into `save_memory` as a hard gate. A model
# instruction is a suggestion; a code check is a guarantee.

# %% [markdown]
# ## Try it yourself
#
# 1. **Add the safety gate** to `save_memory` and to `remember_fact`, then try to
#    talk the agent into saving a credential.
# 2. **Confidence decay.** Store a `last_confirmed` timestamp and down-rank
#    memories older than 90 days rather than deleting them.
# 3. **Procedural feedback loop.** After a run the user rates badly, have the
#    agent write an instruction to `("instructions", ...)` and load those
#    instructions into the next run's system prompt.
# 4. **Measure the benefit.** Run 10 questions with and without recall and count
#    how often the assistant asks for context it already had.

# %% [markdown]
# ## Recap
#
# | Concept | Takeaway |
# |---|---|
# | Checkpointer vs store | Thread state vs cross-thread facts - use both |
# | Namespace | Tuple path; tenant id first, from auth, never from the model |
# | `index={embed, dims, fields}` | Turns `search(query=...)` into vector search |
# | `filter` + `query` | Narrow by metadata, then rank by meaning |
# | Store in a node | `def node(state, *, store)` or `get_store()` |
# | `compile(store=...)` | Also available as `create_agent(store=...)` |
# | Extraction | The hard part is deciding what *not* to save |
# | Reconciliation | Search before writing: insert / update / skip |
# | `InjectedStore` | Memory tools with the store hidden from the model |
# | `SqliteStore` / `PostgresStore` | Durability; `setup()` on first use |
# | TTL | Minutes, durable stores only, and expiry needs a sweep |
# | Deletion, consolidation | Memory must shrink as well as grow |
# | Episodic / procedural / org | The store is not only for user facts |
# | Privacy | Export, erase, and a hard-coded gate on secrets |
#
# ## Next
#
# -> [50_deep_agents.ipynb](50_deep_agents.ipynb)
