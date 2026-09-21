# %% [markdown]
# # 29 - Graphs vs Chains vs Agents
#
# | | |
# |---|---|
# | **Level** | Beginner (LangGraph) |
# | **Time** | 40 minutes |
# | **Prerequisites** | `18_from_agentexecutor_to_graphs`, `28_langchain_hub` |
# | **Checklist ID** | `29_graphs_vs_agents` |
#
# ## Why this matters
#
# You have already built chains (fixed pipelines) and agents (the model decides).
# Both hit the same wall in production, and it is always one of these four:
#
# 1. *"Can a human approve this refund before we send it?"*
# 2. *"The user closed the tab - can they resume tomorrow where they left off?"*
# 3. *"Something went wrong on step 4. Can we rewind and try a different branch?"*
# 4. *"Can the reviewer send it back to the writer, up to three times?"*
#
# A chain cannot pause. An agent cannot be steered. A **graph** can do all four,
# because execution is broken into checkpointed steps that you control.

# %%
from pathlib import Path
import sys

_root = Path.cwd()
while not (_root / "shared").is_dir() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from shared.notebook_setup import setup, require  # noqa: E402

ctx = setup("29_graphs_vs_agents")

# %%
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from shared.llm import get_chat_model

model = get_chat_model()
parser = StrOutputParser()

TICKET = (
    "Subject: Double charged\n"
    "We were billed twice for our January seats - 240 seats charged on the 2nd and again "
    "on the 4th. This is the second time this year. We need the duplicate refunded today; "
    "finance is closing the month."
)

# %% [markdown]
# ## 1. The same task, three ways
#
# Task: **triage a support ticket, then draft a reply.**
#
# ### As a chain - fixed order, no decisions

# %%
from langchain_core.runnables import RunnablePassthrough

chain = (
    RunnablePassthrough.assign(
        triage=(ChatPromptTemplate.from_template(
            "Classify this ticket as billing/technical/security and give a priority.\n"
            "Answer in one line.\n\n{ticket}"
        ) | model | parser)
    )
    .assign(
        reply=(ChatPromptTemplate.from_template(
            "Triage: {triage}\n\nWrite a 3-sentence reply to this ticket:\n{ticket}"
        ) | model | parser)
    )
)

result = chain.invoke({"ticket": TICKET})
print(result["triage"].strip())
print()
print(result["reply"].strip()[:280])

# %% [markdown]
# Predictable and cheap. But the path is frozen: every ticket gets exactly two
# model calls in exactly that order, and nothing can pause for approval.
#
# ### As an agent - the model decides everything

# %%
from langchain.agents import create_agent
from langchain_core.tools import tool


@tool
def lookup_account(company: str) -> str:
    """Look up a customer's plan and billing status."""
    return f"{company}: enterprise plan, 240 seats, 2 open billing disputes this year."


@tool
def issue_refund(company: str, amount_usd: float) -> str:
    """Issue a refund to a customer. This moves real money."""
    return f"Refund of ${amount_usd:,.2f} issued to {company}."


agent = create_agent(model, [lookup_account, issue_refund])
outcome = agent.invoke({"messages": [("user", f"Handle this ticket for Globex:\n{TICKET}")]})

for message in outcome["messages"]:
    kind = message.__class__.__name__.replace("Message", "")
    body = (message.content or "")[:110].replace("\n", " ")
    calls = getattr(message, "tool_calls", None)
    print(f"{kind:10} {body}{'  tool_calls=' + str([c['name'] for c in calls]) if calls else ''}")

# %% [markdown]
# Flexible - but notice what just happened. The agent may have called
# `issue_refund` on its own. **There was no point at which a human could say no**,
# because `invoke` runs the whole loop to completion.
#
# ### As a graph - explicit steps you can interrupt

# %%
from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class TicketState(TypedDict):
    ticket: str
    category: str
    priority: str
    reply: str


def triage_node(state: TicketState) -> dict:
    text = (ChatPromptTemplate.from_template(
        "Reply with exactly two words, lowercase, separated by a space: "
        "category (billing|technical|security) and priority (low|medium|high|critical).\n\n{ticket}"
    ) | model | parser).invoke({"ticket": state["ticket"]})
    parts = text.strip().split()
    return {"category": parts[0] if parts else "billing", "priority": parts[1] if len(parts) > 1 else "medium"}


def draft_node(state: TicketState) -> dict:
    reply = (ChatPromptTemplate.from_template(
        "You are a {category} specialist handling a {priority}-priority ticket.\n"
        "Write a 3-sentence reply. Do not promise a refund.\n\n{ticket}"
    ) | model | parser).invoke(state)
    return {"reply": reply}


builder = StateGraph(TicketState)
builder.add_node("triage", triage_node)
builder.add_node("draft", draft_node)
builder.add_edge(START, "triage")
builder.add_edge("triage", "draft")
builder.add_edge("draft", END)

graph = builder.compile()
final = graph.invoke({"ticket": TICKET, "category": "", "priority": "", "reply": ""})

print(f"category={final['category']} priority={final['priority']}")
print(final["reply"].strip()[:260])

# %% [markdown]
# So far this looks like the chain with more typing. The payoff comes next.

# %% [markdown]
# ## 2. What the graph can do that the others cannot
#
# ### Pause for a human

# %%
from langgraph.checkpoint.memory import InMemorySaver

approval_graph = builder.compile(checkpointer=InMemorySaver(), interrupt_before=["draft"])

config = {"configurable": {"thread_id": "ticket-4821"}}
paused = approval_graph.invoke({"ticket": TICKET, "category": "", "priority": "", "reply": ""}, config)

snapshot = approval_graph.get_state(config)
print(f"triage said: {paused['category']} / {paused['priority']}")
print(f"execution paused before: {snapshot.next}")
print(f"reply so far: {paused['reply']!r}")

# %% [markdown]
# The run stopped mid-graph. The state is durable - this process could exit and
# a different process could pick it up tomorrow.
#
# A human disagrees with the triage, corrects it, and lets it continue:

# %%
approval_graph.update_state(config, {"priority": "critical"})
resumed = approval_graph.invoke(None, config)

print(f"corrected priority: {resumed['priority']}")
print(resumed["reply"].strip()[:260])

# %% [markdown]
# Three capabilities in eight lines: **pause**, **inspect**, **edit state**, then
# **resume**. None of them are possible with `chain.invoke()`.
#
# ### Remember across separate calls

# %%
print("thread ticket-4821 state:", approval_graph.get_state(config).values["priority"])

other = {"configurable": {"thread_id": "ticket-9001"}}
approval_graph.invoke({"ticket": "Password reset email never arrives.", "category": "", "priority": "", "reply": ""}, other)
print("thread ticket-9001 state:", approval_graph.get_state(other).values["category"])
print("\nthread ticket-4821 is untouched:", approval_graph.get_state(config).values["category"])

# %% [markdown]
# Each `thread_id` is an isolated, persistent conversation. This is how LangGraph
# replaces the memory classes from notebook 06.
#
# ### Rewind

# %%
history = list(approval_graph.get_state_history(config))
print(f"{len(history)} checkpoints for ticket-4821 (newest first):\n")
for snap in history:
    print(f"  next={str(snap.next):22} priority={snap.values.get('priority', ''):10} "
          f"reply={'yes' if snap.values.get('reply') else 'no'}")

# %% [markdown]
# Every step was recorded. Notebook 39 shows how to resume from any one of these
# and explore a different branch - "time travel" debugging for LLM systems.

# %% [markdown]
# ## 3. The mental model
#
# | | Chain | Agent | Graph |
# |---|---|---|---|
# | Control flow | Fixed pipeline | Model decides | **You decide, explicitly** |
# | Cycles | No | Yes (hidden in the loop) | Yes (you draw them) |
# | Pause / resume | No | No | **Yes** |
# | Persistent state | No | Via checkpointer | **Yes, first class** |
# | Inspect mid-run | Callbacks only | Callbacks only | **`get_state()`** |
# | Edit mid-run | No | No | **`update_state()`** |
# | Rewind | No | No | **`get_state_history()`** |
# | Complexity cost | Lowest | Low | Highest |
#
# An agent *is* a graph - `create_agent` builds one for you. The difference is
# whether you accept the prebuilt shape or draw your own.

# %%
print(create_agent(model, [lookup_account]).get_graph().draw_ascii())

# %% [markdown]
# ## 4. The three pieces of every graph
#
# **State** - a `TypedDict` describing what flows between steps.
# **Nodes** - functions that take state and return a partial update.
# **Edges** - what runs next; fixed, conditional, or decided by a node.

# %%
def node(state: TicketState) -> dict:
    """The contract: take the whole state, return only the keys you changed."""
    return {"category": "billing"}          # merged into state, other keys untouched

print(graph.get_graph().draw_ascii())

# %% [markdown]
# Returning a *partial* update is the key idea. Nodes never mutate state in
# place; LangGraph merges what they return. That is what makes checkpointing,
# parallel execution and rewind possible.

# %% [markdown]
# ## 5. When NOT to use a graph
#
# Graphs cost you code. Do not pay it when you do not need it.
#
# | Situation | Use |
# |---|---|
# | Prompt -> model -> parse | **Chain**. A graph here is pure overhead. |
# | Fixed multi-step pipeline, no branching | **Chain** with LCEL |
# | Model picks from a handful of tools, no approval needed | **`create_agent`** |
# | Human approval anywhere in the flow | **Graph** |
# | Conversation that must survive a restart | **Graph** (checkpointer) |
# | Several specialists passing work between them | **Graph** |
# | You need to explain the control flow to an auditor | **Graph** - it is a diagram |
#
# The honest rule: start with a chain, upgrade to `create_agent` when the model
# needs to choose, and reach for a graph when you need **control over the loop**
# rather than more intelligence inside it.

# %% [markdown]
# ## Try it yourself
#
# 1. **Add an `escalate` node** to the graph that runs only for critical tickets,
#    and wire it with `add_edge` for now (conditional edges come in notebook 32).
# 2. **Break the state contract.** Make `triage_node` return a key that is not in
#    `TicketState` and read the error - it tells you exactly what the schema is for.
# 3. **Interrupt after instead of before.** Compile with `interrupt_after=["triage"]`
#    and compare the snapshot's `next` value with what you saw above.
# 4. **Count the checkpoints.** Run a fresh thread and print
#    `len(list(graph.get_state_history(config)))` after each stage.

# %% [markdown]
# ## Recap
#
# | Concept | Takeaway |
# |---|---|
# | Chain | Fixed pipeline; cheapest; cannot pause |
# | Agent | Model-driven loop; flexible; cannot be steered mid-run |
# | Graph | Explicit nodes and edges; pause, inspect, edit, resume, rewind |
# | State | A `TypedDict`; nodes return **partial updates**, never mutate |
# | `thread_id` | Isolated durable conversation - replaces memory classes |
# | `interrupt_before` | Stop before a node so a human can intervene |
# | `get_state_history` | Every step recorded; the basis for time travel |
# | When not to | Straight-line work; a graph there is overhead, not architecture |
#
# ## Next
#
# -> [30_state_schemas.ipynb](30_state_schemas.ipynb)
