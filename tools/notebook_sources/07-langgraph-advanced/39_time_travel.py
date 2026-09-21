# %% [markdown]
# # 39 - Time Travel
#
# | | |
# |---|---|
# | **Level** | Advanced (LangGraph) |
# | **Time** | 40 minutes |
# | **Prerequisites** | `38_parallel_fanout_fanin` |
# | **Checklist ID** | `39_time_travel` |
#
# ## Why this matters
#
# A user reports that your agent gave a bad answer twenty minutes ago. With a
# normal system you have logs, and logs tell you *what* happened. Time travel
# lets you go back to the exact moment before the mistake, change one thing, and
# watch what happens instead.
#
# It is also the cheapest way to build "regenerate", "edit and resubmit" and
# "explore both options" features - all of which are just forks of a checkpoint.

# %%
from pathlib import Path
import sys

_root = Path.cwd()
while not (_root / "shared").is_dir() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from shared.notebook_setup import setup, require  # noqa: E402

ctx = setup("39_time_travel")

# %%
import operator
from typing import Annotated, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from shared.llm import get_chat_model

model = get_chat_model()
parser = StrOutputParser()

# %% [markdown]
# ## 1. A pipeline with a decision worth revisiting

# %%
class BriefState(TypedDict):
    topic: str
    audience: str
    outline: str
    draft: str
    trace: Annotated[list[str], operator.add]


def choose_audience(state: BriefState) -> dict:
    return {"audience": "engineers", "trace": ["audience=engineers"]}


def outline_node(state: BriefState) -> dict:
    outline = (ChatPromptTemplate.from_template(
        "Write a 3-bullet outline about {topic} for an audience of {audience}."
    ) | model | parser).invoke(state)
    return {"outline": outline, "trace": ["outline"]}


def draft_node(state: BriefState) -> dict:
    draft = (ChatPromptTemplate.from_template(
        "Write a 100-word brief for {audience} following this outline:\n{outline}"
    ) | model | parser).invoke(state)
    return {"draft": draft, "trace": ["draft"]}


builder = StateGraph(BriefState)
builder.add_sequence([("audience", choose_audience), ("outline", outline_node), ("draft", draft_node)])
builder.add_edge(START, "audience")
builder.add_edge("draft", END)
graph = builder.compile(checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": "brief-1"}}
result = graph.invoke(
    {"topic": "our migration from FAISS to a managed vector store",
     "audience": "", "outline": "", "draft": "", "trace": []},
    config,
)
print("trace:", result["trace"])
print("\n", result["draft"].strip()[:280])

# %% [markdown]
# Suppose the brief should have been written for executives, not engineers. With
# time travel you do not re-run from scratch - you go back to the decision.

# %% [markdown]
# ## 2. Reading history
#
# `get_state_history()` yields snapshots newest first. Each is a complete,
# addressable moment.

# %%
history = list(graph.get_state_history(config))

print(f"{len(history)} checkpoints\n")
print(f"{'step':>5} {'next':14} {'source':8} {'trace':40} checkpoint_id")
for snapshot in history:
    print(f"{str(snapshot.metadata.get('step')):>5} "
          f"{str(snapshot.next):14} "
          f"{str(snapshot.metadata.get('source')):8} "
          f"{str(snapshot.values.get('trace', []))[:38]:40} "
          f"{snapshot.config['configurable']['checkpoint_id'][:8]}")

# %% [markdown]
# Read this from the bottom up and it is the execution story: input arrives,
# `audience` runs, `outline` runs, `draft` runs, done.
#
# The `next` field is the key to time travel. A snapshot with `next=("outline",)`
# is the moment **just before** `outline` ran.

# %% [markdown]
# ## 3. Replay - re-run from a point in the past
#
# Pass a checkpoint's own `config` (which carries its `checkpoint_id`) and
# `invoke(None, ...)`.

# %%
before_outline = next(s for s in history if s.next == ("outline",))
print(f"replaying from step {before_outline.metadata['step']} (before outline)")
print(f"state at that point: audience={before_outline.values['audience']!r}, "
      f"outline={'set' if before_outline.values['outline'] else 'empty'}")

replayed = graph.invoke(None, before_outline.config)
print(f"\nreplayed trace: {replayed['trace']}")

# %% [markdown]
# Note the trace: the replay re-ran `outline` and `draft`, appending to the
# existing trace. Replay **re-executes** - it is not a cached lookup, so
# non-deterministic steps produce different output.

# %% [markdown]
# ## 4. Fork - change the past and run a different future
#
# This is the valuable one. `update_state()` on a **past** checkpoint creates a
# new branch rather than appending to the current one.

# %%
before_outline = next(s for s in graph.get_state_history(config) if s.next == ("outline",))

fork_config = graph.update_state(
    before_outline.config,
    {"audience": "executives", "trace": ["FORK: audience changed to executives"]},
)

print("fork created at:", fork_config["configurable"]["checkpoint_id"][:8])

forked = graph.invoke(None, fork_config)
print("forked trace:", forked["trace"])
print("\nexecutive version:")
print(" ", forked["draft"].strip()[:280])

# %% [markdown]
# Two complete versions of the brief now exist in one thread's history, each
# reachable by `checkpoint_id`. Nothing was overwritten.

# %%
print(f"history is now {len(list(graph.get_state_history(config)))} checkpoints "
      f"(was {len(history)} before the fork)")

# %% [markdown]
# ## 5. Building "regenerate"
#
# The feature every chat UI has, in eight lines.

# %%
def regenerate(graph, config, node_name: str, overrides: dict | None = None) -> dict:
    """Re-run the graph from just before `node_name`, optionally changing state."""
    target = next(
        (s for s in graph.get_state_history(config) if s.next == (node_name,)),
        None,
    )
    if target is None:
        raise ValueError(f"no checkpoint found with next == ({node_name!r},)")

    start_from = graph.update_state(target.config, overrides) if overrides else target.config
    return graph.invoke(None, start_from)


variants = {}
for audience in ["executives", "new hires", "the board"]:
    outcome = regenerate(graph, config, "outline", {"audience": audience})
    variants[audience] = outcome["draft"]

for audience, draft in variants.items():
    print(f"\n--- {audience} ---")
    print(draft.strip()[:190])

# %% [markdown]
# Three drafts, each branching from the same outline decision point, without
# re-running anything before it. For an expensive early step - a large retrieval,
# a slow tool call - this is a real saving.

# %% [markdown]
# ## 6. Debugging a real failure
#
# The workflow: find the bad checkpoint, inspect the state that produced it,
# change the input, confirm the fix.

# %%
class TriageState(TypedDict):
    ticket: str
    category: str
    reply: str
    trace: Annotated[list[str], operator.add]


def bad_triage(state: TriageState) -> dict:
    """Deliberately wrong: everything looks like a how-to question."""
    return {"category": "howto", "trace": ["triage -> howto"]}


def respond(state: TriageState) -> dict:
    reply = (ChatPromptTemplate.from_template(
        "You are a {category} specialist. Reply in 2 sentences.\n\nTicket: {ticket}"
    ) | model | parser).invoke(state)
    return {"reply": reply, "trace": ["respond"]}


builder2 = StateGraph(TriageState)
builder2.add_sequence([("triage", bad_triage), ("respond", respond)])
builder2.add_edge(START, "triage")
builder2.add_edge("respond", END)
triage_graph = builder2.compile(checkpointer=InMemorySaver())

incident_config = {"configurable": {"thread_id": "incident-4821"}}
bad_outcome = triage_graph.invoke(
    {"ticket": "We have been charged twice for January - $18,400 duplicated. Finance needs this today.",
     "category": "", "reply": "", "trace": []},
    incident_config,
)
print("what the customer got:")
print(" ", bad_outcome["reply"].strip()[:200])

# %% [markdown]
# Step 1: find the moment before the bad decision took effect.

# %%
for snapshot in triage_graph.get_state_history(incident_config):
    print(f"  step={snapshot.metadata.get('step'):>2} next={str(snapshot.next):14} "
          f"category={snapshot.values.get('category', '')!r}")

# %% [markdown]
# Step 2: fork with the correct category and confirm the output changes.

# %%
before_respond = next(s for s in triage_graph.get_state_history(incident_config) if s.next == ("respond",))
print(f"the triage node wrote category={before_respond.values['category']!r} - that is the bug")

fixed = triage_graph.invoke(
    None,
    triage_graph.update_state(before_respond.config, {"category": "billing", "trace": ["FIX: category=billing"]}),
)
print("\nwhat they should have got:")
print(" ", fixed["reply"].strip()[:200])

# %% [markdown]
# You have now proved two things without touching production: the triage node is
# at fault, and fixing it fixes the customer-visible output. That is a far
# stronger bug report than a log excerpt.

# %% [markdown]
# ## 7. Comparing branches
#
# A thread's history is a tree after a fork. Walk it with `parent_config`.

# %%
def branch_summary(graph, config) -> None:
    snapshots = list(graph.get_state_history(config))
    by_id = {s.config["configurable"]["checkpoint_id"]: s for s in snapshots}

    children: dict[str, list[str]] = {}
    for snapshot in snapshots:
        parent = (snapshot.parent_config or {}).get("configurable", {}).get("checkpoint_id")
        if parent:
            children.setdefault(parent, []).append(snapshot.config["configurable"]["checkpoint_id"])

    forks = {parent: kids for parent, kids in children.items() if len(kids) > 1}
    print(f"{len(snapshots)} checkpoints, {len(forks)} fork point(s)")
    for parent, kids in forks.items():
        base = by_id.get(parent)
        print(f"\n  fork at {parent[:8]} (step {base.metadata.get('step') if base else '?'}) ->")
        for kid in kids:
            snapshot = by_id[kid]
            print(f"     {kid[:8]}  audience={snapshot.values.get('audience', '')!r:14} "
                  f"trace={str(snapshot.values.get('trace', []))[:44]}")


branch_summary(graph, config)

# %% [markdown]
# ## 8. Time travel with human-in-the-loop
#
# The two features compose. A reviewer can reject, rewind past the decision that
# led to the rejected output, and let the graph try again with a nudge.

# %%
from langgraph.types import Command, interrupt


class ApprovalState(TypedDict):
    request: str
    tone: str
    draft: str
    approved: bool
    trace: Annotated[list[str], operator.add]


def pick_tone(state: ApprovalState) -> dict:
    return {"tone": state["tone"] or "formal", "trace": [f"tone={state['tone'] or 'formal'}"]}


def write_draft(state: ApprovalState) -> dict:
    draft = (ChatPromptTemplate.from_template(
        "Write a 2-sentence reply in a {tone} tone to: {request}"
    ) | model | parser).invoke(state)
    return {"draft": draft, "trace": ["draft"]}


def review(state: ApprovalState) -> dict:
    decision = interrupt({"draft": state["draft"], "tone": state["tone"]})
    return {"approved": decision == "approve", "trace": [f"review={decision}"]}


builder3 = StateGraph(ApprovalState)
builder3.add_sequence([("tone", pick_tone), ("write", write_draft), ("review", review)])
builder3.add_edge(START, "tone")
builder3.add_edge("review", END)
approval_graph = builder3.compile(checkpointer=InMemorySaver())

cfg = {"configurable": {"thread_id": "approval-1"}}
paused = approval_graph.invoke(
    {"request": "Customer is furious about a duplicate $18,400 charge.",
     "tone": "", "draft": "", "approved": False, "trace": []},
    cfg,
)
print("draft for review:", paused["__interrupt__"][0].value["draft"][:160])

# %%
rejected = approval_graph.invoke(Command(resume="reject"), cfg)
print("reviewer rejected. trace:", rejected["trace"])

before_tone = next(s for s in approval_graph.get_state_history(cfg) if s.next == ("tone",))
retry_config = approval_graph.update_state(
    before_tone.config, {"tone": "warm and apologetic", "trace": ["RETRY with a different tone"]}
)
retried = approval_graph.invoke(None, retry_config)
print("\nnew draft:", retried["__interrupt__"][0].value["draft"][:200])

approved = approval_graph.invoke(Command(resume="approve"), retry_config)
print("\napproved:", approved["approved"], "| trace:", approved["trace"])

# %% [markdown]
# ## 9. Costs and cautions
#
# | Caution | Why |
# |---|---|
# | **Replay re-executes** | Model calls cost money again and may differ |
# | **Side effects re-run** | A node that sent an email will send it again |
# | **History grows on every fork** | Three variants triple the checkpoint count |
# | **Forks share a `thread_id`** | Your UI needs to track which `checkpoint_id` is "current" |
# | **Only checkpointed state travels** | Anything stored outside the graph does not rewind |
#
# The side-effect warning is the serious one. If a node has external effects,
# make it idempotent or guard it - the same discipline as notebook 36.

# %%
class GuardedState(TypedDict):
    order_id: str
    emails_sent: Annotated[list[str], operator.add]


def send_email(state: GuardedState) -> dict:
    """Idempotent: refuses to send twice for the same order."""
    if state["order_id"] in state["emails_sent"]:
        print(f"   [skipped] already emailed for {state['order_id']}")
        return {}
    print(f"   [sent] confirmation for {state['order_id']}")
    return {"emails_sent": [state["order_id"]]}


builder4 = StateGraph(GuardedState)
builder4.add_node("email", send_email)
builder4.add_edge(START, "email")
builder4.add_edge("email", END)
email_graph = builder4.compile(checkpointer=InMemorySaver())

email_cfg = {"configurable": {"thread_id": "order-1"}}
print("first run:")
email_graph.invoke({"order_id": "ORD-900", "emails_sent": []}, email_cfg)

print("replay from the start:")
first = list(email_graph.get_state_history(email_cfg))[-1]
email_graph.invoke(None, first.config)

# %% [markdown]
# ## Try it yourself
#
# 1. **Build a "compare two answers" feature.** Fork the same question with two
#    different system prompts and display both drafts side by side.
# 2. **Walk the tree.** Extend `branch_summary` to print an indented tree using
#    `parent_config`.
# 3. **Rewind an agent.** Run a tool-using agent, fork just before the tool call,
#    edit the arguments, and compare the final answers.
# 4. **Prove the side-effect hazard.** Write a node that appends to a file, run
#    it, replay from the start, and count the lines.

# %% [markdown]
# ## Recap
#
# | Concept | Takeaway |
# |---|---|
# | `get_state_history()` | Newest first; every snapshot is addressable |
# | `snapshot.next` | Identifies the moment *before* a given node ran |
# | Replay | `invoke(None, past_config)` re-executes from that point |
# | Fork | `update_state(past_config, ...)` branches instead of appending |
# | `regenerate()` | Rewind to a node, override state, re-run - eight lines |
# | Debugging | Prove the fault and the fix without touching production |
# | `parent_config` | Walk the branch tree after forking |
# | With HITL | Reject, rewind past the cause, retry with a change |
# | Side effects | Replay re-runs them - make such nodes idempotent |
#
# ## Next
#
# -> [40_subgraphs.ipynb](40_subgraphs.ipynb)
