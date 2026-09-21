# %% [markdown]
# # 18 - From AgentExecutor to Graphs
#
# | | |
# |---|---|
# | **Level** | Intermediate to Advanced |
# | **Time** | 45 minutes |
# | **Prerequisites** | `17_agents` |
# | **Checklist ID** | `18_from_agentexecutor_to_graphs` |
#
# ## Why this matters
#
# This notebook is the bridge between the LangChain half of the course and the
# LangGraph half. Its job is to make you *feel* the limitation rather than be told
# about it.
#
# You will take the quality-gate pipeline from notebook 08 - draft, critique,
# rewrite - and rebuild it three ways: as a Python loop, as an agent, and as a
# graph. Then you will do the thing only the graph version can do: **pause it for
# human approval and resume it later**.

# %%
from pathlib import Path
import sys

_root = Path.cwd()
while not (_root / "shared").is_dir() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from shared.notebook_setup import setup, require  # noqa: E402

ctx = setup("18_from_agentexecutor_to_graphs")

# %%
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from shared.llm import get_chat_model

model = get_chat_model()
parser = StrOutputParser()

CUSTOMER_MESSAGE = (
    "since your january release our CSV exports are missing the last row whenever we apply "
    "a filter. we noticed during month-end close and had to redo two reports. we're on starter. "
    "please look into this quickly."
)

draft_prompt = ChatPromptTemplate.from_template(
    "Write a 3-sentence support reply to this customer message.\n\n{message}"
)
critique_prompt = ChatPromptTemplate.from_template(
    "You are a support QA reviewer. Rules:\n"
    "- must not promise a fix date\n"
    "- must be at most 4 sentences\n"
    "- must state a concrete next step\n\n"
    "Reply:\n{draft}\n\nRespond with exactly 'PASS' or 'FAIL: <reason>'."
)
rewrite_prompt = ChatPromptTemplate.from_template(
    "Rewrite this support reply to fix the objection.\n\nReply:\n{draft}\n\nObjection: {critique}\n\nRewritten:"
)

# %% [markdown]
# ## 1. Version A: a plain Python loop
#
# This is what most people write first, and it works.

# %%
def python_loop(message: str, max_attempts: int = 3) -> dict:
    draft = (draft_prompt | model | parser).invoke({"message": message}).strip()
    for attempt in range(1, max_attempts + 1):
        verdict = (critique_prompt | model | parser).invoke({"draft": draft}).strip()
        if verdict.upper().startswith("PASS"):
            return {"reply": draft, "attempts": attempt, "status": "approved"}
        draft = (rewrite_prompt | model | parser).invoke({"draft": draft, "critique": verdict}).strip()
    return {"reply": draft, "attempts": max_attempts, "status": "max attempts"}


outcome = python_loop(CUSTOMER_MESSAGE)
print(f"status: {outcome['status']} after {outcome['attempts']} attempt(s)\n")
print(outcome["reply"])

# %% [markdown]
# **What is wrong with it?** Nothing, until you need any of this:
#
# - the process crashes on attempt 2 and you want to resume, not restart
# - a human must approve the reply before it is sent
# - you want to see, in a trace, that the loop ran twice and why
# - you want to change the draft mid-loop and continue
# - two of the steps could run in parallel
#
# Every one of those requires the loop's **state** to live outside the function.

# %% [markdown]
# ## 2. Version B: an agent
#
# Could an agent do this? Give it the tools and find out.

# %%
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.tools import tool


@tool
def draft_reply(customer_message: str) -> str:
    """Draft a support reply to a customer message."""
    return (draft_prompt | model | parser).invoke({"message": customer_message}).strip()


@tool
def review_reply(draft: str) -> str:
    """Review a draft support reply against QA rules. Returns PASS or FAIL with a reason."""
    return (critique_prompt | model | parser).invoke({"draft": draft}).strip()


@tool
def rewrite_reply(draft: str, objection: str) -> str:
    """Rewrite a draft reply to address a reviewer's objection."""
    return (rewrite_prompt | model | parser).invoke({"draft": draft, "critique": objection}).strip()


agent_version = create_agent(
    model=model,
    tools=[draft_reply, review_reply, rewrite_reply],
    system_prompt=(
        "You produce approved customer support replies.\n"
        "Process: draft_reply, then review_reply. If the review fails, rewrite_reply and "
        "review again. Repeat until PASS, maximum 3 reviews. Return the final approved reply."
    ),
    middleware=[ModelCallLimitMiddleware(run_limit=10, exit_behavior="end")],
)

result = agent_version.invoke({"messages": [{"role": "user", "content": CUSTOMER_MESSAGE}]})

print("steps taken:")
for message in result["messages"]:
    if message.type == "ai" and message.tool_calls:
        print("  ->", [c["name"] for c in message.tool_calls])
    elif message.type == "tool":
        print(f"     {message.name}: {str(message.content)[:70]}")
print("\nfinal:\n", result["messages"][-1].content.strip()[:400])

# %% [markdown]
# It works, but notice the trade: the *control flow itself is now a suggestion in
# a prompt*. The agent might review twice, might skip the review, might rewrite
# without reviewing. For a creative task that flexibility is a feature. For a
# compliance process - "every outbound reply must pass QA" - it is a defect.
#
# **You cannot guarantee a step happened when the model decides the steps.**

# %% [markdown]
# ## 3. Version C: a graph
#
# A graph makes the control flow explicit and the state inspectable. This is your
# first `StateGraph` - Track 05 covers every piece properly, so read it for the
# shape rather than the details.

# %%
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph


class ReviewState(TypedDict):
    """Everything the workflow knows. Each node reads and updates this."""

    message: str
    draft: str
    critique: str
    attempts: int
    status: str


def draft_node(state: ReviewState) -> dict:
    draft = (draft_prompt | model | parser).invoke({"message": state["message"]}).strip()
    return {"draft": draft, "attempts": 0}


def review_node(state: ReviewState) -> dict:
    verdict = (critique_prompt | model | parser).invoke({"draft": state["draft"]}).strip()
    passed = verdict.upper().startswith("PASS")
    return {
        "critique": verdict,
        "attempts": state["attempts"] + 1,
        "status": "approved" if passed else "needs work",
    }


def rewrite_node(state: ReviewState) -> dict:
    improved = (rewrite_prompt | model | parser).invoke(
        {"draft": state["draft"], "critique": state["critique"]}
    ).strip()
    return {"draft": improved}


def route_after_review(state: ReviewState) -> str:
    """The decision that was buried inside an `if` is now a named, traceable edge."""
    if state["status"] == "approved":
        return "done"
    if state["attempts"] >= 3:
        return "give_up"
    return "rewrite"


builder = StateGraph(ReviewState)
builder.add_node("draft", draft_node)
builder.add_node("review", review_node)
builder.add_node("rewrite", rewrite_node)

builder.add_edge(START, "draft")
builder.add_edge("draft", "review")
builder.add_conditional_edges(
    "review",
    route_after_review,
    {"rewrite": "rewrite", "done": END, "give_up": END},
)
builder.add_edge("rewrite", "review")   # <- the cycle LCEL cannot express

graph = builder.compile()
print("graph compiled")

# %%
try:
    print(graph.get_graph().draw_ascii())
except Exception:
    print(graph.get_graph().draw_mermaid())

# %%
final_state = graph.invoke({"message": CUSTOMER_MESSAGE, "draft": "", "critique": "", "attempts": 0, "status": ""})

print(f"status  : {final_state['status']}")
print(f"attempts: {final_state['attempts']}")
print(f"critique: {final_state['critique'][:120]}")
print(f"\nreply:\n{final_state['draft']}")

# %% [markdown]
# ### Watching every step

# %%
for step in graph.stream(
    {"message": CUSTOMER_MESSAGE, "draft": "", "critique": "", "attempts": 0, "status": ""},
    stream_mode="updates",
):
    for node_name, update in step.items():
        summary = {k: (str(v)[:60] + "..." if len(str(v)) > 60 else v) for k, v in update.items()}
        print(f"[{node_name:8}] {summary}")

# %% [markdown]
# Compare that to the Python loop, where the same information only existed in
# local variables that vanished on return.

# %% [markdown]
# ## 4. The thing only the graph can do
#
# Add a checkpointer and the state persists between steps. Add an interrupt and
# the workflow **stops, waits for a human, and resumes** - possibly in a different
# process, hours later.

# %%
from langgraph.checkpoint.memory import InMemorySaver

approval_graph = builder.compile(
    checkpointer=InMemorySaver(),
    interrupt_before=["rewrite"],   # pause before any rewrite
)

thread = {"configurable": {"thread_id": "reply-001"}}

partial = approval_graph.invoke(
    {"message": CUSTOMER_MESSAGE, "draft": "", "critique": "", "attempts": 0, "status": ""},
    thread,
)

snapshot = approval_graph.get_state(thread)
print("execution PAUSED")
print("  next node :", snapshot.next)
print("  attempts  :", snapshot.values["attempts"])
print("  critique  :", snapshot.values["critique"][:140])
print("\n  current draft:\n ", snapshot.values["draft"][:260])

# %% [markdown]
# The workflow is frozen mid-flight. Its entire state is saved. A human can now
# look at it, and - crucially - **edit it**.

# %%
# A human decides the draft needs a specific correction.
approval_graph.update_state(
    thread,
    {"critique": "FAIL: must explicitly acknowledge the month-end impact and offer a workaround."},
)

print("state after human edit:", approval_graph.get_state(thread).values["critique"])

# %%
# Resume by invoking with None - the graph continues from where it stopped.
resumed = approval_graph.invoke(None, thread)

print(f"\nstatus  : {resumed['status']}  after {resumed['attempts']} review(s)")
print(f"\nfinal reply:\n{resumed['draft']}")

# %% [markdown]
# Read that again: the graph paused, a human changed its internal state, and it
# continued with the new value. No amount of LCEL or agent prompting achieves
# that, because neither has durable, external, editable state.

# %% [markdown]
# ## 5. Time travel: rewind and try again
#
# Because every step was checkpointed, the whole history is addressable.

# %%
history = list(approval_graph.get_state_history(thread))
print(f"{len(history)} checkpoints recorded (newest first):\n")
for snapshot in history:
    next_node = snapshot.next[0] if snapshot.next else "END"
    print(f"  attempts={snapshot.values.get('attempts', '-'):<3} next={next_node:<8} "
          f"status={snapshot.values.get('status', '-')!r}")

# %% [markdown]
# Notebook 39 uses this to re-run from any past point with different inputs -
# which is how you debug a production agent that misbehaved last Tuesday.

# %% [markdown]
# ## 6. Your `create_agent` was a graph all along

# %%
from langchain.agents import create_agent as _create_agent

inspect_agent = _create_agent(model=model, tools=[draft_reply, review_reply], system_prompt="Assist.")

print("type:", type(inspect_agent).__name__)
print("nodes:", list(inspect_agent.get_graph().nodes))
print()
print(inspect_agent.get_graph().draw_mermaid())

# %% [markdown]
# `create_agent` builds exactly this shape: a `model` node, a `tools` node, and a
# conditional edge that loops back. Everything you learn in Track 05 applies
# directly to agents you have already built.

# %% [markdown]
# ## 7. Choosing between the three
#
# | | LCEL chain | `create_agent` | `StateGraph` |
# |---|---|---|---|
# | Control flow | you, fixed | the model | you, explicit, can cycle |
# | Cycles | no | yes (tool loop only) | **any shape** |
# | Guaranteed steps | yes | no | yes |
# | Persistence | no | with a checkpointer | with a checkpointer |
# | Pause for a human | no | via middleware | **anywhere** |
# | Parallel branches | `RunnableParallel` | no | **yes, fan-out/fan-in** |
# | Multi-agent | no | no | **yes** |
# | Effort to build | lowest | low | highest |
#
# **Decision rule**
#
# ```
# Fixed sequence?                       -> LCEL chain
# Variable tool use, standard loop?     -> create_agent
# Need cycles you define, approval,
# resumability, parallelism, or
# multiple cooperating agents?          -> StateGraph
# ```
#
# Do not start with a graph. Start with a chain, upgrade when you hit a wall, and
# recognise the wall from this list.

# %% [markdown]
# ## Try it yourself
#
# 1. **Add a parallel branch.** Extend the graph with a `tone_check` node that
#    runs at the same time as `review`, and make `route_after_review` require both
#    to pass. (Fan-out/fan-in is notebook 38 - try it now anyway.)
# 2. **Approve or reject.** Change the interrupt so a human approves the *final*
#    reply. On rejection, set `status="needs work"` via `update_state` and resume.
# 3. **Crash and recover.** Raise an exception inside `rewrite_node`, catch it
#    outside, then re-invoke with `None` on the same thread and confirm it resumes
#    rather than restarting.
# 4. **Rebuild your own.** Take a pipeline from your work, decide which of the
#    three shapes it needs, and write down why.

# %% [markdown]
# ## Recap
#
# | Concept | Takeaway |
# |---|---|
# | Python loop | Works, but state is local and disappears |
# | Agent | Flexible, but control flow becomes a prompt suggestion |
# | `StateGraph` | Explicit nodes, edges and cycles; state is external and durable |
# | `add_conditional_edges` | The `if` statement becomes a named, traceable routing decision |
# | `checkpointer` | State survives between steps and between processes |
# | `interrupt_before` | Pause anywhere; inspect, edit with `update_state`, resume with `invoke(None)` |
# | `get_state_history` | Every step is addressable - the basis of time travel |
# | `create_agent` | Already a graph; Track 05 explains what you have been using |
#
# ## Next
#
# -> [19_langsmith.ipynb](../04-langchain-production/19_langsmith.ipynb)
