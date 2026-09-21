# %% [markdown]
# # 36 - Human in the Loop
#
# | | |
# |---|---|
# | **Level** | Intermediate (LangGraph) |
# | **Time** | 55 minutes |
# | **Prerequisites** | `35_sqlite_postgres_savers` |
# | **Checklist ID** | `36_human_in_the_loop` |
#
# ## Why this matters
#
# There is a category of action you should never let a model take unsupervised:
# issuing a refund, emailing a customer, deleting a record, merging a PR,
# executing a trade. Not because models are bad at deciding - because the cost of
# being wrong is asymmetric.
#
# Human-in-the-loop is the feature that makes LLM systems deployable in those
# domains. LangGraph implements it properly: the graph **pauses**, the state is
# **durable**, and a human can approve, edit or reject hours later from a
# different process.

# %%
from pathlib import Path
import sys

_root = Path.cwd()
while not (_root / "shared").is_dir() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from shared.notebook_setup import setup, require  # noqa: E402

ctx = setup("36_human_in_the_loop")

# %%
import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from shared.llm import get_chat_model

model = get_chat_model()
parser = StrOutputParser()

# %% [markdown]
# ## 1. Static interrupts - `interrupt_before` / `interrupt_after`
#
# The simplest form: name the nodes to pause at when you compile. A checkpointer
# is **required** - without durable state there is nothing to resume.

# %%
class RefundState(TypedDict):
    ticket: str
    amount: float
    decision: str
    email: str
    trace: Annotated[list[str], operator.add]


def assess(state: RefundState) -> dict:
    verdict = (ChatPromptTemplate.from_template(
        "A customer claims: {ticket}\nIn one line, state whether a refund of ${amount} appears justified."
    ) | model | parser).invoke(state)
    return {"decision": verdict.strip(), "trace": ["assess"]}


def issue_refund(state: RefundState) -> dict:
    return {"trace": [f"REFUND ISSUED: ${state['amount']:,.2f}"]}


def notify(state: RefundState) -> dict:
    email = (ChatPromptTemplate.from_template(
        "Write a 2-sentence email confirming a ${amount} refund. Warm, not effusive."
    ) | model | parser).invoke(state)
    return {"email": email, "trace": ["notify"]}


builder = StateGraph(RefundState)
builder.add_sequence([("assess", assess), ("issue_refund", issue_refund), ("notify", notify)])
builder.add_edge(START, "assess")
builder.add_edge("notify", END)

guarded = builder.compile(checkpointer=InMemorySaver(), interrupt_before=["issue_refund"])
print(guarded.get_graph().draw_ascii())

# %%
config = {"configurable": {"thread_id": "refund-1"}}
paused = guarded.invoke(
    {"ticket": "We were charged twice for January - 240 seats, billed on the 2nd and the 4th.",
     "amount": 18400.0, "decision": "", "email": "", "trace": []},
    config,
)

print("paused before:", guarded.get_state(config).next)
print("model's assessment:", paused["decision"][:150])
print("trace so far:", paused["trace"])

# %% [markdown]
# No money moved. The state is on disk (well, in the saver) and the process could
# now exit entirely.
#
# ### Approve

# %%
resumed = guarded.invoke(None, config)
print("trace:", resumed["trace"])
print("\nemail:", resumed["email"].strip()[:180])

# %% [markdown]
# `invoke(None, config)` means *continue from where you paused*. Passing `None`
# rather than input is what distinguishes resuming from starting.
#
# ### Reject
#
# To reject, do not resume. Update state and route elsewhere - or simply stop.

# %%
reject_config = {"configurable": {"thread_id": "refund-2"}}
guarded.invoke(
    {"ticket": "I want a refund because I changed my mind after 6 months.",
     "amount": 18400.0, "decision": "", "email": "", "trace": []},
    reject_config,
)
print("paused at:", guarded.get_state(reject_config).next)

guarded.update_state(reject_config, {"trace": ["REJECTED by reviewer: outside the 30-day window"]})
print("state after rejection:", guarded.get_state(reject_config).values["trace"])
print("thread abandoned; no refund, no email")

# %% [markdown]
# ### Edit before approving
#
# The most useful case in practice: the human agrees in principle but changes a
# detail.

# %%
edit_config = {"configurable": {"thread_id": "refund-3"}}
guarded.invoke(
    {"ticket": "Double charged for January seats.", "amount": 18400.0,
     "decision": "", "email": "", "trace": []},
    edit_config,
)
print("model proposed: $18,400.00")

guarded.update_state(edit_config, {"amount": 9200.0, "trace": ["reviewer halved the amount: partial month"]})
final = guarded.invoke(None, edit_config)
print("actually issued:", [t for t in final["trace"] if "REFUND" in t])

# %% [markdown]
# ## 2. `interrupt()` - dynamic, from inside a node
#
# Static interrupts pause *every* time. Usually you only want to pause when it
# matters - a $50 refund should not wake anyone up. `interrupt()` is called
# inside a node, conditionally, and can carry a payload to the reviewer.

# %%
from langgraph.types import Command, interrupt

APPROVAL_THRESHOLD = 1000.0


class SmartRefundState(TypedDict):
    ticket: str
    amount: float
    approved: bool
    note: str
    trace: Annotated[list[str], operator.add]


def review_node(state: SmartRefundState) -> dict:
    """Pause only when the amount crosses the threshold."""
    if state["amount"] < APPROVAL_THRESHOLD:
        return {"approved": True, "trace": [f"auto-approved ${state['amount']:,.2f} (under threshold)"]}

    decision = interrupt({
        "kind": "refund_approval",
        "amount": state["amount"],
        "ticket": state["ticket"],
        "question": f"Approve a refund of ${state['amount']:,.2f}?",
        "options": ["approve", "reject", "reduce"],
    })

    if isinstance(decision, dict) and decision.get("action") == "reduce":
        return {"approved": True, "amount": decision["amount"],
                "note": decision.get("reason", ""),
                "trace": [f"human reduced to ${decision['amount']:,.2f}"]}
    if decision == "approve":
        return {"approved": True, "trace": ["human approved"]}
    return {"approved": False, "trace": ["human rejected"]}


def execute(state: SmartRefundState) -> dict:
    if not state["approved"]:
        return {"trace": ["no action taken"]}
    return {"trace": [f"REFUND ISSUED: ${state['amount']:,.2f}"]}


builder2 = StateGraph(SmartRefundState)
builder2.add_node("review", review_node)
builder2.add_node("execute", execute)
builder2.add_edge(START, "review")
builder2.add_edge("review", "execute")
builder2.add_edge("execute", END)
smart = builder2.compile(checkpointer=InMemorySaver())

# %%
small = {"configurable": {"thread_id": "small-refund"}}
outcome = smart.invoke({"ticket": "Charged $49 twice.", "amount": 49.0,
                        "approved": False, "note": "", "trace": []}, small)
print("small refund:", outcome["trace"])
print("interrupted?", "__interrupt__" in outcome)

# %%
large = {"configurable": {"thread_id": "large-refund"}}
outcome = smart.invoke({"ticket": "Charged $18,400 twice.", "amount": 18400.0,
                        "approved": False, "note": "", "trace": []}, large)

interrupts = outcome["__interrupt__"]
print("paused. payload sent to the reviewer:")
for key, value in interrupts[0].value.items():
    print(f"   {key:9} {value}")

# %% [markdown]
# That payload is what you render in your approval UI. It travels with the
# checkpoint, so the UI needs nothing but the `thread_id`.
#
# ### Resuming with `Command(resume=...)`

# %%
result = smart.invoke(Command(resume="approve"), large)
print("approved path:", result["trace"])

# %%
reduce_cfg = {"configurable": {"thread_id": "reduce-refund"}}
smart.invoke({"ticket": "Charged $18,400 twice.", "amount": 18400.0,
              "approved": False, "note": "", "trace": []}, reduce_cfg)

result = smart.invoke(
    Command(resume={"action": "reduce", "amount": 9200.0, "reason": "partial month only"}),
    reduce_cfg,
)
print("reduced path:", result["trace"])
print("note:", result["note"])

# %% [markdown]
# **Key detail:** when you resume, the node runs again **from the top**. Anything
# before the `interrupt()` call executes twice.

# %%
class CountState(TypedDict):
    side_effects: Annotated[list[str], operator.add]
    answer: str


def careless(state: CountState) -> dict:
    # This runs on the first pass AND again on resume.
    print("   [side effect] sending an audit email...")
    decision = interrupt("Approve?")
    return {"side_effects": ["email sent"], "answer": str(decision)}


builder3 = StateGraph(CountState)
builder3.add_node("careless", careless)
builder3.add_edge(START, "careless")
builder3.add_edge("careless", END)
careless_graph = builder3.compile(checkpointer=InMemorySaver())

cfg = {"configurable": {"thread_id": "careless-1"}}
print("first pass:")
careless_graph.invoke({"side_effects": [], "answer": ""}, cfg)
print("on resume:")
careless_graph.invoke(Command(resume="yes"), cfg)

# %% [markdown]
# The audit email went out twice. **Put `interrupt()` first in the node, or put
# the side effect in a separate node after it.** This is the single most common
# HITL bug.

# %%
def careful_gate(state: CountState) -> dict:
    """Nothing but the interrupt - safe to re-run."""
    return {"answer": str(interrupt("Approve?"))}


def careful_effect(state: CountState) -> dict:
    """Runs exactly once, after approval."""
    print("   [side effect] sending an audit email...")
    return {"side_effects": ["email sent"]}


builder4 = StateGraph(CountState)
builder4.add_node("gate", careful_gate)
builder4.add_node("effect", careful_effect)
builder4.add_edge(START, "gate")
builder4.add_edge("gate", "effect")
builder4.add_edge("effect", END)
careful = builder4.compile(checkpointer=InMemorySaver())

cfg = {"configurable": {"thread_id": "careful-1"}}
print("first pass:")
careful.invoke({"side_effects": [], "answer": ""}, cfg)
print("on resume:")
outcome = careful.invoke(Command(resume="yes"), cfg)
print("side effects:", outcome["side_effects"])

# %% [markdown]
# ## 3. The four HITL patterns
#
# ### Approve or reject a tool call

# %%
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition


@tool
def search_tickets(query: str) -> str:
    """Search the support ticket database. Safe, read-only."""
    return f"3 tickets match {query!r}: T-101 (open), T-099 (closed), T-088 (closed)"


@tool
def close_ticket(ticket_id: str, reason: str) -> str:
    """Close a support ticket. This is visible to the customer."""
    return f"{ticket_id} closed: {reason}"


DANGEROUS = {"close_ticket"}
tools = [search_tickets, close_ticket]
model_with_tools = model.bind_tools(tools)


def agent_node(state: MessagesState) -> dict:
    return {"messages": [model_with_tools.invoke(state["messages"])]}


def guarded_tools(state: MessagesState) -> dict:
    """Ask for approval before running anything in DANGEROUS."""
    last = state["messages"][-1]
    risky = [c for c in last.tool_calls if c["name"] in DANGEROUS]

    if risky:
        decision = interrupt({
            "kind": "tool_approval",
            "calls": [{"name": c["name"], "args": c["args"]} for c in risky],
            "question": "Approve these tool calls?",
        })
        if decision != "approve":
            from langchain_core.messages import ToolMessage

            return {"messages": [
                ToolMessage(content="Rejected by a human reviewer. Do not retry.", tool_call_id=c["id"])
                for c in last.tool_calls
            ]}

    return ToolNode(tools).invoke(state)


builder5 = StateGraph(MessagesState)
builder5.add_node("agent", agent_node)
builder5.add_node("tools", guarded_tools)
builder5.add_edge(START, "agent")
builder5.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
builder5.add_edge("tools", "agent")
tool_agent = builder5.compile(checkpointer=InMemorySaver())

# %%
cfg = {"configurable": {"thread_id": "tools-safe"}}
outcome = tool_agent.invoke({"messages": [HumanMessage("Search for tickets about duplicate billing.")]}, cfg)
print("read-only path interrupted?", "__interrupt__" in outcome)
print(outcome["messages"][-1].content.strip()[:140])

# %%
cfg = {"configurable": {"thread_id": "tools-risky"}}
outcome = tool_agent.invoke(
    {"messages": [HumanMessage("Close ticket T-101, the customer confirmed it is resolved.")]}, cfg
)
if "__interrupt__" in outcome:
    print("paused for approval:", outcome["__interrupt__"][0].value["calls"])
    rejected = tool_agent.invoke(Command(resume="reject"), cfg)
    print("\nafter rejection:", rejected["messages"][-1].content.strip()[:160])

# %% [markdown]
# ### Edit the model's proposed arguments
#
# Approval is binary; editing is better. Let the reviewer fix the arguments.

# %%
def editable_tools(state: MessagesState) -> dict:
    last = state["messages"][-1]
    if not any(c["name"] in DANGEROUS for c in last.tool_calls):
        return ToolNode(tools).invoke(state)

    decision = interrupt({"kind": "edit_args", "calls": [dict(c) for c in last.tool_calls]})

    if isinstance(decision, dict) and decision.get("action") == "edit":
        edited = last.model_copy(update={"tool_calls": decision["tool_calls"]})
        return ToolNode(tools).invoke({"messages": state["messages"][:-1] + [edited]})
    if decision == "approve":
        return ToolNode(tools).invoke(state)

    from langchain_core.messages import ToolMessage

    return {"messages": [ToolMessage(content="Rejected by reviewer.", tool_call_id=c["id"]) for c in last.tool_calls]}


builder6 = StateGraph(MessagesState)
builder6.add_node("agent", agent_node)
builder6.add_node("tools", editable_tools)
builder6.add_edge(START, "agent")
builder6.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
builder6.add_edge("tools", "agent")
editable = builder6.compile(checkpointer=InMemorySaver())

cfg = {"configurable": {"thread_id": "edit-args"}}
outcome = editable.invoke({"messages": [HumanMessage("Close ticket T-101 as resolved.")]}, cfg)

if "__interrupt__" in outcome:
    proposed = outcome["__interrupt__"][0].value["calls"]
    print("model proposed:", proposed[0]["args"])

    corrected = [dict(c, args={**c["args"], "reason": "Duplicate charge refunded; confirmed with finance."})
                 for c in proposed]
    final = editable.invoke(Command(resume={"action": "edit", "tool_calls": corrected}), cfg)
    print("executed with :", corrected[0]["args"])
    print("\n", final["messages"][-1].content.strip()[:160])

# %% [markdown]
# ### Ask the user a question mid-run
#
# `interrupt()` is not only for approval - it is a general "I need input" signal.

# %%
class BookingState(TypedDict):
    request: str
    details: dict
    confirmation: str


def gather(state: BookingState) -> dict:
    missing = interrupt({
        "kind": "clarification",
        "question": "Which date and how many attendees?",
        "context": state["request"],
    })
    return {"details": missing}


def book(state: BookingState) -> dict:
    return {"confirmation": f"Booked for {state['details']['date']}, {state['details']['attendees']} attendees."}


builder7 = StateGraph(BookingState)
builder7.add_node("gather", gather)
builder7.add_node("book", book)
builder7.add_edge(START, "gather")
builder7.add_edge("gather", "book")
builder7.add_edge("book", END)
booking = builder7.compile(checkpointer=InMemorySaver())

cfg = {"configurable": {"thread_id": "booking-1"}}
paused = booking.invoke({"request": "Book the large meeting room", "details": {}, "confirmation": ""}, cfg)
print("asked:", paused["__interrupt__"][0].value["question"])

answered = booking.invoke(Command(resume={"date": "2026-10-02", "attendees": 12}), cfg)
print("result:", answered["confirmation"])

# %% [markdown]
# ### Review and rewrite the model's output

# %%
class ReviewState(TypedDict):
    request: str
    draft: str
    final: str


def draft_node(state: ReviewState) -> dict:
    draft = (ChatPromptTemplate.from_template(
        "Write a 2-sentence reply to: {request}"
    ) | model | parser).invoke(state)
    return {"draft": draft}


def human_review(state: ReviewState) -> dict:
    decision = interrupt({"kind": "review_text", "draft": state["draft"]})
    if isinstance(decision, dict) and "rewrite" in decision:
        return {"final": decision["rewrite"]}
    return {"final": state["draft"]}


builder8 = StateGraph(ReviewState)
builder8.add_node("draft", draft_node)
builder8.add_node("review", human_review)
builder8.add_edge(START, "draft")
builder8.add_edge("draft", "review")
builder8.add_edge("review", END)
review_graph = builder8.compile(checkpointer=InMemorySaver())

cfg = {"configurable": {"thread_id": "review-1"}}
paused = review_graph.invoke(
    {"request": "Customer is angry about a double charge of $18,400.", "draft": "", "final": ""}, cfg
)
print("model draft:", paused["__interrupt__"][0].value["draft"][:150])

edited = review_graph.invoke(
    Command(resume={"rewrite": "We have confirmed the duplicate charge of $18,400 and refunded it in full today; "
                               "it will appear within 3 business days. I am sorry this happened twice."}),
    cfg,
)
print("\nsent:", edited["final"])

# %% [markdown]
# ## 4. Building an approval queue
#
# In production, the human is not in your Python process. The pattern is:
# pause -> persist -> a UI lists pending threads -> a human decides -> a
# *different* process resumes.

# %%
class ApprovalQueue:
    """The minimum viable approval inbox on top of a checkpointer."""

    def __init__(self, graph):
        self.graph = graph
        self.threads: list[str] = []

    def submit(self, thread_id: str, payload: dict) -> dict | None:
        self.threads.append(thread_id)
        outcome = self.graph.invoke(payload, {"configurable": {"thread_id": thread_id}})
        return outcome.get("__interrupt__", [None])[0]

    def pending(self) -> list[dict]:
        items = []
        for thread_id in self.threads:
            snapshot = self.graph.get_state({"configurable": {"thread_id": thread_id}})
            for task in snapshot.tasks:
                for pending in task.interrupts:
                    items.append({"thread_id": thread_id, "node": task.name, "payload": pending.value})
        return items

    def decide(self, thread_id: str, decision) -> dict:
        return self.graph.invoke(Command(resume=decision), {"configurable": {"thread_id": thread_id}})


queue = ApprovalQueue(smart)
for i, amount in enumerate([49.0, 18400.0, 5200.0], start=1):
    queue.submit(f"q-{i}", {"ticket": f"Refund request {i}", "amount": amount,
                            "approved": False, "note": "", "trace": []})

print("pending approvals:")
for item in queue.pending():
    print(f"   {item['thread_id']}  {item['node']:8}  ${item['payload']['amount']:>10,.2f}")

# %%
print("\nreviewer works through the queue:")
for item in queue.pending():
    decision = "approve" if item["payload"]["amount"] < 10000 else "reject"
    outcome = queue.decide(item["thread_id"], decision)
    print(f"   {item['thread_id']}  {decision:8} -> {outcome['trace'][-1]}")

print("\nremaining:", queue.pending())

# %% [markdown]
# `snapshot.tasks[*].interrupts` is the API your approval UI is built on. Every
# paused thread advertises what it is waiting for, and for what reason.

# %% [markdown]
# ## 5. `HumanInTheLoopMiddleware` for agents
#
# If you are using `create_agent` rather than a hand-built graph, middleware gives
# you the same thing declaratively.

# %%
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware

hitl_agent = create_agent(
    model,
    [search_tickets, close_ticket],
    middleware=[HumanInTheLoopMiddleware(interrupt_on={"close_ticket": True, "search_tickets": False})],
    checkpointer=InMemorySaver(),
)

cfg = {"configurable": {"thread_id": "mw-1"}}
outcome = hitl_agent.invoke({"messages": [HumanMessage("Close ticket T-101, it is resolved.")]}, cfg)

if "__interrupt__" in outcome:
    print("middleware paused. payload:")
    print("  ", str(outcome["__interrupt__"][0].value)[:260])

# %% [markdown]
# `interrupt_on` maps tool name to whether it needs approval, and accepts an
# `InterruptOnConfig` for finer control (allowing edit or rejection, custom
# descriptions). It is the right choice when the only thing you need to gate is
# tool execution.

# %% [markdown]
# ## 6. Design guidance
#
# **Gate on consequence, not on confidence.** Pause for anything irreversible,
# externally visible, or expensive - regardless of how sure the model is.
#
# | Gate it | Let it run |
# |---|---|
# | Money moving | Reading a database |
# | Anything a customer sees | Searching documents |
# | Deleting or overwriting | Drafting text for internal review |
# | Actions with legal weight | Calculations |
#
# **Give reviewers enough to decide.** An approval payload should contain the
# action, the arguments, the reasoning, and the relevant context. "Approve?" with
# no detail trains people to click yes.
#
# **Have a timeout policy.** Threads pause forever by default. Decide what
# happens after 24 hours: auto-reject, escalate, or notify - and implement it.
#
# **Measure your approval rate.** If reviewers approve 99% of requests you are
# gating too much and the review has become a rubber stamp. If they reject 40%,
# the model needs fixing, not supervising.

# %%
def summarise_queue(decisions: list[tuple[str, str]]) -> None:
    approved = sum(1 for _, d in decisions if d == "approve")
    total = len(decisions)
    rate = approved / total
    verdict = ("gating too much - consider raising the threshold" if rate > 0.95
               else "model needs work, not supervision" if rate < 0.7
               else "healthy")
    print(f"approval rate {rate:.0%} over {total} reviews -> {verdict}")


summarise_queue([("q-1", "approve"), ("q-2", "reject"), ("q-3", "approve"),
                 ("q-4", "approve"), ("q-5", "approve")])

# %% [markdown]
# ## Try it yourself
#
# 1. **Add a 24-hour timeout.** Write a job that finds threads paused longer than
#    a cutoff and resumes them with `Command(resume="reject")`.
# 2. **Multi-approver.** Require two distinct approvals for amounts over $50,000
#    by keeping an `approvals: list[str]` in state and interrupting until it has two.
# 3. **Find the double-execution bug.** Put a `print` before an `interrupt()` in a
#    node that also writes to a file, and watch the file get two entries.
# 4. **Persist the queue.** Swap `InMemorySaver` for `SqliteSaver`, submit
#    approvals from one script and resolve them from another.

# %% [markdown]
# ## Recap
#
# | Concept | Takeaway |
# |---|---|
# | Checkpointer required | No durable state, no resume |
# | `interrupt_before` / `interrupt_after` | Static pause points, every time |
# | `invoke(None, config)` | Resume from a static interrupt |
# | `interrupt(payload)` | Dynamic, conditional, carries data to the reviewer |
# | `Command(resume=value)` | Resume and deliver the human's answer |
# | **Node re-runs from the top** | Put `interrupt()` first or isolate side effects |
# | `__interrupt__` in the result | How you detect a pause |
# | `snapshot.tasks[*].interrupts` | What an approval UI reads |
# | Four patterns | Approve, edit arguments, ask a question, rewrite output |
# | `HumanInTheLoopMiddleware` | Declarative tool gating for `create_agent` |
# | Gate on consequence | Irreversible, visible, expensive - not on model confidence |
#
# ## Next
#
# -> [37_message_history_and_deletion.ipynb](37_message_history_and_deletion.ipynb)
