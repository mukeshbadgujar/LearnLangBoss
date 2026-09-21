# %% [markdown]
# # 41 - Multi-Agent Systems and Handoffs
#
# | | |
# |---|---|
# | **Level** | Advanced (LangGraph) |
# | **Time** | 55 minutes |
# | **Prerequisites** | `40_subgraphs` |
# | **Checklist ID** | `41_multi_agent_handoffs` |
#
# ## Why this matters
#
# One agent with twenty tools is worse than four agents with five tools each.
# Not for philosophical reasons - because tool selection accuracy degrades as the
# tool list grows, and a system prompt covering four domains is a compromise in
# all four.
#
# Notebook 25 did routing in LangChain: pick a specialist once, at the start.
# Multi-agent systems go further - specialists can **hand off** to each other
# mid-task, share state, and come back. This notebook builds the three
# architectures you will actually use.

# %%
from pathlib import Path
import sys

_root = Path.cwd()
while not (_root / "shared").is_dir() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from shared.notebook_setup import setup, require  # noqa: E402

ctx = setup("41_multi_agent_handoffs")

# %%
import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field

from shared.llm import get_chat_model

model = get_chat_model()
parser = StrOutputParser()

# %% [markdown]
# ## 1. The three architectures
#
# ```
# SUPERVISOR              NETWORK                 HIERARCHICAL
#
#      supervisor          a <---> b              top supervisor
#     /     |     \         \     /                /           \
#    a      b      c          \  /            sub-sup-1     sub-sup-2
#     \     |     /            c               /    \        /    \
#      supervisor                             a      b      c      d
# ```
#
# | | Control | Cost | Use when |
# |---|---|---|---|
# | **Supervisor** | Central, predictable | One extra call per hop | **Default choice** |
# | **Network** | Distributed, emergent | Cheapest per hop | Agents genuinely know who is next |
# | **Hierarchical** | Layered | Highest | More than ~6 specialists |

# %% [markdown]
# ## 2. Shared tools and specialists

# %%
@tool
def search_docs(query: str) -> str:
    """Search internal policy and product documentation."""
    corpus = {
        "leave": "Employees accrue 24 days annual leave; 12 may carry forward until 30 June.",
        "notice": "Notice periods: L1-L3 30 days, L4 60 days, L5+ 90 days.",
        "rate limit": "Growth plan: 600 requests/minute. Enterprise: negotiated.",
        "residency": "Default data region is ap-south-1 (Mumbai); EU customers may request eu-west-1.",
    }
    for key, value in corpus.items():
        if key in query.lower():
            return value
    return "No matching documentation found."


@tool
def lookup_invoice(invoice_id: str) -> str:
    """Look up an invoice by ID."""
    return f"{invoice_id}: $18,400 charged 2026-01-02, duplicate suspected, status=UNRESOLVED"


@tool
def check_system_status(service: str) -> str:
    """Check the current status of a service."""
    return {"dashboards": "degraded - p95 latency 38s since 09:10",
            "api": "healthy", "reports": "healthy"}.get(service.lower(), "unknown service")


def make_specialist(name: str, instructions: str, tools: list):
    """A specialist is just a small agent with a narrow prompt and few tools."""
    from langchain.agents import create_agent

    return create_agent(model, tools, system_prompt=f"You are the {name}. {instructions}")


billing_agent = make_specialist(
    "billing specialist",
    "Handle invoices, charges and refunds. Quote policy exactly. Never promise a refund you cannot authorise. "
    "Answer in at most 3 sentences.",
    [lookup_invoice, search_docs],
)

technical_agent = make_specialist(
    "support engineer",
    "Diagnose errors, slowness and outages. Check system status before speculating. "
    "Answer in at most 3 sentences.",
    [check_system_status, search_docs],
)

policy_agent = make_specialist(
    "HR policy specialist",
    "Answer questions about leave, notice periods and internal policy from documentation only. "
    "Answer in at most 3 sentences.",
    [search_docs],
)

SPECIALISTS = {"billing": billing_agent, "technical": technical_agent, "policy": policy_agent}

# %% [markdown]
# ## 3. Supervisor architecture
#
# A coordinator decides who works next, sees each result, and decides again.
# Control is central, which makes the system predictable and easy to audit.

# %%
class SupervisorState(MessagesState):
    next_agent: str
    completed: Annotated[list[str], operator.add]
    hops: Annotated[int, operator.add]


MAX_HOPS = 4


class Routing(BaseModel):
    """Which specialist should work on this next."""

    next_agent: Literal["billing", "technical", "policy", "FINISH"] = Field(
        description="billing: invoices, charges, refunds. "
        "technical: errors, outages, performance. "
        "policy: leave, notice periods, HR rules. "
        "FINISH: the question is fully answered."
    )
    reason: str = Field(description="One short sentence")


supervisor_model = model.with_structured_output(Routing)


def supervisor(state: SupervisorState) -> dict:
    if state.get("hops", 0) >= MAX_HOPS:
        return {"next_agent": "FINISH", "messages": [AIMessage("Reached the coordination limit.")]}

    transcript = "\n".join(
        f"{m.__class__.__name__}: {m.content}" for m in state["messages"] if isinstance(m.content, str) and m.content
    )
    decision = supervisor_model.invoke([
        ("system",
         "You coordinate a support team. Route to the specialist who should work next, "
         f"or FINISH when the user's question is fully answered. Already consulted: "
         f"{state.get('completed') or 'nobody'}. Do not consult the same specialist twice."),
        ("human", transcript),
    ])
    return {"next_agent": decision.next_agent, "hops": 1}


def make_worker(name: str):
    def worker(state: SupervisorState) -> dict:
        result = SPECIALISTS[name].invoke({"messages": state["messages"]})
        answer = result["messages"][-1].content
        return {
            "messages": [AIMessage(content=f"[{name}] {answer}", name=name)],
            "completed": [name],
        }
    return worker


def route(state: SupervisorState) -> str:
    return END if state["next_agent"] == "FINISH" else state["next_agent"]


builder = StateGraph(SupervisorState)
builder.add_node("supervisor", supervisor)
for name in SPECIALISTS:
    builder.add_node(name, make_worker(name))
    builder.add_edge(name, "supervisor")          # always report back
builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor", route, {**{n: n for n in SPECIALISTS}, END: END})

supervisor_graph = builder.compile()
print(supervisor_graph.get_graph().draw_ascii())

# %%
QUESTION = ("We were double charged $18,400 on invoice INV-2026-0102, and separately our "
            "dashboards have been unusable since this morning. What is going on with both?")

outcome = supervisor_graph.invoke(
    {"messages": [HumanMessage(QUESTION)], "next_agent": "", "completed": [], "hops": 0}
)

print(f"consulted: {outcome['completed']} in {outcome['hops']} supervisor turns\n")
for message in outcome["messages"][1:]:
    print(f"  {(message.content or '')[:150]}\n")

# %% [markdown]
# Two specialists contributed, each using its own tools, coordinated centrally.
# Note `MAX_HOPS` - a supervisor loop without one is how you get a $400 bill from
# a single request.

# %% [markdown]
# ## 4. Network architecture with `Command`
#
# No coordinator. Each agent decides for itself whether to answer or hand off.
# One fewer model call per hop, at the cost of predictability.

# %%
class NetworkState(MessagesState):
    visited: Annotated[list[str], operator.add]
    hops: Annotated[int, operator.add]


class Handoff(BaseModel):
    """Either answer, or pass the work to a colleague."""

    action: Literal["answer", "handoff"]
    answer: str = Field(default="", description="Your answer, if action is 'answer'")
    handoff_to: Literal["billing", "technical", "policy", "none"] = Field(default="none")
    context: str = Field(default="", description="What the next specialist needs to know")


handoff_model = model.with_structured_output(Handoff)


def networked(name: str, remit: str):
    def node(state: NetworkState) -> Command:
        if state.get("hops", 0) >= MAX_HOPS:
            return Command(update={"messages": [AIMessage(f"[{name}] hop limit reached")]}, goto=END)

        transcript = "\n".join(
            f"{m.__class__.__name__}: {m.content}" for m in state["messages"]
            if isinstance(m.content, str) and m.content
        )
        decision = handoff_model.invoke([
            ("system",
             f"You are the {name}. Your remit: {remit}\n"
             f"If the request is within your remit, answer it in 2 sentences. "
             f"If a colleague (billing, technical, policy) should handle it, hand off. "
             f"Already visited: {state.get('visited') or 'nobody'} - do not hand back to them."),
            ("human", transcript),
        ])

        if decision.action == "answer" or decision.handoff_to in ("none", name):
            return Command(
                update={"messages": [AIMessage(f"[{name}] {decision.answer}", name=name)],
                        "visited": [name], "hops": 1},
                goto=END,
            )

        return Command(
            update={"messages": [AIMessage(f"[{name} -> {decision.handoff_to}] {decision.context}", name=name)],
                    "visited": [name], "hops": 1},
            goto=decision.handoff_to,
        )
    return node


REMITS = {
    "billing": "invoices, charges, refunds, plan changes",
    "technical": "errors, outages, latency, integrations that stopped working",
    "policy": "leave, notice periods, HR and internal policy",
}

network = StateGraph(NetworkState)
for name, remit in REMITS.items():
    network.add_node(name, networked(name, remit),
                     destinations=tuple(n for n in REMITS if n != name) + (END,))
network.add_edge(START, "policy")                  # every request lands at the front desk
network_graph = network.compile()

print(network_graph.get_graph().draw_ascii())

# %%
outcome = network_graph.invoke(
    {"messages": [HumanMessage("Our dashboards have been timing out since 9am. Is something down?")],
     "visited": [], "hops": 0}
)
print("path:", " -> ".join(outcome["visited"]))
for message in outcome["messages"][1:]:
    print(f"  {(message.content or '')[:150]}")

# %% [markdown]
# `destinations=` on `add_node` tells LangGraph where a `Command` node can jump,
# so the diagram is accurate. Without it the graph still runs but draws no edges.
#
# **The network failure mode is ping-pong.** Two agents each convinced the other
# owns the problem. The `visited` list and the hop cap are not optional
# refinements - they are what makes this architecture safe.

# %% [markdown]
# ## 5. Handoff as a tool
#
# The most idiomatic form: give each agent a `transfer_to_x` tool. The model
# decides to hand off the same way it decides anything else, and a
# `Command` returned from the tool moves the graph.

# %%
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from typing_extensions import Annotated as TAnnotated


def make_handoff_tool(target: str, description: str):
    @tool(f"transfer_to_{target}", description=description)
    def handoff(
        reason: str,
        tool_call_id: TAnnotated[str, InjectedToolCallId],
    ) -> Command:
        """Transfer the conversation to a colleague."""
        return Command(
            goto=target,
            graph=Command.PARENT,
            update={"messages": [ToolMessage(content=f"Transferred to {target}: {reason}",
                                             tool_call_id=tool_call_id)]},
        )

    return handoff


to_billing = make_handoff_tool("billing", "Transfer to the billing specialist for invoices, charges or refunds.")
to_technical = make_handoff_tool("technical", "Transfer to the support engineer for errors, outages or slowness.")
to_policy = make_handoff_tool("policy", "Transfer to the HR policy specialist for leave or notice periods.")

print("handoff tools:", [t.name for t in (to_billing, to_technical, to_policy)])
print("schema:", to_billing.args_schema.model_json_schema()["properties"].keys())

# %% [markdown]
# `InjectedToolCallId` means the model never sees or fills that argument -
# LangGraph supplies it. `Command(graph=Command.PARENT)` is what lets a tool
# running inside a child agent move the **parent** graph.

# %%
from langchain.agents import create_agent


def agent_node(name: str, instructions: str, tools: list):
    agent = create_agent(model, tools, system_prompt=f"You are the {name}. {instructions}")

    def node(state: MessagesState) -> Command:
        result = agent.invoke({"messages": state["messages"]})
        last = result["messages"][-1]
        return Command(update={"messages": [AIMessage(f"[{name}] {last.content}", name=name)]}, goto=END)

    return node


swarm = StateGraph(MessagesState)
swarm.add_node("front_desk", agent_node(
    "front desk",
    "You triage requests. If a request belongs to billing, technical support or HR policy, "
    "transfer it immediately using the appropriate tool. Otherwise answer briefly yourself.",
    [to_billing, to_technical, to_policy],
), destinations=("billing", "technical", "policy", END))
swarm.add_node("billing", agent_node("billing specialist", "Handle invoices and charges. 2 sentences.",
                                     [lookup_invoice, search_docs]))
swarm.add_node("technical", agent_node("support engineer", "Diagnose outages and slowness. 2 sentences.",
                                       [check_system_status, search_docs]))
swarm.add_node("policy", agent_node("HR policy specialist", "Answer policy questions. 2 sentences.",
                                    [search_docs]))
swarm.add_edge(START, "front_desk")
swarm_graph = swarm.compile()

print(swarm_graph.get_graph().draw_ascii())

# %%
for question in [
    "What is the notice period for an L5 engineer?",
    "Invoice INV-2026-0102 looks like a duplicate charge.",
]:
    outcome = swarm_graph.invoke({"messages": [HumanMessage(question)]})
    print(f"\nQ: {question}")
    for message in outcome["messages"][1:]:
        print(f"   {(message.content or '')[:130]}")

# %% [markdown]
# ## 6. Shared state between agents
#
# Agents that only exchange messages have to re-derive facts. A shared scratchpad
# in state is cheaper and more reliable.

# %%
class SharedState(MessagesState):
    facts: Annotated[dict, lambda a, b: {**a, **b}]
    visited: Annotated[list[str], operator.add]


def fact_gathering_agent(name: str, gather):
    def node(state: SharedState) -> dict:
        new_facts = gather(state)
        return {"facts": new_facts, "visited": [name],
                "messages": [AIMessage(f"[{name}] recorded {list(new_facts)}", name=name)]}
    return node


def final_answer(state: SharedState) -> dict:
    summary = (ChatPromptTemplate.from_template(
        "Write a 3-sentence customer reply using only these verified facts:\n{facts}"
    ) | model | parser).invoke({"facts": state["facts"]})
    return {"messages": [AIMessage(summary)]}


shared = StateGraph(SharedState)
shared.add_node("billing", fact_gathering_agent(
    "billing", lambda s: {"invoice": lookup_invoice.invoke({"invoice_id": "INV-2026-0102"})}))
shared.add_node("technical", fact_gathering_agent(
    "technical", lambda s: {"dashboards": check_system_status.invoke({"service": "dashboards"})}))
shared.add_node("respond", final_answer)
shared.add_edge(START, "billing")
shared.add_edge(START, "technical")
shared.add_edge("billing", "respond")
shared.add_edge("technical", "respond")
shared.add_edge("respond", END)

outcome = shared.compile().invoke({"messages": [HumanMessage(QUESTION)], "facts": {}, "visited": []})
print("facts gathered in parallel:")
for key, value in outcome["facts"].items():
    print(f"   {key:12} {value[:80]}")
print(f"\n{outcome['messages'][-1].content.strip()[:280]}")

# %% [markdown]
# Both specialists ran concurrently (notebook 38), wrote to a shared dict via a
# merge reducer, and a final node composed the answer. No agent had to read
# another's prose to extract a fact.

# %% [markdown]
# ## 7. Cost and when not to
#
# Multi-agent systems are expensive. Measure before committing.

# %%
from langchain_core.callbacks import BaseCallbackHandler


class CallCounter(BaseCallbackHandler):
    def __init__(self):
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def on_llm_end(self, response, **kwargs):
        self.calls += 1
        try:
            usage = response.generations[0][0].message.usage_metadata or {}
        except (AttributeError, IndexError):
            return
        self.input_tokens += usage.get("input_tokens", 0)
        self.output_tokens += usage.get("output_tokens", 0)


single = create_agent(model, [lookup_invoice, check_system_status, search_docs],
                      system_prompt="You are a support assistant. Answer in 3 sentences.")

print(f"{'approach':22} {'llm calls':>10} {'in tokens':>10} {'out tokens':>11}")
for label, run in [
    ("single agent", lambda cb: single.invoke({"messages": [HumanMessage(QUESTION)]}, config={"callbacks": [cb]})),
    ("supervisor", lambda cb: supervisor_graph.invoke(
        {"messages": [HumanMessage(QUESTION)], "next_agent": "", "completed": [], "hops": 0},
        config={"callbacks": [cb]})),
    ("shared-state parallel", lambda cb: shared.compile().invoke(
        {"messages": [HumanMessage(QUESTION)], "facts": {}, "visited": []}, config={"callbacks": [cb]})),
]:
    counter = CallCounter()
    run(counter)
    print(f"{label:22} {counter.calls:>10} {counter.input_tokens:>10} {counter.output_tokens:>11}")

# %% [markdown]
# The supervisor typically costs 3-5x a single agent for the same question.
# That is worth paying when specialisation genuinely improves accuracy, and pure
# waste when it does not.
#
# **Start with one agent.** Split only when you can point at a specific failure:
# the model picks the wrong tool from a long list, or one domain's instructions
# keep leaking into another's answers.
#
# | Symptom | Response |
# |---|---|
# | Wrong tool chosen from 15+ tools | Split by domain |
# | System prompt over ~1,500 tokens and growing | Split by domain |
# | One domain needs a different model | Split |
# | Different teams own different capabilities | Split |
# | "It would be cleaner architecture" | **Do not split** |

# %% [markdown]
# ## 8. Debugging multi-agent systems

# %%
def trace_run(graph, payload: dict, label: str) -> None:
    """Print who ran, in order, with what they produced."""
    print(f"--- {label} ---")
    for chunk in graph.stream(payload, stream_mode="updates"):
        for node_name, update in chunk.items():
            messages = update.get("messages") or []
            snippet = (messages[-1].content or "")[:90] if messages else str(update)[:90]
            print(f"  {node_name:12} {snippet}")


trace_run(
    supervisor_graph,
    {"messages": [HumanMessage("How many days of annual leave do we get?")],
     "next_agent": "", "completed": [], "hops": 0},
    "supervisor on a simple question",
)

# %% [markdown]
# Three things to check when a multi-agent system misbehaves:
#
# 1. **Is it looping?** Print the visited list; a repeated name means a missing
#    guard.
# 2. **Is context being lost?** Agents that only see the last message often are.
#    Pass the full `messages` list or use shared state.
# 3. **Is the supervisor's prompt specific enough?** Vague remits produce
#    thrashing. The `Routing` schema's field descriptions are the fix.

# %% [markdown]
# ## Try it yourself
#
# 1. **Add an escalation agent** that any specialist can hand off to, and make it
#    the only one allowed to end the conversation.
# 2. **Force a ping-pong loop** by removing the `visited` guard from the network
#    graph, then fix it two different ways.
# 3. **Hierarchical.** Put the three specialists under a "support" supervisor and
#    add a second "sales" supervisor under a top-level coordinator.
# 4. **Measure accuracy, not just cost.** Build 10 questions with known correct
#    desks and compare single-agent against supervisor routing accuracy.

# %% [markdown]
# ## Recap
#
# | Concept | Takeaway |
# |---|---|
# | Supervisor | Central control, predictable, one extra call per hop - **the default** |
# | Network | Agents hand off directly; cheaper, less predictable |
# | Hierarchical | Supervisors of supervisors; for more than ~6 specialists |
# | `Command(goto=, update=)` | A node that both writes state and chooses the next agent |
# | `destinations=` | Keeps the diagram accurate for `Command` nodes |
# | Handoff tools | `InjectedToolCallId` + `Command(graph=Command.PARENT)` |
# | Shared state | A merged scratchpad beats re-deriving facts from prose |
# | Hop caps and `visited` | Mandatory - ping-pong is the characteristic failure |
# | Cost | 3-5x a single agent; measure before committing |
# | When to split | Wrong tool from a long list, or a bloated system prompt - not aesthetics |
#
# ### Structured output from a subagent
#
# Only the subagent's **final message** returns to the parent - that is the
# isolation guarantee. If the parent needs typed data, give the subagent a
# `response_format=` (notebook 24) or have it write a JSON file the parent
# reads. Do not try to scrape intermediate tool calls out of the child's
# trajectory; they are not part of the contract.
#
# ## Next
#
# -> [42_custom_stream_channels.ipynb](42_custom_stream_channels.ipynb)
