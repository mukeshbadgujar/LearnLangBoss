# %% [markdown]
# # 17 - Agents: Types, Execution and Failure Modes
#
# | | |
# |---|---|
# | **Level** | Intermediate to Advanced |
# | **Time** | 55 minutes |
# | **Prerequisites** | `16_tools_builtin_custom_toolkits` |
# | **Checklist ID** | `17_agents` |
#
# ## Why this matters
#
# A chain has a fixed path you decided at build time. An **agent** decides the
# path at runtime: which tool, how many times, when to stop.
#
# That flexibility is genuinely powerful and genuinely risky. An agent can loop
# forever, call an expensive tool forty times, or confidently answer from a tool
# result it misread. This notebook covers how agents work, how to control them,
# and - importantly - **when not to use one**.

# %%
from pathlib import Path
import sys

_root = Path.cwd()
while not (_root / "shared").is_dir() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from shared.notebook_setup import setup, require  # noqa: E402

ctx = setup("17_agents")

# %%
import sqlite3
from contextlib import closing

from langchain.tools import tool

from shared.llm import get_chat_model, get_embeddings
from shared.sample_data import ensure_all

assets = ensure_all()
DB_PATH = str(assets["sqlite"])
model = get_chat_model()


@tool
def get_employee(employee_id: str) -> str:
    """Look up an employee's department, level, location and leave balances by id (e.g. E-104)."""
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM employees WHERE employee_id = ?", (employee_id.strip().upper(),)).fetchone()
    if row is None:
        return f"No employee '{employee_id}'. Ids look like E-104. Use find_employees to browse by department."
    return (
        f"{row['name']} ({row['employee_id']}) - {row['department']}, level {row['level']}, {row['location']}. "
        f"Annual leave remaining: {row['annual_leave_balance']} days. Sick leave: {row['sick_leave_balance']} days."
    )


@tool
def find_employees(department: str) -> str:
    """List employees in a department: Engineering, Data Science, Product, Support or Leadership."""
    with closing(sqlite3.connect(DB_PATH)) as conn:
        rows = conn.execute(
            "SELECT employee_id, name, level FROM employees WHERE department = ?", (department,)
        ).fetchall()
    return "\n".join(f"{r[0]} {r[1]} ({r[2]})" for r in rows) or f"No employees in '{department}'."


@tool
def open_tickets(customer: str = "") -> str:
    """List open support tickets, optionally filtered by customer name."""
    query = "SELECT ticket_id, customer, priority, summary FROM tickets WHERE status='open'"
    params: tuple = ()
    if customer:
        query += " AND customer LIKE ?"
        params = (f"%{customer}%",)
    with closing(sqlite3.connect(DB_PATH)) as conn:
        rows = conn.execute(query, params).fetchall()
    return "\n".join(f"{r[0]} [{r[2]}] {r[1]}: {r[3]}" for r in rows) or "No open tickets."


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression such as '24 - 18' or '6000 * 1.18'."""
    import numexpr

    try:
        return str(numexpr.evaluate(expression.strip()).item())
    except Exception as exc:
        return f"Could not evaluate {expression!r}: {exc}"


from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=70)
policy_docs = [
    Document(chunk, metadata={"source": name})
    for name in ("leave_policy.txt", "company_handbook.md")
    for chunk in splitter.split_text(Path(ctx.data(name)).read_text(encoding="utf-8"))
]
policy_store = FAISS.from_documents(policy_docs, get_embeddings())


@tool
def search_policy(query: str) -> str:
    """Search Northwind HR policy and handbook: leave rules, expenses, remote work,
    notice periods, promotions, equipment and security."""
    hits = policy_store.similarity_search(query, k=3)
    return "\n\n".join(f"[{d.metadata['source']}] {d.page_content}" for d in hits) or "No policy found."


TOOLS = [get_employee, find_employees, open_tickets, calculator, search_policy]
print("tools:", [t.name for t in TOOLS])

# %% [markdown]
# ## 1. `create_agent`: the modern standard
#
# In LangChain v1 this replaces `AgentExecutor`, `initialize_agent` and
# `langgraph.prebuilt.create_react_agent`. It returns a **compiled LangGraph**,
# which is why it gets streaming, persistence and human-in-the-loop for free.

# %%
from langchain.agents import create_agent

agent = create_agent(
    model=model,
    tools=TOOLS,
    system_prompt=(
        "You are the Northwind Analytics Ops Assistant.\n"
        "Use tools to get facts - never guess employee data, ticket data or policy.\n"
        "For arithmetic, always use the calculator tool.\n"
        "Cite the tool you used. If a tool says no data exists, say so plainly."
    ),
)

result = agent.invoke({"messages": [{"role": "user", "content": "How many annual leave days does E-103 have left?"}]})
print(result["messages"][-1].content)

# %% [markdown]
# ## 2. Watching the loop
#
# The agent loop is: **model -> tools -> model -> ... -> answer**. Print every
# message to see it.

# %%
def trace_run(agent_graph, question: str, **invoke_kwargs) -> dict:
    """Run an agent and print each step of its reasoning."""
    outcome = agent_graph.invoke({"messages": [{"role": "user", "content": question}]}, **invoke_kwargs)
    print(f"QUESTION: {question}\n" + "-" * 78)
    for message in outcome["messages"]:
        kind = message.type
        if kind == "human":
            print(f"[user]      {message.content}")
        elif kind == "ai":
            if message.tool_calls:
                for call in message.tool_calls:
                    print(f"[model]     -> calls {call['name']}({call['args']})")
            if message.content:
                text = message.content if isinstance(message.content, str) else str(message.content)
                print(f"[model]     {text.strip()[:400]}")
        elif kind == "tool":
            print(f"[tool:{message.name}] {str(message.content).strip()[:180]}")
    print("-" * 78)
    return outcome


trace_run(agent, "Daniel Fernandes wants 10 days off in March. Is he eligible and what does policy require?")

# %% [markdown]
# Look at what happened: the agent decided it needed *both* an employee lookup and
# a policy search, combined them, and applied the handover rule. No branching
# logic was written for that - it planned it.

# %%
trace_run(agent, "Which Engineering employee has the most annual leave left, and how many days more than Karthik Rao?")

# %% [markdown]
# ## 3. Agent types, and where they came from
#
# You will see these names in older code and blog posts. They are all strategies
# for the same problem - getting a model to use tools - and history has settled on
# the last one.
#
# | Type | How | Status |
# |---|---|---|
# | **ReAct (prompt-based)** | Model writes `Thought:/Action:/Action Input:` text, you parse it | Legacy. Fragile parsing. Only for models without tool calling |
# | **OpenAI Functions** | Provider-native function calling, one call at a time | Superseded |
# | **OpenAI Tools** | Provider-native, supports parallel calls | Became the standard |
# | **XML Agent** | Model emits `<tool>...</tool>`, you parse XML | For older Claude models |
# | **`create_agent`** | Provider-native tool calling on LangGraph | **Use this** |
#
# ### What ReAct text parsing looked like

# %%
react_style = """Answer the question using these tools:

get_employee: Look up an employee by id.
calculator: Evaluate arithmetic.

Use exactly this format:

Question: the input question
Thought: your reasoning
Action: the tool name
Action Input: the tool input
Observation: the tool result
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: the answer

Question: How many leave days does E-103 have?
Thought:"""

raw = model.invoke(react_style)
print(raw.content[:500])

# %% [markdown]
# It works - and it breaks the moment the model writes `Action: get_employee tool`
# or forgets the `Action Input:` line. Every one of those is an
# `OutputParserException` in production. Native tool calling returns structured
# JSON from the provider, which is why it won.
#
# **When ReAct still matters:** small local models with no tool-calling support.
# The modern way to express it is
# `create_agent(model, tools, middleware=[LLMToolEmulator()])` or a prompt-based
# fallback you maintain yourself.

# %% [markdown]
# ### The legacy `AgentExecutor` lifecycle
#
# ```python
# # LangChain <= 0.2 - now in langchain-classic
# from langchain.agents import AgentExecutor, create_tool_calling_agent
#
# agent = create_tool_calling_agent(llm, tools, prompt)
# executor = AgentExecutor(
#     agent=agent,
#     tools=tools,
#     max_iterations=10,
#     max_execution_time=60,
#     early_stopping_method="force",
#     handle_parsing_errors=True,
#     return_intermediate_steps=True,
#     verbose=True,
# )
# executor.invoke({"input": "..."})
# ```
#
# The lifecycle was: plan -> `AgentAction` -> execute -> `AgentStep` -> repeat
# until `AgentFinish` or a limit. Its problems: no persistence, no pausing, no
# branching, and everything after the loop was opaque. `create_agent` keeps the
# loop and fixes all four - because it is a graph.

# %% [markdown]
# ## 4. Controlling the loop
#
# ### Limiting model calls

# %%
from langchain.agents.middleware import ModelCallLimitMiddleware

limited = create_agent(
    model=model,
    tools=TOOLS,
    system_prompt="You are the Ops Assistant. Use tools for all facts.",
    middleware=[ModelCallLimitMiddleware(run_limit=4, exit_behavior="end")],
)

outcome = trace_run(
    limited,
    "For every department, list each employee and their leave balance, then total them all.",
)

# %% [markdown]
# `exit_behavior="end"` stops gracefully and returns what it has;
# `"error"` raises. Choose `end` for user-facing systems and `error` for batch
# jobs where a partial answer is worse than a failure.

# %%
from langchain.agents.middleware import ToolCallLimitMiddleware

tool_capped = create_agent(
    model=model,
    tools=TOOLS,
    system_prompt="You are the Ops Assistant.",
    middleware=[ToolCallLimitMiddleware(tool_name="get_employee", run_limit=2, exit_behavior="continue")],
)
print([t.name for t in TOOLS], "- get_employee capped at 2 calls per run")

# %% [markdown]
# ### Retrying flaky tools

# %%
from langchain.agents.middleware import ToolRetryMiddleware

attempts = {"n": 0}


@tool
def flaky_service(query: str) -> str:
    """Query the (unreliable) upstream billing service."""
    attempts["n"] += 1
    if attempts["n"] < 3:
        raise ConnectionError("upstream timeout")
    return f"billing service responded to {query!r} on attempt {attempts['n']}"


retrying = create_agent(
    model=model,
    tools=[flaky_service],
    system_prompt="Use the billing service tool to answer.",
    middleware=[ToolRetryMiddleware(max_retries=3, initial_delay=0.1, backoff_factor=1.5)],
)

outcome = retrying.invoke({"messages": [{"role": "user", "content": "Check billing status for Acme Retail."}]})
print(outcome["messages"][-1].content.strip()[:200])
print("total tool attempts:", attempts["n"])

# %% [markdown]
# ### Handling tool errors instead of crashing

# %%
from langchain.agents.middleware import ToolErrorMiddleware


@tool
def always_broken(anything: str) -> str:
    """A tool that always fails - to demonstrate error handling."""
    raise RuntimeError("database connection refused")


tolerant = create_agent(
    model=model,
    tools=[always_broken, search_policy],
    system_prompt="Answer the user. If a tool fails, explain what you could not do and try another route.",
    middleware=[ToolErrorMiddleware()],
)

outcome = tolerant.invoke({
    "messages": [{"role": "user", "content": "Use always_broken to check the leave carry-forward rule."}]
})
print(outcome["messages"][-1].content.strip()[:400])

# %% [markdown]
# ### Falling back to another model

# %%
from langchain.agents.middleware import ModelFallbackMiddleware

from shared.llm import available_providers, get_chat_model as _model

providers = available_providers()
if len(providers) > 1:
    resilient = create_agent(
        model=model,
        tools=TOOLS,
        system_prompt="You are the Ops Assistant.",
        middleware=[ModelFallbackMiddleware(_model(provider=providers[1]))],
    )
    print(f"primary={providers[0]}, fallback={providers[1]}")
    print(resilient.invoke({"messages": [{"role": "user", "content": "How many open tickets are there?"}]})["messages"][-1].content.strip()[:200])
else:
    print("[skipped] add a second provider key to .env to demo ModelFallbackMiddleware")

# %% [markdown]
# ## 5. Structured output from an agent
#
# An agent that returns prose is hard to integrate. `response_format` makes the
# final answer a validated object - generated in the main loop, so no extra call.

# %%
from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel, Field


class EligibilityVerdict(BaseModel):
    """Structured decision on a leave request."""

    employee_name: str = Field(description="Full name of the employee")
    days_requested: int = Field(description="Number of days requested")
    days_available: int = Field(description="Their current annual leave balance")
    eligible: bool = Field(description="Whether the request can be approved")
    requirements: list[str] = Field(description="Policy requirements that apply, e.g. handover note")
    reason: str = Field(description="One-sentence justification")


structured_agent = create_agent(
    model=model,
    tools=TOOLS,
    system_prompt=(
        "You are the Northwind leave assessor. Look up the employee's balance and the "
        "relevant policy before deciding. Never guess."
    ),
    response_format=ToolStrategy(EligibilityVerdict),
)

outcome = structured_agent.invoke({
    "messages": [{"role": "user", "content": "Can E-104 take 10 days of annual leave starting 2 March 2026?"}]
})

verdict = outcome["structured_response"]
print(type(verdict).__name__)
print(f"  employee    : {verdict.employee_name}")
print(f"  requested   : {verdict.days_requested} days (available: {verdict.days_available})")
print(f"  eligible    : {verdict.eligible}")
print(f"  requirements: {verdict.requirements}")
print(f"  reason      : {verdict.reason}")

# %% [markdown]
# `ToolStrategy` uses tool calling to produce the object; `ProviderStrategy` uses
# the provider's native JSON mode where available. Notebook 24 compares them.

# %% [markdown]
# ## 6. Streaming agent progress
#
# Users will not wait 20 seconds in silence. Stream the steps.

# %%
question = "Compare open ticket counts for Acme Retail and Everline Bank, then tell me which needs attention first."

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": question}]},
    stream_mode="updates",
):
    for node_name, update in chunk.items():
        for message in update.get("messages", []):
            if message.type == "ai" and message.tool_calls:
                print(f"  [{node_name}] calling {[c['name'] for c in message.tool_calls]}")
            elif message.type == "tool":
                print(f"  [tool] {message.name} -> {str(message.content)[:80]}")
            elif message.type == "ai" and message.content:
                print(f"  [answer] {str(message.content)[:300]}")

# %%
# Token-level streaming of just the final answer.
print("\ntoken stream: ", end="")
for token, metadata in agent.stream(
    {"messages": [{"role": "user", "content": "In one sentence, what is the notice period for L4?"}]},
    stream_mode="messages",
):
    if token.content and metadata.get("langgraph_node") == "model":
        print(token.content, end="", flush=True)
print()

# %% [markdown]
# ## 7. Memory across turns
#
# Same pattern as notebook 06: a checkpointer plus a `thread_id`.

# %%
from langgraph.checkpoint.memory import InMemorySaver

remembering = create_agent(
    model=model,
    tools=TOOLS,
    system_prompt="You are the Northwind Ops Assistant. Be concise.",
    checkpointer=InMemorySaver(),
)

thread = {"configurable": {"thread_id": "ops-session-1"}}

for question in [
    "Who is E-106?",
    "How much annual leave do they have left?",
    "Is that more or less than their manager?",
]:
    answer = remembering.invoke({"messages": [{"role": "user", "content": question}]}, thread)
    print(f"Q: {question}\nA: {answer['messages'][-1].content.strip()[:220]}\n")

# %% [markdown]
# Turn 3 required the agent to remember who E-106 is, look up their manager id,
# and compare - across three separate invocations.

# %% [markdown]
# ## 8. Failure modes you will actually hit

# %%
# (a) Hallucinated tool arguments
trace_run(agent, "What is the leave balance for employee number 42?")

# %%
# (b) The agent answering from parametric knowledge instead of the tool
loose = create_agent(
    model=model,
    tools=TOOLS,
    system_prompt="You are a helpful assistant.",   # no instruction to use tools
)
outcome = loose.invoke({"messages": [{"role": "user", "content": "What is Northwind's notice period for L4 engineers?"}]})
used_tools = any(m.type == "tool" for m in outcome["messages"])
print(f"tools used: {used_tools}")
print("answer:", outcome["messages"][-1].content.strip()[:250])

# %% [markdown]
# With a weak system prompt the agent may answer plausibly and completely wrongly,
# with no tool call at all. **The system prompt is a control surface, not
# decoration.** Compare with our strict prompt:

# %%
outcome = agent.invoke({"messages": [{"role": "user", "content": "What is Northwind's notice period for L4 engineers?"}]})
print("tools used:", any(m.type == "tool" for m in outcome["messages"]))
print("answer:", outcome["messages"][-1].content.strip()[:250])

# %% [markdown]
# ### Failure mode table
#
# | Failure | Cause | Fix |
# |---|---|---|
# | Infinite loop | Tool keeps returning something unsatisfying | `ModelCallLimitMiddleware` |
# | Wrong tool chosen | Vague descriptions, too many tools | Rewrite docstrings; reduce tool count |
# | Bad arguments | No validation | Pydantic `args_schema` with constraints |
# | Answers without tools | Weak system prompt | Explicit "never guess, always use tools" |
# | Crash on tool error | Tool raises | Return error strings; `ToolErrorMiddleware` |
# | Runaway cost | No budget | Call limits + `max_tokens` + tracing |
# | Misreads tool output | Output is verbose or ambiguous | Return compact, labelled strings |

# %% [markdown]
# ## 9. When **not** to use an agent
#
# Agents are slower, pricier and less predictable than chains. Use one only when
# the *sequence of steps genuinely cannot be known in advance*.
#
# | Task | Better choice |
# |---|---|
# | Always: retrieve then answer | LCEL RAG chain (notebook 14) |
# | Classify into 5 categories | Single call + `with_structured_output` |
# | Fixed 3-step pipeline | LCEL chain (notebook 8) |
# | Route to one of 4 handlers | `RunnableBranch` (notebook 25) |
# | Variable steps, tool-dependent | **Agent** |
# | Needs approval mid-run, or must resume after a crash | **LangGraph graph** (Track 05+) |
#
# A useful test: if you can draw the flowchart, build the flowchart. Agents are
# for when you cannot.

# %%
import time

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

fixed_chain = (
    {"context": (lambda q: search_policy.invoke({"query": q})), "question": RunnablePassthrough()}
    | ChatPromptTemplate.from_template("Answer from the context only.\n\n{context}\n\nQ: {question}")
    | model
    | StrOutputParser()
)

question = "What is the notice period for L4 engineers?"

start = time.perf_counter()
chain_answer = fixed_chain.invoke(question)
chain_time = time.perf_counter() - start

start = time.perf_counter()
agent_out = agent.invoke({"messages": [{"role": "user", "content": question}]})
agent_time = time.perf_counter() - start

print(f"chain: {chain_time:.2f}s, 1 LLM call")
print(f"agent: {agent_time:.2f}s, {sum(1 for m in agent_out['messages'] if m.type == 'ai')} LLM calls")
print(f"\nsame answer? both mention 90 days: "
      f"{'90' in chain_answer} / {'90' in agent_out['messages'][-1].content}")

# %% [markdown]
# For a question whose path is fixed, the chain is faster and cheaper for an
# identical answer. Reach for the agent when the path varies.

# %% [markdown]
# ## Try it yourself
#
# 1. **Force a loop.** Write a tool that always returns "partial data, try again"
#    and run it without a call limit (interrupt the kernel when you have seen
#    enough). Then add `ModelCallLimitMiddleware` and confirm the stop.
# 2. **Improve tool selection.** Add two deliberately similar tools
#    (`search_policy` and `lookup_handbook`) and measure how often the agent picks
#    the right one. Rewrite the descriptions until it is reliable.
# 3. **Structured triage.** Build an agent with `response_format` returning
#    `{ticket_id, category, priority, suggested_owner, needs_escalation}` and run
#    it over the open tickets.
# 4. **Justify the agent.** Take one of your real work tasks and write down
#    whether a chain would do. Only build the agent if the answer is no.

# %% [markdown]
# ## Recap
#
# | Concept | Takeaway |
# |---|---|
# | `create_agent` | The v1 standard; returns a compiled LangGraph |
# | Agent loop | model -> tools -> model -> ... -> answer |
# | ReAct / Functions / XML agents | Historical strategies; native tool calling won |
# | `AgentExecutor` | Legacy; no persistence, pausing or branching |
# | `ModelCallLimitMiddleware` | The loop guard you should always set |
# | `ToolRetryMiddleware` / `ToolErrorMiddleware` | Survive flaky and failing tools |
# | `response_format=ToolStrategy(Schema)` | Validated final answer, no extra call |
# | `checkpointer` + `thread_id` | Multi-turn memory |
# | System prompt | A control surface - weak prompts cause silent hallucination |
# | When not to agent | If you can draw the flowchart, build the flowchart |
#
# ### See also
#
# - [17a - The Agent Loop From Scratch](17a_agent_loop_from_scratch.ipynb) rebuilds
#   this loop at three abstraction levels (bind_tools, raw schemas, raw ReAct
#   text) so you can see what `create_agent` is doing for you. It also shows the
#   cheapest debugger: a `BaseCallbackHandler` that prints the prompt the model
#   is about to see.
#
# ## Next
#
# -> [17a_agent_loop_from_scratch.ipynb](17a_agent_loop_from_scratch.ipynb)
