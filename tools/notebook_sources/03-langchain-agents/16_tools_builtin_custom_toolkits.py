# %% [markdown]
# # 16 - Tools: Built-in, Custom and Toolkits
#
# | | |
# |---|---|
# | **Level** | Intermediate |
# | **Time** | 50 minutes |
# | **Prerequisites** | `15_advanced_rag` |
# | **Checklist ID** | `16_tools_builtin_custom_toolkits` |
#
# ## Why this matters
#
# A model can only produce text. A **tool** is how it reaches the world: query a
# database, call your API, do arithmetic it would otherwise get wrong.
#
# The critical insight, and the one that trips everyone up: **the model never
# executes anything**. It emits a structured request - "call `get_leave_balance`
# with `employee_id='E-104'`" - and *your code* decides whether and how to run it.
# That separation is your entire security boundary.
#
# Running example for this track: an **Ops Assistant** that looks up employees and
# tickets, does calculations, and searches policy documents.

# %%
from pathlib import Path
import sys

_root = Path.cwd()
while not (_root / "shared").is_dir() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from shared.notebook_setup import setup, require  # noqa: E402

ctx = setup("16_tools_builtin_custom_toolkits")

# %%
from shared.llm import get_chat_model
from shared.sample_data import ensure_all

assets = ensure_all()
model = get_chat_model()
print("sqlite:", assets["sqlite"].name)

# %% [markdown]
# ## 1. Your first tool with `@tool`

# %%
from langchain.tools import tool


@tool
def working_days_between(start_date: str, end_date: str) -> int:
    """Count working days (Mon-Fri) between two ISO dates, inclusive.

    Args:
        start_date: Start date in YYYY-MM-DD format.
        end_date: End date in YYYY-MM-DD format.
    """
    from datetime import date, timedelta

    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    days = 0
    current = start
    while current <= end:
        if current.weekday() < 5:
            days += 1
        current += timedelta(days=1)
    return days


print("name        :", working_days_between.name)
print("description :", working_days_between.description)
print("args schema :", working_days_between.args)
print("\ndirect call :", working_days_between.invoke({"start_date": "2026-03-02", "end_date": "2026-03-13"}))

# %% [markdown]
# ### The docstring is not documentation - it is the prompt
#
# The model chooses tools by reading `name`, `description` and the argument
# schema. That text is the *only* thing it knows about your tool. A vague
# docstring is the number one cause of "the agent didn't use my tool" and
# "the agent called it with nonsense arguments".
#
# | Bad | Good |
# |---|---|
# | `"""Gets data."""` | `"""Look up an employee's remaining annual leave balance by employee id."""` |
# | `def f(x)` | `def get_leave_balance(employee_id: str)` |
# | no type hints | full type hints - they become the JSON schema |

# %% [markdown]
# ## 2. Precise schemas with Pydantic
#
# Type hints get you a schema. Pydantic gets you a *validated, described* schema -
# and validation happens before your code runs.

# %%
from typing import Literal

from pydantic import BaseModel, Field


class LeaveRequestInput(BaseModel):
    """Arguments for submitting a leave request."""

    employee_id: str = Field(description="Employee id such as E-104")
    leave_type: Literal["annual", "sick", "casual", "parental", "bereavement"] = Field(
        description="Category of leave being requested"
    )
    start_date: str = Field(description="First day of leave, YYYY-MM-DD")
    days: int = Field(description="Number of calendar days requested", ge=1, le=60)
    reason: str = Field(default="", description="Optional note for the approver")


@tool("submit_leave_request", args_schema=LeaveRequestInput)
def submit_leave_request(
    employee_id: str, leave_type: str, start_date: str, days: int, reason: str = ""
) -> str:
    """Submit a leave request for approval. Returns the created request id."""
    # In a real system this would POST to the HR service.
    return (
        f"Created request LR-NEW for {employee_id}: {days} day(s) of {leave_type} "
        f"leave from {start_date}. Status: pending manager approval."
    )


import json

print(json.dumps(submit_leave_request.args, indent=2))

# %%
# Validation fires before your function body executes.
from pydantic import ValidationError

try:
    submit_leave_request.invoke({
        "employee_id": "E-104",
        "leave_type": "vacation",   # not in the Literal
        "start_date": "2026-03-02",
        "days": 5,
    })
except (ValidationError, Exception) as exc:
    print(f"rejected: {type(exc).__name__}")
    print(str(exc)[:300])

# %% [markdown]
# `ge=1, le=60` and the `Literal` mean a hallucinated argument is caught at the
# boundary rather than corrupting your HR system.

# %% [markdown]
# ## 3. Tools that touch real data

# %%
import sqlite3
from contextlib import closing

DB_PATH = str(assets["sqlite"])


@tool
def get_employee(employee_id: str) -> str:
    """Look up an employee's department, level, location and leave balances by employee id."""
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM employees WHERE employee_id = ?", (employee_id.upper(),)
        ).fetchone()
    if row is None:
        return f"No employee found with id {employee_id}."
    return (
        f"{row['name']} ({row['employee_id']}) - {row['department']}, level {row['level']}, "
        f"{row['location']}. Joined {row['joined_on']}. "
        f"Annual leave remaining: {row['annual_leave_balance']} days. "
        f"Sick leave remaining: {row['sick_leave_balance']} days."
    )


@tool
def find_employees(department: str) -> str:
    """List all employees in a department. Valid departments: Engineering, Data Science, Product, Support, Leadership."""
    with closing(sqlite3.connect(DB_PATH)) as conn:
        rows = conn.execute(
            "SELECT employee_id, name, level FROM employees WHERE department = ? ORDER BY level DESC",
            (department,),
        ).fetchall()
    if not rows:
        return f"No employees found in department '{department}'."
    return "\n".join(f"{r[0]} {r[1]} ({r[2]})" for r in rows)


@tool
def open_tickets(customer: str = "") -> str:
    """List open support tickets, optionally filtered to one customer."""
    query = "SELECT ticket_id, customer, priority, summary FROM tickets WHERE status = 'open'"
    params: tuple = ()
    if customer:
        query += " AND customer LIKE ?"
        params = (f"%{customer}%",)
    with closing(sqlite3.connect(DB_PATH)) as conn:
        rows = conn.execute(query, params).fetchall()
    if not rows:
        return "No open tickets found."
    return "\n".join(f"{r[0]} [{r[2]}] {r[1]}: {r[3]}" for r in rows)


print(get_employee.invoke({"employee_id": "E-104"}))
print()
print(open_tickets.invoke({"customer": "Acme"}))

# %% [markdown]
# ## 4. Binding tools to a model
#
# `bind_tools` sends the schemas to the provider. The model then answers *or*
# asks for a tool call.

# %%
tools = [working_days_between, get_employee, find_employees, open_tickets, submit_leave_request]
model_with_tools = model.bind_tools(tools)

response = model_with_tools.invoke("How many annual leave days does employee E-103 have left?")

print("content     :", repr(response.content))
print("tool_calls  :", response.tool_calls)

# %% [markdown]
# `content` is empty and `tool_calls` is populated. The model did **not** run
# anything - it asked you to. Here is the full manual loop:

# %%
from langchain.messages import HumanMessage, ToolMessage

tools_by_name = {t.name: t for t in tools}
conversation = [HumanMessage("How many annual leave days does employee E-103 have left?")]

ai_message = model_with_tools.invoke(conversation)
conversation.append(ai_message)

for call in ai_message.tool_calls:
    print(f"model requested: {call['name']}({call['args']})")
    result = tools_by_name[call["name"]].invoke(call["args"])
    print(f"we executed it : {result[:110]}")
    conversation.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

final = model_with_tools.invoke(conversation)
print("\nfinal answer   :", final.content.strip())

# %% [markdown]
# That four-step cycle - **model asks -> you execute -> you return a `ToolMessage`
# -> model answers** - is the agent loop. Notebook 17 automates it; every agent
# framework in existence is a wrapper around this.

# %% [markdown]
# ### Parallel tool calls

# %%
multi = model_with_tools.invoke(
    "Compare the leave balances of E-101 and E-105, and list open tickets for Everline Bank."
)
print(f"{len(multi.tool_calls)} tool calls requested:")
for call in multi.tool_calls:
    print(f"   {call['name']}({call['args']})")

# %% [markdown]
# ## 5. Built-in tools
#
# ### Calculator
#
# LLMs are unreliable at arithmetic. Give them a calculator - but never `eval()`.

# %%
@tool
def calculator(expression: str) -> str:
    """Evaluate a arithmetic expression, e.g. '24 * 0.6 + 12'. Supports + - * / ** and parentheses."""
    import numexpr

    try:
        return str(numexpr.evaluate(expression.strip()).item())
    except Exception as exc:
        return f"Could not evaluate {expression!r}: {exc}"


try:
    print("62 * 1.18 / 3 =", calculator.invoke({"expression": "62 * 1.18 / 3"}))
except ImportError:
    print("[skipped] pip install numexpr for the calculator tool")
    print("Never use eval() here - a model-supplied string is untrusted input.")

# %% [markdown]
# ### Wikipedia

# %%
try:
    from langchain_community.tools import WikipediaQueryRun
    from langchain_community.utilities import WikipediaAPIWrapper

    wikipedia = WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
    )
    print(wikipedia.invoke("retrieval augmented generation")[:400])
except Exception as exc:
    print(f"[skipped] Wikipedia needs network + `pip install wikipedia` ({type(exc).__name__})")

# %% [markdown]
# ### Web search

# %%
try:
    from langchain_community.tools import DuckDuckGoSearchRun

    search = DuckDuckGoSearchRun()
    print(search.invoke("LangGraph checkpointer")[:400])
except Exception as exc:
    print(f"[skipped] DuckDuckGo needs network + `pip install duckduckgo-search` ({type(exc).__name__})")

# %%
if require("TAVILY_API_KEY", feature="Tavily search (built for LLM agents)"):
    from langchain_tavily import TavilySearch

    tavily = TavilySearch(max_results=3)
    print(tavily.invoke({"query": "LangGraph human in the loop pattern"}))

# %% [markdown]
# ## 6. Converting a retriever into a tool
#
# This is how RAG joins an agent: the agent *decides* whether to search documents,
# instead of always searching.

# %%
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from shared.llm import get_embeddings

policy = Path(ctx.data("leave_policy.txt")).read_text(encoding="utf-8")
handbook = Path(ctx.data("company_handbook.md")).read_text(encoding="utf-8")
splitter = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=70)
policy_docs = [
    Document(chunk, metadata={"source": name})
    for name, text in [("leave_policy.txt", policy), ("company_handbook.md", handbook)]
    for chunk in splitter.split_text(text)
]
policy_store = FAISS.from_documents(policy_docs, get_embeddings())


@tool
def search_policy(query: str) -> str:
    """Search Northwind HR policy and the employee handbook. Use for questions about
    leave rules, expenses, remote work, notice periods, promotions and equipment."""
    hits = policy_store.similarity_search(query, k=3)
    if not hits:
        return "No relevant policy found."
    return "\n\n".join(f"[{d.metadata['source']}] {d.page_content}" for d in hits)


print(search_policy.invoke({"query": "carry forward unused leave"})[:400])

# %% [markdown]
# There is also a helper that does this for you:
#
# ```python
# from langchain_classic.tools.retriever import create_retriever_tool
#
# tool = create_retriever_tool(retriever, name="search_policy",
#                              description="Search Northwind HR policy...")
# ```

# %% [markdown]
# ## 7. Error handling inside tools
#
# A tool that raises can kill the agent loop. A tool that returns an *informative
# error string* lets the model correct itself - which it is surprisingly good at.

# %%
@tool
def fragile_lookup(employee_id: str) -> str:
    """Look up an employee. Raises on a bad id - the wrong way to do this."""
    with closing(sqlite3.connect(DB_PATH)) as conn:
        row = conn.execute("SELECT name FROM employees WHERE employee_id = ?", (employee_id,)).fetchone()
    if row is None:
        raise ValueError(f"No such employee: {employee_id}")
    return row[0]


@tool
def robust_lookup(employee_id: str) -> str:
    """Look up an employee by id. Returns guidance when the id is not found."""
    normalised = employee_id.strip().upper()
    with closing(sqlite3.connect(DB_PATH)) as conn:
        row = conn.execute("SELECT name FROM employees WHERE employee_id = ?", (normalised,)).fetchone()
        if row is None:
            known = [r[0] for r in conn.execute("SELECT employee_id FROM employees LIMIT 5")]
            return (
                f"No employee with id '{employee_id}'. Ids look like 'E-104'. "
                f"Examples that exist: {', '.join(known)}. "
                "If you only have a name, call find_employees with their department instead."
            )
    return row[0]


try:
    fragile_lookup.invoke({"employee_id": "104"})
except Exception as exc:
    print("fragile ->", f"{type(exc).__name__}: {exc}")

print("robust  ->", robust_lookup.invoke({"employee_id": "104"}))

# %% [markdown]
# The second message is *actionable*. The model reads it and retries with `E-104`.
# **Write tool errors for the model to read, not for a log file.**

# %% [markdown]
# ## 8. Toolkits
#
# A toolkit is a bundle of related tools. `SQLDatabaseToolkit` is the most useful:
# it gives an agent the ability to list tables, inspect schemas, and run queries.

# %%
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
sql_toolkit = SQLDatabaseToolkit(db=db, llm=model)

print("dialect:", db.dialect)
print("tables :", db.get_usable_table_names())
print("\ntools in the toolkit:")
for t in sql_toolkit.get_tools():
    print(f"  {t.name:24} {t.description.splitlines()[0][:90]}")

# %%
print(db.run("SELECT department, COUNT(*) FROM employees GROUP BY department"))

# %% [markdown]
# ### SQL tools are dangerous by default
#
# An LLM-generated `DELETE FROM employees` will execute happily. Mitigations, in
# order of importance:
#
# 1. **Connect with a read-only database user.** Nothing else is as reliable.
# 2. Restrict visible tables: `SQLDatabase.from_uri(uri, include_tables=[...])`
# 3. Add human approval for writes (notebook 36)
# 4. Validate the SQL before running it

# %%
@tool
def run_readonly_sql(query: str) -> str:
    """Run a read-only SELECT query against the Northwind HR database.
    Tables: employees(employee_id, name, department, level, location, manager, joined_on,
    annual_leave_balance, sick_leave_balance), tickets(ticket_id, customer, plan, category,
    priority, status, opened_on, summary, resolution_hours),
    leave_requests(request_id, employee_id, leave_type, start_date, days, status)."""
    cleaned = query.strip().rstrip(";")
    lowered = cleaned.lower()

    if not lowered.startswith(("select", "with")):
        return "Rejected: only SELECT queries are permitted."
    forbidden = ["insert", "update", "delete", "drop", "alter", "create", "attach", "pragma"]
    if any(f" {word} " in f" {lowered} " for word in forbidden):
        return f"Rejected: query contains a forbidden keyword. Allowed: SELECT only."
    if ";" in cleaned:
        return "Rejected: multiple statements are not allowed."

    try:
        with closing(sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)) as conn:
            rows = conn.execute(cleaned + " LIMIT 50").fetchall()
    except Exception as exc:
        return f"SQL error: {exc}. Check the table and column names in the tool description."

    return "\n".join(str(r) for r in rows) if rows else "No rows returned."


print(run_readonly_sql.invoke({"query": "SELECT name, annual_leave_balance FROM employees WHERE annual_leave_balance > 18"}))
print()
print(run_readonly_sql.invoke({"query": "DELETE FROM employees"}))

# %% [markdown]
# Note `mode=ro` in the connection URI - that is the defence that actually holds
# if the keyword filter is bypassed. Layer them.

# %% [markdown]
# ### Other toolkits
#
# | Toolkit | Gives the agent | Import |
# |---|---|---|
# | `SQLDatabaseToolkit` | list tables, schema, query, checker | `langchain_community.agent_toolkits` |
# | Pandas agent | run pandas on a DataFrame | `langchain_experimental.agents` |
# | `RequestsToolkit` | HTTP GET/POST/PATCH/DELETE | `langchain_community.agent_toolkits` |
# | OpenAPI toolkit | call any documented REST API | `langchain_community.agent_toolkits.openapi` |
# | `FileManagementToolkit` | read/write/list files in a sandbox dir | `langchain_community.agent_toolkits` |
# | MCP tools | any Model Context Protocol server | `langchain-mcp-adapters` |
#
# The pandas and Python REPL toolkits **execute arbitrary generated code**. Use
# them only in a sandbox or container you are willing to lose.

# %%
try:
    import pandas as pd

    tickets_df = pd.read_csv(ctx.data("support_tickets.csv"))

    @tool
    def ticket_stats(group_by: str) -> str:
        """Aggregate support tickets. group_by must be one of: category, priority, plan, customer, status."""
        allowed = {"category", "priority", "plan", "customer", "status"}
        if group_by not in allowed:
            return f"Invalid group_by '{group_by}'. Choose one of: {', '.join(sorted(allowed))}."
        counts = tickets_df.groupby(group_by).size().sort_values(ascending=False)
        return counts.to_string()

    print(ticket_stats.invoke({"group_by": "category"}))
except ImportError:
    print("[skipped] pandas not installed")

# %% [markdown]
# That is the safe alternative to a pandas agent: a **narrow tool** that exposes
# the specific analysis you need, with validated parameters, instead of arbitrary
# code execution.

# %% [markdown]
# ## 9. Tool design checklist
#
# 1. **Name** - a verb phrase: `get_employee`, not `employee_data`
# 2. **Description** - say *when* to use it, not just what it does
# 3. **Types** - full hints; use `Literal` for closed sets
# 4. **Validate** - Pydantic constraints catch hallucinated arguments
# 5. **Return strings the model can read** - not raw objects or stack traces
# 6. **Errors are guidance** - tell the model how to fix its call
# 7. **Small surface** - 5 focused tools beat 1 tool with a `mode` parameter
# 8. **Keep the tool count low** - accuracy drops noticeably past ~15 tools;
#    beyond that, route to sub-agents (notebook 41) or use `LLMToolSelectorMiddleware`
# 9. **Assume the arguments are adversarial** - they come from a model that can be
#    influenced by retrieved text

# %%
# Bad vs good, side by side.
@tool
def data(q: str) -> str:
    """Gets data."""
    return "..."


@tool
def get_ticket_by_id(ticket_id: str) -> str:
    """Fetch full details of one support ticket by its id (format: TCK-1234).
    Use when the user references a specific ticket number. For searching by
    customer or status, use open_tickets instead."""
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id.upper(),)).fetchone()
    if row is None:
        return f"No ticket {ticket_id}. Ids look like TCK-1004. Use open_tickets to list current ones."
    return " | ".join(f"{k}={row[k]}" for k in row.keys())


for t in (data, get_ticket_by_id):
    print(f"{t.name:20} args={list(t.args)}  desc={t.description[:70]!r}")

print("\n", get_ticket_by_id.invoke({"ticket_id": "TCK-1006"}))

# %% [markdown]
# ## Try it yourself
#
# 1. **Build `check_leave_eligibility(employee_id, days)`** that reads the balance
#    from SQLite, applies the 7-day-notice and 5-day-handover rules from the
#    policy, and returns an eligibility verdict with reasons.
# 2. **Break a tool on purpose.** Give `get_employee` the docstring `"""Gets
#    info."""`, bind it, and ask a question. Observe the model failing to select
#    it. Restore the docstring and watch it work.
# 3. **Defeat the SQL guard.** Try to get `run_readonly_sql` to mutate data. When
#    you cannot, explain which layer stopped you.
# 4. **Tool overload.** Bind 20 tools (duplicate some with slight variations) and
#    measure how often the model picks the right one compared to 5 tools.

# %% [markdown]
# ## Recap
#
# | Concept | Takeaway |
# |---|---|
# | `@tool` | Function + docstring + type hints becomes a callable schema |
# | Docstring | It is a prompt, not documentation - the model's only information |
# | `args_schema` | Pydantic validation rejects hallucinated arguments before execution |
# | `bind_tools` | Model returns `tool_calls`; **it never executes anything** |
# | `ToolMessage` | How you hand a result back, matched by `tool_call_id` |
# | Retriever as a tool | RAG becomes optional and agent-decided |
# | Error strings | Return actionable guidance, do not raise |
# | `SQLDatabaseToolkit` | Powerful and dangerous - read-only connection first |
# | Narrow tools | Safer than code-executing agents for most analysis |
#
# ## Next
#
# -> [16a_code_execution_and_sandboxing.ipynb](16a_code_execution_and_sandboxing.ipynb)
