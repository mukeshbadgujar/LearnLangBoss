# %% [markdown]
# # 16a - Code Execution Tools and Sandboxing
#
# | | |
# |---|---|
# | **Level** | Intermediate |
# | **Time** | 45 minutes |
# | **Prerequisites** | `16_tools_builtin_custom_toolkits` |
# | **Checklist ID** | `16a_code_execution_and_sandboxing` |
# | **Sourced from** | [emarco177/langchain-course `project/code-interpreter`](https://github.com/emarco177/langchain-course/tree/project/code-interpreter) |
#
# ## Why this matters
#
# Notebook 16 names `PythonREPLTool` in a table and never runs it. That is the
# gap. Code interpreters are one of the three most common production agent
# patterns (alongside search and RAG) - and the one with the worst blast radius.
# An agent that can `os.system` on your laptop is a liability, not a feature.
#
# This lesson teaches three things the Eden Marco course shows in one happy-path
# script, then adds what that script skips:
#
# 1. A modern `create_agent` REPL (not the legacy `AgentExecutor` the course uses)
# 2. A CSV analysis agent that answers questions over a DataFrame
# 3. A router that picks between them
# 4. The sandboxing and allow-list rules that make the pattern shippable

# %%
from pathlib import Path
import sys

_root = Path.cwd()
while not (_root / "shared").is_dir() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from shared.notebook_setup import setup, require_package  # noqa: E402

ctx = setup("16a_code_execution_and_sandboxing")

# %%
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from shared.llm import get_chat_model

model = get_chat_model()

# %% [markdown]
# ## 1. The danger, stated plainly
#
# A Python REPL tool can:
#
# - read any file the process can read (`.env`, SSH keys, customer data)
# - write anywhere the process can write
# - make network calls
# - run forever
#
# That is why every serious deployment either **does not ship a REPL**, or
# runs it inside a sandbox (Docker, gVisor, Firecracker, Deno, Pyodide). The
# lesson below starts with the unsafe local version so you understand the
# surface, then shows the patterns that shrink it.

# %% [markdown]
# ## 2. A constrained calculator - safer than a REPL
#
# Before reaching for `exec`, ask whether a **narrow tool** is enough. Most
# "run some Python" requests are arithmetic or pandas. Narrow tools have a
# fixed schema and no code execution.

# %%
@tool
def calculate(expression: str) -> str:
    """Evaluate a numeric expression using only +, -, *, /, **, and parentheses.

    Args:
        expression: A pure arithmetic expression, e.g. '(3 + 5) * 12'.
    """
    allowed = set("0123456789.+-*/() %")
    if not set(expression) <= allowed:
        return "Rejected: only digits and + - * / ** ( ) are allowed."
    try:
        # No builtins, no names - pure arithmetic only.
        return str(eval(expression, {"__builtins__": {}}, {}))  # noqa: S307
    except Exception as exc:  # noqa: BLE001
        return f"Calculation error: {type(exc).__name__}: {exc}"


print(calculate.invoke({"expression": "(3 + 5) * 12"}))
print(calculate.invoke({"expression": "__import__('os').system('ls')"}))

# %% [markdown]
# That second call is the point. A REPL would have run it. A calculator cannot.

# %% [markdown]
# ## 3. The modern REPL agent (optional install)
#
# The Eden Marco course uses `AgentExecutor` + `create_react_agent` from
# `langchain.agents` - that path is legacy. The same capability today is
# `create_agent` plus `PythonREPLTool` from `langchain_experimental`.

# %%
if require_package("langchain_experimental", feature="PythonREPLTool",
                   pip="langchain-experimental"):
    from langchain_experimental.tools import PythonREPLTool

    repl = PythonREPLTool()
    # Soften the blast radius for a notebook demo: short timeout, no network.
    # Production: run inside a container with no host mounts and no network.
    repl_agent = create_agent(
        model,
        tools=[repl],
        system_prompt=(
            "You write and execute Python to answer questions. "
            "Prefer short scripts. Print the final answer. "
            "Never import os, subprocess, socket, pathlib, or shutil. "
            "Never read or write files outside /tmp. "
            "If you cannot answer safely, say so."
        ),
        middleware=[ModelCallLimitMiddleware(run_limit=6, exit_behavior="end")],
        name="python_repl_agent",
    )
    result = repl_agent.invoke({
        "messages": [{"role": "user",
                      "content": "Compute the 12th Fibonacci number and print only that number."}]
    })
    print(result["messages"][-1].content)
else:
    print("Install langchain-experimental to run the REPL demo, or stick with calculate().")

# %% [markdown]
# ### What the course does that we deliberately do not
#
# Their script asks the agent to *generate 15 QR codes and save them to the
# working directory*. That is a fine Udemy demo and a terrible production
# pattern: the agent writes arbitrary files wherever the process can write.
# If you need file output, write to a dedicated scratch directory and pass
# that path in; never the project root.

# %% [markdown]
# ## 4. CSV analysis without a REPL
#
# `create_csv_agent` from `langchain_experimental` wraps pandas. It is
# convenient and still dangerous (`allow_dangerous_code=True` is required).
# Prefer a **typed tool over a fixed DataFrame** - the model never sees
# `exec`, only your function.

# %%
import pandas as pd

from shared.sample_data import DATA_DIR

tickets = pd.read_csv(DATA_DIR / "support_tickets.csv")
print(tickets.head(3))
print(f"\n{len(tickets)} tickets, columns: {list(tickets.columns)}")

# %%
class TicketQuery(BaseModel):
    """A structured question over the support tickets table."""

    group_by: str = Field(description="Column to group by, e.g. 'priority' or 'status'")
    metric: str = Field(description="'count' or the name of a numeric column to sum")


@tool(args_schema=TicketQuery)
def analyse_tickets(group_by: str, metric: str) -> str:
    """Aggregate the support tickets table. No free-form Python."""
    if group_by not in tickets.columns:
        return f"Unknown column {group_by!r}. Available: {list(tickets.columns)}"
    if metric == "count":
        return tickets.groupby(group_by).size().to_string()
    if metric not in tickets.columns:
        return f"Unknown metric {metric!r}."
    return tickets.groupby(group_by)[metric].sum().to_string()


print(analyse_tickets.invoke({"group_by": "priority", "metric": "count"}))

# %% [markdown]
# ## 5. A router over specialised agents
#
# The course's "grand agent" is a ReAct agent whose tools are *other agents*.
# That pattern is still valid - wrap each specialist as a tool and let a thin
# router decide. Prefer this over one mega-agent with every tool.

# %%
@tool
def ask_calculator(question: str) -> str:
    """Answer a pure arithmetic question by evaluating an expression."""
    # Let the model extract the expression, then run our safe calculator.
    from langchain_core.prompts import ChatPromptTemplate

    extract = ChatPromptTemplate.from_template(
        "Extract the arithmetic expression from this question. "
        "Return ONLY the expression, nothing else.\n\n{question}"
    ) | model
    expression = extract.invoke({"question": question}).content.strip()
    return calculate.invoke({"expression": expression})


@tool
def ask_tickets(question: str) -> str:
    """Answer a question about the support tickets CSV."""
    # Tiny structured extractor - never free-form code.
    class Plan(BaseModel):
        group_by: str
        metric: str = "count"

    plan = model.with_structured_output(Plan).invoke(
        f"Map this question onto group_by and metric for the tickets table "
        f"with columns {list(tickets.columns)}.\n\nQuestion: {question}"
    )
    return analyse_tickets.invoke({"group_by": plan.group_by, "metric": plan.metric})


router = create_agent(
    model,
    tools=[ask_calculator, ask_tickets],
    system_prompt=(
        "You are a router. Use ask_calculator for arithmetic. "
        "Use ask_tickets for anything about support tickets. "
        "Never invent numbers - always call a tool."
    ),
    middleware=[ModelCallLimitMiddleware(run_limit=4, exit_behavior="end")],
)

for question in [
    "What is (18 + 6) / 3?",
    "How many tickets are there per priority?",
]:
    answer = router.invoke({"messages": [{"role": "user", "content": question}]})
    print(f"\nQ: {question}")
    print(f"A: {answer['messages'][-1].content}")

# %% [markdown]
# ## 6. Sandboxing checklist
#
# Before you ship any code-execution tool:
#
# | Control | Why |
# |---|---|
# | **Prefer a narrow tool** | Most requests never needed `exec` |
# | **Allow-list imports** | Block `os`, `subprocess`, `socket`, `ctypes` |
# | **No host filesystem** | Mount a scratch volume; never the project root |
# | **No network** | Outbound calls exfiltrate data |
# | **CPU / memory / time limits** | An infinite loop is a denial of service |
# | **Separate process or container** | Contain the blast radius |
# | **Audit log every snippet** | You will need to explain what ran |
# | **Human approval for writes** | Notebook 36's interrupt pattern |
#
# If you cannot tick most of these, do not ship a REPL. Ship the calculator.

# %% [markdown]
# ## Try it yourself
#
# 1. Add a `plot_tickets` tool that returns a matplotlib chart of ticket counts
#    by priority, saved under `ctx.artifact(...)` - never the repo root.
# 2. Wrap `calculate` in a human-approval interrupt before evaluating any
#    expression longer than 40 characters.
# 3. Deliberately try to make the REPL read `.env`. Confirm your allow-list
#    stops it. If it does not, do not proceed.
#
# ## Recap
#
# | Concept | Takeaway |
# |---|---|
# | Narrow tools first | A calculator beats a REPL for most "run some code" asks |
# | `PythonREPLTool` | Exists, is powerful, and is dangerous by default |
# | Typed DataFrame tools | Safer than `create_csv_agent` with `allow_dangerous_code` |
# | Router over specialists | Thin agent whose tools are other agents |
# | Sandbox or do not ship | No host FS, no network, allow-listed imports, time limits |
#
# ## Next
#
# -> [16b_mcp_servers_and_clients.ipynb](16b_mcp_servers_and_clients.ipynb)
