# %% [markdown]
# # 16b - MCP: Model Context Protocol Servers and Clients
#
# | | |
# |---|---|
# | **Level** | Intermediate to Advanced |
# | **Time** | 50 minutes |
# | **Prerequisites** | `16_tools_builtin_custom_toolkits` |
# | **Checklist ID** | `16b_mcp_servers_and_clients` |
# | **Sourced from** | `LLG/mcplanggraph` (FastMCP + MultiServerMCPClient) |
#
# ## Why this matters
#
# Notebook 16 mentions MCP in one table row. MCP is how tools are shipped
# across teams now: one team writes a server, any agent framework consumes it
# as tools, without copying Python functions around.
#
# Two transports matter in practice:
#
# | Transport | When to use |
# |---|---|
# | **stdio** | Local process, one client, simple tools (math, file ops) |
# | **streamable-http** | Shared service, multiple clients, network-reachable tools |
#
# This lesson writes both kinds of server, then wires them into a LangGraph
# agent via `MultiServerMCPClient` - the same shape as the LLG demo, updated
# for `create_agent` instead of the older `create_react_agent`.

# %%
from pathlib import Path
import sys

_root = Path.cwd()
while not (_root / "shared").is_dir() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from shared.notebook_setup import setup, require_package  # noqa: E402

ctx = setup("16b_mcp_servers_and_clients")

# %% [markdown]
# ## 1. What MCP actually is
#
# MCP is a **protocol**, not a LangChain feature. A server advertises tools
# (and optionally resources and prompts). A client connects, lists them, and
# calls them. LangChain's `langchain-mcp-adapters` turns those tools into
# LangChain `BaseTool` objects so an agent can bind them like any other tool.
#
# ```
#   FastMCP server  --stdio or http-->  MultiServerMCPClient  -->  create_agent
# ```
#
# The value is **decoupling**: the weather team ships a weather server; your
# agent pulls it in without importing their code.

# %% [markdown]
# ## 2. Write a stdio server
#
# Save this as a real file so the client can spawn it as a subprocess. The
# LLG course puts `mathserver.py` next to the client; we write ours under
# `artifacts/` so the repo stays clean.

# %%
math_server = ctx.artifact("mcp", "math_server.py")
math_server.write_text('''\
"""A tiny MCP math server over stdio."""
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


if __name__ == "__main__":
    # stdio: the client spawns this process and talks over stdin/stdout.
    mcp.run(transport="stdio")
''', encoding="utf-8")
print(f"wrote {math_server}")

# %% [markdown]
# ## 3. Write a streamable-http server
#
# Same FastMCP API, different transport. This one listens on a port so any
# number of clients can connect.

# %%
weather_server = ctx.artifact("mcp", "weather_server.py")
weather_server.write_text('''\
"""A tiny MCP weather server over streamable HTTP."""
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather")


@mcp.tool()
async def get_weather(location: str) -> str:
    """Return a short weather summary for a location."""
    # Demo stub - replace with a real weather API in production.
    return f"It is 28 C and clear in {location}."


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
''', encoding="utf-8")
print(f"wrote {weather_server}")

# %% [markdown]
# ## 4. Connect as a client and bind tools to an agent
#
# `MultiServerMCPClient` takes a dict of server configs. Each entry picks a
# transport. The client returns LangChain tools you pass straight to
# `create_agent`.

# %%
if not require_package("langchain_mcp_adapters", "mcp",
                       feature="MCP client + server",
                       pip="langchain-mcp-adapters mcp"):
    pass
else:
    import asyncio

    from langchain.agents import create_agent
    from langchain.agents.middleware import ModelCallLimitMiddleware
    from langchain_mcp_adapters.client import MultiServerMCPClient

    from shared.llm import get_chat_model

    async def run_mcp_agent():
        client = MultiServerMCPClient({
            "math": {
                "command": sys.executable,
                "args": [str(math_server)],
                "transport": "stdio",
            },
            # Uncomment when weather_server.py is running in another terminal:
            # "weather": {
            #     "url": "http://localhost:8000/mcp",
            #     "transport": "streamable_http",
            # },
        })
        tools = await client.get_tools()
        print("tools from MCP:", [t.name for t in tools])

        agent = create_agent(
            get_chat_model(),
            tools=tools,
            system_prompt="Use the math tools for any arithmetic. Show your steps briefly.",
            middleware=[ModelCallLimitMiddleware(run_limit=6, exit_behavior="end")],
        )
        result = await agent.ainvoke({
            "messages": [{"role": "user", "content": "What is (3 + 5) times 12?"}]
        })
        print(result["messages"][-1].content)

    awaitable = run_mcp_agent()
    try:
        asyncio.get_running_loop()
        # Inside Jupyter an event loop is already running.
        import nest_asyncio

        nest_asyncio.apply()
        asyncio.get_event_loop().run_until_complete(awaitable)
    except RuntimeError:
        asyncio.run(awaitable)

# %% [markdown]
# ### Running the HTTP server
#
# In a second terminal:
#
# ```powershell
# python artifacts\mcp\weather_server.py
# ```
#
# Then uncomment the `weather` entry above. The agent will discover
# `get_weather` alongside `add` and `multiply` without any code change to the
# agent itself - that is the whole point of MCP.

# %% [markdown]
# ## 5. Design rules
#
# | Rule | Why |
# |---|---|
# | One concern per server | Math and weather are separate processes so they scale and fail independently |
# | Tool descriptions are prompts | The model only sees the docstring - write it like a prompt |
# | Prefer stdio for local demos | No ports, no CORS, no auth to configure |
# | Prefer HTTP for shared services | Multiple agents, remote hosts, real auth |
# | Never put secrets in tool output | MCP responses can be logged and traced |
# | Version your servers | Breaking a tool schema breaks every client overnight |
#
# ## 6. How this differs from `@tool`
#
# | | `@tool` in-process | MCP server |
# |---|---|---|
# | Deployment | Same process as the agent | Separate process or host |
# | Language | Must be Python (for LangChain) | Any language with an MCP SDK |
# | Sharing | Copy the function | Point the client at the server |
# | Isolation | Shares memory and credentials | Process boundary |
# | Latency | Function call | IPC or network hop |
#
# Use `@tool` for agent-private helpers. Use MCP when another team owns the
# capability, or when you need isolation.

# %% [markdown]
# ## Try it yourself
#
# 1. Add a `divide(a, b)` tool to the math server and confirm the agent picks
#    it up after a restart - no agent code change.
# 2. Start the weather server and enable the HTTP entry. Ask a mixed question
#    ("what is 7*8, and what is the weather in Pune?").
# 3. Point `MultiServerMCPClient` at a public MCP server (filesystem, GitHub)
#    and list its tools. Do not grant write access until you understand the
#    blast radius.
#
# ## Recap
#
# | Concept | Takeaway |
# |---|---|
# | MCP | Protocol for tooling across processes and languages |
# | FastMCP | Tiny Python server; `@mcp.tool()` is enough |
# | stdio vs streamable-http | Local subprocess vs shared network service |
# | `MultiServerMCPClient` | Discovers tools and returns LangChain `BaseTool`s |
# | `create_agent` | Consumes MCP tools like any other tool |
# | Decoupling | The weather team ships a server; you do not import their code |
#
# ## Next
#
# -> [17_agents.ipynb](17_agents.ipynb)
