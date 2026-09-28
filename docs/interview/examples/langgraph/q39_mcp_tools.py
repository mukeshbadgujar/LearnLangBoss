"""Q39. LangGraph and MCP tool servers.

THE PROBLEM
The bookstore wants shipping lookup and return-policy tools hosted remotely so
the support graph does not hard-code every vendor API.

WHAT WE ARE GOING TO SOLVE
Show the real shape of tools you would pass to an agent, and how MCP adapters
would load them - without requiring a running MCP server.

WHAT THIS EXAMPLE IS ABOUT
Order A100 support desk: a local @tool stands in for a remote MCP shipping tool,
then a tiny graph calls that tool from a plain Python node.

WHAT IT SOLVES
You see how MCP fits: adapters turn remote MCP tools into LangChain tools; the
graph/agent only sees a normal tool list.

KEYWORDS
- MCP (Model Context Protocol): a standard for exposing tools and resources from a server to an AI client.
- langchain-mcp-adapters: optional package that loads MCP server tools as LangChain tools (not installed here).
- MultiServerMCPClient: the client class you would use to connect to one or more MCP servers.
- Tool: a callable the agent or node can invoke with a name, args schema, and return value.
"""

from typing import TypedDict

from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph

# If langchain_mcp_adapters *were* installed, the real wiring looks like:
#
#   from langchain_mcp_adapters.client import MultiServerMCPClient
#   client = MultiServerMCPClient({
#       "bookstore": {
#           "command": "python",
#           "args": ["mcp_bookstore_server.py"],
#           "transport": "stdio",
#       }
#   })
#   mcp_tools = await client.get_tools()  # list[BaseTool]
#   # then: create_react_agent(model, mcp_tools) or bind_tools(mcp_tools)
#
# We do NOT import MultiServerMCPClient here - the package is not installed.


@tool
def lookup_order_shipping(order_id: str) -> str:
    """LOCAL STAND-IN for a remote MCP tool named lookup_order_shipping.

    In production this would live on an MCP server; the agent would call it
    the same way after MultiServerMCPClient.get_tools().
    """
    if order_id == "A100":
        return "Order A100 The Little Prince: shipped, $18"
    return f"Unknown order {order_id}"


# Tool list you would pass to an agent - same shape whether local or MCP-loaded.
BOOKSTORE_TOOLS = [lookup_order_shipping]


class TicketState(TypedDict):
    order_id: str
    answer: str


def call_shipping_tool(state: TicketState) -> dict:
    # Plain node calling a tool - same as an agent tool step, without an LLM.
    text = lookup_order_shipping.invoke({"order_id": state["order_id"]})
    return {"answer": text}


def build_graph():
    g = StateGraph(TicketState)
    g.add_node("shipping", call_shipping_tool)
    g.add_edge(START, "shipping")
    g.add_edge("shipping", END)
    return g.compile()


def main() -> None:
    print("Tools ready for an agent:", [t.name for t in BOOKSTORE_TOOLS])
    out = build_graph().invoke({"order_id": "A100", "answer": ""})
    print(out["answer"])
    print("MCP note: reconnect the adapter to swap the remote implementation;")
    print("the graph keeps calling tools by name.")


if __name__ == "__main__":
    print(__doc__)
    main()
