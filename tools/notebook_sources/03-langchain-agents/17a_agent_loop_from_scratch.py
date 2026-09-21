# %% [markdown]
# # 17a - The Agent Loop From Scratch
#
# | | |
# |---|---|
# | **Level** | Intermediate to Advanced |
# | **Time** | 55 minutes |
# | **Prerequisites** | `17_agents` |
# | **Checklist ID** | `17a_agent_loop_from_scratch` |
# | **Sourced from** | [emarco177/langchain-course `project/agents-under-the-hood`](https://github.com/emarco177/langchain-course/tree/project/agents-under-the-hood), `project/ReAct-Algo` |
#
# ## Why this matters
#
# Notebook 17 teaches `create_agent`. That is the right API to ship. It is
# also a black box. Until you have written the loop yourself - three times, at
# three levels of abstraction - you cannot debug an agent that is looping,
# calling the wrong tool, or inventing arguments.
#
# Eden Marco's `agents-under-the-hood` branch does exactly that. We rebuild it
# here with our provider helper and a shopping scenario, then contrast it with
# `create_agent` so you see what the framework buys you.

# %%
from pathlib import Path
import sys

_root = Path.cwd()
while not (_root / "shared").is_dir() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from shared.notebook_setup import setup  # noqa: E402

ctx = setup("17a_agent_loop_from_scratch")

# %%
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from shared.llm import get_chat_model

model = get_chat_model()
MAX_ITERATIONS = 8

# %% [markdown]
# ## Shared tools
#
# Same catalogue for every implementation. Prices are fixed so you can predict
# the answer and catch hallucination.

# %%
@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalogue."""
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product.lower(), 0.0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price. Tiers: bronze, silver, gold."""
    rates = {"bronze": 0.05, "silver": 0.12, "gold": 0.23}
    rate = rates.get(discount_tier.lower(), 0.0)
    return round(float(price) * (1 - rate), 2)


TOOLS = [get_product_price, apply_discount]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}
QUESTION = "What is the price of a laptop after applying a gold discount?"

print(f"expected answer: {apply_discount.invoke({'price': 1299.99, 'discount_tier': 'gold'})}")

# %% [markdown]
# ## Level 1 - `bind_tools` loop (what `create_agent` wraps)
#
# The model returns structured `tool_calls`. You execute them, append
# `ToolMessage`s, and call the model again until it stops requesting tools.

# %%
SYSTEM = (
    "You are a shopping assistant.\n"
    "STRICT RULES:\n"
    "1. NEVER guess a price - call get_product_price first.\n"
    "2. Only call apply_discount with a price returned by get_product_price.\n"
    "3. NEVER compute discounts yourself.\n"
    "4. If no tier is specified, ask - do not assume one."
)


def run_bind_tools_loop(question: str) -> str:
    llm = model.bind_tools(TOOLS)
    messages: list[Any] = [SystemMessage(SYSTEM), HumanMessage(question)]

    for step in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- bind_tools iteration {step} ---")
        ai: AIMessage = llm.invoke(messages)
        messages.append(ai)

        if not ai.tool_calls:
            print(f"final: {ai.content}")
            return ai.content or ""

        # Force one tool per turn so the trajectory is readable.
        call = ai.tool_calls[0]
        print(f"  tool: {call['name']} {call['args']}")
        observation = TOOLS_BY_NAME[call["name"]].invoke(call["args"])
        print(f"  result: {observation}")
        messages.append(ToolMessage(str(observation), tool_call_id=call["id"]))

    return "ERROR: max iterations"


bind_answer = run_bind_tools_loop(QUESTION)

# %% [markdown]
# That loop is the entire agent. `create_agent` adds middleware, streaming,
# checkpointing and retries around the same structure.

# %% [markdown]
# ## Level 2 - raw provider tool schemas
#
# Some providers want the JSON schema handed over explicitly rather than via
# LangChain's `bind_tools`. The loop is identical; only the binding changes.
# Useful when you are wrapping a provider LangChain does not know yet.

# %%
def tool_to_openai_schema(t) -> dict:
    """Convert a LangChain tool into an OpenAI-style function schema."""
    return {
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description,
            "parameters": t.args_schema.model_json_schema() if t.args_schema else {
                "type": "object", "properties": {}
            },
        },
    }


schemas = [tool_to_openai_schema(t) for t in TOOLS]
print("schema for get_product_price:")
print(schemas[0]["function"]["parameters"])

# %% [markdown]
# With Groq / OpenAI / OpenRouter the schemas go through `bind_tools` under the
# hood anyway. The point of writing them out is to see that a "tool" is just a
# JSON schema plus a Python callable - nothing magical.

# %% [markdown]
# ## Level 3 - raw ReAct text + regex parsing
#
# The historical ReAct pattern: the model writes
# `Thought / Action / Action Input / Observation` as plain text, and you parse
# it with regex. This is fragile. It is also how every early agent library
# worked, and why native tool calling replaced it.

# %%
TOOL_BLURB = "\n".join(
    f"- {t.name}{t.args}: {t.description}" for t in TOOLS
)

REACT_PROMPT = f"""Answer the question using these tools:

{TOOL_BLURB}

Use this format exactly:

Thought: reason about the next step
Action: one of [{', '.join(TOOLS_BY_NAME)}]
Action Input: comma-separated argument values
Observation: (the runtime will fill this in)
... repeat Thought/Action/Action Input/Observation as needed ...
Thought: I now know the final answer
Final Answer: the answer

Question: {{question}}
Thought:"""


def run_react_text_loop(question: str) -> str:
    scratchpad = ""
    for step in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- raw ReAct iteration {step} ---")
        prompt = REACT_PROMPT.format(question=question) + scratchpad
        # Stop before the model invents its own Observation.
        output = model.invoke(prompt).content
        print(output[:400])

        final = re.search(r"Final Answer:\s*(.+)", output)
        if final:
            print(f"final: {final.group(1).strip()}")
            return final.group(1).strip()

        action = re.search(r"Action:\s*(.+)", output)
        action_input = re.search(r"Action Input:\s*(.+)", output)
        if not action or not action_input:
            print("  parse failure - model did not follow the format")
            break

        name = action.group(1).strip()
        raw_args = [a.strip().strip("'\"") for a in action_input.group(1).split(",")]
        # Positional args mapped onto the tool's parameter names.
        params = list(TOOLS_BY_NAME[name].args.keys())
        kwargs = dict(zip(params, raw_args))
        print(f"  tool: {name} {kwargs}")
        observation = TOOLS_BY_NAME[name].invoke(kwargs)
        print(f"  result: {observation}")
        scratchpad += f"{output}\nObservation: {observation}\nThought:"

    return "ERROR: max iterations or parse failure"


# Raw ReAct is optional - many small models refuse the format.
# Uncomment to see the fragility yourself:
# react_answer = run_react_text_loop(QUESTION)
print("raw ReAct left as an exercise - run it and watch a parse failure at least once")

# %% [markdown]
# ## Level 4 - what `create_agent` adds on top

# %%
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware

agent = create_agent(
    model,
    tools=TOOLS,
    system_prompt=SYSTEM,
    middleware=[ModelCallLimitMiddleware(run_limit=MAX_ITERATIONS, exit_behavior="end")],
)
result = agent.invoke({"messages": [{"role": "user", "content": QUESTION}]})
print(result["messages"][-1].content)

# %% [markdown]
# | Concern | Hand-rolled loop | `create_agent` |
# |---|---|---|
# | Tool binding | You call `bind_tools` | Built in |
# | Tool execution | You append `ToolMessage` | Built in |
# | Max steps | Your `for` loop | `ModelCallLimitMiddleware` |
# | Streaming | You wire it | `stream_mode=` |
# | Memory | You manage messages | `checkpointer` + `thread_id` |
# | Human approval | You check a flag | `HumanInTheLoopMiddleware` |
# | Retries / fallbacks | You write them | Middleware |
#
# Hand-roll the loop once so you understand it. Ship `create_agent`.

# %% [markdown]
# ## Debugging tip from the ReAct-Algo branch
#
# Eden Marco's `callbacks.py` prints every prompt the model sees. That is still
# the cheapest debugging tool. When an agent misbehaves, log the messages list
# before each `invoke` - you will usually find the model never saw the tool
# result you think it saw.

# %%
from langchain_core.callbacks import BaseCallbackHandler


class PromptPrinter(BaseCallbackHandler):
    """Print the messages about to be sent to the model."""

    def on_chat_model_start(self, serialized, messages, **kwargs):
        flat = messages[0] if messages and isinstance(messages[0], list) else messages
        print(f"\n[prompt] {len(flat)} messages")
        for message in flat[-4:]:
            role = getattr(message, "type", type(message).__name__)
            content = getattr(message, "content", str(message))
            print(f"  {role}: {str(content)[:120]}")


_ = agent.invoke(
    {"messages": [{"role": "user", "content": "Price of headphones with silver discount?"}]},
    {"callbacks": [PromptPrinter()]},
)

# %% [markdown]
# ## Try it yourself
#
# 1. Break the system prompt (remove rule 1) and re-run Level 1. Confirm the
#    model invents a price. That is why the rules exist.
# 2. Implement Level 3 and count how often the regex fails on your provider.
# 3. Add a third tool `convert_currency(amount, to)` and watch how little of
#    Levels 1 and 4 change - versus how much of Level 3's prompt you must edit.
#
# ## Recap
#
# | Concept | Takeaway |
# |---|---|
# | Agent loop | model → tool_calls → execute → ToolMessage → model → ... |
# | `bind_tools` | Structured tool calls; the modern default |
# | Raw JSON schemas | Same loop, explicit provider payload |
# | Raw ReAct text | Fragile regex parsing; historical, not recommended |
# | `create_agent` | The loop plus middleware, memory, streaming |
# | Prompt printer | Cheapest debugger: log what the model actually saw |
#
# ## Next
#
# -> [18_from_agentexecutor_to_graphs.ipynb](18_from_agentexecutor_to_graphs.ipynb)
