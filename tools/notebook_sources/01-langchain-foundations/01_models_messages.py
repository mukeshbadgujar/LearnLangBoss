# %% [markdown]
# # 01 - Models and Messages
#
# | | |
# |---|---|
# | **Level** | Beginner |
# | **Time** | 45-60 minutes |
# | **Prerequisites** | `00_environment_providers_and_keys` |
# | **Checklist ID** | `01_models_messages` |
#
# ## Why this matters
#
# You are asked to build a support assistant for Northwind Analytics. Before any
# RAG, agents or graphs, you need to answer three practical questions:
#
# 1. How do I tell the model *who it is* versus *what the user asked*?
# 2. How do I keep a conversation going across turns?
# 3. How do I know what a call cost and why it stopped?
#
# All three come down to **messages**. Messages are the atom of everything else in
# LangChain - agents, memory, RAG and graphs are all just structured ways of
# building and passing message lists.

# %%
from pathlib import Path
import sys

_root = Path.cwd()
while not (_root / "shared").is_dir() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from shared.notebook_setup import setup, require  # noqa: E402

ctx = setup("01_models_messages")

# %% [markdown]
# ## 1. Two kinds of models: LLM vs Chat Model
#
# | | **LLM (text completion)** | **Chat Model** |
# |---|---|---|
# | Input | a single string | a list of messages |
# | Output | a string | an `AIMessage` |
# | Roles | none - you fake them with text | built-in: system / user / assistant |
# | Tool calling | no | yes |
# | Class | `BaseLLM` | `BaseChatModel` |
# | Status in 2026 | legacy, rarely used | **the default for everything** |
#
# Text-completion LLMs are what `text-davinci-003` was. Almost every modern model
# (GPT-4o, Llama 3.3, Claude) is served as a chat model, and tool calling - the
# foundation of agents - only exists on the chat interface.
#
# **Rule of thumb: always use a chat model.** We cover the old interface only so
# you can read older codebases.

# %%
from shared.llm import get_chat_model

model = get_chat_model()
print("Class :", type(model).__name__)
print("Base  :", [c.__name__ for c in type(model).__mro__ if c.__name__ in {"BaseChatModel", "BaseLLM"}])

# %% [markdown]
# ### The legacy LLM interface, for recognition only
#
# ```python
# from langchain_openai import OpenAI          # note: OpenAI, not ChatOpenAI
# llm = OpenAI(model="gpt-3.5-turbo-instruct")
# llm.invoke("Translate to French: hello")     # -> plain str, no roles, no tools
# ```
#
# If you see `OpenAI(...)`, `HuggingFaceHub(...)` or `Cohere(...)` used directly in
# a tutorial, that tutorial predates the chat-model era. The concepts still apply,
# but rewrite the code with a chat model.

# %% [markdown]
# ## 2. The message types
#
# | Message | Role | What you use it for |
# |---|---|---|
# | `SystemMessage` | `system` | Identity, rules, tone, output format. Set once, at the front. |
# | `HumanMessage` | `user` | What the user typed. |
# | `AIMessage` | `assistant` | What the model previously said. You replay these for context. |
# | `ToolMessage` | `tool` | The result of a tool call, fed back to the model (notebook 16). |
#
# In LangChain v1 these are importable from `langchain.messages` (re-exported from
# `langchain_core.messages`, which is what older code imports).

# %%
from langchain.messages import AIMessage, HumanMessage, SystemMessage

messages = [
    SystemMessage(
        "You are the Northwind Analytics support assistant. "
        "Answer in at most two sentences. If you are unsure, say so plainly."
    ),
    HumanMessage("Our webhook deliveries started failing with 401 after we rotated our API key."),
]

reply = model.invoke(messages)
print(reply.content)

# %% [markdown]
# ### The system message is the highest-leverage line you will write
#
# Same question, different system message. Watch the whole character of the answer
# change without touching the user's words.

# %%
personas = {
    "terse engineer": "You are a senior engineer. Answer in one line. No pleasantries.",
    "support agent": "You are a warm customer support agent. Reassure the customer, then give steps.",
    "compliance officer": "You are a compliance officer. Answer only with what policy permits, and cite the risk.",
}

question = HumanMessage("Can I email a customer's exported data to my personal Gmail to work on it at home?")

for label, system_prompt in personas.items():
    answer = model.invoke([SystemMessage(system_prompt), question])
    print(f"--- {label} ---\n{answer.content.strip()}\n")

# %% [markdown]
# ## 3. Shorthand formats you will see everywhere
#
# LangChain accepts three equivalent input shapes. Know all of them, because docs
# and Stack Overflow answers mix them freely.

# %%
# (a) Message objects - most explicit, best for production code
form_a = [SystemMessage("You are terse."), HumanMessage("Capital of Japan?")]

# (b) (role, content) tuples - compact, very common in prompt templates
form_b = [("system", "You are terse."), ("human", "Capital of Japan?")]

# (c) OpenAI-style dicts - what you get from a web API payload
form_c = [
    {"role": "system", "content": "You are terse."},
    {"role": "user", "content": "Capital of Japan?"},
]

for name, payload in [("objects", form_a), ("tuples", form_b), ("dicts", form_c)]:
    print(f"{name:8} -> {model.invoke(payload).content.strip()}")

# %% [markdown]
# A bare string is also accepted and is silently wrapped in a `HumanMessage`:
#
# ```python
# model.invoke("Capital of Japan?")   # same as [HumanMessage("Capital of Japan?")]
# ```

# %% [markdown]
# ## 4. Conversation state is your job, not the model's
#
# This is the single most important idea in this notebook.
#
# **The model is stateless.** It has no memory of your previous call. A
# conversation only exists because *you* resend the whole message list each turn.
# Every memory feature in LangChain and every checkpointer in LangGraph is just
# machinery for managing that list.

# %%
conversation = [
    SystemMessage("You are the Northwind support assistant. Be concise."),
    HumanMessage("I'm on the Growth plan. What is my API rate limit?"),
]

first = model.invoke(conversation)
print("Turn 1:", first.content.strip())

# Append the model's own reply, otherwise turn 2 has no idea what was discussed.
conversation.append(first)
conversation.append(HumanMessage("And what happens if I exceed it?"))

second = model.invoke(conversation)
print("\nTurn 2:", second.content.strip())
print("\nMessages now in the conversation:", len(conversation) + 1)

# %% [markdown]
# ### Prove it: drop the history and the follow-up becomes nonsense

# %%
stateless = model.invoke([
    SystemMessage("You are the Northwind support assistant. Be concise."),
    HumanMessage("And what happens if I exceed it?"),
])
print("Without history:", stateless.content.strip()[:300])

# %% [markdown]
# ## 5. Reading the response object
#
# `.content` is what you show the user. Everything else is what you need for
# debugging, costing and reliability.

# %%
import json

sample = model.invoke([HumanMessage("List three benefits of hybrid work. Be brief.")])

print("content:\n", sample.content.strip(), "\n")
print("usage_metadata   :", sample.usage_metadata)
print("finish reason    :", sample.response_metadata.get("finish_reason"))
print("id               :", sample.id)
print("\nfull response_metadata:")
print(json.dumps(sample.response_metadata, indent=2, default=str)[:700])

# %% [markdown]
# **`finish_reason` is your early-warning system:**
#
# | Value | Meaning | What to do |
# |---|---|---|
# | `stop` | Model finished naturally | Nothing |
# | `length` | Hit `max_tokens` - **answer is truncated** | Raise `max_tokens` or shorten input |
# | `tool_calls` | Model wants to call a tool | Execute it and send a `ToolMessage` back |
# | `content_filter` | Provider blocked it | Rephrase, or handle as a refusal |
#
# Truncation is the classic silent bug: the JSON your parser receives is cut in
# half and you get a confusing parse error three layers away from the real cause.

# %%
truncating_model = get_chat_model(max_tokens=16)
cut_off = truncating_model.invoke("Explain the CAP theorem in detail.")
print("content     :", cut_off.content)
print("finish_reason:", cut_off.response_metadata.get("finish_reason"), "  <- 'length' means truncated")

# %% [markdown]
# ## 6. Parameters that actually change behaviour
#
# - **`temperature`** (0.0-2.0): randomness. `0` for extraction, classification,
#   SQL and routing. `0.7`+ for brainstorming and copywriting. This course uses
#   `0` by default so your results are reproducible.
# - **`max_tokens`**: cap on the *output* only. Input is limited by the context window.
# - **`timeout`** and **`max_retries`**: reliability knobs. Always set a timeout in
#   a web service or one slow call will pin a worker.
# - **`model_kwargs` / `seed`**: provider-specific extras.

# %%
creative = get_chat_model(temperature=1.2, max_tokens=60)
precise = get_chat_model(temperature=0.0, max_tokens=60)
prompt = "Invent a name for an internal tool that finds stale feature flags."

print("temperature 1.2:")
for _ in range(3):
    print("  -", creative.invoke(prompt).content.strip().replace("\n", " ")[:90])

print("\ntemperature 0.0:")
for _ in range(3):
    print("  -", precise.invoke(prompt).content.strip().replace("\n", " ")[:90])

# %% [markdown]
# Note that temperature `0` is *near*-deterministic, not guaranteed-deterministic:
# provider-side batching and floating-point non-determinism can still cause drift.
# Never write a test that asserts on exact model output.

# %% [markdown]
# ## 7. `batch` and `stream`: the other two ways to call a model
#
# Every LangChain component implements the same **Runnable** interface. You will
# study it properly in notebook 07, but you already have three of its methods.

# %%
import time

tickets = [
    "Invoice shows a duplicate charge for January seats",
    "Webhook deliveries failing with 401 after key rotation",
    "How do I export a dashboard to PDF?",
]
classifier_prompt = "Reply with exactly one word - billing, integration, or howto - for this ticket: "

start = time.perf_counter()
sequential = [model.invoke(classifier_prompt + t).content.strip() for t in tickets]
sequential_time = time.perf_counter() - start

start = time.perf_counter()
batched = [r.content.strip() for r in model.batch([classifier_prompt + t for t in tickets])]
batched_time = time.perf_counter() - start

print(f"sequential {sequential_time:5.2f}s -> {sequential}")
print(f"batch      {batched_time:5.2f}s -> {batched}")
print(f"\nbatch ran concurrently and was ~{sequential_time / max(batched_time, 0.01):.1f}x faster")

# %%
print("Streaming (what a chat UI does):\n")
for chunk in model.stream("Write two sentences about why observability matters for LLM apps."):
    print(chunk.content, end="", flush=True)
print("\n\n[stream complete]")

# %% [markdown]
# ## 8. Swapping providers mid-flight
#
# Because all of this is the same interface, comparing vendors is a one-line change.
# In practice teams do this to route cheap traffic to a small model and hard
# traffic to a large one.

# %%
from shared.llm import available_providers

hard_question = (
    "A customer on Growth downgrades to Starter on day 12 of a 30-day cycle. "
    "Per our policy, when does it take effect and is there a refund? Answer in one sentence."
)

for provider in available_providers():
    answer = get_chat_model(provider=provider).invoke([
        SystemMessage(
            "Northwind policy: downgrades take effect at the start of the next billing "
            "cycle and no partial refunds are issued for the current cycle."
        ),
        HumanMessage(hard_question),
    ])
    print(f"[{provider}] {answer.content.strip()}\n")

# %% [markdown]
# ## Try it yourself
#
# 1. **Build a 4-turn conversation** about a leave request, appending each
#    `AIMessage` as you go. Then print `len(conversation)` and reason about how
#    token cost grows every turn - this is exactly the problem memory solves.
# 2. **Force a truncation bug.** Ask for JSON with `max_tokens=20`, then try to
#    `json.loads()` the content. Notice how the error message points at JSON, not
#    at the real cause. Then fix it by checking `finish_reason` first.
# 3. **Write a router system prompt** that classifies a ticket into
#    `billing | integration | bug | howto | security` and returns *only* that word.
#    Test it on all 15 rows of `shared/sample_data/support_tickets.csv` with
#    `model.batch(...)` and count how many match the `category` column.

# %%
# Space for exercise 3 - here is the data loading part to get you started.
import csv

with open(ctx.data("support_tickets.csv"), newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))

print(f"{len(rows)} tickets loaded. Categories present:", sorted({r["category"] for r in rows}))

# %% [markdown]
# ## Recap
#
# | Concept | Takeaway |
# |---|---|
# | Chat model vs LLM | Always use chat models; tool calling only exists there |
# | `SystemMessage` | Cheapest, strongest lever on behaviour |
# | Statelessness | Conversations exist only because you resend the message list |
# | `AIMessage` | `.content` for users, `.usage_metadata` and `.response_metadata` for you |
# | `finish_reason` | `length` means truncated - check it before parsing |
# | `temperature` | `0` for extraction/routing, higher for creative work |
# | `invoke` / `batch` / `stream` | Same interface on every component; `batch` is concurrent |
#
# **When not to use a raw model call:** the moment your prompt has variables in
# it, stop concatenating strings and move to a prompt template - which is exactly
# the next notebook.
#
# ## Next
#
# -> [02_prompt_templates.ipynb](02_prompt_templates.ipynb)
