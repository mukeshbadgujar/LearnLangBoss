# %% [markdown]
# # 06 - Memory: Classic Types and the Modern Replacement
#
# | | |
# |---|---|
# | **Level** | Intermediate |
# | **Time** | 50 minutes |
# | **Prerequisites** | `03_chat_prompt_templates` |
# | **Checklist ID** | `06_memory` |
#
# ## Why this matters
#
# Your support bot answers turn 1 perfectly and has amnesia by turn 2. Fixing that
# is "memory", and it is where most LangChain tutorials you will find online are
# **out of date**.
#
# The `ConversationBufferMemory` family was deprecated in LangChain 0.3 and moved
# to `langchain-classic` in v1. The replacement is LangGraph persistence: state
# plus a checkpointer plus a `thread_id`.
#
# This notebook does two things:
#
# 1. Teaches the **five memory strategies** (buffer, window, summary, token,
#    vector) as *concepts* - because the trade-offs are permanent even though the
#    classes changed.
# 2. Shows the **modern implementation** of each, so you write 2026 code.

# %%
from pathlib import Path
import sys

_root = Path.cwd()
while not (_root / "shared").is_dir() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from shared.notebook_setup import setup, require  # noqa: E402

ctx = setup("06_memory")

# %%
from shared.llm import get_chat_model, get_embeddings

model = get_chat_model()

# %% [markdown]
# ## 1. The five strategies and their trade-offs
#
# Every memory design answers one question: **the history is too long to send -
# what do you drop?**
#
# | Strategy | Keeps | Loses | Cost per turn | Use when |
# |---|---|---|---|---|
# | **Buffer** | everything | nothing | grows forever | Short conversations (<10 turns) |
# | **Window (last k)** | last k turns | everything older | constant | Support chat, task bots |
# | **Summary** | an LLM summary of old turns | exact wording | constant + 1 LLM call | Long advisory conversations |
# | **Token buffer** | as many recent turns as fit a token budget | oldest turns | bounded, predictable | Hard context/cost limits |
# | **Summary + buffer** | summary of old + verbatim recent | exact old wording | constant + occasional call | **Best general default** |
# | **Vector-backed** | semantically relevant past turns | chronological order | retrieval + constant | Very long-lived assistants |
#
# There is no universally right answer. A billing bot needs the last 3 turns. A
# 6-month customer relationship assistant needs semantic recall.

# %% [markdown]
# ## 2. The legacy API (recognise it, do not write it)
#
# You *will* meet this code. Here is what it looked like:
#
# ```python
# # LangChain <= 0.2 - deprecated, moved to langchain-classic in v1
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationChain
#
# memory = ConversationBufferMemory()
# chain = ConversationChain(llm=llm, memory=memory)
# chain.predict(input="Hi, I'm on the Growth plan")
# chain.predict(input="What's my rate limit?")   # memory injected automatically
# ```
#
# Why it was replaced:
#
# - memory lived **inside** the chain, invisible and hard to inspect
# - it was in-process only - a restart or a second server lost everything
# - it could not represent branching, retries or human approval
# - it did not compose with tools and multi-step agents
#
# The modern equivalent makes state **explicit and persistent**.

# %%
legacy_available = False
try:
    from langchain_classic.memory import ConversationBufferMemory  # noqa: F401

    legacy_available = True
    print("langchain-classic is installed - the legacy classes are importable for reference.")
except ImportError:
    print("langchain-classic not installed. That is fine: nothing below depends on it.")

# %% [markdown]
# ## 3. Modern baseline: a checkpointer and a `thread_id`
#
# This is the whole modern memory story in one cell. `InMemorySaver` stores state;
# `thread_id` says *which conversation*; the agent reloads history automatically.

# %%
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model=model,
    tools=[],
    system_prompt="You are the Northwind Analytics support assistant. Be concise.",
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": "customer-bluepeak"}}

turn1 = agent.invoke({"messages": [{"role": "user", "content": "I'm on the Growth plan."}]}, config)
print("Turn 1:", turn1["messages"][-1].content.strip())

turn2 = agent.invoke({"messages": [{"role": "user", "content": "What's my API rate limit then?"}]}, config)
print("\nTurn 2:", turn2["messages"][-1].content.strip())

print(f"\nStored messages on this thread: {len(turn2['messages'])}")

# %% [markdown]
# ### Threads are isolated - this is what sessions/users map to

# %%
other = agent.invoke(
    {"messages": [{"role": "user", "content": "What plan am I on?"}]},
    {"configurable": {"thread_id": "customer-granite"}},
)
print("Different thread ->", other["messages"][-1].content.strip()[:200])

# %% [markdown]
# A different `thread_id` sees none of the first conversation. In a web service
# `thread_id` is your session id, conversation id, or `user_id:channel`.

# %% [markdown]
# ## 4. Strategy 1 and 2 - Buffer and Window, implemented today
#
# A buffer is simply "do nothing" - the default above. A window means trimming
# before the model call. In LangGraph you do that with **middleware** or a
# pre-model hook.

# %%
from langchain.agents.middleware import AgentMiddleware
from langchain.messages import trim_messages


class WindowMemory(AgentMiddleware):
    """Keep only the last `k` messages - the modern ConversationBufferWindowMemory."""

    def __init__(self, k: int = 4):
        super().__init__()
        self.k = k

    def before_model(self, state, runtime):
        messages = state["messages"]
        if len(messages) <= self.k:
            return None  # nothing to change
        kept = trim_messages(
            messages,
            max_tokens=self.k,
            token_counter=len,      # count messages, not tokens
            strategy="last",
            start_on="human",
            include_system=True,
            allow_partial=False,
        )
        print(f"    [WindowMemory] {len(messages)} -> {len(kept)} messages sent to the model")
        return {"messages": kept}


windowed_agent = create_agent(
    model=model,
    tools=[],
    system_prompt="You are a concise support assistant.",
    middleware=[WindowMemory(k=4)],
    checkpointer=InMemorySaver(),
)

window_config = {"configurable": {"thread_id": "window-demo"}}
for question in [
    "My name is Rahul and I work at Bluepeak Health.",
    "We use the Salesforce connector.",
    "Our sandbox was refreshed last night.",
    "Sync is now failing. What should I check first?",
    "By the way, what is my name?",
]:
    answer = windowed_agent.invoke({"messages": [{"role": "user", "content": question}]}, window_config)
    print(f"USER: {question}\nBOT : {answer['messages'][-1].content.strip()[:160]}\n")

# %% [markdown]
# The last answer probably failed to recall "Rahul" - the window dropped it. That
# is the trade-off made visible: constant cost, bounded recall.

# %% [markdown]
# ## 5. Strategy 3 and 5 - Summarisation (the production default)
#
# `SummarizationMiddleware` is the direct descendant of
# `ConversationSummaryBufferMemory`: when history crosses a token threshold it
# summarises the old part and keeps recent turns verbatim.

# %%
from langchain.agents.middleware import SummarizationMiddleware

summarising_agent = create_agent(
    model=model,
    tools=[],
    system_prompt="You are the Northwind support assistant. Be concise.",
    middleware=[
        SummarizationMiddleware(
            model=model,
            trigger=("tokens", 400),   # summarise once history passes 400 tokens
            keep=("messages", 4),      # keep the last 4 messages verbatim
        )
    ],
    checkpointer=InMemorySaver(),
)

# The trigger/keep pair accepts ("tokens", n), ("messages", n) or ("fraction", 0.0-1.0).
# Older tutorials pass max_tokens_before_summary=/messages_to_keep=; those names are
# deprecated aliases for exactly these two arguments.

summary_config = {"configurable": {"thread_id": "summary-demo"}}
script = [
    "Hi, I'm Priya from Cedar Logistics, we're on the Starter plan.",
    "We have about 40 dashboards and 3 data sources connected.",
    "Our main problem is CSV export dropping the last row when a filter is applied.",
    "It started after the January release and affects every user in our workspace.",
    "We also want to know if upgrading to Growth would give us faster refresh.",
    "Remind me - which company did I say I was from, and what was my original bug?",
]

for question in script:
    answer = summarising_agent.invoke({"messages": [{"role": "user", "content": question}]}, summary_config)

print("Final answer:\n", answer["messages"][-1].content.strip())
print(f"\nMessages retained in state: {len(answer['messages'])}")

# %% [markdown]
# The bot recalled the company and the bug even though the earliest turns were
# compressed. That is the summary doing its job.

# %% [markdown]
# ## 6. Strategy 4 - Token budget
#
# Counting *messages* is crude; long messages blow the budget. Count tokens.

# %%
from langchain.messages import AIMessage, HumanMessage, SystemMessage


class TokenBudgetMemory(AgentMiddleware):
    """Modern ConversationTokenBufferMemory: keep history under a token ceiling."""

    def __init__(self, max_tokens: int = 300):
        super().__init__()
        self.max_tokens = max_tokens

    def before_model(self, state, runtime):
        messages = state["messages"]
        kept = trim_messages(
            messages,
            max_tokens=self.max_tokens,
            token_counter=model,
            strategy="last",
            start_on="human",
            include_system=True,
        )
        if len(kept) != len(messages):
            print(f"    [TokenBudget] trimmed {len(messages)} -> {len(kept)} messages")
        return {"messages": kept} if len(kept) != len(messages) else None


demo_history = [SystemMessage("You are concise.")]
for i in range(6):
    demo_history.append(HumanMessage(f"Turn {i}: " + "detail about invoices and billing cycles " * 8))
    demo_history.append(AIMessage(f"Acknowledged turn {i} with a similarly long explanation of billing."))

print("total tokens in raw history:", model.get_num_tokens("\n".join(m.content for m in demo_history)))
budgeted = trim_messages(demo_history, max_tokens=300, token_counter=model, strategy="last",
                         start_on="human", include_system=True)
print(f"messages {len(demo_history)} -> {len(budgeted)}")
print("tokens after trim   :", model.get_num_tokens("\n".join(m.content for m in budgeted)))

# %% [markdown]
# ## 7. Strategy 6 - Vector-backed memory (long-term recall)
#
# Window and summary both lose detail. Sometimes you need "what did this customer
# say about SSO three months ago?" - which is retrieval, not recency.
#
# The pattern: write past turns into a vector store, and retrieve the relevant few
# at request time. Notebook 49 builds the full LangGraph version with the `Store`
# API; here is the concept in isolation.

# %%
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore


class VectorMemory:
    """Semantic recall over past conversation turns."""

    def __init__(self, embeddings, k: int = 2):
        self.store = InMemoryVectorStore(embeddings)
        self.k = k

    def remember(self, speaker: str, text: str) -> None:
        self.store.add_documents([Document(page_content=f"{speaker}: {text}")])

    def recall(self, query: str) -> list[str]:
        return [d.page_content for d in self.store.similarity_search(query, k=self.k)]


vector_memory = VectorMemory(get_embeddings())

past_conversation = [
    ("user", "We use Okta for SSO and our metadata certificate rotates every year in March."),
    ("user", "Our finance contact for invoices is Meera Iyer."),
    ("user", "We decided against the Enterprise plan last quarter because of budget."),
    ("user", "Our primary data source is Snowflake in ap-south-1."),
    ("user", "The team size is 42 people across three offices."),
]
for speaker, text in past_conversation:
    vector_memory.remember(speaker, text)

query = "who should we contact about the invoice?"
print("Query:", query)
for hit in vector_memory.recall(query):
    print("  recalled ->", hit)

print("\nQuery: when does our SSO certificate expire?")
for hit in vector_memory.recall("when does our SSO certificate expire?"):
    print("  recalled ->", hit)

# %%
from langchain_core.prompts import ChatPromptTemplate

recall_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a support assistant. Use these remembered facts if relevant:\n{facts}"),
    ("human", "{question}"),
])

question = "Who do I send the invoice correction to, and what SSO provider do they use?"
answer = model.invoke(recall_prompt.invoke({
    "facts": "\n".join(vector_memory.recall(question)),
    "question": question,
}))
print(answer.content.strip())

# %% [markdown]
# ## 8. Inspecting and editing memory
#
# A real advantage of the modern approach: state is **inspectable**. You can read
# it, audit it, and correct it - impossible with the old opaque memory objects.

# %%
snapshot = agent.get_state(config)
print("Thread:", config["configurable"]["thread_id"])
print("Messages stored:", len(snapshot.values["messages"]))
for message in snapshot.values["messages"]:
    print(f"  [{message.type:9}] {message.content[:70]}")
print("\nNext node to run:", snapshot.next or "(idle)")

# %% [markdown]
# ## 9. Choosing in practice
#
# ```
# Conversation < 10 turns?            -> buffer (do nothing)
# Need constant cost, recent context? -> window / token budget
# Long conversation, facts matter?    -> SummarizationMiddleware   <- default
# Recall across days/months?          -> vector store + Store API (notebook 49)
# Regulated domain, need the exact transcript? -> persist raw history separately
#                                         and still summarise for the prompt
# ```
#
# **Persistence tiers** (notebook 35 covers these):
#
# | Saver | Survives | Use |
# |---|---|---|
# | `InMemorySaver` | nothing | Notebooks, tests |
# | `SqliteSaver` | process restart | Local apps, single-node |
# | `PostgresSaver` | everything, multi-node | Production |

# %% [markdown]
# ## Try it yourself
#
# 1. **Recreate `ConversationBufferWindowMemory(k=2)`** with `WindowMemory` and
#    prove the loss: state a fact in turn 1 and ask for it in turn 5.
# 2. **Tune the summariser.** Set `trigger=("tokens", 150)` and
#    `keep=("messages", 2)`, re-run the 6-turn script, and note what gets lost.
# 3. **Hybrid memory.** Combine `SummarizationMiddleware` with the `VectorMemory`
#    class: summarise for recency, retrieve for long-term facts. Test with a
#    question that needs both.
# 4. **Persist it.** Swap `InMemorySaver` for `SqliteSaver`, restart the kernel,
#    and continue the same `thread_id`. (Full walkthrough in notebook 35.)

# %% [markdown]
# ## Recap
#
# | Concept | Takeaway |
# |---|---|
# | Legacy `ConversationBufferMemory` etc. | Deprecated; lives in `langchain-classic`. Read-only knowledge |
# | Checkpointer + `thread_id` | The modern memory primitive; state is explicit and persistent |
# | Buffer | Simplest, unbounded cost |
# | Window / token budget | Constant cost, loses old facts |
# | `SummarizationMiddleware` | Best general default; compresses old, keeps recent verbatim |
# | Vector memory | Semantic recall across long time spans |
# | `agent.get_state(config)` | Memory you can read and audit |
#
# ## Next
#
# -> [07_lcel.ipynb](07_lcel.ipynb)
