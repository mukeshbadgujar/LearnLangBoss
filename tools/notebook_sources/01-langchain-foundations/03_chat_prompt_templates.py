# %% [markdown]
# # 03 - ChatPromptTemplates and MessagesPlaceholder
#
# | | |
# |---|---|
# | **Level** | Beginner |
# | **Time** | 40 minutes |
# | **Prerequisites** | `02_prompt_templates` |
# | **Checklist ID** | `03_chat_prompt_templates` |
#
# ## Why this matters
#
# In notebook 01 you kept a conversation alive by manually appending messages to a
# list. That breaks down the moment a prompt has both *fixed instructions* and
# *variable history*: you end up with fragile code that splices lists together in
# the right order.
#
# `ChatPromptTemplate` + `MessagesPlaceholder` is the standard solution, and it is
# the exact structure that agents, RAG chains and LangGraph nodes use internally.
# Learn it properly here and the later notebooks will feel familiar.

# %%
from pathlib import Path
import sys

_root = Path.cwd()
while not (_root / "shared").is_dir() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from shared.notebook_setup import setup, require  # noqa: E402

ctx = setup("03_chat_prompt_templates")

# %%
from shared.llm import get_chat_model

model = get_chat_model()

# %% [markdown]
# ## 1. From string template to chat template
#
# A `ChatPromptTemplate` is a list of message templates. Each entry knows its role.

# %%
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are the Northwind Analytics support assistant.\n"
     "Tone: {tone}.\n"
     "Answer in at most {max_sentences} sentences."),
    ("human", "{question}"),
])

print("Inputs:", prompt.input_variables)

rendered = prompt.invoke({
    "tone": "direct and factual",
    "max_sentences": 2,
    "question": "What is the API rate limit on the Growth plan?",
})

for message in rendered.to_messages():
    print(f"\n[{message.type}] {message.content}")

# %%
print(model.invoke(rendered).content.strip())

# %% [markdown]
# ## 2. All three roles in one template
#
# Including an `ai` turn in the template is a powerful trick: it shows the model
# the *shape* of a good answer, and it can be used to "put words in its mouth" so
# the real reply continues in that style.

# %%
styled = ChatPromptTemplate.from_messages([
    ("system", "You are a Northwind support agent. Always answer with: VERDICT, then REASON, then NEXT STEP."),
    ("human", "A customer wants a refund for a mid-cycle downgrade."),
    ("ai",
     "VERDICT: Not eligible.\n"
     "REASON: Downgrades take effect next cycle and no partial refunds are issued.\n"
     "NEXT STEP: Confirm the new rate starts on their next invoice date."),
    ("human", "{question}"),
])

print(model.invoke(styled.invoke({
    "question": "A customer on a 14-day trial asks for a 7-day extension."
})).content.strip())

# %% [markdown]
# ## 3. `MessagesPlaceholder`: the slot for dynamic history
#
# The problem: you want fixed instructions at the top, then *however many* prior
# turns exist, then the new question. The number of history messages changes every
# request, so you cannot hard-code them.

# %%
from langchain_core.prompts import MessagesPlaceholder

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are the Northwind support assistant. Be concise and never invent policy."),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])

print("Inputs:", chat_prompt.input_variables)

# %%
from langchain.messages import AIMessage, HumanMessage

history = [
    HumanMessage("I'm on the Growth plan."),
    AIMessage("Understood - Growth includes 15 data sources and a 3-hour refresh."),
]

rendered = chat_prompt.invoke({"history": history, "question": "How often can I refresh if I upgrade?"})
for message in rendered.to_messages():
    print(f"[{message.type:9}] {message.content}")

print("\n->", model.invoke(rendered).content.strip())

# %% [markdown]
# ### Empty history works too - that is the point

# %%
first_turn = chat_prompt.invoke({"history": [], "question": "What are your support SLAs?"})
print(f"{len(first_turn.to_messages())} messages when history is empty")
print(model.invoke(first_turn).content.strip())

# %% [markdown]
# ### `optional=True` lets you omit the key entirely

# %%
lenient = ChatPromptTemplate.from_messages([
    ("system", "You are terse."),
    MessagesPlaceholder("history", optional=True),
    ("human", "{question}"),
])

print([m.type for m in lenient.invoke({"question": "Hi"}).to_messages()])

# %% [markdown]
# ## 4. A working multi-turn chatbot in ~15 lines
#
# This is the whole pattern: render the template, call the model, append both
# sides to history, repeat.

# %%
from langchain_core.output_parsers import StrOutputParser

chain = chat_prompt | model | StrOutputParser()


class SimpleChat:
    """Minimal conversation manager. LangGraph checkpointers replace this later."""

    def __init__(self, runnable):
        self.runnable = runnable
        self.history: list = []

    def ask(self, question: str) -> str:
        answer = self.runnable.invoke({"history": self.history, "question": question})
        self.history.extend([HumanMessage(question), AIMessage(answer)])
        return answer


bot = SimpleChat(chain)

for turn in [
    "We're a 40-person company evaluating Northwind. Which plan fits?",
    "What's the refresh interval on that one?",
    "And if we need 15-minute refresh instead?",
]:
    print(f"\nUSER: {turn}")
    print(f"BOT : {bot.ask(turn).strip()}")

print(f"\nHistory now holds {len(bot.history)} messages "
      f"({len(bot.history) // 2} exchanges) and grows every turn.")

# %% [markdown]
# **Notice the growth problem.** Every turn adds two messages, and all of them are
# resent on the next call. Ten turns in, you are paying for the whole transcript
# on every request. Notebook 06 (memory) and notebook 47 (summarisation nodes)
# deal with this directly.

# %% [markdown]
# ## 5. Trimming history before it gets expensive
#
# `trim_messages` is the built-in tool. It is token-aware and, crucially, keeps the
# message list *valid* (it won't orphan a tool call from its result).

# %%
from langchain.messages import trim_messages

long_history = []
for i in range(1, 9):
    long_history.append(HumanMessage(f"Question number {i} about billing and invoices."))
    long_history.append(AIMessage(f"Detailed answer number {i} about billing and invoices."))

trimmed = trim_messages(
    long_history,
    max_tokens=120,
    token_counter=model,      # ask the model to count - accurate per provider
    strategy="last",          # keep the most recent turns
    start_on="human",         # a valid window starts with a human turn
    include_system=True,
)

print(f"before: {len(long_history)} messages")
print(f"after : {len(trimmed)} messages")
for message in trimmed:
    print(f"  [{message.type:6}] {message.content[:60]}")

# %% [markdown]
# ## 6. Placeholders can also carry examples or retrieved context
#
# `MessagesPlaceholder` is not only for chat history. Any list of messages fits -
# few-shot examples (notebook 04), retrieved documents (notebook 14), or tool
# results (notebook 17).

# %%
rag_style = ChatPromptTemplate.from_messages([
    ("system",
     "Answer strictly from the provided context. If the context does not contain "
     "the answer, reply exactly: 'Not covered by the documents I have.'"),
    MessagesPlaceholder("context_messages"),
    ("human", "{question}"),
])

policy_text = Path(ctx.data("leave_policy.txt")).read_text(encoding="utf-8")
excerpt = policy_text[policy_text.index("2. SICK LEAVE"): policy_text.index("3. CASUAL LEAVE")]

context_messages = [HumanMessage(f"CONTEXT:\n{excerpt}")]

print("Answerable  ->", model.invoke(rag_style.invoke({
    "context_messages": context_messages,
    "question": "When do I need a medical certificate?",
})).content.strip())

print("\nUnanswerable ->", model.invoke(rag_style.invoke({
    "context_messages": context_messages,
    "question": "How many days of paternity leave do I get?",
})).content.strip())

# %% [markdown]
# ## 7. Templates from message objects, and re-usable fragments

# %%
from langchain_core.prompts import (
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

# The verbose form - identical result, occasionally clearer when building dynamically.
verbose = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("You are {role}."),
    HumanMessagePromptTemplate.from_template("{question}"),
])
print(verbose.invoke({"role": "a database expert", "question": "What is a covering index?"}).to_messages())

# %%
# Building a prompt list programmatically - common when roles/rules come from config.
def build_prompt(rules: list[str], include_history: bool = True) -> ChatPromptTemplate:
    system_text = "You are the Northwind support assistant.\n" + "\n".join(f"- {r}" for r in rules)
    pieces: list = [("system", system_text)]
    if include_history:
        pieces.append(MessagesPlaceholder("history", optional=True))
    pieces.append(("human", "{question}"))
    return ChatPromptTemplate.from_messages(pieces)


strict = build_prompt([
    "Never quote a price you are not certain of.",
    "Escalate anything involving security or compliance.",
    "Answer in one sentence.",
])

print(strict.invoke({"question": "Can you confirm our contract's data residency region?"}).to_messages()[0].content)
print("\n->", model.invoke(strict.invoke({"question": "Can you confirm our contract's data residency region?"})).content.strip())

# %% [markdown]
# ## Try it yourself
#
# 1. **Add a summary slot.** Extend `chat_prompt` with a `{running_summary}`
#    variable in the system message, and have `SimpleChat` regenerate that summary
#    every 4 turns while trimming raw history to the last 4 messages. You have just
#    hand-built `ConversationSummaryBufferMemory`.
# 2. **Compare trimming strategies.** Run `trim_messages` with `strategy="first"`
#    and `strategy="last"` on the same history and explain which one you want for
#    a support bot, and which for a document-analysis bot.
# 3. **Two placeholders.** Build a template with both `examples` and `history`
#    placeholders and confirm the rendered order is: system, examples, history,
#    question.

# %% [markdown]
# ## Recap
#
# | Concept | Takeaway |
# |---|---|
# | `ChatPromptTemplate.from_messages` | List of `(role, template)` pairs; roles are `system`/`human`/`ai` |
# | `("ai", ...)` in a template | Demonstrates the answer shape; steers format strongly |
# | `MessagesPlaceholder("history")` | Slot for a *list* of messages of unknown length |
# | `optional=True` | Placeholder key may be omitted entirely |
# | `trim_messages` | Token-aware history trimming that keeps the list valid |
# | Placeholders are generic | Same mechanism carries examples, context and tool results |
#
# **When not to hand-roll this:** once you need persistence across processes,
# stop managing `self.history` yourself and use a LangGraph checkpointer
# (notebook 34). Hand-rolled history is fine for a notebook, fragile in a service.
#
# ## Next
#
# -> [04_few_shot_and_example_selectors.ipynb](04_few_shot_and_example_selectors.ipynb)
