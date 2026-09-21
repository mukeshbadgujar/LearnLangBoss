# %% [markdown]
# # 04 - Few-Shot Prompting and Example Selectors
#
# | | |
# |---|---|
# | **Level** | Beginner to Intermediate |
# | **Time** | 45 minutes |
# | **Prerequisites** | `03_chat_prompt_templates` |
# | **Checklist ID** | `04_few_shot_and_example_selectors` |
#
# ## Why this matters
#
# You wrote a ticket classifier with a carefully worded instruction, and it is 70%
# accurate. You add three worked examples and it jumps to 90% - with no
# fine-tuning, no extra infrastructure, and one extra second of latency.
#
# That is few-shot prompting: **showing beats telling**. The catch is cost. Twenty
# examples in every prompt is twenty examples you pay for on every request. Example
# *selectors* solve that by choosing only the handful of examples relevant to the
# current input.

# %%
from pathlib import Path
import sys

_root = Path.cwd()
while not (_root / "shared").is_dir() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from shared.notebook_setup import setup, require  # noqa: E402

ctx = setup("04_few_shot_and_example_selectors")

# %%
from shared.llm import get_chat_model, get_embeddings

model = get_chat_model()

# %% [markdown]
# ## 1. The baseline: zero-shot
#
# Our task: turn a messy customer message into a structured internal triage line.
# Zero-shot, the model invents its own format every time.

# %%
raw_tickets = [
    "hi we cant log in since this morning, whole team blocked, we pay for enterprise",
    "quick q - where do i change the timezone on reports?",
    "you charged us twice in january, please fix",
]

zero_shot = "Convert this customer message into an internal triage line:\n\n{msg}"

for ticket in raw_tickets:
    print("->", model.invoke(zero_shot.format(msg=ticket)).content.strip().replace("\n", " ")[:140])

# %% [markdown]
# Inconsistent. Now watch what examples do.

# %% [markdown]
# ## 2. `FewShotPromptTemplate` for text-style prompts

# %%
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

examples = [
    {
        "message": "our dashboards have been spinning for 20 mins, nothing loads",
        "triage": "category=performance | priority=high | summary=Dashboards fail to load for the customer",
    },
    {
        "message": "how do I invite a read-only user?",
        "triage": "category=howto | priority=low | summary=Needs guidance on inviting read-only users",
    },
    {
        "message": "invoice 4412 has the wrong GST number, we need it reissued",
        "triage": "category=billing | priority=medium | summary=Invoice reissue requested with corrected GST number",
    },
    {
        "message": "we rotated keys and now webhooks return 401",
        "triage": "category=integration | priority=high | summary=Webhook auth failing after key rotation",
    },
]

example_prompt = PromptTemplate.from_template("Message: {message}\nTriage: {triage}")

few_shot = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=(
        "You are a support triage bot. Convert each customer message into a triage line.\n"
        "Use exactly the format shown. Categories: performance, howto, billing, integration, bug, security.\n"
    ),
    suffix="Message: {message}\nTriage:",
    input_variables=["message"],
)

print(few_shot.format(message="we cant log in since this morning, whole team blocked"))

# %%
for ticket in raw_tickets:
    print("->", model.invoke(few_shot.format(message=ticket)).content.strip().splitlines()[0])

# %% [markdown]
# Same model, same effort, consistent machine-parseable output. The examples did
# the work that a long specification could not.

# %% [markdown]
# ## 3. `FewShotChatMessagePromptTemplate` for chat models
#
# For chat models the better shape is **alternating human/ai turns**, not one big
# text block. The model sees a mini conversation it is expected to continue.

# %%
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

chat_example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{message}"),
    ("ai", "{triage}"),
])

few_shot_block = FewShotChatMessagePromptTemplate(
    example_prompt=chat_example_prompt,
    examples=examples,
)

final_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a support triage bot. Reply with one line in the format shown. "
     "Categories: performance, howto, billing, integration, bug, security."),
    few_shot_block,
    ("human", "{message}"),
])

for message in final_prompt.invoke({"message": "you charged us twice in january"}).to_messages():
    print(f"[{message.type:6}] {message.content}")

# %%
from langchain_core.output_parsers import StrOutputParser

triage_chain = final_prompt | model | StrOutputParser()
for ticket in raw_tickets:
    print("->", triage_chain.invoke({"message": ticket}).strip())

# %% [markdown]
# ## 4. Why you cannot just add 50 examples
#
# Every example is tokens in every request. Measure it.

# %%
def count_tokens(text: str) -> int:
    try:
        return model.get_num_tokens(text)
    except Exception:
        return len(text) // 4  # rough fallback if the provider has no counter


big_pool = examples * 12  # pretend we curated 48 examples
small = few_shot.format(message=raw_tickets[0])
large = FewShotPromptTemplate(
    examples=big_pool,
    example_prompt=example_prompt,
    prefix=few_shot.prefix,
    suffix=few_shot.suffix,
    input_variables=["message"],
).format(message=raw_tickets[0])

print(f"4 examples  : {count_tokens(small):>5} tokens")
print(f"48 examples : {count_tokens(large):>5} tokens")
print(f"Overhead    : {count_tokens(large) - count_tokens(small):>5} extra tokens on EVERY request")

# %% [markdown]
# ## 5. `LengthBasedExampleSelector`: fit the budget
#
# Simple and cheap: include as many examples as fit in a length budget. Good when
# your inputs vary a lot in size and all examples are roughly equally useful.

# %%
from langchain_core.example_selectors import LengthBasedExampleSelector

length_selector = LengthBasedExampleSelector(
    examples=big_pool,
    example_prompt=example_prompt,
    max_length=180,  # measured in words by the default get_text_length
)

adaptive = FewShotPromptTemplate(
    example_selector=length_selector,
    example_prompt=example_prompt,
    prefix=few_shot.prefix,
    suffix=few_shot.suffix,
    input_variables=["message"],
)

short_input = "help"
long_input = (
    "Since the maintenance window last night our scheduled reports are duplicating, "
    "the Salesforce connector keeps asking us to reauthorize, and two of our admins "
    "cannot log in at all. This is affecting our month-end close and we are on Enterprise."
)

print("short input ->", adaptive.format(message=short_input).count("Triage:") - 1, "examples included")
print("long input  ->", adaptive.format(message=long_input).count("Triage:") - 1, "examples included")

# %% [markdown]
# ## 6. `SemanticSimilarityExampleSelector`: pick the *relevant* examples
#
# This is the one you will actually use. Examples are embedded into a vector store;
# at request time the selector retrieves the `k` nearest to the current input.
#
# A billing question then gets billing examples, and a security question gets
# security examples - so a small `k` outperforms a large unfiltered set.

# %%
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.vectorstores import InMemoryVectorStore

diverse_examples = [
    {"message": "invoice 4412 has the wrong GST number", "triage": "category=billing | priority=medium | summary=Invoice reissue with corrected GST"},
    {"message": "we were charged twice for january seats", "triage": "category=billing | priority=high | summary=Duplicate charge on January invoice"},
    {"message": "can we downgrade in the middle of the cycle?", "triage": "category=billing | priority=low | summary=Mid-cycle downgrade question"},
    {"message": "webhooks return 401 after rotating the key", "triage": "category=integration | priority=high | summary=Webhook auth failing after key rotation"},
    {"message": "google sheets connector asks to reauthorize daily", "triage": "category=integration | priority=medium | summary=Sheets connector reauthorization loop"},
    {"message": "salesforce sync stopped after sandbox refresh", "triage": "category=integration | priority=high | summary=Salesforce sync broken post sandbox refresh"},
    {"message": "we need your SOC2 report for our vendor review", "triage": "category=security | priority=high | summary=SOC2 report requested for vendor review"},
    {"message": "where is our data physically stored?", "triage": "category=security | priority=medium | summary=Data residency question"},
    {"message": "dashboards take 30+ seconds on big datasets", "triage": "category=performance | priority=high | summary=Slow dashboard load on large datasets"},
    {"message": "api latency spiked this morning in europe", "triage": "category=performance | priority=high | summary=Elevated API latency in EU region"},
    {"message": "how do I export a dashboard to PDF?", "triage": "category=howto | priority=low | summary=PDF export guidance"},
    {"message": "where do I change report timezone?", "triage": "category=howto | priority=low | summary=Report timezone setting guidance"},
]

semantic_selector = SemanticSimilarityExampleSelector.from_examples(
    diverse_examples,
    get_embeddings(),
    InMemoryVectorStore,
    k=3,
)

for probe in ["our card was billed twice last month", "is the platform ISO certified?", "reports are really slow today"]:
    picked = semantic_selector.select_examples({"message": probe})
    print(f"\nInput: {probe}")
    for example in picked:
        print(f"   picked -> {example['message']}")

# %% [markdown]
# The selector pulled billing examples for a billing question and security examples
# for a compliance question, entirely from embedding similarity. No rules written.

# %%
semantic_few_shot = ChatPromptTemplate.from_messages([
    ("system", "You are a support triage bot. Reply with one line in the format shown."),
    FewShotChatMessagePromptTemplate(
        example_prompt=chat_example_prompt,
        example_selector=semantic_selector,
        input_variables=["message"],
    ),
    ("human", "{message}"),
])

smart_chain = semantic_few_shot | model | StrOutputParser()

for ticket in ["our card got billed twice", "need the SOC2 doc for procurement", "how do i add a read only user"]:
    print(f"{ticket:38} -> {smart_chain.invoke({'message': ticket}).strip()}")

# %% [markdown]
# ### Inspect what was actually sent
#
# Always check the rendered prompt when a few-shot system misbehaves - usually the
# selector picked something misleading.

# %%
for message in semantic_few_shot.invoke({"message": "need the SOC2 doc for procurement"}).to_messages():
    print(f"[{message.type:6}] {message.content[:100]}")

# %% [markdown]
# ## 7. Adding examples at runtime (a cheap feedback loop)
#
# When a human corrects the bot, append that correction as a new example. The next
# similar question is then handled correctly. This is the poor man's fine-tuning,
# and it is often enough.

# %%
before = smart_chain.invoke({"message": "our SSO metadata certificate expires next week"})
print("before:", before.strip())

semantic_selector.add_example({
    "message": "our SSO metadata certificate expires next week",
    "triage": "category=security | priority=high | summary=SSO certificate rotation required before expiry",
})

after = smart_chain.invoke({"message": "the SAML cert for our SSO is about to expire"})
print("after :", after.strip())

# %% [markdown]
# ## 8. Choosing a selector
#
# | Selector | Picks by | Use when |
# |---|---|---|
# | *(none - all examples)* | n/a | You have <= 5 examples and they are all relevant |
# | `LengthBasedExampleSelector` | fits a size budget | Inputs vary wildly in length; examples interchangeable |
# | `SemanticSimilarityExampleSelector` | embedding similarity | **Default choice** for a large, diverse example bank |
# | `MaxMarginalRelevanceExampleSelector` | similarity + diversity | Similar examples cluster and you want variety |
# | `NGramOverlapExampleSelector` | word overlap | Domain jargon matters more than semantics; no embeddings available |
#
# ### Practical rules
#
# 1. **3-5 examples is usually the sweet spot.** Gains flatten fast after that.
# 2. **Cover the edge cases, not the easy path.** Examples of ambiguous inputs teach
#    more than examples of obvious ones.
# 3. **Keep example outputs perfectly consistent.** The model copies your format
#    including your mistakes - one example with a trailing period teaches a
#    trailing period.
# 4. **Balance the classes.** Six billing examples and one security example biases
#    the classifier toward billing.

# %% [markdown]
# ## Try it yourself
#
# 1. **Measure the lift.** Score zero-shot vs 3-shot vs semantic-3-shot on the 15
#    tickets in `shared/sample_data/support_tickets.csv` against their `category`
#    column. Report accuracy for each.
# 2. **Poison an example.** Change one example's category to something wrong and
#    re-run. Observe how a single bad example propagates - this is why example
#    banks need review.
# 3. **Swap in MMR.** Replace the semantic selector with
#    `MaxMarginalRelevanceExampleSelector` and compare which examples get picked
#    for "billing was wrong again this month".

# %% [markdown]
# ## Recap
#
# | Concept | Takeaway |
# |---|---|
# | Few-shot | Showing the format beats describing it; biggest cheap accuracy win |
# | `FewShotPromptTemplate` | Text-style: prefix + rendered examples + suffix |
# | `FewShotChatMessagePromptTemplate` | Chat-style: alternating human/ai example turns |
# | Cost | Every example is tokens on every request - measure before scaling up |
# | `SemanticSimilarityExampleSelector` | Retrieve only relevant examples; default choice |
# | Runtime `add_example` | Human corrections become future examples |
#
# **When not to use few-shot:** when the task needs a guaranteed schema. Examples
# make the format *likely*; `.with_structured_output()` (notebook 24) makes it
# *enforced*. Use both together for best results.
#
# ## Next
#
# -> [05_output_parsers.ipynb](05_output_parsers.ipynb)
