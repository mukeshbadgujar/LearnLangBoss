# %% [markdown]
# # 05 - Output Parsers and Error Recovery
#
# | | |
# |---|---|
# | **Level** | Intermediate |
# | **Time** | 45 minutes |
# | **Prerequisites** | `04_few_shot_and_example_selectors` |
# | **Checklist ID** | `05_output_parsers` |
#
# ## Why this matters
#
# Your triage bot returns beautiful prose. Your database needs
# `{"category": "billing", "priority": "high"}`. Between those two facts lives a
# whole class of production bugs: the model wraps JSON in a code fence, adds
# "Sure, here's the JSON!", uses `High` instead of `high`, or truncates mid-object.
#
# Output parsers turn model text into typed Python objects **and** tell the model
# what shape you expect. This notebook also covers what to do when parsing fails,
# because at scale it will.
#
# > **Read this first:** in 2026 the primary tool for structured output is
# > `.with_structured_output()` (notebook 24), which uses the provider's native
# > JSON/tool-calling mode. Parsers remain essential for models without that
# > support, for non-JSON formats, and for repairing bad output - so learn both.

# %%
from pathlib import Path
import sys

_root = Path.cwd()
while not (_root / "shared").is_dir() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from shared.notebook_setup import setup, require  # noqa: E402

ctx = setup("05_output_parsers")

# %%
from shared.llm import get_chat_model

model = get_chat_model()

# %% [markdown]
# ## 1. The problem, demonstrated

# %%
naive = model.invoke(
    "Return the category and priority of this ticket as JSON: "
    "'we were charged twice for january seats'"
)
print(repr(naive.content))

import json

try:
    json.loads(naive.content)
    print("\nparsed fine (you got lucky - it is not guaranteed)")
except json.JSONDecodeError as exc:
    print(f"\njson.loads failed: {exc}")

# %% [markdown]
# Depending on the provider you either got clean JSON or JSON wrapped in
# ```` ```json ```` fences with a sentence of preamble. Writing your own string
# cleanup for this is a trap - every model has a different habit.

# %% [markdown]
# ## 2. `PydanticOutputParser`: schema in, typed object out
#
# Two jobs in one class:
# 1. `get_format_instructions()` generates the schema description for the prompt
# 2. `parse()` validates the response and returns a real Python object

# %%
from typing import Literal

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class Triage(BaseModel):
    """Structured triage of an inbound support ticket."""

    category: Literal["billing", "integration", "performance", "bug", "security", "howto"] = Field(
        description="The single best-fitting category"
    )
    priority: Literal["low", "medium", "high", "critical"] = Field(
        description="Urgency based on customer impact"
    )
    summary: str = Field(description="One-line internal summary, max 90 characters")
    needs_human: bool = Field(description="True if this requires a human specialist")


parser = PydanticOutputParser(pydantic_object=Triage)
print(parser.get_format_instructions())

# %% [markdown]
# Those format instructions get injected into the prompt via a partial - exactly
# the pattern from notebook 02.

# %%
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a support triage bot.\n{format_instructions}"),
    ("human", "Ticket: {ticket}"),
]).partial(format_instructions=parser.get_format_instructions())

chain = prompt | model | parser

result = chain.invoke({"ticket": "we were charged twice for january seats, this is the second month running"})

print(type(result).__name__)
print(result)
print("\nreal attribute access:", result.category, "|", result.priority, "|", result.needs_human)

# %% [markdown]
# `result` is a validated Pydantic object. `result.category` can only ever be one
# of the six literals - if the model returned `"Billing"` the parser would raise
# rather than silently hand you bad data.

# %%
tickets = [
    "our SOC2 report is needed for procurement by friday",
    "how do i rename a data source",
    "prod dashboards have been down for 40 minutes, entire ops team blocked",
]

for raw in tickets:
    triaged = chain.invoke({"ticket": raw})
    flag = "HUMAN" if triaged.needs_human else "auto "
    print(f"[{flag}] {triaged.category:12} {triaged.priority:8} {triaged.summary}")

# %% [markdown]
# ## 3. `JsonOutputParser`: dicts when you do not need a class

# %%
from langchain_core.output_parsers import JsonOutputParser

json_parser = JsonOutputParser()

loose_chain = ChatPromptTemplate.from_messages([
    ("system", "Reply with JSON only: {{\"category\": string, \"priority\": string, \"tags\": [string]}}"),
    ("human", "{ticket}"),
]) | model | json_parser

payload = loose_chain.invoke({"ticket": "salesforce sync stopped after our sandbox refresh"})
print(type(payload).__name__, payload)

# %% [markdown]
# `JsonOutputParser` also accepts a Pydantic model for its format instructions
# while still returning a plain dict - useful when you want the schema guidance
# but prefer dicts downstream.

# %%
schema_guided = JsonOutputParser(pydantic_object=Triage)
print(schema_guided.get_format_instructions()[:300], "...")

# %% [markdown]
# ### It streams, and that is its superpower
#
# `JsonOutputParser` emits partial objects as tokens arrive - which is how you
# build a UI that fills in fields progressively instead of waiting for the whole
# response.

# %%
stream_chain = ChatPromptTemplate.from_template(
    'Reply with JSON only: {{"customer": string, "issue": string, "impact": string, "recommended_action": string}}\n'
    "Ticket: {ticket}"
) | model | JsonOutputParser()

for partial in stream_chain.stream({"ticket": "Everline Bank reports 4s API p99 latency during EU hours"}):
    print(partial)

# %% [markdown]
# ## 4. `CommaSeparatedListOutputParser` and friends

# %%
from langchain_core.output_parsers import CommaSeparatedListOutputParser

list_parser = CommaSeparatedListOutputParser()
print("instructions:", list_parser.get_format_instructions())

tags = (
    ChatPromptTemplate.from_template("List 5 tags for this ticket.\n{format_instructions}\nTicket: {ticket}")
    .partial(format_instructions=list_parser.get_format_instructions())
    | model
    | list_parser
)

out = tags.invoke({"ticket": "webhook deliveries failing with 401 after key rotation"})
print(type(out).__name__, out)

# %% [markdown]
# Other built-ins worth knowing:
#
# | Parser | Output | Typical use |
# |---|---|---|
# | `StrOutputParser` | `str` | Strip the `AIMessage` wrapper - used in almost every chain |
# | `CommaSeparatedListOutputParser` | `list[str]` | Tags, keywords, options |
# | `DatetimeOutputParser` | `datetime` | Extracting dates from free text |
# | `EnumOutputParser` | `Enum` | Strict classification into a fixed set |
# | `XMLOutputParser` | `dict` | Models that are better at XML than JSON (older Claude) |
# | `MarkdownListOutputParser` | `list[str]` | Bulleted output |
#
# > **Import location matters in v1.** `StrOutputParser`, `JsonOutputParser`,
# > `PydanticOutputParser`, `CommaSeparatedListOutputParser` and `XMLOutputParser`
# > live in `langchain_core.output_parsers`. The less-used ones - `EnumOutputParser`,
# > `DatetimeOutputParser`, `OutputFixingParser`, `RetryOutputParser` - moved to
# > **`langchain_classic.output_parsers`**. Older tutorials import them from
# > `langchain.output_parsers`, which no longer exists.

# %%
from enum import Enum

from langchain_classic.output_parsers import EnumOutputParser


class Sentiment(str, Enum):
    ANGRY = "angry"
    FRUSTRATED = "frustrated"
    NEUTRAL = "neutral"
    HAPPY = "happy"


enum_parser = EnumOutputParser(enum=Sentiment)
sentiment_chain = (
    ChatPromptTemplate.from_template("Classify the sentiment.\n{format_instructions}\n\nMessage: {msg}")
    .partial(format_instructions=enum_parser.get_format_instructions())
    | model
    | enum_parser
)

for msg in ["this is the third time this month and nobody has called me back", "thanks, that fixed it!"]:
    print(f"{sentiment_chain.invoke({'msg': msg})!r:28} <- {msg[:50]}")

# %% [markdown]
# ## 5. When parsing fails
#
# Let us create a genuine failure and look at the exception, because recognising
# `OutputParserException` in a stack trace is half the battle.

# %%
from langchain_core.exceptions import OutputParserException

broken_output = 'Sure! Here you go:\n```json\n{"category": "Billing", "priority": "urgent"}\n```'

try:
    parser.parse(broken_output)
except OutputParserException as exc:
    print("OutputParserException raised.\n")
    print(str(exc)[:500])

# %% [markdown]
# Two separate problems in that output: `"Billing"` is not in our literal set,
# `"urgent"` is not a valid priority, and `summary`/`needs_human` are missing.

# %% [markdown]
# ## 6. `OutputFixingParser`: ask a model to repair it
#
# Wraps any parser. On failure it sends the broken text *and* the error to a model
# and asks for a corrected version. One extra LLM call, no prompt redesign.

# %%
from langchain_classic.output_parsers import OutputFixingParser

fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=model, max_retries=2)

repaired = fixing_parser.parse(broken_output)
print("repaired ->", repaired)

# %% [markdown]
# Notice it normalised `"Billing"` to `"billing"`, mapped `"urgent"` to a valid
# priority, and invented the missing fields. That last part is the danger: a fixing
# parser will **hallucinate values to satisfy a schema**. Only use it for format
# repair, never where a wrong-but-valid value is costly.

# %% [markdown]
# ## 7. `RetryOutputParser`: repair *with the original question*
#
# `OutputFixingParser` only sees the bad text. When the model omitted information
# it never had, fixing is guesswork. `RetryOutputParser` also passes the original
# prompt, so the retry can actually answer properly.

# %%
from langchain_classic.output_parsers import RetryOutputParser

retry_parser = RetryOutputParser.from_llm(parser=parser, llm=model, max_retries=2)

bad_prompt_value = prompt.invoke({"ticket": "prod is down for everyone since the 09:00 deploy"})
incomplete = '{"category": "bug"}'  # model stopped early

recovered = retry_parser.parse_with_prompt(incomplete, bad_prompt_value)
print("recovered ->", recovered)

# %% [markdown]
# | | `OutputFixingParser` | `RetryOutputParser` |
# |---|---|---|
# | Sees | broken output + error | broken output + error + **original prompt** |
# | Good at | format problems (fences, quotes, casing) | missing or wrong *content* |
# | Cost | 1 extra call | 1 extra call, larger prompt |
# | Risk | invents values | lower, but still an LLM guess |
# | API | `.parse(text)` | `.parse_with_prompt(text, prompt_value)` |

# %% [markdown]
# ## 8. Production pattern: validate, then fall back
#
# In a real service you want: try the cheap path, escalate only on failure, and
# always have a safe default so one bad response cannot take down a queue worker.

# %%
from langchain_core.runnables import RunnableLambda

strict_chain = prompt | model | parser
forgiving_chain = prompt | model | OutputFixingParser.from_llm(parser=parser, llm=model)

SAFE_DEFAULT = Triage(
    category="howto",
    priority="low",
    summary="Could not parse ticket automatically - needs manual triage",
    needs_human=True,
)

resilient = strict_chain.with_fallbacks(
    [forgiving_chain, RunnableLambda(lambda _: SAFE_DEFAULT)]
)

for raw in ["the thing is broken", "we need the SOC2 report and a DPA before we can renew"]:
    outcome = resilient.invoke({"ticket": raw})
    print(f"{outcome.category:12} {outcome.priority:8} human={outcome.needs_human}  {outcome.summary[:60]}")

# %% [markdown]
# `with_fallbacks` tries each option in order until one succeeds. Notebook 07
# covers it as a general LCEL primitive; here it gives you a parser pipeline that
# cannot raise.

# %% [markdown]
# ## 9. The modern alternative, previewed
#
# For any provider with native structured output, this replaces the whole prompt +
# parser dance:

# %%
try:
    structured_model = model.with_structured_output(Triage)
    direct = structured_model.invoke("Ticket: our webhook signatures stopped validating this morning")
    print(type(direct).__name__, "->", direct)
except Exception as exc:
    print(f"Provider does not support native structured output here ({type(exc).__name__}). "
          "Parsers remain the fallback - that is exactly why this notebook matters.")

# %% [markdown]
# Notebook 24 covers `.with_structured_output()` in depth, including the
# `ToolStrategy` vs `ProviderStrategy` distinction.
#
# **Decision guide:**
#
# | Situation | Use |
# |---|---|
# | Provider supports native structured output | `.with_structured_output()` |
# | Local/open model with no tool calling | `PydanticOutputParser` |
# | Output is not JSON (list, date, enum, XML) | The matching built-in parser |
# | Need streaming partial objects | `JsonOutputParser` |
# | Occasional malformed output in production | Wrap with `OutputFixingParser` |
# | Model omits information it should have had | `RetryOutputParser` |

# %% [markdown]
# ## Try it yourself
#
# 1. **Nested schema.** Add `class Customer(BaseModel)` with `name` and `plan`, and
#    nest it inside `Triage`. Re-run and inspect how `get_format_instructions()`
#    changes.
# 2. **Add a validator.** Use Pydantic's `field_validator` to reject a `summary`
#    longer than 90 characters, then find a ticket that trips it and watch the
#    fixing parser shorten it.
# 3. **Measure failure rate.** Run the strict chain over all 15 tickets in the CSV
#    with `temperature=1.0`, count `OutputParserException`s, then repeat with the
#    resilient chain and compare.

# %% [markdown]
# ## Recap
#
# | Concept | Takeaway |
# |---|---|
# | `PydanticOutputParser` | Schema -> format instructions + validated object |
# | `JsonOutputParser` | Dicts, and the only parser that streams partial objects |
# | `CommaSeparatedListOutputParser` / `EnumOutputParser` | Non-JSON shapes |
# | `OutputParserException` | The error to catch; read it, it names the failing field |
# | `OutputFixingParser` | Repairs *format*; may hallucinate missing values |
# | `RetryOutputParser` | Repairs *content* because it sees the original prompt |
# | `with_fallbacks` | Strict -> forgiving -> safe default; a chain that cannot raise |
# | `.with_structured_output()` | The modern default when the provider supports it |
#
# ## Next
#
# -> [06_memory.ipynb](06_memory.ipynb)
