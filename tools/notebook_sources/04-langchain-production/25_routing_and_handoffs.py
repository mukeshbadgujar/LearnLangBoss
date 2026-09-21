# %% [markdown]
# # 25 - Routing and Handoffs
#
# | | |
# |---|---|
# | **Level** | Intermediate |
# | **Time** | 40 minutes |
# | **Prerequisites** | `24_structured_outputs` |
# | **Checklist ID** | `25_routing_and_handoffs` |
#
# ## Why this matters
#
# One prompt cannot be good at everything. A system prompt that tries to cover
# billing rules, security policy, technical troubleshooting and small talk becomes
# a 2,000-token compromise that is mediocre at all four.
#
# Routing fixes this: classify the request, then send it to a specialist that does
# one thing well. The benefits are concrete - shorter prompts, cheaper calls, and
# the ability to use a small model for easy paths and a large one for hard paths.

# %%
from pathlib import Path
import sys

_root = Path.cwd()
while not (_root / "shared").is_dir() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from shared.notebook_setup import setup, require  # noqa: E402

ctx = setup("25_routing_and_handoffs")

# %%
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from shared.llm import get_chat_model

model = get_chat_model()
parser = StrOutputParser()

QUESTIONS = [
    "We were charged twice for January seats.",
    "Where is our data physically stored for GDPR purposes?",
    "Our dashboards take 40 seconds to load.",
    "How do I invite a read-only user?",
    "What's the weather like in Bengaluru?",
]

# %% [markdown]
# ## 1. The specialists
#
# Each one is short, focused and independently testable.

# %%
def specialist(role: str, rules: str):
    return (
        ChatPromptTemplate.from_messages([
            ("system", f"You are {role} at Northwind Analytics.\n{rules}\nAnswer in at most 3 sentences."),
            ("human", "{question}"),
        ])
        | model
        | parser
    )


billing = specialist(
    "a billing specialist",
    "Quote policy precisely. Downgrades take effect next cycle with no partial refund. "
    "Upgrades are immediate and pro-rated. Never promise a refund you cannot authorise.",
)

security = specialist(
    "a security and compliance officer",
    "Primary region is ap-south-1 (Mumbai); EU customers may request eu-west-1. "
    "SOC 2 Type II is available under NDA. Never confirm anything you cannot verify; "
    "escalate contract-specific questions to the compliance team.",
)

technical = specialist(
    "a senior support engineer",
    "Ask for the specific symptom, dataset size and time window. "
    "Give concrete diagnostic steps. Never promise a fix date.",
)

howto = specialist(
    "a friendly product guide",
    "Give short numbered steps. Link to documentation conceptually rather than inventing URLs.",
)

out_of_scope = (
    ChatPromptTemplate.from_template(
        "The user asked something outside Northwind support: {question}\n"
        "Politely say you can only help with Northwind Analytics topics. One sentence."
    )
    | model
    | parser
)

ROUTES = {
    "billing": billing,
    "security": security,
    "technical": technical,
    "howto": howto,
    "other": out_of_scope,
}

# %% [markdown]
# ## 2. Keyword routing: cheapest, and often enough

# %%
from langchain_core.runnables import RunnableBranch

KEYWORDS = {
    "billing": ["invoice", "charge", "charged", "refund", "billing", "plan", "price", "downgrade", "upgrade"],
    "security": ["soc2", "soc 2", "gdpr", "security", "residency", "compliance", "dpa", "audit", "encryption"],
    "technical": ["slow", "latency", "error", "failing", "timeout", "broken", "down", "500", "401"],
    "howto": ["how do i", "how to", "where do i", "can i find", "steps"],
}


def keyword_route(payload: dict) -> str:
    text = payload["question"].lower()
    scores = {name: sum(word in text for word in words) for name, words in KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "other"


keyword_router = RunnableBranch(
    *[(lambda p, n=name: keyword_route(p) == n, chain) for name, chain in ROUTES.items() if name != "other"],
    ROUTES["other"],
)

for question in QUESTIONS:
    print(f"[{keyword_route({'question': question}):9}] {question}")

# %% [markdown]
# Keyword routing is free, instant and fully predictable. Its weakness is
# vocabulary: "you took money twice" contains no billing keyword.

# %%
print(keyword_route({"question": "you took money out of our account twice this month"}))
print("  <- misrouted; no keyword matched")

# %% [markdown]
# ## 3. LLM routing with structured output
#
# Robust to phrasing, costs one small model call.

# %%
from typing import Literal

from pydantic import BaseModel, Field


class Route(BaseModel):
    """Which specialist should handle this request."""

    destination: Literal["billing", "security", "technical", "howto", "other"] = Field(
        description="billing: invoices, charges, refunds, plan changes. "
        "security: compliance, certifications, data residency, access control. "
        "technical: errors, slowness, outages, integrations that stopped working. "
        "howto: usage questions where nothing is broken. "
        "other: anything unrelated to Northwind Analytics."
    )
    confidence: Literal["low", "medium", "high"] = Field(description="How certain the routing is")
    reason: str = Field(description="Short justification")


classifier = (
    ChatPromptTemplate.from_messages([
        ("system", "You route Northwind support requests to the right specialist."),
        ("human", "{question}"),
    ])
    | model.with_structured_output(Route)
)

for question in QUESTIONS + ["you took money out of our account twice this month"]:
    route = classifier.invoke({"question": question})
    print(f"[{route.destination:9} {route.confidence:6}] {question[:52]:54} {route.reason[:45]}")

# %% [markdown]
# The paraphrase that defeated keyword routing is handled correctly.

# %%
from langchain_core.runnables import RunnableLambda, RunnablePassthrough


def dispatch(payload: dict) -> str:
    return ROUTES[payload["route"].destination].invoke({"question": payload["question"]})


llm_router = (
    RunnablePassthrough.assign(route=classifier)
    .assign(answer=RunnableLambda(dispatch))
)

for question in QUESTIONS[:3]:
    outcome = llm_router.invoke({"question": question})
    print(f"\n[{outcome['route'].destination}] {question}")
    print(f"  {outcome['answer'].strip()[:200]}")

# %% [markdown]
# ## 4. Semantic routing with embeddings
#
# No LLM call at all: embed a few example utterances per route, embed the query,
# pick the nearest. Faster and cheaper than LLM routing, more flexible than
# keywords.

# %%
import math

from shared.llm import get_embeddings

embeddings = get_embeddings()

ROUTE_EXAMPLES = {
    "billing": [
        "we were charged twice this month",
        "can we get a refund for the unused period",
        "the invoice has the wrong tax details",
        "how much does the enterprise plan cost",
        "we want to downgrade our subscription",
    ],
    "security": [
        "where is our data stored",
        "do you have SOC 2 certification",
        "we need a data processing agreement",
        "how is data encrypted at rest",
        "who can access our workspace",
    ],
    "technical": [
        "the dashboard will not load",
        "we are getting 401 errors from the API",
        "the salesforce sync stopped working",
        "everything is extremely slow today",
        "reports are being sent twice",
    ],
    "howto": [
        "how do I add a new user",
        "where do I change the timezone",
        "what are the steps to export data",
        "how can I schedule a report",
        "where do I find my API key",
    ],
}

route_centroids: dict[str, list[float]] = {}
for name, examples in ROUTE_EXAMPLES.items():
    vectors = embeddings.embed_documents(examples)
    route_centroids[name] = [sum(values) / len(values) for values in zip(*vectors)]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def semantic_route(question: str, floor: float = 0.35) -> tuple[str, float]:
    query = embeddings.embed_query(question)
    scores = {name: cosine(query, centroid) for name, centroid in route_centroids.items()}
    best = max(scores, key=scores.get)
    return (best, scores[best]) if scores[best] >= floor else ("other", scores[best])


for question in QUESTIONS + ["you took money out of our account twice this month"]:
    destination, score = semantic_route(question)
    print(f"[{destination:9} {score:.3f}] {question}")

# %% [markdown]
# The `floor` is what gives you an "other" bucket. Without it, every off-topic
# question gets forced into the nearest route - which is how a weather question
# ends up with the billing team.

# %% [markdown]
# ## 5. Comparing the three

# %%
import time

LABELLED = [
    ("We were charged twice for January seats.", "billing"),
    ("you took money out of our account twice", "billing"),
    ("Where is our data physically stored?", "security"),
    ("do you hold any certifications our auditors would want", "security"),
    ("Our dashboards take 40 seconds to load.", "technical"),
    ("nothing loads since this morning", "technical"),
    ("How do I invite a read-only user?", "howto"),
    ("what are the steps to schedule a weekly report", "howto"),
    ("What's the weather in Bengaluru?", "other"),
    ("can you write me a poem", "other"),
]

strategies = {
    "keyword": lambda q: keyword_route({"question": q}),
    "semantic": lambda q: semantic_route(q)[0],
    "llm": lambda q: classifier.invoke({"question": q}).destination,
}

print(f"{'strategy':10} {'accuracy':>9} {'sec/query':>10}")
for name, route_fn in strategies.items():
    start = time.perf_counter()
    correct = sum(route_fn(question) == expected for question, expected in LABELLED)
    elapsed = (time.perf_counter() - start) / len(LABELLED)
    print(f"{name:10} {correct}/{len(LABELLED):<7} {elapsed:>10.3f}")

# %% [markdown]
# | Strategy | Cost | Latency | Robustness | Use when |
# |---|---|---|---|---|
# | Keyword | free | ~0 ms | low | Controlled vocabulary, internal tools |
# | Semantic | one embedding | ~10-50 ms | good | **Best default** for most routers |
# | LLM | one model call | 300-2000 ms | best | Nuanced or overlapping categories |
#
# A common production shape is **cascading**: keywords catch the obvious cases,
# semantics handles the rest, and the LLM is consulted only when semantic
# confidence is low.

# %%
def cascading_route(question: str) -> tuple[str, str]:
    keyword_hit = keyword_route({"question": question})
    if keyword_hit != "other":
        return keyword_hit, "keyword"

    destination, score = semantic_route(question)
    if score >= 0.55:
        return destination, f"semantic ({score:.2f})"

    return classifier.invoke({"question": question}).destination, "llm"


for question, expected in LABELLED:
    destination, how = cascading_route(question)
    mark = "OK " if destination == expected else "X  "
    print(f"{mark} {destination:9} via {how:18} {question[:48]}")

# %% [markdown]
# ## 6. Model routing: cheap model for easy work
#
# Routing is not only about prompts. Send simple requests to a small fast model
# and hard ones to a large model - often the largest cost saving available.

# %%
from shared.llm import available_providers


class Complexity(BaseModel):
    """How hard is this request?"""

    level: Literal["simple", "complex"] = Field(
        description="simple: a factual lookup or a short how-to. "
        "complex: multi-step reasoning, policy interpretation, or an unhappy customer."
    )


complexity_checker = ChatPromptTemplate.from_template("{question}") | model.with_structured_output(Complexity)

# Substitute real model names for your provider; these are illustrative.
try:
    fast_model = get_chat_model("llama-3.1-8b-instant") if available_providers()[0] == "groq" else model
except Exception:
    fast_model = model
strong_model = model


def model_routed_answer(question: str) -> tuple[str, str]:
    level = complexity_checker.invoke({"question": question}).level
    chosen = fast_model if level == "simple" else strong_model
    answer = (ChatPromptTemplate.from_template("Answer briefly: {question}") | chosen | parser).invoke(
        {"question": question}
    )
    return answer, level


for question in ["Where do I change the report timezone?",
                 "We downgraded mid-cycle, were charged full price, and our SLA was missed twice. What are we owed?"]:
    answer, level = model_routed_answer(question)
    print(f"[{level:7}] {question[:55]}\n           {answer.strip()[:170]}\n")

# %% [markdown]
# ## 7. Handoffs: when one specialist passes to another
#
# Routing decides once at the start. A **handoff** happens mid-conversation when
# the specialist realises the request belongs elsewhere.

# %%
class HandoffDecision(BaseModel):
    """Whether the current specialist can finish, or must hand off."""

    can_handle: bool = Field(description="True if the current specialist can fully answer")
    handoff_to: Literal["billing", "security", "technical", "howto", "none"] = Field(
        description="Where to hand off, or 'none' if no handoff is needed"
    )
    context_for_next: str = Field(description="What the next specialist needs to know")


handoff_checker = (
    ChatPromptTemplate.from_messages([
        ("system", "You are {current_role}. Decide whether you can fully handle this request."),
        ("human", "{question}"),
    ])
    | model.with_structured_output(HandoffDecision)
)


def handle_with_handoff(question: str, start_at: str = "howto", max_hops: int = 3) -> dict:
    """Route, and allow the specialist to hand off if it is the wrong desk."""
    trail = []
    current = start_at
    context = question

    for hop in range(max_hops):
        trail.append(current)
        decision = handoff_checker.invoke({"current_role": f"the {current} specialist", "question": context})

        if decision.can_handle or decision.handoff_to in ("none", current):
            answer = ROUTES[current].invoke({"question": context})
            return {"answer": answer, "trail": trail, "hops": hop + 1}

        context = f"{question}\n\n[Handoff note from {current}: {decision.context_for_next}]"
        current = decision.handoff_to

    return {"answer": ROUTES[current].invoke({"question": context}), "trail": trail, "hops": max_hops}


outcome = handle_with_handoff(
    "How do I export our data? We need it because we are cancelling and want it before you delete it.",
    start_at="howto",
)
print("trail :", " -> ".join(outcome["trail"]))
print("answer:", outcome["answer"].strip()[:280])

# %% [markdown]
# Note the loop guard. Without `max_hops`, two specialists can bounce a request
# between them indefinitely - a real failure mode in multi-agent systems, and one
# LangGraph handles more cleanly (notebook 41).

# %% [markdown]
# ## 8. Fallbacks and low-confidence handling
#
# A router that is 90% accurate sends 1 in 10 users to the wrong desk. Handle low
# confidence explicitly rather than pretending it does not happen.

# %%
def routed_answer(question: str) -> dict:
    route = classifier.invoke({"question": question})

    if route.confidence == "low":
        clarifier = ChatPromptTemplate.from_template(
            "A user asked: {question}\nYou are unsure whether this is a billing, security, "
            "technical or how-to question. Ask ONE short clarifying question."
        ) | model | parser
        return {"destination": "clarify", "answer": clarifier.invoke({"question": question}).strip()}

    return {
        "destination": route.destination,
        "answer": ROUTES[route.destination].invoke({"question": question}).strip(),
    }


for question in ["it's not working", "We were charged twice for January."]:
    outcome = routed_answer(question)
    print(f"[{outcome['destination']}] {question}\n   {outcome['answer'][:200]}\n")

# %% [markdown]
# ## 9. Where routing hits its limits
#
# `RunnableBranch` picks one path and runs it to completion. It cannot:
#
# - let a specialist come back and re-route after discovering something
# - run two specialists in parallel and merge their answers
# - remember, on turn 5, which desk handled turn 2
# - pause mid-handoff for human approval
#
# Those are graph problems. Notebook 32 does conditional routing in LangGraph and
# notebook 41 does true multi-agent handoffs with shared state.

# %% [markdown]
# ## Try it yourself
#
# 1. **Grow the labelled set to 30 examples**, including five genuinely ambiguous
#    ones, and re-measure all three strategies.
# 2. **Tune the semantic floor.** Sweep it from 0.2 to 0.6 and plot correct
#    routing against false "other" classifications.
# 3. **Add a fifth route** for "account management" (renewals, seats, contacts)
#    and see how much it degrades the others - category overlap is the main enemy
#    of routers.
# 4. **Cost model.** For 10,000 daily requests, compute the monthly cost of LLM
#    routing versus semantic routing at your provider's prices.

# %% [markdown]
# ## Recap
#
# | Concept | Takeaway |
# |---|---|
# | Why route | Short focused prompts beat one long general prompt |
# | `RunnableBranch` | Condition/runnable pairs, default last |
# | Keyword routing | Free and predictable; fails on unexpected vocabulary |
# | Semantic routing | Embed examples, compare centroids; **best default** |
# | LLM routing | Most robust, slowest; use structured output for the decision |
# | Cascading | Keywords -> semantics -> LLM only when uncertain |
# | Model routing | Small model for easy paths; often the biggest cost saving |
# | Handoffs | Mid-conversation re-routing; **always cap the hops** |
# | Low confidence | Ask a clarifying question instead of guessing |
#
# ## Next
#
# -> [26_evaluation.ipynb](26_evaluation.ipynb)
