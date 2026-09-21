# %% [markdown]
# # 48a - Reflection and Reflexion Agents
#
# | | |
# |---|---|
# | **Level** | Advanced |
# | **Time** | 55 minutes |
# | **Prerequisites** | `48_self_reflective_rag`, `43a_five_workflow_patterns` |
# | **Checklist ID** | `48a_reflection_and_reflexion` |
# | **Sourced from** | [emarco177/langgraph-course `project/reflection-agent`](https://github.com/emarco177/langgraph-course/tree/project/reflection-agent), [`project/reflexion-agent`](https://github.com/emarco177/langgraph-course/tree/project/reflexion-agent) |
#
# ## Why this matters
#
# Notebook 48 reflects on **retrieval** - are the documents relevant, is the
# answer grounded? Reflection and Reflexion reflect on the agent's **own
# output**. Same family, different target.
#
# | Pattern | Critiques | Then |
# |---|---|---|
# | Self-reflective RAG (48) | Retrieved docs + answer faithfulness | Rewrite query / regenerate / refuse |
# | **Reflection** (this lesson) | The draft itself | Revise the draft |
# | **Reflexion** (this lesson) | The draft, as structured critique + search queries | Research, then revise with citations |
#
# Reflection is evaluator-optimizer (43a) with a fixed critic persona.
# Reflexion adds tool-executed research driven by the critique.

# %%
from pathlib import Path
import sys

_root = Path.cwd()
while not (_root / "shared").is_dir() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from shared.notebook_setup import setup  # noqa: E402

ctx = setup("48a_reflection_and_reflexion")

# %%
import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from shared.llm import get_chat_model

model = get_chat_model()

# %% [markdown]
# ## 1. Reflection - generate ↔ critique over MessagesState
#
# The Eden Marco reflection agent writes a tweet, a critic grades it, the
# writer revises, until a message-count cap. The critic's output is fed back
# as a `HumanMessage` so the writer treats it as feedback, not as its own prior
# answer.

# %%
generate_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a concise tech writer. Produce the best short LinkedIn post you can "
     "for the user's request. If critique is provided, revise - do not defend."),
    MessagesPlaceholder("messages"),
])

reflect_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a demanding editor grading a LinkedIn post. Critique specificity, "
     "hook, length (aim under 80 words), and credibility. Give concrete revisions."),
    MessagesPlaceholder("messages"),
])

generate_chain = generate_prompt | model
reflect_chain = reflect_prompt | model


class ReflectState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def generation_node(state: ReflectState) -> dict:
    return {"messages": [generate_chain.invoke({"messages": state["messages"]})]}


def reflection_node(state: ReflectState) -> dict:
    critique = reflect_chain.invoke({"messages": state["messages"]})
    # Feed critique back as HumanMessage so the writer sees it as feedback.
    return {"messages": [HumanMessage(content=critique.content)]}


def should_continue(state: ReflectState) -> Literal["reflect", "__end__"]:
    # Cap by total messages: request + (draft + critique)*N
    return END if len(state["messages"]) > 6 else "reflect"


builder = StateGraph(ReflectState)
builder.add_node("generate", generation_node)
builder.add_node("reflect", reflection_node)
builder.add_edge(START, "generate")
builder.add_conditional_edges("generate", should_continue,
                              {"reflect": "reflect", END: END})
builder.add_edge("reflect", "generate")
reflection_agent = builder.compile()

print(reflection_agent.get_graph().draw_ascii())

# %%
seed = HumanMessage(
    "Write a LinkedIn post about why hybrid search (BM25 + dense) beats dense-only "
    "retrieval for company policy chatbots."
)
outcome = reflection_agent.invoke({"messages": [seed]})
print(f"{len(outcome['messages'])} messages in the trajectory\n")
for message in outcome["messages"]:
    role = message.__class__.__name__
    print(f"--- {role} ---\n{message.content[:280]}\n")

# %% [markdown]
# ## 2. Reflexion - structured self-critique that drives research
#
# Shinn et al.'s Reflexion pattern: the actor answers, critiques itself with a
# structured schema, emits search queries to fill the gaps, a tool runs those
# searches, and a revisor produces a cited answer. The structure is what makes
# the loop reliable.

# %%
class Reflection(BaseModel):
    missing: str = Field(description="What important information is missing")
    superfluous: str = Field(description="What should be cut")


class AnswerQuestion(BaseModel):
    """First-pass answer plus self-critique and research plan."""

    answer: str = Field(description="~120 word answer")
    reflection: Reflection
    search_queries: list[str] = Field(description="1-3 queries to improve the answer")


class ReviseAnswer(AnswerQuestion):
    """Revised answer with citations."""

    references: list[str] = Field(description="Sources motivating the update")


# Tiny local "web" so the lesson runs without Tavily.
RESEARCH = {
    "hybrid search": "Hybrid retrieval (BM25 + dense) typically adds 8-15 points of recall@5 over dense alone on policy corpora.",
    "bm25": "BM25 excels at exact tokens - product codes, error IDs, policy clause numbers - that embeddings blur.",
    "reranking": "A cross-encoder reranker adds 5-10 points of recall at ~200ms and ~$0.001 per query.",
    "rag evaluation": "Improving recall@5 from 60% to 85% moves answer accuracy more than swapping a 7B model for a 70B one.",
}


def fake_search(query: str) -> str:
    q = query.lower()
    hits = [v for k, v in RESEARCH.items() if k in q]
    return hits[0] if hits else f"No local result for {query!r}."

# %%
actor_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert researcher.\n"
     "1. {instruction}\n"
     "2. Reflect and critique your answer. Be severe.\n"
     "3. Recommend search queries to improve it."),
    MessagesPlaceholder("messages"),
    ("system", "Answer using the required schema."),
])

first_responder = actor_prompt.partial(
    instruction="Provide a detailed ~120 word answer."
) | model.with_structured_output(AnswerQuestion)

revisor = actor_prompt.partial(
    instruction=(
        "Revise using the new information. Include numerical citations like [1]. "
        "Add a References section. Keep the answer under 120 words."
    )
) | model.with_structured_output(ReviseAnswer)

# %%
class ReflexionState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    latest: dict
    research: Annotated[list[str], operator.add]
    revisions: Annotated[int, operator.add]


MAX_REVISIONS = 1


def respond(state: ReflexionState) -> dict:
    result = first_responder.invoke({"messages": state["messages"]})
    return {
        "latest": result.model_dump(),
        "messages": [AIMessage(content=result.answer)],
    }


def research(state: ReflexionState) -> dict:
    queries = state["latest"].get("search_queries") or []
    notes = [f"[{i}] {fake_search(q)} (query: {q})" for i, q in enumerate(queries, 1)]
    return {
        "research": notes,
        "messages": [HumanMessage(
            content="Research results:\n" + "\n".join(notes)
                    + f"\n\nPrior critique - missing: {state['latest']['reflection']['missing']}"
                      f"\nsuperfluous: {state['latest']['reflection']['superfluous']}"
        )],
    }


def revise(state: ReflexionState) -> dict:
    result = revisor.invoke({"messages": state["messages"]})
    return {
        "latest": result.model_dump(),
        "messages": [AIMessage(content=result.answer + "\n\nReferences:\n"
                               + "\n".join(result.references))],
        "revisions": 1,
    }


def after_respond(state: ReflexionState) -> Literal["research", "__end__"]:
    return "research" if state["latest"].get("search_queries") else END


def after_revise(state: ReflexionState) -> Literal["research", "__end__"]:
    return END if state["revisions"] >= MAX_REVISIONS else "research"


builder2 = StateGraph(ReflexionState)
builder2.add_node("respond", respond)
builder2.add_node("research", research)
builder2.add_node("revise", revise)
builder2.add_edge(START, "respond")
builder2.add_conditional_edges("respond", after_respond, {"research": "research", END: END})
builder2.add_edge("research", "revise")
builder2.add_conditional_edges("revise", after_revise, {"research": "research", END: END})
reflexion_agent = builder2.compile()

print(reflexion_agent.get_graph().draw_ascii())

# %%
final = reflexion_agent.invoke({
    "messages": [HumanMessage(
        "Why should a policy chatbot use hybrid search instead of dense-only retrieval?"
    )],
    "latest": {}, "research": [], "revisions": 0,
})
print(f"revisions: {final['revisions']}")
print(f"research notes: {len(final['research'])}")
print(f"\n{final['messages'][-1].content}")

# %% [markdown]
# ## 3. When to use which
#
# | Situation | Pattern |
# |---|---|
# | Draft quality is the bottleneck (copy, email, tweet) | Reflection |
# | Factual gaps are the bottleneck (research, analysis) | Reflexion |
# | Retrieved context is the bottleneck | Self-reflective RAG (48) |
# | A human must approve before shipping | Evaluator-optimizer + interrupt (43a + 36) |
#
# Reflection without a cap is an infinite loop. Reflexion without structured
# critique degenerates into "search for something related". The schema is the
# control surface.

# %% [markdown]
# ## Try it yourself
#
# 1. Replace `fake_search` with Tavily (if you have a key) and compare citation
#    quality.
# 2. Add a second revision pass and measure whether answer length stays under
#    the 120-word budget - Reflexion tends to bloat without a hard check.
# 3. Combine with notebook 48: retrieve first, then Reflexion-revise the answer
#    against the retrieved docs instead of web search.
#
# ## Recap
#
# | Concept | Takeaway |
# |---|---|
# | Reflection | Generate ↔ critique over `MessagesState`; critique fed back as `HumanMessage` |
# | Message-count cap | The simplest reliable stop condition |
# | Reflexion | Structured self-critique emits search queries; tools fill the gaps; revisor cites |
# | Schema as control | `AnswerQuestion` / `ReviseAnswer` keep the loop honest |
# | Reflection vs RAG reflection | Output quality vs retrieval quality - pick for the actual bottleneck |
#
# ## Next
#
# -> [49_long_term_semantic_memory.ipynb](49_long_term_semantic_memory.ipynb)
