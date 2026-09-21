# %% [markdown]
# # 28 - LangChain Hub and Prompt Management
#
# | | |
# |---|---|
# | **Level** | Intermediate |
# | **Time** | 35 minutes |
# | **Prerequisites** | `27_usage_tracking` |
# | **Checklist ID** | `28_langchain_hub` |
#
# ## Why this matters
#
# Prompts start life hardcoded in a Python file. That works until the day a
# product manager wants to tweak the tone, and the change requires a pull request,
# a code review and a deployment.
#
# Prompt management separates the prompt from the code: version it, review it,
# roll it back, and let non-engineers edit it. LangChain Hub is the hosted answer;
# this notebook also covers the file-based approach that works with no account at
# all, because a hosted registry is not always the right answer.

# %%
from pathlib import Path
import sys

_root = Path.cwd()
while not (_root / "shared").is_dir() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from shared.notebook_setup import setup, require  # noqa: E402

ctx = setup("28_langchain_hub")

# %%
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from shared.llm import get_chat_model

model = get_chat_model()
parser = StrOutputParser()

# %% [markdown]
# ## 1. Pulling a public prompt
#
# `hub.pull()` downloads a prompt by `owner/name` and returns a real
# `ChatPromptTemplate` you can pipe immediately. Public prompts need no API key;
# private ones and any `push` need `LANGSMITH_API_KEY`.

# %%
from langchain import hub

try:
    rag_prompt = hub.pull("rlm/rag-prompt")
    pulled = True
except Exception as exc:
    print(f"[offline] hub.pull failed ({type(exc).__name__}: {exc}).")
    print("          Using a local equivalent so the rest of the notebook runs.")
    rag_prompt = ChatPromptTemplate.from_template(
        "You are an assistant for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, just say "
        "that you don't know. Use three sentences maximum and keep the answer concise.\n"
        "Question: {question}\nContext: {context}\nAnswer:"
    )
    pulled = False

print("input variables:", rag_prompt.input_variables)
print()
print(rag_prompt.invoke({"context": "<CONTEXT>", "question": "<QUESTION>"}).to_messages()[0].content[:400])

# %% [markdown]
# `rlm/rag-prompt` is the most-pulled prompt on the Hub and a good example of why
# these prompts are worth reading: "If you don't know the answer, just say that
# you don't know" is doing the heavy lifting on hallucination, and the three
# sentence cap is what keeps RAG answers readable.

# %% [markdown]
# ## 2. Pinning a version
#
# **Never pull an unpinned prompt in production.** `hub.pull("owner/name")` gives
# you the latest version, which means someone else's edit can change your
# application's behaviour with no deploy on your side.

# %%
# Pinned by commit hash - reproducible forever:
#     hub.pull("rlm/rag-prompt:50442af1")
#
# Unpinned - whatever is latest today:
#     hub.pull("rlm/rag-prompt")

if pulled:
    try:
        pinned = hub.pull("rlm/rag-prompt:50442af1")
        print("pinned version pulled; identical to latest:", pinned.messages[0].prompt.template
              == rag_prompt.messages[0].prompt.template)
    except Exception as exc:
        print(f"pinned pull unavailable: {type(exc).__name__}")

# %% [markdown]
# ## 3. Using a pulled prompt in a real chain

# %%
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

from shared.llm import get_embeddings

text = Path(ctx.data("leave_policy.txt")).read_text(encoding="utf-8")
chunks = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=70).split_text(text)
store = FAISS.from_documents([Document(c, metadata={"source": "leave_policy.txt"}) for c in chunks], get_embeddings())
retriever = store.as_retriever(search_kwargs={"k": 3})


def format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | model
    | parser
)

print(rag_chain.invoke("How many days of casual leave can I take?").strip())

# %% [markdown]
# The prompt came from a registry, the retriever and model are yours. That
# separation is the whole point: the prompt becomes configuration.

# %% [markdown]
# ## 4. Prompts worth knowing
#
# | Prompt | What it is | Where it appears |
# |---|---|---|
# | `rlm/rag-prompt` | Minimal grounded QA prompt | Almost every RAG tutorial |
# | `hwchase17/react` | The original ReAct Thought/Action/Observation loop | Notebook 17's history section |
# | `hwchase17/openai-tools-agent` | Tool-calling agent scaffold | Pre-`create_agent` agents |
# | `hwchase17/structured-chat-agent` | Multi-input tool agent | Legacy agent code |
# | `langchain-ai/retrieval-qa-chat` | RAG with chat history | History-aware retrieval |
#
# Reading `hwchase17/react` is genuinely educational - it shows exactly how
# agentic behaviour was coaxed out of models that had no native tool calling.

# %%
if pulled:
    try:
        react = hub.pull("hwchase17/react")
        template = react.messages[0].prompt.template if hasattr(react, "messages") else react.template
        print(template[:700])
    except Exception as exc:
        print(f"[skipped] {type(exc).__name__}: {exc}")

# %% [markdown]
# ## 5. Pushing your own prompt
#
# Needs `LANGSMITH_API_KEY`. Pushing creates a new immutable version each time and
# returns its URL.

# %%
triage_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You triage inbound support tickets for Northwind Analytics.\n"
     "Return: category (billing|integration|performance|bug|security|howto), "
     "priority (low|medium|high|critical), and a one-line summary.\n"
     "Critical is reserved for data loss, security incidents, or a full outage.\n"
     "Never guess a customer's plan tier - if it is not stated, say unknown."),
    ("human", "Ticket from {customer} ({plan} plan):\n{ticket}"),
])

if require("LANGSMITH_API_KEY", feature="pushing prompts to LangChain Hub"):
    try:
        url = hub.push("northwind-ticket-triage", triage_prompt)
        print("pushed:", url)
    except Exception as exc:
        print(f"push failed ({type(exc).__name__}: {exc})")
        print("Prompts push to your LangSmith workspace; check the key has write access.")

# %% [markdown]
# Each push is a new version with its own hash. Roll back by pulling an older
# hash - no code change, no deploy.

# %% [markdown]
# ## 6. File-based prompt management (no account needed)
#
# For many teams this is the better answer: prompts live in the repo, so they get
# code review, git history and atomic rollback with the code that uses them - and
# there is no runtime dependency on a third-party service.

# %%
import json

PROMPT_DIR = ctx.artifact("prompts")
PROMPT_DIR.mkdir(parents=True, exist_ok=True)

registry = {
    "ticket_triage": {
        "version": "2.1.0",
        "description": "Classify and prioritise inbound support tickets",
        "messages": [
            ["system",
             "You triage inbound support tickets for Northwind Analytics.\n"
             "Return category, priority and a one-line summary.\n"
             "Critical is reserved for data loss, security incidents, or a full outage."],
            ["human", "Ticket from {customer} ({plan} plan):\n{ticket}"],
        ],
        "changelog": "2.1.0 narrowed 'critical' after too many false alarms in v2.0.0",
    },
    "policy_qa": {
        "version": "1.3.0",
        "description": "Answer HR policy questions from retrieved context only",
        "messages": [
            ["system",
             "Answer using only the provided context. If the context does not contain the "
             "answer, say so. Cite the source filename. At most three sentences."],
            ["human", "Context:\n{context}\n\nQuestion: {question}"],
        ],
        "changelog": "1.3.0 added source citation requirement",
    },
}

(PROMPT_DIR / "registry.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")


def load_prompt(name: str) -> ChatPromptTemplate:
    """Load a versioned prompt from the local registry."""
    data = json.loads((PROMPT_DIR / "registry.json").read_text(encoding="utf-8"))
    if name not in data:
        raise KeyError(f"Unknown prompt {name!r}. Available: {', '.join(data)}")
    entry = data[name]
    prompt = ChatPromptTemplate.from_messages([(role, body) for role, body in entry["messages"]])
    prompt.metadata = {"name": name, "version": entry["version"]}
    return prompt


triage = load_prompt("ticket_triage")
print(f"loaded {triage.metadata}")
print("inputs:", triage.input_variables)

chain = triage | model | parser
print()
print(chain.invoke({
    "customer": "Globex",
    "plan": "enterprise",
    "ticket": "Our scheduled reports stopped sending three days ago. Finance is blocked on month-end close.",
}).strip())

# %% [markdown]
# ### Hub vs files
#
# | | LangChain Hub | Files in the repo |
# |---|---|---|
# | Non-engineers can edit | Yes, in the UI | No |
# | Change without deploying | Yes | No |
# | Code review on changes | Separate flow | Standard PR review |
# | Rollback | Pull an older hash | `git revert` |
# | Runtime dependency | Network + service uptime | None |
# | Works offline / air-gapped | No | Yes |
# | Prompt and code stay in sync | Manual discipline | Automatic |
#
# Use the Hub when prompt iteration is owned by people who do not deploy code.
# Use files when prompts and code change together - which is most of the time.
# **Never use the Hub unpinned in production**, whichever you choose.

# %% [markdown]
# ## 7. A/B testing prompt versions
#
# Versioned prompts plus notebook 26's evaluation harness gives you a real
# workflow: change the prompt, measure, keep or revert.

# %%
VARIANTS = {
    "v1_terse": ChatPromptTemplate.from_messages([
        ("system", "Answer using only the context. Be brief."),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]),
    "v2_grounded": ChatPromptTemplate.from_messages([
        ("system",
         "Answer using only the provided context. If the context does not contain the answer, "
         "reply exactly: 'That is not covered in the documents I have.' Cite the source filename. "
         "At most three sentences."),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]),
}

PROBES = [
    ("How many days of casual leave?", "6", True),
    ("How many sick leave days per year?", "12", True),
    ("What is the pet insurance policy?", None, False),
]

print(f"{'variant':14} {'facts':>7} {'refusals':>10}")
for name, prompt in VARIANTS.items():
    variant_chain = {"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | model | parser
    facts = refusals = 0
    for question, fact, answerable in PROBES:
        answer = variant_chain.invoke(question).lower()
        if answerable:
            facts += fact in answer
        else:
            refusals += any(m in answer for m in ["not covered", "don't have", "do not have", "no information"])
    print(f"{name:14} {facts:>3}/2    {refusals:>6}/1")

# %% [markdown]
# The explicit refusal instruction in `v2_grounded` is usually what separates a
# demo from something you can put in front of employees. Small prompt edits,
# measured - that is the whole discipline.

# %% [markdown]
# ## 8. Prompts as configuration in a service
#
# The shape most production systems converge on: load prompts at startup, allow
# a version override per request, and log which version produced each answer.

# %%
class PromptRegistry:
    """Loads versioned prompts once and reports which version served each call."""

    def __init__(self, path: Path):
        self.path = path
        self._cache: dict[str, ChatPromptTemplate] = {}
        self._data = json.loads(path.read_text(encoding="utf-8"))

    def get(self, name: str) -> ChatPromptTemplate:
        if name not in self._cache:
            entry = self._data[name]
            prompt = ChatPromptTemplate.from_messages([(r, b) for r, b in entry["messages"]])
            prompt.metadata = {"name": name, "version": entry["version"]}
            self._cache[name] = prompt
        return self._cache[name]

    def reload(self) -> None:
        """Pick up edits without restarting the process."""
        self._data = json.loads(self.path.read_text(encoding="utf-8"))
        self._cache.clear()

    def versions(self) -> dict[str, str]:
        return {name: entry["version"] for name, entry in self._data.items()}


prompts = PromptRegistry(PROMPT_DIR / "registry.json")
print("deployed versions:", prompts.versions())

prompt = prompts.get("policy_qa")
answer = ({"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | model | parser).invoke(
    "Can I carry forward unused annual leave?",
    config={"metadata": {"prompt_name": "policy_qa", "prompt_version": prompt.metadata["version"]}},
)
print(f"\n[policy_qa v{prompt.metadata['version']}] {answer.strip()}")

# %% [markdown]
# Logging `prompt_version` as run metadata (notebook 19) is what lets you answer
# "did quality drop when we shipped v1.3.0?" three weeks later. Without it, prompt
# changes are invisible in your traces.

# %% [markdown]
# ## Try it yourself
#
# 1. **Pull and read three Hub prompts** - `rlm/rag-prompt`,
#    `hwchase17/react`, `langchain-ai/retrieval-qa-chat` - and write down one
#    technique from each that you would reuse.
# 2. **Add a third variant** to the A/B test that requires a confidence statement,
#    and measure whether it hurts fact recall.
# 3. **Version bump workflow.** Edit `ticket_triage` to 2.2.0, call `reload()`,
#    and confirm the running registry serves the new version without a restart.
# 4. **Wire it to evaluation.** Run notebook 26's harness across two prompt
#    versions and produce a keep-or-revert recommendation from the numbers.

# %% [markdown]
# ## Recap
#
# | Concept | Takeaway |
# |---|---|
# | `hub.pull("owner/name")` | Returns a real `ChatPromptTemplate`; public prompts need no key |
# | **Pinning** | `owner/name:hash` - never run unpinned prompts in production |
# | `hub.push(...)` | Needs `LANGSMITH_API_KEY`; each push is an immutable version |
# | Notable prompts | `rlm/rag-prompt`, `hwchase17/react` are worth reading closely |
# | File registry | Git history, PR review, offline, no runtime dependency |
# | Hub vs files | Hub when non-engineers own iteration; files when prompts ship with code |
# | A/B testing | Version prompts, measure with notebook 26, keep or revert |
# | `prompt_version` metadata | Makes prompt changes visible in traces weeks later |
#
# ## Next
#
# -> [29_graphs_vs_agents.ipynb](../05-langgraph-beginner/29_graphs_vs_agents.ipynb)
#
# That completes the LangChain half of the curriculum. Track 05 starts LangGraph:
# why graphs exist, and what they do that chains and agents cannot.
