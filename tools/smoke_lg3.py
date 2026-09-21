"""Runtime smoke test for the non-LLM patterns in notebooks 47-50.

Verifies store semantics, the state-backed workspace tools, todo/planning
middleware wiring and the depth guard, all without calling a real model.
"""
import operator
import sys
import time
import uuid
from pathlib import Path
from typing import Annotated, TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PASS, FAIL = 0, 0


def check(label, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ok   {label}" + (f"  ({detail})" if detail else ""))
    else:
        FAIL += 1
        print(f"  FAIL {label}  {detail}")


print("== 47: token budget + map-reduce fan-out ==")
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send


def token_budget(window, system_tokens, tool_tokens=0, reserve=1500, margin=200):
    return window - system_tokens - tool_tokens - reserve - margin


check("budget shrinks with tools", token_budget(8192, 60, 600) < token_budget(8192, 60, 0))
check("budget is positive on 8k", token_budget(8192, 60, 600) > 0,
      str(token_budget(8192, 60, 600)))


class MR(TypedDict):
    document: str
    chunks: list
    summaries: Annotated[list, operator.add]
    answer: str


def split(state: MR) -> dict:
    return {"chunks": RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
            .split_text(state["document"])}


def fan_out(state: MR):
    return [Send("one", {"chunk": c, "index": i}) for i, c in enumerate(state["chunks"])]


def one(state: dict) -> dict:
    if "skip" in state["chunk"]:
        return {"summaries": []}
    return {"summaries": [f"{state['index']}"]}


def reduce_(state: MR) -> dict:
    return {"answer": ",".join(state["summaries"])}


b = StateGraph(MR)
b.add_node("split", split)
b.add_node("one", one)
b.add_node("reduce", reduce_)
b.add_edge(START, "split")
b.add_conditional_edges("split", fan_out, ["one"])
b.add_edge("one", "reduce")
b.add_edge("reduce", END)
mr = b.compile()

doc = ("alpha " * 60) + "skip " * 60 + ("beta " * 60)
out = mr.invoke({"document": doc, "chunks": [], "summaries": [], "answer": ""})
check("map-reduce fans out", len(out["chunks"]) > 1, f"{len(out['chunks'])} chunks")
check("NOTHING-RELEVANT filter drops chunks",
      len(out["summaries"]) < len(out["chunks"]),
      f"{len(out['summaries'])}/{len(out['chunks'])} kept")

print("\n== 47: refine loop terminates ==")


class RF(TypedDict):
    chunks: list
    index: int
    running: str
    steps: Annotated[int, operator.add]


def prep(state: RF) -> dict:
    return {"chunks": ["a", "b", "c"], "index": 0, "running": ""}


def step(state: RF) -> dict:
    return {"running": state["running"] + state["chunks"][state["index"]],
            "index": state["index"] + 1, "steps": 1}


def more(state: RF):
    return "refine" if state["index"] < len(state["chunks"]) else END


b2 = StateGraph(RF)
b2.add_node("prep", prep)
b2.add_node("refine", step)
b2.add_edge(START, "prep")
b2.add_edge("prep", "refine")
b2.add_conditional_edges("refine", more, {"refine": "refine", END: END})
rg = b2.compile()
r = rg.invoke({"chunks": [], "index": 0, "running": "", "steps": 0}, {"recursion_limit": 20})
check("refine visits every chunk", r["running"] == "abc" and r["steps"] == 3, r["running"])

print("\n== 47/37: RemoveMessage pruning ==")
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.graph.message import add_messages


class CS(TypedDict):
    messages: Annotated[list, add_messages]
    summary: str


def compress(state: CS) -> dict:
    older = state["messages"][:-2]
    return {"summary": f"summary of {len(older)}",
            "messages": [RemoveMessage(id=m.id) for m in older]}


b3 = StateGraph(CS)
b3.add_node("compress", compress)
b3.add_edge(START, "compress")
b3.add_edge("compress", END)
cg = b3.compile()
seed = [HumanMessage(f"m{i}", id=f"h{i}") for i in range(6)]
res = cg.invoke({"messages": seed, "summary": ""})
check("RemoveMessage prunes to keep window", len(res["messages"]) == 2, str(len(res["messages"])))
check("summary written", res["summary"] == "summary of 4", res["summary"])

print("\n== 48: reflective routing and loop caps ==")


class RS(TypedDict):
    documents: list
    answer: str
    retries: Annotated[int, operator.add]
    generations: Annotated[int, operator.add]
    trace: Annotated[list, operator.add]


MAX_RETRIES, MAX_GENERATIONS = 2, 2
ATTEMPT = {"n": 0}


def retrieve(state: RS) -> dict:
    ATTEMPT["n"] += 1
    return {"documents": [], "trace": [f"retrieve{ATTEMPT['n']}"]}


def rewrite(state: RS) -> dict:
    return {"retries": 1, "trace": ["rewrite"]}


def give_up(state: RS) -> dict:
    return {"answer": "cannot answer", "trace": ["give_up"]}


def after_grading(state: RS):
    if state["documents"]:
        return "generate"
    return "give_up" if state["retries"] >= MAX_RETRIES else "rewrite"


b4 = StateGraph(RS)
b4.add_node("retrieve", retrieve)
b4.add_node("rewrite", rewrite)
b4.add_node("give_up", give_up)
b4.add_node("generate", lambda s: {"answer": "ok", "generations": 1})
b4.add_edge(START, "retrieve")
b4.add_conditional_edges("retrieve", after_grading,
                         {"generate": "generate", "rewrite": "rewrite", "give_up": "give_up"})
b4.add_edge("rewrite", "retrieve")
b4.add_edge("generate", END)
b4.add_edge("give_up", END)
refl = b4.compile()
out = refl.invoke({"documents": [], "answer": "", "retries": 0, "generations": 0, "trace": []})
check("retry cap stops the loop", out["retries"] == MAX_RETRIES, str(out["retries"]))
check("falls through to give_up", out["answer"] == "cannot answer", str(out["trace"]))

print("\n== 49: store API ==")
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()
ns = ("memories", "u1")
store.put(ns, "a", {"text": "one", "kind": "profile"})
store.put(ns, "b", {"text": "two", "kind": "preference"})
item = store.get(ns, "a")
check("get returns Item with value", item.value["text"] == "one")
check("Item carries namespace/key/timestamps",
      item.key == "a" and item.namespace == ns and item.created_at is not None)
check("search returns all in namespace", len(store.search(ns)) == 2)
check("metadata filter narrows", len(store.search(ns, filter={"kind": "preference"})) == 1)
check("same key overwrites",
      (store.put(ns, "a", {"text": "one-v2", "kind": "profile"}) or
       store.get(ns, "a").value["text"]) == "one-v2")
store.delete(ns, "b")
check("delete removes", len(store.search(ns)) == 1)

print("\n== 49: namespace isolation ==")
for org, user in [("northwind", "mukesh"), ("northwind", "priya"), ("acme", "mukesh")]:
    store.put(("org", org, "user", user, "facts"), "k", {"text": f"{org}/{user}"})
check("prefix search scopes to tenant", len(store.search(("org", "northwind"))) == 2)
check("cross-tenant is isolated",
      len(store.search(("org", "acme", "user", "mukesh", "facts"))) == 1)

print("\n== 49: semantic index + TTL ==")
from shared.llm import get_embeddings

emb = get_embeddings()
sem = InMemoryStore(index={"embed": emb, "dims": len(emb.embed_query("x")), "fields": ["text"]})
for i, text in enumerate(["deploys to ap-south-1", "prefers short answers", "leads Kestrel"]):
    sem.put(("m",), f"k{i}", {"text": text, "kind": "x"})
hits = sem.search(("m",), query="where do they deploy", limit=2)
check("semantic search returns scored hits",
      len(hits) == 2 and all(getattr(h, "score", None) is not None for h in hits))
check("filter + query combine", len(sem.search(("m",), query="anything", filter={"kind": "x"})) == 3)

try:
    InMemoryStore().put(("s",), "t", {"text": "temp"}, ttl=1.0)
    check("InMemoryStore rejects ttl", False, "expected NotImplementedError")
except NotImplementedError:
    check("InMemoryStore rejects ttl", True)

print("\n== 49: store inside a graph ==")
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_store


class GS(TypedDict):
    text: str
    recalled: list


def writer(state: GS, *, store) -> dict:
    store.put(("g",), str(uuid.uuid4()), {"text": state["text"]})
    return {}


def reader(state: GS) -> dict:
    return {"recalled": [i.value["text"] for i in get_store().search(("g",))]}


b5 = StateGraph(GS)
b5.add_node("writer", writer)
b5.add_node("reader", reader)
b5.add_edge(START, "writer")
b5.add_edge("writer", "reader")
b5.add_edge("reader", END)
shared_store = InMemoryStore()
gg = b5.compile(checkpointer=InMemorySaver(), store=shared_store)

gg.invoke({"text": "first", "recalled": []}, {"configurable": {"thread_id": "t1"}})
out = gg.invoke({"text": "second", "recalled": []}, {"configurable": {"thread_id": "t2"}})
check("injected store= kwarg works", len(shared_store.search(("g",))) == 2)
check("get_store() sees cross-thread writes", sorted(out["recalled"]) == ["first", "second"],
      str(out["recalled"]))

print("\n== 49: SqliteStore durability ==")
from langgraph.store.sqlite import SqliteStore

path = Path(__file__).resolve().parents[1] / "artifacts" / "_smoke_store.sqlite"
path.parent.mkdir(exist_ok=True)
path.unlink(missing_ok=True)
with SqliteStore.from_conn_string(str(path)) as st:
    st.setup()
    st.put(("mem",), "k", {"text": "durable"})
with SqliteStore.from_conn_string(str(path)) as st2:
    check("survives reopen", st2.get(("mem",), "k").value["text"] == "durable")
path.unlink(missing_ok=True)

# TTL is in minutes and the sweep compares against SQLite's CURRENT_TIMESTAMP,
# which has one-second granularity - hence the generous margin.
with SqliteStore.from_conn_string(":memory:", ttl={"default_ttl": 0.02,
                                                   "refresh_on_read": False}) as ttl_store:
    ttl_store.setup()
    ttl_store.put(("s",), "t", {"text": "temp"})
    present = ttl_store.get(("s",), "t") is not None
    time.sleep(3.0)
    lazy = ttl_store.get(("s",), "t") is not None
    swept = ttl_store.sweep_ttl()
    check("ttl needs an explicit sweep", present and lazy and swept == 1, f"swept={swept}")
    check("swept item is gone", ttl_store.get(("s",), "t") is None)

print("\n== 49: InjectedStore hides the store from the model ==")
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedStore
from langgraph.store.base import BaseStore


@tool
def remember_fact(fact: str, kind: str, store: Annotated[BaseStore, InjectedStore()]) -> str:
    """Save a fact."""
    store.put(("memories", "u"), fact[:6], {"text": fact, "kind": kind})
    return "saved"


check("store hidden from tool schema", set(remember_fact.args) == {"fact", "kind"},
      str(list(remember_fact.args)))

print("\n== 50: workspace tools over state ==")
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command


def merge_files(existing, new):
    return {**(existing or {}), **(new or {})}


class WS(TypedDict):
    messages: Annotated[list, operator.add]
    files: Annotated[dict, merge_files]


@tool
def write_file(path: str, content: str,
               tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Write a file."""
    return Command(update={"files": {path: content},
                           "messages": [ToolMessage(f"wrote {path}", tool_call_id=tool_call_id)]})


@tool
def read_file(path: str, state: Annotated[dict, InjectedState]) -> str:
    """Read a file."""
    files = state.get("files") or {}
    return files.get(path, f"No such file: {path}")


check("write_file schema hides plumbing", set(write_file.args) == {"path", "content"},
      str(list(write_file.args)))
check("read_file schema hides state", set(read_file.args) == {"path"}, str(list(read_file.args)))
check("merge reducer keeps both files",
      merge_files({"a.md": "1"}, {"b.md": "2"}) == {"a.md": "1", "b.md": "2"})
check("merge reducer overwrites same path",
      merge_files({"a.md": "1"}, {"a.md": "2"}) == {"a.md": "2"})

# The tools run through a real ToolNode so InjectedState/Command are exercised.
from langgraph.prebuilt import ToolNode


class WSGraph(TypedDict):
    messages: Annotated[list, add_messages]
    files: Annotated[dict, merge_files]


tool_node = ToolNode([write_file, read_file])
b6 = StateGraph(WSGraph)
b6.add_node("tools", tool_node)
b6.add_edge(START, "tools")
b6.add_edge("tools", END)
wg = b6.compile()

call = AIMessage("", tool_calls=[{"name": "write_file", "id": "c1",
                                  "args": {"path": "notes.md", "content": "hello"}}])
state = wg.invoke({"messages": [call], "files": {}})
check("ToolNode applies Command state update", state["files"] == {"notes.md": "hello"},
      str(state["files"]))

call2 = AIMessage("", tool_calls=[{"name": "read_file", "id": "c2", "args": {"path": "notes.md"}}])
state2 = wg.invoke({"messages": [call2], "files": {"notes.md": "hello"}})
check("InjectedState reaches the tool", state2["messages"][-1].content == "hello",
      state2["messages"][-1].content)

print("\n== 50: planning + limits middleware wiring ==")
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    TodoListMiddleware,
)


class WorkspaceMiddleware(AgentMiddleware):
    state_schema = WSGraph

    def __init__(self):
        super().__init__()
        self.tools = [write_file, read_file]


fake = GenericFakeChatModel(messages=iter([AIMessage("done")]))
agent = create_agent(
    fake,
    tools=[],
    middleware=[TodoListMiddleware(), WorkspaceMiddleware(),
                ModelCallLimitMiddleware(thread_limit=30, exit_behavior="end")],
    checkpointer=InMemorySaver(),
)
channels = set(agent.stream_channels_list)
check("todos key added to state", "todos" in channels, str(sorted(channels)))
check("files key added to state", "files" in channels)
check("workspace tools registered",
      "tools" in agent.get_graph().nodes, str(list(agent.get_graph().nodes)))

approving = create_agent(
    GenericFakeChatModel(messages=iter([AIMessage("done")])),
    tools=[write_file],
    middleware=[HumanInTheLoopMiddleware(interrupt_on={"write_file": False}),
                ModelCallLimitMiddleware(thread_limit=5, exit_behavior="end")],
    checkpointer=InMemorySaver(),
)
check("HITL middleware compiles into the agent",
      any("HumanInTheLoop" in n for n in approving.get_graph().nodes),
      str(list(approving.get_graph().nodes)))

print("\n== 50: depth guard via wrap_tool_call ==")


class DepthGuard(AgentMiddleware):
    def __init__(self, budget: int = 2):
        super().__init__()
        self.budget = budget
        self.blocked = 0

    def wrap_tool_call(self, request, handler):
        spawned = sum(1 for m in (request.state.get("messages") or [])
                      for tc in getattr(m, "tool_calls", [])
                      if tc["name"].startswith("delegate"))
        if request.tool_call["name"].startswith("delegate") and spawned >= self.budget:
            self.blocked += 1
            return ToolMessage("budget exhausted", tool_call_id=request.tool_call["id"])
        return handler(request)


@tool
def delegate_research(topic: str) -> str:
    """Delegate research."""
    return f"findings on {topic}"


from langchain.agents.middleware import ToolCallRequest

guard = DepthGuard(budget=2)


def make_request(spawned: int, name: str = "delegate_research") -> ToolCallRequest:
    history = [AIMessage("", tool_calls=[{"name": "delegate_research", "id": f"p{i}",
                                          "args": {"topic": "x"}}]) for i in range(spawned)]
    return ToolCallRequest(
        tool_call={"name": name, "id": "call-9", "args": {"topic": "y"}},
        tool=delegate_research,
        state={"messages": history},
        runtime=None,
    )


def handler(request):
    return ToolMessage("real findings", tool_call_id=request.tool_call["id"])


under = guard.wrap_tool_call(make_request(spawned=1), handler)
over = guard.wrap_tool_call(make_request(spawned=2), handler)
other = guard.wrap_tool_call(make_request(spawned=5, name="research"), handler)

check("under budget passes through to the handler", under.content == "real findings")
check("over budget is blocked", "budget exhausted" in over.content, over.content)
check("non-delegate tools are unaffected", other.content == "real findings")
check("guard counts blocks", guard.blocked == 1, f"blocked={guard.blocked}")

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
