🔥 2025 EDITION
Top
Interview
Questions
on
Master graph-based AI agents and ace your
next interview
📊 State Management 🔄 Checkpointing
🤖 Multi-Agent 👤 Human-in-Loop
Naved Khan
Gen AI Engineer
+ Follow
💡 ANSWER
LangGraph is a framework for building stateful, multi-actor applications
with LLMs. It represents workflows as graphs where nodes are functions
and edges define transitions.
LangChain
Sequential chains (DAGs)
No built-in cycles
Simple workflows
LangGraph
Graph-based (cycles OK)
Built-in state management
Complex workflows
KEY DIFFERENCES
→ LangGraph supports cycles (retry loops, iterative agents)
→ Built-in checkpointing for persistence
→ Native multi-agent coordination
Naved Khan
Gen AI Engineer
2 / 22
Q1 / 20 EASY
What is LangGraph and how does it differ from
LangChain?
💡 ANSWER
LangGraph workflows are defined by three core building blocks that work
together to create complex agent flows.
BUILDING BLOCKS
📦 Nodes: Python functions that process and return state updates
🔗 Edges: Define transitions between nodes (normal or conditional)
📊 State: TypedDict that flows through graph, accumulating results
START → Node A → Node B → END
graph = StateGraph(State)
graph.add_node("agent", agent_fn)
graph.add_edge(START, "agent")
graph.add_edge("agent", END)
Naved Khan
Gen AI Engineer
3 / 22
Q2 / 20 EASY
What are the core concepts: Nodes, Edges, and
State?
💡 ANSWER
State is a shared data structure (TypedDict) that persists across nodes.
Each node receives current state and returns updates that get merged
back.
from typing import TypedDict
class State(TypedDict):
messages: list
current_step: str
results: dict
STATE FLOW
1. Initial state provided at invoke()
2. Each node receives current state
3. Node returns partial updates
4. Updates merged into state → next node
💡 INTERVIEW TIP
Naved Khan
Gen AI Engineer
4 / 22
Q3 / 20 MEDIUM
How does state management work in
LangGraph?
💡 ANSWER
TypedDict defines the state schema. Annotated with a reducer function
specifies how state updates are merged (append vs replace).
from typing import Annotated
import operator
class State(TypedDict):
# Appends new messages to list
messages: Annotated[list, operator.add]
# Replaces value (default)
current_step: str
REDUCER FUNCTIONS
→ operator.add : Append lists together
→ add_messages : Smart message merging (deduplication)
→ No annotation: Replace value entirely
⚠️ COMMON TRAP
With t t dd t i li t l it i t d f di !
Naved Khan
Gen AI Engineer
5 / 22
Q4 / 20 HARD
What is TypedDict and Annotated in state
definition?
💡 ANSWER
MessagesState is a pre-built state with a messages list using
add_messages reducer. It handles message deduplication and proper
merging.
from langgraph.graph import MessagesState
# Equivalent to:
from langgraph.graph.message import add_messages
class State(TypedDict):
messages: Annotated[list, add_messages]
ADD_MESSAGES FEATURES
✓ Appends new messages to existing list
✓ Deduplicates by message ID
✓ Updates existing messages if same ID
✓ Handles BaseMessage objects properly
💡 BEST PRACTICE
Use MessagesState for chatbots - it's optimized for conversation flows!
Naved Khan
Gen AI Engineer
6 / 22
Q5 / 20 MEDIUM
What is MessagesState and add_messages?
💡 ANSWER
Conditional edges route to different nodes based on state. A routing
function examines state and returns the next node name.
def should_continue(state):
if state["done"]:
return "end"
return "continue"
graph.add_conditional_edges(
"agent",
should_continue,
{"end": END, "continue": "tools"})
USE CASES
→ Tool calling decisions (call tool or respond)
→ Loop until condition (retry logic)
→ Route by classification result
Naved Khan
Gen AI Engineer
7 / 22
Q6 / 20 MEDIUM
What are conditional edges and how to
implement them?
💡 ANSWER
tools_condition is a prebuilt routing function that checks if the last message
has tool calls. Routes to "tools" node or END.
from langgraph.prebuilt import tools_condition
graph.add_conditional_edges(
"chatbot",
tools_condition
)
# Routes to "tools" or END automatically
Agent → tools_condition → Tools / END
HOW IT WORKS
✓ Checks tool_calls in last AI message
✓ Returns "tools" if tool calls exist
✓ Returns END if no tool calls (final response)
Naved Khan
Gen AI Engineer
8 / 22
Q7 / 20 MEDIUM
What is tools_condition in LangGraph?
💡 ANSWER
LangGraph enables cycles - edges that loop back to previous nodes. The
agent continues until a condition routes to END.
Agent → Tools → Check ↩️
graph.add_edge("tools", "agent")
# Creates cycle: agent → tools → agent
graph.add_conditional_edges(
"agent", tools_condition)
AGENT LOOP PATTERN
1. Agent decides: call tool or respond
2. If tool → execute → return to agent
3. Repeat until no more tool calls → END
Naved Khan
Gen AI Engineer
9 / 22
Q8 / 20 HARD
How to build an agent loop that retries until
success?
💡 ANSWER
Checkpointing saves graph state at each step, enabling persistence,
resumption after failures, and time-travel debugging.
from langgraph.checkpoint.memory import InMemorySaver
memory = InMemorySaver()
graph = builder.compile(checkpointer=memory)
CHECKPOINTING ENABLES
💾 Persistence: Save conversation state across sessions
🔄 Resumption: Continue from last point after failure
⏰ Time-travel: Go back to any previous state
👤 Human-in-loop: Pause for approval
Naved Khan
Gen AI Engineer
10 / 22
Q9 / 20 MEDIUM
What is checkpointing and why is itimportant?
💡 ANSWER
LangGraph provides multiple checkpointers for different persistence needs
- from development to production scale.
CHECKPOINTER TYPES
🧪 InMemorySaver: Development & testing (non-persistent)
📁 SqliteSaver: Local file persistence
🐘 PostgresSaver: Production database persistence
🍃 MongoDBSaver: Document-based persistence
# Development
from langgraph.checkpoint.memory import InMemorySaver
memory = InMemorySaver()
# Production
from langgraph.checkpoint.postgres import PostgresSaver
checkpointer = PostgresSaver.from_conn_string(DB_URI)
Naved Khan
Gen AI Engineer
11/ 22
Q10 / 20 MEDIUM
What are the different checkpointer options?
💡 ANSWER
thread_id is a unique identifier for a conversation session. Each thread
maintains its own state history, enabling multi-user support.
config = {
"configurable": {
"thread_id": "user_123"
}
}
# First message
graph.invoke(input_1, config)
# Same thread - has memory!
graph.invoke(input_2, config)
THREAD ID USE CASES
→ User-specific conversation history
→ Multiple parallel conversations
→ Resume sessions across requests
Naved Khan
Gen AI Engineer
12 / 22
Q11 / 20 MEDIUM
How does thread_id work for persistence?
💡 ANSWER
Human-in-the-loop pauses graph execution at specific points for human
review, approval, or input before continuing.
Agent → ⏸️ PAUSE → Human → Continue
USE CASES
✅ Approve before executing sensitive actions
✏️ Edit agent's proposed response
🔧 Correct mistakes before they propagate
⚠️ Quality control checkpoints
💡 REQUIRES CHECKPOINTING
Human-in-the-loop needs a checkpointer to save state while waiting for human
input!
Naved Khan
Gen AI Engineer
13 / 22
Q12 / 20 HARD
What is human-in-the-loop in LangGraph?
💡 ANSWER
Use interrupt_before or interrupt_after at compile time to specify which
nodes trigger a pause.
graph = builder.compile(
checkpointer=memory,
interrupt_before=["tools"]
)
# Resume after human approval
graph.invoke(None, config)
interrupt_before
Pause BEFORE node executes
Review what will happen
interrupt_after
Pause AFTER node executes
Review results before next
⚡ RESUME EXECUTION
Call invoke(None, config) to continue from interrupt point!
Naved Khan
Gen AI Engineer
14 / 22
Q13 / 20 HARD
How to implement interrupt points?
💡 ANSWER
Multi-agent systems use multiple agent nodes that coordinate through
shared state. Common patterns include supervisor, collaborative, and
debate.
MULTI-AGENT PATTERNS
👔 Supervisor: One agent delegates to worker agents
🤝 Collaborative: Agents build on each other's work
⚔️ Debate: Agents argue positions to refine output
Supervisor → Worker A Worker B
graph.add_node("supervisor", supervisor_fn)
graph.add_node("researcher", researcher_fn)
graph.add_node("writer", writer_fn)
Naved Khan
Gen AI Engineer
15 / 22
Q14 / 20 HARD
How to build multi-agent systems in
LangGraph?
💡 ANSWER
Subgraphs are nested graphs that encapsulate reusable workflows. Use a
compiled graph as a node in a parent graph.
# Create subgraph
subgraph = subgraph_builder.compile()
# Use as node in parent
main_graph.add_node("research", subgraph)
main_graph.add_edge("research", "write")
BENEFITS OF SUBGRAPHS
♻️ Reusability: Same subgraph in multiple workflows
🧪 Testability: Test subgraphs in isolation
📦 Organization: Modular, maintainable code
🔒 Encapsulation: Hide internal complexity
💡 CHECKPOINTER PROPAGATION
Parent's checkpointer automatically propagates to subgraphs!
Naved Khan
Gen AI Engineer
16 / 22
Q15 / 20 HARD
What are subgraphs and when to use them?
💡 ANSWER
LangGraph supports multiple streaming modes to output results as they're
generated, improving perceived latency.
for chunk in graph.stream(
inputs,
config,
stream_mode="values"
):
print(chunk)
STREAM MODES
→ "values" : Full state after each node
→ "updates" : Only state changes per node
→ "messages" : Stream individual tokens
→ "debug" : Detailed execution info
💡 BEST PRACTICE
Use "updates" for efficient streaming, "values" for debugging full state
Naved Khan
Gen AI Engineer
17 / 22
Q16 / 20 MEDIUM
How does streaming work in LangGraph?
💡 ANSWER
Time-travel lets you go back to any previous checkpoint, inspect or modify
state, and re-run from that point.
# Get all checkpoints
states = list(graph.get_state_history(config))
# Go back to specific checkpoint
to_replay = states[2]
# Resume from checkpoint
graph.invoke(None, to_replay.config)
TIME-TRAVEL USE CASES
🔍 Debug issues by inspecting past states
↩️ Undo and retry with different input
🔀 Branch from past state for "what-if" analysis
Naved Khan
Gen AI Engineer
18 / 22
Q17 / 20 HARD
What is time-travel debugging in LangGraph?
💡 ANSWER
ToolNode is a prebuilt node that executes tool calls from the last AI
message. It handles tool execution and returns results.
from langgraph.prebuilt import ToolNode
tools = [search_tool, calc_tool]
tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)
TOOLNODE FEATURES
✓ Extracts tool_calls from AI message
✓ Executes matching tool with arguments
✓ Returns ToolMessage with results
✓ Handles multiple tool calls in parallel
🔧 COMPLETE AGENT SETUP
ToolNode + tools_condition = Complete ReAct agent pattern!
Naved Khan
Gen AI Engineer
19 / 22
Q18 / 20 MEDIUM
What is ToolNode in LangGraph?
💡 ANSWER
Choose based on workflow complexity. LangChain for simple chains,
LangGraph for stateful, complex agent workflows.
Use LangChain
Simple A→B→C flows
Basic RAG pipelines
Quick prototypes
No cycles needed
Use LangGraph
Loops & retries
Multi-agent systems
Human-in-the-loop
Persistent state
KEY DECISION POINTS
🔄 Need cycles/loops? → LangGraph
💾 Need checkpointing? → LangGraph
👥 Multiple agents? → LangGraph
⚡ Simple chain? → LangChain LCEL
Naved Khan
Gen AI Engineer
20 / 22
Q19 / 20 EASY
LangGraph vs LangChain - when to use which?
💡 ANSWER
Production LangGraph applications require careful attention to state
design, error handling, persistence, and observability.
PRODUCTION CHECKLIST
✅ Persistent Checkpointer: Use Postgres/MongoDB, not InMemorySaver
✅ Error Handling: Wrap nodes in try/catch, use fallback edges
✅ Timeouts: Set max iterations to prevent infinite loops
✅ Observability: Use LangSmith for tracing & debugging
✅ State Validation: Validate state at node boundaries
⚠️ COMMON PITFALL
Always set recursion_limit to prevent runaway agent loops!
Naved Khan
Gen AI Engineer
21/ 22
Q20 / 20 HARD
Best practices for production LangGraph apps?
ThankYou for
Reading! 🎉
Naved Khan
Gen AI Engineer
💾 Save this for interview prep.
🔄 Share with someone who needs it!
❤️
Like
💬
Comment
🔄
Repost
🔖
Save

1 · LangGraph and Agentic AI Fundamentals
Beginner — Q1–Q9.

Q1. What is LangGraph, and what problem does it solve that plain prompt-chaining can’t?
LangGraph is a low-level orchestration library for building stateful, multi-step applications with LLMs, modeled on Pregel-style graph computation (the same lineage as Apache Beam). A plain chain of prompts executes once, start to finish, in a straight line. Real agent behavior isn't linear — an agent needs to loop while it keeps calling tools, branch differently depending on what a tool returns, pause and wait for a human, and pick back up exactly where it left off after a crash. LangGraph models an application as a graph of nodes and edges with a shared, persisted state object, which is what makes cycles, branching, and recovery possible instead of something you hand-roll with flags and retries.

Q2. How does LangGraph differ from a standard LangChain LCEL chain?
An LCEL chain is a directed acyclic graph (DAG) — data flows forward through a fixed pipeline and the chain ends. LangGraph explicitly supports cycles, so a node can route back to an earlier node (an agent re-planning after a failed tool call, for example) as many times as the logic requires. LangGraph also adds a first-class, checkpointed state object and a persistence layer, so execution can be paused, inspected, replayed, or resumed — none of which a stateless chain gives you out of the box.

Q3. What are the three core building blocks of every LangGraph graph?
State — a schema (typically a TypedDict, dataclass, or Pydantic model) describing the data that flows through the graph.
Nodes — plain Python (or JS/TS) functions that receive the current state and return a partial update to it.
Edges — the connections that decide which node runs next: fixed edges, conditional edges (a routing function), or dynamic routing returned from inside a node via Command.
Q4. What is a “superstep,” and why does LangGraph’s execution model matter for parallelism?
LangGraph executes in discrete supersteps, borrowed from the Pregel/Bulk Synchronous Parallel model: every node scheduled to run in the current step executes (potentially concurrently), all of their state updates are collected, merged through the state's reducers, and only then is the next superstep scheduled. This is why a fan-out of five parallel nodes produces one combined checkpoint for that step, not five separate ones — and why reducers, not manual locking, are what make parallel branches safe to write to shared state.

Q5. Why does LangGraph support cycles when tools like Airflow are strictly DAG-based?
Airflow orchestrates pipelines where the shape of the work is known in advance. Agentic workflows aren't like that — an LLM decides at runtime whether to call a tool again, ask a clarifying question, or hand off to another agent, and that decision can send execution back to a node it already visited. Cycles are what let a node say "call me again with updated state" instead of the graph author having to pre-enumerate every possible path before the agent ever runs.

Q6. What is MessagesState, and when should you use it instead of a custom TypedDict?
MessagesState is a prebuilt state schema that ships with a messages key already wired to the add_messages reducer, covering the common case of a chat-style agent that just needs to accumulate a conversation. Reach for it when your agent's state genuinely is "the conversation so far." Define your own TypedDict (which can still include a messages field) the moment you need additional structured fields — severity, assigned team, approval status — that don't belong in the chat history itself.

Q7. What programming languages and runtimes does LangGraph support?
LangGraph ships as both a Python package (langgraph) and a JavaScript/TypeScript package (@langchain/langgraph), with feature parity on the core graph API and some ecosystem packages (like the supervisor and swarm prebuilts) more mature on the Python side first. Most Indian training content and job postings default to the Python SDK, since it pairs naturally with the wider Python ML/data tooling most backend and DevOps engineers already know.

Q8. Write the minimal code to define, compile, and invoke a two-node LangGraph graph.
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    ticket: str
    triage_notes: str

def triage(state: State) -> dict:
    return {"triage_notes": f"Reviewed: {state['ticket']}"}

def resolve(state: State) -> dict:
    return {"triage_notes": state["triage_notes"] + " -> resolution drafted"}

builder = StateGraph(State)
builder.add_node("triage", triage)
builder.add_node("resolve", resolve)
builder.add_edge(START, "triage")
builder.add_edge("triage", "resolve")
builder.add_edge("resolve", END)

graph = builder.compile()
result = graph.invoke({"ticket": "Pod CrashLoopBackOff in payments-api", "triage_notes": ""})
Notice each node returns only the fields it changed, not the whole state — LangGraph merges the partial update into the existing state for you.

Q9. What does .compile() actually do under the hood?
compile() turns the StateGraph builder into a runnable object: it validates that every node referenced by an edge actually exists, resolves the reducers for each state channel, and wires in whatever you pass for persistence (checkpointer=) and interruption points (interrupt_before=, interrupt_after=). An uncompiled StateGraph is just a definition; nothing executes until compile() produces the runnable graph.

Interview tip: interviewers ask this specifically to check you understand that checkpointer and interrupt_before/interrupt_after are compile-time arguments, not something you bolt on per-invocation.

2 · State, Schema and Reducers
Beginner–Intermediate — Q10–Q17.

Q10. What is a reducer, and what breaks in a graph if you don’t define one?
A reducer is the function LangGraph uses to combine a node's returned update with the existing value for that state key. Without one, the default behavior is overwrite — the newest write simply replaces whatever was there. That's fine for a scalar field like severity, but disastrous for something like a running log list: every node that touches it would need to return the entire accumulated list, and two parallel nodes writing to the same un-reduced key in the same superstep will conflict.

Q11. How does add_messages differ from a plain operator.add reducer?
from typing import Annotated, TypedDict
import operator
from langgraph.graph.message import add_messages

class IncidentState(TypedDict):
    messages: Annotated[list, add_messages]       # smart merge by message ID
    logs: Annotated[list[str], operator.add]      # simple append-only concatenation
operator.add on a list just concatenates — call it twice with the same item and you get a duplicate. add_messages is purpose-built for chat history: it matches incoming messages to existing ones by ID and replaces a message with the same ID instead of appending it. That's what lets you stream a message in progress (updating it token by token) without ending up with dozens of duplicate partial messages in state.

Q12. TypedDict vs. dataclass vs. Pydantic model for state — how do you choose?
TypedDict is the default in most LangGraph examples: zero runtime overhead, works everywhere, but gives you no validation — a node can return a malformed value and you won't find out until something downstream breaks. A Pydantic model adds runtime validation and coercion at the cost of a small performance hit on every state update, which matters if a node runs thousands of times in a batch job. A dataclass sits in between — you get attribute access and defaults without full validation. For an interview answer: default to TypedDict for speed and simplicity, upgrade to Pydantic when the state crosses a trust boundary (user input, an external API response) that genuinely needs validation.

Q13. What are input_schema and output_schema for on a StateGraph?
They let the externally visible shape of a graph differ from its full internal state. A graph might track a dozen internal fields — intermediate reasoning, tool call scratch space, routing flags — but you only want callers to have to pass in ticket and get back resolution. Declaring narrower input_schema/output_schema types keeps that internal complexity from leaking into the graph's public contract, which matters a lot once other services start calling your graph.

Q14. How do you keep large payloads, like a 40-page log dump, out of your checkpointed state?
Store a reference, not the content. If a node fetches a large log file or document, write the storage key (an S3 path, a document ID) into state and have the node that actually needs the content fetch it fresh using that key. Every field in state gets serialized into every checkpoint, so a state object that holds full document text turns a lightweight, kilobyte-sized checkpoint into a multi-megabyte write on every single superstep — and that cost compounds across a long-running thread.

Q15. What happens when two parallel nodes write to the same un-reduced state key?
By default, LangGraph raises an InvalidUpdateError — it refuses to silently pick a winner between two concurrent writes to a scalar-typed channel, because that would make the graph's behavior non-deterministic based on execution timing. This is precisely the failure mode a reducer exists to prevent: with Annotated[list, operator.add] or a custom reducer, LangGraph knows exactly how to merge the two writes instead of guessing.

Q16. What’s the difference between the graph-level state channel and a private, node-scoped channel?
Every field declared on the main state schema is visible to every node in the graph — that's the shared channel. LangGraph also supports narrower schemas on subgraphs, effectively giving a nested subgraph its own private state that the parent graph never sees except through whatever fields are explicitly passed in and returned out. Use shared state for anything genuinely cross-cutting (the conversation, the overall ticket); use subgraph-private state to stop one agent's scratch work from cluttering, or accidentally colliding with, another agent's fields.

Q17. What’s the runtime cost of validating state with a Pydantic model on every node transition?
Every node return gets re-validated and re-coerced against the Pydantic schema, which is meaningfully slower than a TypedDict's zero-cost static typing — the difference is small per call but adds up in tight agentic loops that might execute a node hundreds of times. The honest interview answer is: it's a deliberate trade of throughput for safety, so it belongs at the edges of a graph where untrusted data enters, not necessarily on every internal hop.

3 · Persistence: Checkpoints, Threads and Stores
Intermediate — Q18–Q24.

Q18. What is a checkpointer, and what three capabilities does it unlock?
A checkpointer is the persistence layer you attach at compile time (builder.compile(checkpointer=...)); it saves a snapshot of the graph's state after every superstep. That single mechanism unlocks three things interviewers expect you to name: conversational memory across separate invocations of the same thread, human-in-the-loop workflows (pause, inspect, resume), and fault tolerance — a crashed process can resume a thread from its last durable checkpoint instead of starting over.

Q19. What is a thread_id, and why is persistence impossible without one?
A thread is the unit of persistence in LangGraph — a series of checkpoints tied together by a unique thread_id passed inside the config on every call ({"configurable": {"thread_id": "incident-4521"}}). Without a thread_id, the checkpointer has no key to save or load state under, so every invocation would behave as a fresh, memory-less run even with a checkpointer attached. In a multi-tenant app, thread_id is also how you keep one user's or one incident's state from bleeding into another's.

Q20. Compare MemorySaver, SqliteSaver, and PostgresSaver — when would you choose each in production?
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()  # one-time schema creation
    graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "incident-4521"}}
graph.invoke({"ticket": "Pod CrashLoopBackOff in payments-api"}, config)
MemorySaver keeps checkpoints in process memory — fast, zero setup, gone the moment the process restarts, so it belongs in local development only. SqliteSaver gives you a durable file on disk, a reasonable step up for a single-process app or a prototype. PostgresSaver is the production choice: durable, supports concurrent connections from multiple app instances, and survives a pod restart — which matters a great deal for exactly the kind of incident-response agent used throughout this guide, since losing an in-flight thread mid-incident is not an acceptable failure mode.

Q21. What’s the difference between a Checkpointer and a Store?
A checkpointer persists one thread's state — short-term, thread-scoped memory: conversation continuity, time travel, resuming after a crash. A Store (implementing BaseStore) persists data that's meant to live across threads — long-term memory such as a user's stated preferences or facts learned in a previous, unrelated conversation. Most production agents use both together: the checkpointer tracks "this specific incident," the store tracks "everything we've learned about this customer's infrastructure across every incident we've ever handled."

Q22. Explain “time travel” in LangGraph. How would you replay a thread from an earlier step with modified state?
Because every superstep produces a checkpoint, you can list a thread's full checkpoint history and re-invoke the graph starting from any point in it — optionally editing the state first. That's time travel: useful for debugging ("what did the agent actually see right before it made the wrong call?") and for exploring an alternate path without re-running the whole thread from scratch.

history = list(graph.get_state_history(config))
earlier_checkpoint = history[3]

graph.update_state(earlier_checkpoint.config, {"severity": "P1"})
graph.invoke(None, earlier_checkpoint.config)
Q23. How would you encrypt sensitive fields inside a checkpoint at rest?
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph.checkpoint.postgres import PostgresSaver

encrypted_serde = EncryptedSerializer.from_pycryptodome_aes(encryption_key)
checkpointer = PostgresSaver(conn, serde=encrypted_serde)
Checkpointers accept a custom serializer (serde), and LangGraph ships an EncryptedSerializer that wraps the default one with AES encryption before anything is written to the backing store. This matters the moment state includes anything sensitive — customer PII, credentials pulled from a tool call — since a checkpoint is, by default, stored as plain serialized data.

Q24. What are “pending writes” in checkpoint metadata, and what failure do they protect against?
If a process crashes after a node finishes running but before the next step's scheduling is fully recorded, LangGraph needs to know, on resume, exactly which writes were already durably committed versus which still need to be recomputed. The "pending writes" recorded in a checkpoint's metadata are what let it make that distinction — preventing the two failure modes that would otherwise be possible: silently losing a completed node's output, or double-applying it when the thread resumes.

4 · Control Flow: Conditional Edges, Command and Send
Intermediate–Advanced — Q25–Q32.

Q25. add_conditional_edges vs. returning a Command from a node — what’s the real difference?
add_conditional_edges keeps routing logic external to the node: you register a separate function whose only job is to look at the state and return the name of the next node. A Command lets a node make that same routing decision and update state in a single return value, from inside the node that just did the work needed to make the decision — no separate routing function required. Command also does something conditional edges structurally can't: it can override a statically defined edge and it can route across a subgraph boundary into the parent graph.

Q26. Write a node that uses Command to update state and route dynamically in the same return.
from typing import Literal
from langgraph.types import Command

def triage(state: IncidentState) -> Command[Literal["network_agent", "database_agent"]]:
    if "connection timeout" in state["ticket"].lower():
        return Command(
            update={"assigned_team": "network"},
            goto="network_agent",
        )
    return Command(
        update={"assigned_team": "database"},
        goto="database_agent",
    )
The Command[Literal[...]] return type annotation isn't decorative — LangGraph reads it at graph-build time to know which destination nodes this function might route to, so always annotate it with every possible goto target.

Q27. What does Command(graph=Command.PARENT) do, and when do you actually need it?
By default, a goto inside Command targets a node in the same graph the node belongs to. When a node lives inside a subgraph but needs to hand control back out to a node in the parent graph — a specialist sub-agent signaling "I'm done, return to the supervisor" — you set graph=Command.PARENT, which tells LangGraph to resolve goto against the closest parent graph instead of the current one.

Q28. Can a tool return a Command? Why is that useful?
Yes — a tool function can return a Command just like a node can, letting a tool both update graph state (say, persisting a customer record it just looked up) and route execution to a specific node once the tool call completes. It's especially useful for tool-triggered handoffs in a multi-agent system, since it means the routing decision can live right next to the tool logic that determined it, instead of being inferred afterward by a separate conditional edge.

Q29. What is the Send API for, and how is it different from a normal edge?
A normal edge, even a conditional one, routes to a fixed, known set of next nodes with the existing state. Send lets a routing function dynamically create any number of parallel invocations of a node, each with its own distinct input — the fan-out count doesn't need to be known when you build the graph, only at runtime. It's the primitive behind map-reduce-style patterns: process N items, where N is only known once a prior node has run.

Q30. Design a fan-out/fan-in (map-reduce) pattern in LangGraph using Send.
from langgraph.types import Send

def fan_out_pods(state: IncidentState) -> list[Send]:
    return [
        Send("check_pod_logs", {"pod_name": pod})
        for pod in state["affected_pods"]
    ]

def check_pod_logs(payload: dict) -> dict:
    findings = analyze_logs(payload["pod_name"])
    return {"findings": [findings]}   # merged via an operator.add reducer

builder.add_conditional_edges("triage", fan_out_pods, ["check_pod_logs"])
builder.add_edge("check_pod_logs", "aggregate_findings")
triage decides which pods are affected; fan_out_pods spins up one parallel check_pod_logs invocation per pod; every parallel branch appends to the same reducer-backed findings list, and aggregate_findings runs once all the fanned-out branches from that superstep have completed.

Q31. How do you stop an agent from looping forever on a “retry” conditional edge?
Two layers, and interviewers generally want both: design-level, put an explicit attempt counter in state and route to a terminal "give up, escalate to a human" node once it crosses a threshold, rather than trusting the LLM to eventually decide to stop. Safety-net level, set recursion_limit in the invocation config, which hard-caps the number of supersteps a single run can execute regardless of what the graph's own logic does — the backstop for the retry loop nobody designed correctly.

Q32. What is a subgraph, and when is isolated state better than shared state?
A subgraph is a compiled StateGraph used as a node inside a larger graph — a way to encapsulate a self-contained piece of logic (an entire specialist agent, a multi-step validation routine) as one reusable unit, optionally with its own private state schema instead of reading and writing the parent's shared channels directly. Reach for isolated subgraph state when an agent's internal scratch work (its own reasoning trace, intermediate tool outputs) genuinely shouldn't be visible to, or overwritable by, every other agent in the system; keep shared state when agents genuinely need to coordinate off the same information, like a running incident timeline every specialist should see.

5 · Multi-Agent Architectures
Advanced — Q33–Q40.

Q33. What are the main multi-agent topology patterns available in LangGraph?
The three you should be able to name and contrast: Supervisor — a central orchestrator agent that reads the request and routes each turn to the right specialist; Swarm — decentralized peer-to-peer handoff, where any agent can transfer control directly to any other agent it has a handoff tool for; and Network / custom graph — you hand-wire the topology yourself with regular edges and Command-based routing when neither prebuilt pattern fits, common in pipeline-shaped workflows with a fixed, known sequence of specialists.

Q34. Explain the Supervisor pattern and where create_supervisor fits in.
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

k8s_agent = create_react_agent(model=model, tools=[restart_pod, get_pod_logs], name="k8s_agent")
db_agent = create_react_agent(model=model, tools=[check_connection_pool], name="db_agent")

incident_supervisor = create_supervisor(
    agents=[k8s_agent, db_agent],
    model=model,
    prompt="Route each incident to the specialist best suited to the reported symptoms.",
).compile()
create_supervisor, from the langgraph-supervisor package, wires up a central agent whose only job is deciding, on every turn, which specialist agent should act next — the specialists never talk to each other directly, only through the supervisor. It's the pattern to reach for when you need a predictable, auditable chain of command: exactly one agent decides "who goes next" at any point.

Q35. Explain the Swarm pattern and how create_handoff_tool enables peer-to-peer handoff.
from langgraph.prebuilt import create_react_agent
from langgraph_swarm import create_handoff_tool, create_swarm

handoff_to_db = create_handoff_tool(agent_name="db_agent")
k8s_agent = create_react_agent(model=model, tools=[restart_pod, handoff_to_db], name="k8s_agent")
db_agent = create_react_agent(model=model, tools=[check_connection_pool], name="db_agent")

swarm = create_swarm(agents=[k8s_agent, db_agent], default_active_agent="k8s_agent").compile()
create_handoff_tool generates a regular tool that, when the agent calls it, hands control directly to the named peer agent rather than routing through a central decision-maker. Swarm suits workflows that genuinely feel organic — the agent currently active is best placed to judge who should take over next — at the cost of being harder to trace: debugging a chain of peer handoffs without a tracing tool like LangSmith is close to impossible once the graph is nontrivial.

Q36. Supervisor vs. Swarm — how would you justify picking one in a system-design interview?
Frame it as centralized-control vs. distributed-judgment, not "which is better." Supervisor when the business genuinely needs one throat to choke — a single, inspectable decision point, useful when routing needs to be auditable or governed by rules a compliance team can review. Swarm when the specialists themselves are best positioned to judge the handoff — a k8s agent that discovers mid-investigation the real issue is a database connection pool can transfer directly instead of bouncing back up to a supervisor and waiting to be re-routed. Naming that trade-off, rather than just describing the two APIs, is what separates a strong answer from a memorized one.

Q37. What is create_react_agent, and why was it split into the langgraph-prebuilt package?
create_react_agent is a prebuilt wrapper that gives you a working tool-calling agent — following the Reason-and-Act loop — in one function call, without hand-building a graph. It originally lived inside the core langgraph package; it was later split out into its own langgraph-prebuilt package alongside a growing family of other prebuilt agent patterns (supervisor, swarm), so that the core library could stay focused on the low-level graph runtime while opinionated, higher-level agent patterns live and version independently on top of it.

Q38. In a supervisor architecture, how do sub-agents typically share, or isolate, context?
The default in both create_supervisor and create_swarm is shared state — every agent reads from and writes to the same state channels, most commonly the same messages list, so each specialist sees the full conversation history including other agents' turns. When that's too leaky (one agent's internal tool scratch-work polluting another's context), you isolate an agent as a subgraph with its own private schema and pass only the specific fields it needs in and its result back out, trading some context-sharing convenience for a much cleaner separation of concerns.

Q39. How does LangGraph integrate with MCP (Model Context Protocol) tool servers?
Through the langchain-mcp adapters, which let a LangGraph agent load an MCP server's exposed tools as regular LangChain tools it can bind to a create_react_agent or a custom node — the agent calls them exactly like any locally defined tool. The practical benefit for an interview answer: MCP decouples tool implementation from agent logic, so adding a new capability, or pointing at an updated API, doesn't require rewriting the agent, only reconnecting to a different (or updated) MCP server.

Q40. What are “deep agents,” and how do they extend the basic ReAct loop for long-horizon tasks?
A plain ReAct loop (reason, call a tool, observe, repeat) tends to degrade on long, multi-hour tasks — the agent loses track of the overall plan, and the context window fills with intermediate tool output. "Deep agent" architectures extend the pattern with an explicit planning step up front, the ability to delegate sub-tasks to focused sub-agents rather than doing everything in one context, and an external scratchpad (often described as a virtual file system) to offload intermediate work instead of keeping it all in the live conversation. It's a genuinely current topic — expect senior-track interviews in 2026 to probe whether you know why long-horizon agents need more structure than a bare ReAct loop, even if you haven't personally built one yet.

6 · Human-in-the-Loop, Streaming and Production
Advanced — Q41–Q48.

Q41. How do you pause a graph for human approval before a high-risk action, like a production deployment?
from langgraph.types import interrupt, Command

def request_approval(state: IncidentState) -> dict:
    decision = interrupt({"question": f"Approve restart of pod {state['pod_name']}?"})
    return {"approved": decision == "yes"}

# ...later, once a human has responded:
graph.invoke(Command(resume="yes"), config)
Calling interrupt() inside a node pauses the graph mid-step and surfaces whatever payload you pass it to your application layer — a UI, a Slack approval message, wherever a human actually reviews it. Because the graph is checkpointed, it can sit paused indefinitely; resuming is just a normal invoke call passing Command(resume=...) with the human's decision, and the node picks up exactly where interrupt() left off.

Q42. Static interrupts (interrupt_before/interrupt_after) vs. the dynamic interrupt() function — what’s the difference?
interrupt_before/interrupt_after are compile-time arguments — you name specific nodes the graph should always pause before or after, decided when you build the graph, not by runtime logic. The interrupt() function is dynamic: it's called from inside a node's own code, so whether the graph pauses at all, and what data accompanies the pause, can depend on the current state — approve every restart, say, but only pause for approval above a certain severity.

Q43. What stream modes does LangGraph support, and when do you use “messages” vs. “values”?
LangGraph's .stream()/.astream() support several stream modes, and you can request more than one at once: values emits the full state after every superstep, updates emits just the partial diff each node returned, messages emits token-level LLM output as it's generated (paired with metadata about which node produced it), and custom lets a node emit arbitrary application-defined events. Use values or updates when a UI needs to reflect state changes (a new log line landed, a field changed); use messages specifically when you need to stream an LLM's response token-by-token to a chat interface.

Q44. How would you stream token-by-token output from a multi-agent graph to a frontend?
for stream_mode, payload in graph.stream(
    {"ticket": "Pod CrashLoopBackOff in payments-api"},
    config,
    stream_mode=["messages", "updates"],
):
    if stream_mode == "messages":
        token, metadata = payload
        # metadata["langgraph_node"] tells you which agent produced this token
        send_to_frontend(token.content, source=metadata.get("langgraph_node"))
Requesting messages mode gives you each token alongside metadata identifying which node (and therefore which specialist agent, in a supervisor or swarm setup) it came from — essential for a frontend that wants to visibly attribute output to "the database agent is now responding" rather than showing one undifferentiated stream.

Q45. What is recursion_limit, and what production incident does it guard against?
recursion_limit caps the number of supersteps a single graph invocation is allowed to execute before LangGraph raises an error and halts it, passed either at compile time or per-invocation in the config ({"recursion_limit": 25}). It exists because a conditional loop with a subtly wrong exit condition — a "check again" edge that routes back to itself on any finding, say — doesn't fail loudly, it just keeps running, burning LLM calls and, eventually, budget, until something else notices. It's the difference between an incident that costs a few extra tokens and one that costs a very large API bill.

Q46. How do you unit test a single LangGraph node in isolation?
A node is a plain function — that's deliberate, and it's the whole reason node functions should stay free of hidden global state. Call it directly with a hand-built state dict and assert on what it returns, without ever building or compiling a graph:

def test_triage_routes_network_issues():
    result = triage({"ticket": "connection timeout to payments-db", "assigned_team": ""})
    assert result.update["assigned_team"] == "network"
    assert result.goto == "network_agent"
For nodes that return a Command, assert against its .update and .goto attributes, as above. Reserve full graph.invoke()-level tests for integration coverage of routing and multi-node behavior, not for a single node's core logic.

Q47. How do you observe and debug a LangGraph agent running in production?
Two complementary tools come up constantly in interview answers, and naming both signals real production experience: LangSmith for tracing and evaluation — every node execution, tool call, and token gets logged with full inputs and outputs, so you can inspect exactly why an agent made a given decision after the fact. LangGraph Studio for interactive, step-by-step debugging during development — a visual graph view where you can watch state change node by node and manually trigger a resume from any point. In production, tracing is what you rely on; Studio is what you reach for while building.

Q48. What’s a realistic LangGraph system-design prompt, and how should you structure your answer?
A common one: "Design an agent that triages a production incident, investigates using specialist tools, and requires human approval before taking a destructive action." Structure the answer top-down rather than diving straight into code: (1) state schema — what fields does the whole system need to share; (2) topology — supervisor, swarm, or a hand-wired sequence, and why; (3) persistence — which checkpointer, and what a thread_id maps to in this domain; (4) the human-in-the-loop point — exactly which action triggers interrupt() and what payload a reviewer needs to decide; (5) failure modes — what recursion_limit and reducer choices you'd set and why. Interviewers are grading whether you treat checkpointing and human review as first-class design decisions made up front, not details bolted on after the "happy path" is drawn.

7 · LangGraph vs. The Ecosystem
Q49–Q50.

Q49. In 2026, is it still accurate to frame this as “LangGraph vs. LangChain”?
Not really, and saying so is itself a strong interview answer. LangChain's own prebuilt agents, including create_react_agent, now execute on the LangGraph runtime under the hood — LangGraph is the low-level engine, LangChain is the batteries-included layer on top of it, not a competing choice. An interviewer who asks "LangGraph or LangChain?" is generally checking whether you know that distinction: reach for LangChain's prebuilt agents to move fast on a standard tool-calling agent; drop down to LangGraph's graph API directly the moment you need custom control flow, multi-agent routing, or fine-grained persistence that the prebuilt layer doesn't expose.

Q50. LangGraph vs. CrewAI vs. AutoGen — what’s the one-line differentiator for each?
Framework	Control level	Best fit	Learning curve
LangGraph	Low-level, explicit graph and state control	Custom stateful agents needing fine-grained control over flow, persistence, and human-in-the-loop	Moderate–steep
CrewAI	High-level, role-based abstraction	Fast-to-assemble "crews" of role-playing agents where you don't need to hand-design control flow	Gentle
AutoGen	High-level, conversation-driven	Multi-agent conversation patterns and research-style experimentation	Gentle–moderate
LangChain (prebuilt agents)	High-level, runs on the LangGraph runtime	Getting a standard tool-calling agent running fast without hand-building a graph	Gentle
The honest framing for an interview: LangGraph trades a steeper learning curve for control that the higher-level frameworks deliberately give up in exchange for speed. Pick LangGraph specifically when the state and flow of the agent are the hard part of the problem, not just getting an agent to call tools at all.

FAQ: Quick Answers Before You Go In
Is LangGraph hard to learn if I already know LangChain?
Not particularly — the state, node, and edge concepts are new, but they build directly on ideas (chains, tools, messages) you already know from LangChain. Most developers with solid LangChain and Python fundamentals get comfortable with core LangGraph concepts within one to two focused weeks.

Do I need LangChain to use LangGraph, or can I use it standalone?
LangGraph works standalone — you can define state, nodes, and edges without touching LangChain at all. In practice, most real agents still use LangChain's model wrappers and tool abstractions inside LangGraph nodes, since rebuilding that plumbing yourself adds little value.

Is LangGraph asked about in fresher/entry-level interviews, or only for experienced hires?
Both, but the depth expected differs sharply. Freshers are usually asked to define core concepts and maybe sketch a simple graph; experienced candidates are expected to justify architecture choices, discuss production failure modes, and defend a topology decision under pushback.

Should I learn the Python or JavaScript version of LangGraph first?
Python, for most learners — it is the more mature SDK, the one most tutorials and job postings assume, and it pairs naturally with the rest of the Python-based AI/ML tooling you will be expected to know alongside it.

How long does it realistically take to become interview-ready in LangGraph?
Budget two to four weeks if you are already comfortable with Python and have used an LLM API before: roughly a week on core concepts (state, nodes, edges, reducers), a week on persistence and control flow, and the remainder actually building one non-trivial multi-agent project you can discuss in depth — a real project you can explain beats broad, shallow familiarity with every API.

Where can I practice building real LangGraph projects before an interview?
Building something with actual moving parts — persistence, a multi-agent handoff, a human-approval gate — matters far more than reading through API references. That hands-on, project-based approach is exactly what the Agentic AI training track at Cloud Soft Solutions is built around, working with free tooling so you can practice without needing a paid API key.


2 02 6 I N T E RV I EW G U I D E
30
LangGraph
Interview
Questions
with Answers
Spoken answers with real code. The 5 areas agent
interviews testin 2026.
Naved Khan
Senior Gen-AI Engineer
+ Follow
WHY THIS MATTERS N OW
In 2026,
LangGraph
stopped being
optional.
LangChain's own agents now run on the
LangGraph runtime. It became the default
way to ship agents, and interviews caught up
fast. Checkpointers, interrupts, and state
reducers are standard questions now.
Naved Khan
Senior Gen-AI Engineer
+ Follow
H OW AGENT INTERV IEWS RUN IN 2026
5 areas. 30 questions.
1 Fundamentals Q1–6
2 State & Memory Q7–12
3 Control Flow Q13–18
4 Human-in-the-Loop Q19–24
5 Production Q25–30
State and human-in-the-loop are where senior
candidates separate from juniors. That's where this
guide goes deepest.
Naved Khan
Senior Gen-AI Engineer
+ Follow
Q1 Why does an agent need a graph instead of a
chain?
Q2 What are the core primitives?
Q1–2 AREA 1 · FUNDAMENTALS
A chain runs once, left to right. An agent has to act, observe the result,
and decide again. That's a loop, and a chain can't express cycles.
▸
LangGraph makes the control flow explicit code: nodes, edges, state I
can read, test, and constrain, instead of hoping a prompt behaves.
▸
State: a typed schema every node reads and writes. Nodes: plain
functions that do the work. Edges: fixed or conditional routing
between them.
▸
I define them on a StateGraph builder, mark START and END, and
compile. The compiled graph is a runnable I can invoke or stream.
▸
Naved Khan
Senior Gen-AI Engineer
+ Follow
Q3 How does state actually move through the graph?
Q4 Whatis a reducer, and why does add_messages
exist?
class State(TypedDict):
messages: Annotated[list, add_messages]
Q3–4 AREA 1 · FUNDAMENTALS
Every node receives the current state and returns a partial update,
not a mutation. LangGraph merges updates at the end of each step.
▸
That keeps nodes close to pure functions, which is why graphs are
testable and replayable. I'd say that part out loud, it's the design
insight.
▸
A reducer defines how updates merge into a state key. Default is
overwrite. For chat history, overwriting would destroy the
conversation.
▸
add_messages appends new messages and updates existing ones by
ID. Custom reducers do the same for any accumulating key.
▸
Naved Khan
Senior Gen-AI Engineer
+ Follow
Q5 LangGraph vs CrewAI or AutoGen. How do you
choose?
Q6 What does create_react_agent actually build?
agent = create_react_agent(model, tools=[search, calc])
Q5–6 AREA 1 · FUNDAMENTALS
They sit at different altitudes. LangGraph is a low-level orchestration
runtime: explicit state, durability, full control. CrewAI is a role-based
abstraction, faster to start, less control.
▸
The 2026 data point I'd drop: LangChain's own prebuilt agents run on
the LangGraph runtime now. It's infrastructure, not a framework
choice.
▸
A complete tool-calling loop: model node, tool node, and a conditional
edge that exits when the model stops calling tools.
▸
I'd add that I can rebuild it by hand in about 20 lines, because
interviewers often ask exactly that as the follow-up.
▸
Naved Khan
Senior Gen-AI Engineer
+ Follow
Q7 Short-term vs long-term memory in LangGraph?
Q8 What exactly does a checkpointer save, and when?
graph = builder.compile(checkpointer=PostgresSaver(conn))
Q7–8 AREA 2 · STATE & MEMORY
Short-term is the checkpointer: thread-scoped state, so one
conversation can pause and resume with full context.
▸
Long-term is the Store: namespaced key-value memory that survives
across threads, like a user's preferences visible to every future
conversation.
▸
A snapshot of the full state after every step, keyed by thread. That
single mechanism is what enables resume, replay, time travel, and
human-in-the-loop.
▸
MemorySaver for development, Postgres in production. Saying that
pair signals you've deployed one.
▸
Naved Khan
Senior Gen-AI Engineer
+ Follow
Q9 Whatis thread_id and why does it matter?
graph.invoke(inputs, config={"configurable": {"thread_id":
"42"}})
Q10 The message history keeps growing. What do you
do?
Q9–10 AREA 2 · STATE & MEMORY
It's the session key. Same thread_id continues a conversation from its
last checkpoint. New thread_id starts clean.
▸
In production it maps to something real: a user session, a support
ticket, a workflow run.
▸
Three tools: trim_messages before each model call, a summarization
node that compresses old turns, and RemoveMessage to prune state
itself.
▸
I treat context as a budget, the same discipline as RAG. Unbounded
history is a cost and quality bug, not a feature.
▸
Naved Khan
Senior Gen-AI Engineer
+ Follow
Q11 Whatis time travel, practically?
Q12 How would you personalize an agent across
sessions?
Q11–12 AREA 2 · STATE & MEMORY
Every checkpoint is addressable. I can pull the state history, fork from
any step with modified state, and re-run from there.
▸
Practically it's debugging: reproduce the exact state where the agent
went wrong, fix one value, and test the counterfactual.
▸
The Store: I write memories under a namespace like the user ID, and
read them at the start of every new thread.
▸
With semantic search over the store, the agent recalls relevant facts,
not everything. Write selectively: store decisions and preferences, not
transcripts.
▸
Naved Khan
Senior Gen-AI Engineer
+ Follow
Q13 How do conditional edges work?
builder.add_conditional_edges("agent", route_fn)
Q14 What does Command add over normal edges?
Q13–14 AREA 3 · CONTROL FLOW
A router function inspects the state and returns the name of the next
node. The LLM decides the content, my code decides the routes.
▸
That separation is the whole point: the set of possible paths is fixed
and reviewable, even when the model is unpredictable.
▸
A node can return Command(update=..., goto=...): change state and
choose the next node in one move.
▸
It's how dynamic multi-agent handoffs work: an agent decides midtask to pass control, with context attached, without pre-wiring every
path.
▸
Naved Khan
Senior Gen-AI Engineer
+ Follow
Q15 How do you run work in parallel inside a graph?
return [Send("grade", {"doc": d}) for d in state["docs"]]
Q16 What are subgraphs for?
Q15–16 AREA 3 · CONTROL FLOW
For a known fan-out, multiple edges from one node run branches in
the same step, and reducers merge the results.
▸
For a dynamic fan-out, Send spawns one node instance per item, each
with its own payload. That's map-reduce inside the graph, like grading
20 documents at once.
▸
A whole graph becomes one node in a parent graph. I use them to
encapsulate a skill or a team member with its own internal state.
▸
Shared state keys are the interface, private keys stay internal. It's the
module system for agents.
▸
Naved Khan
Senior Gen-AI Engineer
+ Follow
Q17 Which multi-agent architecture would you pick?
Q18 What stops an agent from looping forever?
Q17–18 AREA 3 · CONTROL FLOW
I start with a supervisor: one router agent delegating to specialized
workers. Predictable, easy to debug.
▸
I move to swarm-style handoffs with Command when agents
genuinely need to pass control sideways. Hierarchies of subgraphs
come last, only at real scale.
▸
The built-in recursion_limit kills the run with GraphRecursionError
past the step budget.
▸
But I don't rely on it alone: I keep loop counters in state and hard
budgets on tool calls and cost. The limit is a circuit breaker, not a
strategy.
▸
Naved Khan
Senior Gen-AI Engineer
+ Follow
Q19 How does interrupt() actually work?
decision = interrupt({"approve": plan})
# later: graph.invoke(Command(resume="yes"), config)
Q20 Where do you put approval gates?
Q19–20 AREA 4 · HUMAN- IN-THE-LO OP
Called inside a node, it pauses the run and surfaces a payload to the
caller. The graph sleeps in its checkpoint, holding no compute.
▸
Resume happens with Command(resume=value), and that value
becomes interrupt()'s return inside the node. None of it works without a
checkpointer.
▸
Before writes, not reads: payments, emails, deletions, deployments.
Reading data rarely needs a human.
▸
The human can approve, editthe tool arguments, or reject with
feedback, and the edited version flows back through the resume.
▸
Naved Khan
Senior Gen-AI Engineer
+ Follow
Q21 What does "durable execution" give you?
Q22 A node keeps failing on a flaky API.Your options?
Q21–22 AREA 4 · HUMAN- IN-THE-LO OP
State persists at every step, so a crash, a deploy, or a week-long pause
resumes from the last checkpoint, not from scratch.
▸
The senior nuance: recovery is not exactly-once for side effects. A
resumed node can re-run, so I add idempotency keys anywhere money
or emails move.
▸
▸ A retry policy on that node with backoff handles the transient stuff.
For expected failures, I return the error into state and route it to a
handler node. The failure becomes a visible path in the graph, not a
stack trace.
▸
Naved Khan
Senior Gen-AI Engineer
+ Follow
Q23 What streaming modes exist, and why care?
Q24 How do you test a graph?
Q23–24 AREA 4 · HUMAN- IN-THE-LO OP
values streams full state, updates streams each node's delta,
messages streams LLM tokens. They can combine.
▸
An agent that thinks for 30 seconds in silence feels broken. Streaming
progress is a UX requirement, not a nice-to-have.
▸
Nodes are functions, so I unittestthem with a fixed state and a fake
model, asserting on the returned update.
▸
Then integration: invoke the compiled graph and assert on the path it
took, not just the final answer. Evals on real traces close the loop.
▸
Naved Khan
Senior Gen-AI Engineer
+ Follow
Q25 How do you deploy LangGraph?
Q26 The agent feels slow. Where do you look?
Q25–26 AREA 5 · PRODUCTION
Two honest options: LangGraph Platform for managed infra with
Studio debugging, or self-host: a containerized FastAPI service with a
Postgres checkpointer.
▸
Self-hosting scales horizontally because instances share the
checkpoint store. Any replica can resume any thread.
▸
First, parallelize independent nodes and stream tokens so perceived
latency drops immediately.
▸
Then right-size models per node: routing and extraction on a small
model, synthesis on the big one. Most graphs overpay on every hop.
▸
Naved Khan
Senior Gen-AI Engineer
+ Follow
Q27 How do you keep agent costs under control?
Q28 What does observability look like for a graph?
Q27–28 AREA 5 · PRODUCTION
Budget counters in state: tokens, tool calls, and loops all have caps the
graph enforces.
▸
Plus the same levers as any LLM system: trimmed context, cached
stable prefixes, and a model cascade so cheap queries stay cheap.
▸
Traces per node and per step, so I can see exactly which hop burned
the tokens or made the bad call.
▸
And because state is checkpointed, incident review is replay: load the
thread, walk the steps, find the turn where it went wrong.
▸
Naved Khan
Senior Gen-AI Engineer
+ Follow
Q29 When would you NOT use LangGraph?
Q30 What actually changed with LangGraph 1.0?
Q29–30 AREA 5 · PRODUCTION
A single LLM call or a fixed pipeline is just code. No cycles, no
persistence, no humans in the loop means the framework is overhead.
▸
LangGraph earns its keep the moment I need state, loops, pauses, or
durability. Before that, simpler wins.
▸
A stable core API, plus production features: durability modes, node
caching, deferred nodes.
▸
The bigger shift is positioning: with LangChain agents standardized on
this runtime, LangGraph became the boring, safe choice. In
infrastructure, boring is a compliment.
▸
Naved Khan
Senior Gen-AI Engineer
+ Follow
Preparing for AI
interviews?
Naved Khan
Senior Gen-AI Engineer
I post practical GenAI interview prep, RAG,
agents, and production AI, every week.
Follow along if that's useful to you.


LangGraph Advanced Interview Questions & Answers
June 29, 2026 by tnrajesh80
Here are the Top 100 LangGraph Interview Question and Answers:

Core Concepts & Architecture:
1. What is LangGraph and how does it differ from LangChain’s LCEL?

LangGraph is a library for building stateful, multi-actor applications with LLMs using a graph based execution model. LCEL (LangChain Expression Language) is a declarative chain composition system — linear or branching but stateless between steps. LangGraph adds: persistent state across nodes, cycles/loops, human-in-the-loop, and fine-grained control over agent execution flow. LCEL is for pipelines; LangGraph is for agents and workflows that need memory and iteration.

2. What is a StateGraph and how is it different from a MessageGraph?

StateGraph: The primary graph type. You define a custom TypedDict state schema. Every node reads from and writes to this shared state object. Full control over what data flows through the graph.
MessageGraph: A convenience subclass where the state is simply a list of messages (list[BaseMessage]). Simpler for pure chat agents but less flexible for complex workflows needing custom state fields.
3. Explain the concept of “state” in LangGraph. How is it managed?

State is a typed dictionary(using Python TypedDict) that is passed between all nodes in the graph. Each node receives the current state, performs work and returns a partial update. LangGraph merges the updates into the state using reducers. State persists across the entire graph execution and, with checkpointing, across multiple invocations (turns in a conversation).

4.  What are reducers in LangGraph and why are they important?

Reducers define how state fields are updated when a node returns a value. By default, a returned value overwrites the existing field. Custom reducers allow different merge behaviors:

operator.add on a list field appends instead of replacing
Custom functions can implement deduplication, max, merge logic
from typing import Annotated

from operator import add

class State(TypedDict):
messages: Annotated[list, add] # appends instead of overwrites      count: int # overwrites

5. What is the role of START and END nodes in LangGraph?

START is a virtual entry point — edges from START define which node(s) execute first when the graph is invoked. END is a virtual terminal node — when execution reaches END, the graph stops and returns the final state. They are imported from langgraph.graph and used to define the graph’s entry and exit points without creating actual node functions.

6. How do you add nodes and edges to a StateGraph?

from langgraph.graph import StateGraph, START, END

graph=StateGraph(State)

graph.add_node(“node_a”, function_a)

graph.add_node(“node_b”, function_b)

graph.add_edge(START, “node_a”)

graph.add_edge(“node_a”, “node_b”)

graph.add_edge(“node_b”, END)

compiles =graph.compile()

Nodes are Python callables. Edges define execution order. Conditional edges use a routing function.

7. What is graph compilation and what does it do? graph.compile() validates the graph structure (checks for unreachable nodes, missing edges), sets up the runtime execution engine, and optionally attaches a checkpointer and interrupt configuration. The compiled graph is what you actually invoke. Compilation catches structural errors early before runtime.

8. What is the difference between invoke, stream and astream on a compiled graph?

invoke(input): Synchronous, runs the full graph, returns final state.
stream(input): Synchronous generator, yields state updates after each node execution. Good for observability.
astream(input): Async generator version of stream. Use in async frameworks (FastAPI, etc.).
astream_events: Streams granular events including LLM token-level streaming.
9. How does LangGraph handle cycles, and why are they useful? LangGraph explicitly supports cycles via edges that point back to earlier nodes. This enables agent loops — the agent acts, observes tool results, decides to act again, and repeats until a termination condition is met. Without cycles, you’d need to pre-define the number of steps. Cycles make true agentic behavior possible.

10. What is a conditional edge and how do you implement one? A conditional edge routes execution to different nodes based on the current state. You provide a routing function that returns a node name (or list of node names for parallel execution).

def route(state: State) -> str:
    if state["next_action"] == "tool":
        return "tool_node"
    return END

graph.add_conditional_edges("agent", route) 

Checkpointer & Persistance:
11. What is a checkpointer in LangGraph and why is it critical?
A checkpointer persists the graph state after every node execution to a storage backend. This enables: multi-turn conversations (state survives between calls), human-in-the-loop (pause and resume), fault tolerance (resume from last checkpoint on failure), and time-travel debugging (replay from any past state). Without a checkpointer, state is lost after each invoke call.

12. What checkpointer backends does LangGraph support?

MemorySaver: In-memory, for development/testing only. Lost on restart.
SqliteSaver: SQLite-backed, good for local persistence.
PostgresSaver/AsyncPostgresSaver:  Production-grade, from langgraph-checkpoint-postgres
RedisSaver: Redis-backed for high-throughput scenarios
Custom: Implement the BaseCheckpointSaver interface for any backend
13. What is a thread_id and how does it relate to checkpointing?

thread_id is a unique identifier for a conversation or execution session. When you invoke a graph with config={“configurable”: {“thread_id”: “user-123”}}, the checkpointer saves/loads state scoped to that thread. Different thread IDs=independent conversation histories. Same thread ID=continued conversation with full state history.

14. Explain the concept of checkpoints, threads, and runs in LangGraph.

Thread: A persistent conversation session identified by thread_id. Contains the full history of states.
Checkpoint: A snapshot of the graph state at a specific point in execution. Multiple checkpoints exist per thread (one per node execution).
Run: A single invocation of the graph within a thread. A thread can have many runs, each adding new checkpoints.
15. How does time-travel work in LangGraph? Because every node execution is checkpointed, you can replay the graph from any past checkpoint. Use graph.get_state_history(config) to list all checkpoints for a thread, then invoke with a specific checkpoint_id to branch from that point. Useful for debugging, A/B testing different agent decisions, and correcting mistakes.

history = list(graph.get_state_history(config))
past_config = history[2].config  # 3rd checkpoint
graph.invoke(None, past_config)  # resume from there
16. How do you update state manually between graph invocations?

Use graph.update_state(config, values) to inject state changes outside of node execution. This is useful for human-in-the-loop corrections — a human reviews the state, modifies it, then resumes execution.

graph.update_state(config, {"messages": [HumanMessage("Actually, use Python")]})
graph.invoke(None, config) # continues with updated state

17. What is checkpoint_ns and when does it matter?

checkpoint_ns (namespace) is used to scope checkpoints within subgraphs. When a parent graph calls a subgraph, the subgraph’s checkpoints are namespaced separately to avoid collisions. It matters when debugging subgraph execution or implementing fine-grained state inspection in nested graph architectures.

18. How do you implement cross-thread memory (shared state across conversations)?

LangGraph’s built-in checkpointing is per-thread. For cross-thread memory (e.g., user preferences shared across sessions), use the Store interface (InMemoryStore, AsyncPostgresStore). The store is a key-value system accessible from any node via the RunnableConfig or by injecting it as a node parameter.

from langgraph.store.memory import InMemoryStore
store = InMemoryStore()
graph.compile(checkpointer=checkpointer, store=store)

Human in the loop:
19. How do you implement human-in-the-loop in LangGraph?

Use interrupt_before or interrupt_after in graph.compile() to pause execution at specific nodes. The graph raises an Interrupt exception, saves state and waits. A human reviews/modifies state, then resumes with graph.invoke (None, config).

graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["tool_node"]  # pause before tool execution
)
20. What is the interrupt() function and how does it differ from compile-time interrupts?

interrupt() (introduced in newer LangGraph versions) is called inside a node function to pause execution dynamically based on runtime conditions. Unlike compile-time interrupt_before/after  which always pauses at a node, interrupt() lets you conditionally pause based on state values — more flexible for dynamic approval workflows.

def human_review_node(state):
decision = interrupt({"question": "Approve this action?", "data": state})
return {"approved": decision}

21. How do you resume a graph after a human-in-the-loop interrupt?

After an interrupt, the graph state is saved in the checkpointer. To resume:

Optionally update state with human feedback: graph.update_state(config, new_values)
Resume execution: graph.invoke(None, config)— LangGraph detects the interrupted checkpoint and continues from where it stopped.
22. What is the difference between interrupt_before and interrupt_after?

 interrupt_before=[“node_name”] : Pauses BEFORE the node executes. The node hasn’t run yet — human can prevent or modify the action before it happens. Ideal for approval workflows.
interrupt_after=[“node_name”] : Pauses AFTER the node executes. Human reviews the output of the node. Ideal for review-and-correct workflows.
Tools & Tool Nodes
23. What is a ToolNode and how does it work?

ToolNode is a pre-built LangGraph node that executes tool calls found in the last AIMessage in the state. It automatically:

Extracts tool calls from the message
Executes each tool (in parallel if multiple)
Returns ToolMessage results appended to the messages list
from langgraph.prebuilt import ToolNode
tools = [search_tool, calculator_tool]
tool_node = ToolNode(tools)
graph.add_node("tools", tool_node)

24. How do you handle tool errors in LangGraph?

ToolNode has a tool_handle_error parameter (default True) that catches exceptions and returns them as ToolMessage with error content instead of crashing the graph. You can also pass a custom error handler function. For custom tool nodes, wrap tool calls in try/except and return error messages in the state.

25. How do you implement parallel tool execution in LangGraph?

ToolNode executes multiple tool calls from a single AIMessage in parallel using asyncio.gather (async) or ThreadPoolExecutor (sync). For custom parallel execution, use Send API to fan out to multiple nodes simultaneously.

26. What is create_react_agent and when should you use it vs building a custom graph?

create_react_agent is a prebuilt function that creates a standard ReAct (Reason + Act) agent graph with an LLM node and a ToolNode in a loop. Use it for standard tool-calling agents. Build a custom graph when you need: custom state fields, multiple agents, complex routing logic, specialized node behavior, or non-standard agent patterns.

27. How do you bind tools to an LLM in LangGraph?

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools)

def agent_node(state):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

The LLM generates AIMessage with tool_calls field populated when it decides to use a tool.

28. How do you implement a tool that modifies graph state directly? 

Use the InjectedState annotation to pass state into a tool, or use InjectedStore for store access. For direct state modification, return an Command object from the tool with state updates — this is the “tool-as-node” pattern in newer LangGraph versions.

from langgraph.prebuilt import InjectedState
from typing import Annotated

def my_tool(query: str, state: Annotated[dict, InjectedState]) -> str:
    # access state inside tool
    return f"User context: {state['user_id']}, query: {query}"


Multi-Agent Architectures
29. What are the main multi-agent patterns in LangGraph?

Supervisor: A central LLM routes tasks to specialized sub-agents and aggregates results.
Swarm/Handoff: Agents pass control to each other peer-to-peer using handoff tools.
Hierarchical: Supervisor agents manage other supervisor agents in a tree structure.
Parallel fan-out: Multiple agents run simultaneously on different subtasks, results merged.
Sequential pipeline: Output of one agent feeds into the next.
30. How do you implement a supervisor agent in LangGraph?

The supervisor is an LLM node that decides which worker agent to call next (or to finish). Workers are subgraphs or nodes. The supervisor uses a structured output or tool call to route.

def supervisor_node(state):
    response = supervisor_llm.invoke(state["messages"])
    return {"next": response.next_agent}  # "researcher", "coder", or "FINISH"

graph.add_conditional_edges("supervisor", lambda s: s["next"], {
    "researcher": "researcher_node",
    "coder": "coder_node",
    "FINISH": END
})

Fundamentals & State Graphs (8 Questions)
The first 20 minutes. Short answers, clean tradeoffs, no hand-waving on state.

Q1. What is LangGraph and when do you reach for it over plain LangChain?
LangGraph is a low-level runtime for building stateful, multi-step LLM applications as a graph: nodes do work, edges decide what runs next, and a typed state object flows through. You reach for it over a plain LangChain chain when control flow stops being linear — when you need cycles (retry until a quality bar), branching on model output, persistence across steps, or a human approval gate. A chain is a straight line; LangGraph is a state machine.

Testing: Whether you pick LangGraph for a real reason (cycles, state, persistence) instead of because it's trendy.

Q2. Explain the state graph model: nodes, edges, typed state.
The state is a typed dict (usually a TypedDict or Pydantic model) that defines the shape of everything passing through the graph. Each node is a function that takes the current state and returns a partial update. Edges connect nodes and decide order. The graph runs a node, merges its returned update into state, then follows edges to the next node. State is the contract; nodes only ever see and update that shared object.

Testing: Whether you can describe the run loop precisely — node returns a partial update, runtime merges it — not just name the parts.

Q3. What is a reducer, and what happens when parallel branches write the same field without one?
A reducer is a function attached to a state field that says how to merge updates instead of overwriting. The classic is add_messages (or operator.add) on a message list, so each node appends rather than replaces.

Without a reducer, the default behavior is last-write-wins overwrite. If two parallel branches both write the same field in the same step, you get a conflict — LangGraph raises an InvalidUpdateError because it can't decide which write wins. With a reducer, both writes merge deterministically.

from typing import Annotated, TypedDict
import operator

class State(TypedDict):
    # parallel branches can both append; results concatenate
    results: Annotated[list, operator.add]
    # no reducer: two branches writing this in one step -> error
    summary: str
Testing: This is the trap question. They want to hear "without a reducer, concurrent writes to the same key error out," not a vague "it merges somehow."

Q4. StateGraph vs MessageGraph vs prebuilt create_react_agent.
StateGraph is the general low-level API — you define your own state schema and full control flow. MessageGraph is the older specialization where state is just a list of messages; it's largely superseded by StateGraph with a messages field plus add_messages. create_react_agent is the prebuilt: one call gives you a ReAct tool-calling agent already running on the LangGraph runtime. Reach for the prebuilt to ship fast, drop to StateGraph when you need custom state or non-ReAct control flow.

Testing: Whether you know the prebuilt exists and when its convenience stops being worth the loss of control.

Q5. How does state flow and merge through nodes?
A node receives the full current state and returns a dict containing only the keys it wants to change. The runtime takes that partial dict and merges it into the running state — overwrite by default, or via the reducer if the key has one. Keys the node didn't return are untouched. This is why you return {"count": state["count"] + 1} and not the whole object: you describe the delta, the runtime applies it.

Testing: Whether you understand partial updates and merge semantics, the source of most LangGraph bugs.

Q6. Conditional edges vs normal edges — show the routing logic.
A normal edge always goes from A to B. A conditional edge runs a function that reads state and returns the name of the next node, so the graph branches on data or model output.

def route(state: State) -> str:
    if state["score"] >= 0.8:
        return "finish"
    return "retry"

graph.add_conditional_edges("grade", route, {"finish": END, "retry": "generate"})
Testing: Whether you can write the router cleanly and know the return value maps to a node name.

Q7. START, END, entry points, and graph compilation.
START and END are sentinel nodes. An edge from START sets the entry point; an edge to END terminates that path. You build the graph by adding nodes and edges, then call .compile() to get a runnable. Compilation validates the graph (reachability, dangling edges) and is also where you attach a checkpointer. You invoke the compiled object, not the builder.

Testing: Whether you know .compile() is a real step and where the checkpointer gets wired in.

Q8. How do you keep state small and typed in production?
Put only what nodes actually need in state, and type every field. Don't stuff giant tool outputs or full documents into state — store them by reference (an ID or URL) and fetch on demand, because every checkpoint serializes the whole state. Use reducers deliberately so lists don't grow unbounded. A bloated, loosely-typed state is the thing that makes checkpoints expensive and bugs invisible.

Testing: Whether you've felt the cost of fat state — serialized on every checkpoint — in a real system.

Nodes, Edges & Conditional Routing (6 Questions)
Now they want you writing graph code, not describing it.

Q9. Write a node function — signature and return shape.
A node takes state and returns a partial update dict. That's the whole contract.

def generate(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}  # appended via add_messages reducer
It can also take a second config arg for runtime data (thread_id, user info). It must return a dict whose keys exist in the state schema — return nothing you don't intend to merge.

Testing: Whether you return a partial dict, not the mutated full state, and whether you know the optional config param.

Q10. Build a conditional router that branches on state.
from langgraph.graph import StateGraph, START, END

def needs_tool(state: State) -> str:
    last = state["messages"][-1]
    return "tools" if last.tool_calls else "respond"

g = StateGraph(State)
g.add_node("agent", agent_node)
g.add_node("tools", tool_node)
g.add_node("respond", respond_node)
g.add_edge(START, "agent")
g.add_conditional_edges("agent", needs_tool, {"tools": "tools", "respond": "respond"})
g.add_edge("tools", "agent")  # loop back
g.add_edge("respond", END)
app = g.compile()
Testing: Whether you wire the router, the branch map, and the loop-back edge correctly in one pass.

Q11. Loops and retries: the quality-score retry pattern.
Generate, grade, and loop back if the grade is below a bar — with a max-attempts guard so it can't spin forever.

def route(state: State) -> str:
    if state["score"] >= 0.8 or state["attempts"] >= 3:
        return END
    return "generate"

g.add_conditional_edges("grade", route, {"generate": "generate", END: END})
The attempt counter lives in state and increments in the node. Without the cap, a model that never clears the bar loops until it blows your token budget.

Testing: Whether you add the hard attempt cap. Anyone who's run a retry loop in prod adds it reflexively.

Q12. Cycles vs DAGs — when LangGraph beats a chain.
A plain LangChain chain is a DAG: it flows one direction and stops. LangGraph allows cycles — a node can route back to an earlier node. That's the whole reason it exists. Any pattern that needs "keep going until a condition holds" — ReAct loops, reflection, retry-until-valid — is a cycle. If your task is genuinely one-directional, a chain is simpler and you don't need LangGraph.

Testing: Whether you can name the dividing line: cycles are the value, linear flows don't need the runtime.

Q13. Tool nodes and the ToolNode / tools_condition pattern.
ToolNode is a prebuilt node that takes the tool calls from the last AI message, runs the matching tools, and appends the results as tool messages. tools_condition is the matching prebuilt router: it sends flow to the tool node if the last message has tool calls, otherwise to the end.

from langgraph.prebuilt import ToolNode, tools_condition

g.add_node("tools", ToolNode(tools))
g.add_conditional_edges("agent", tools_condition)
g.add_edge("tools", "agent")
Testing: Whether you know the prebuilt tool-calling loop instead of hand-rolling tool dispatch.

Q14. Subgraphs: when and how to nest a graph.
A subgraph is a compiled graph used as a node inside a parent graph. Reach for it when a chunk of logic is self-contained and reusable — a research subroutine, a per-agent loop in a multi-agent system. The key question is state: if the subgraph shares state keys with the parent, they merge directly; if its schema differs, you wrap it in a node that maps parent state in and subgraph results out. Subgraphs keep big graphs readable and let teams own pieces independently.

Testing: Whether you understand state mapping at the boundary, the part people get wrong.

Checkpointing & Persistence (6 Questions)
This is where LangGraph earns its keep, and where weak candidates get exposed.

Q15. What is a checkpointer and why does it need a thread_id?
A checkpointer saves the full graph state after every step. That snapshot is what lets you pause, resume, recover from a crash, and do human-in-the-loop. The thread_id is the key that scopes a conversation or run: all checkpoints for one user session share a thread_id, so when you resume you pass that id and the runtime loads the right history. No thread_id, no way to know which run's state to restore.

from langgraph.checkpoint.memory import InMemorySaver
app = g.compile(checkpointer=InMemorySaver())
app.invoke(inp, config={"configurable": {"thread_id": "user-42"}})
Testing: Whether you connect the checkpointer to the thread_id as the scoping key — not just "it saves state."

Q16. In-memory vs durable checkpointers (Postgres / SQLite) — tradeoffs.
InMemorySaver keeps checkpoints in process memory: zero setup, gone on restart, fine for tests and notebooks. A durable saver (PostgresSaver, SqliteSaver) persists to a database so state survives process restarts, scales across workers, and supports real recovery. In production you use a durable one — in-memory means a crash loses every in-flight run. Postgres also lets multiple app instances share thread state.

Testing: Whether you reach for durable in prod and know in-memory is a dev-only convenience.

Q17. Checkpointer (within-thread) vs Store (cross-thread long-term memory).
These are two different memory systems. The checkpointer is short-term, within-thread memory: it holds the state of one conversation, keyed by thread_id. The Store is long-term, cross-thread memory: a key-value store, namespaced (often by user), that persists facts across separate threads — so something learned in Monday's session is available in Friday's. Checkpointer answers "where were we in this run"; Store answers "what do we know about this user across all runs."

Testing: Whether you keep the two straight. Mixing them up — trying to use thread state for long-term user memory — is a common, telling mistake.

Q18. Resume after a crash: load the checkpoint, re-enter at the next node.
With a durable checkpointer, resume is just invoking again with the same thread_id and no new input. The runtime loads the latest checkpoint and continues from the node after the last successful one — it does not replay completed nodes.

# crash happened mid-run; relaunch with the same thread_id
app.invoke(None, config={"configurable": {"thread_id": "user-42"}})
Passing None as input signals "continue from saved state" rather than "start fresh."

Testing: Whether you know resume re-enters at the next pending node and doesn't re-run finished work.

Q19. Partial side effects on resume — idempotency keys.
Resume re-enters at the node after the last checkpoint. But if a node did real work — charged a card, sent an email, wrote a row — and then the process died before checkpointing, a resume can re-run that node and repeat the side effect. The fix is idempotency keys: derive a stable key (thread_id + step + action) and have the downstream system dedupe on it, or check "did I already do this" before acting. Checkpoints make state recoverable; they don't make external side effects exactly-once on their own.

Testing: Whether you've thought about the gap between "state recovered" and "side effect not repeated." Senior-level signal.

Q20. Cost of checkpointing and checkpoint-by-reference for big tool results.
Every checkpoint serializes the whole state, so fat state means slow writes and a big database. If a node produces a large blob — a scraped page, a big tool result, a document — don't put the blob in state. Write it to object storage or a cache, put the reference (an ID or URL) in state, and have downstream nodes fetch by reference. You checkpoint a short pointer instead of megabytes on every step.

Testing: Whether you'd keep checkpoints cheap under real payloads instead of serializing huge tool outputs repeatedly.

Human-in-the-Loop (5 Questions)
HITL is the round that maps directly to "would I let your agent touch prod."

Q21. When do you interrupt for human approval (the blast-radius rule)?
Gate any action whose blast radius is real: spending money, sending external communication, writing to a system of record, deleting data, anything irreversible or expensive. Reads and reversible internal steps run unattended. Draw the line by consequence, not by step type — if a wrong action costs money, leaks data, or pages someone, put a human in front of it.

Testing: Whether you gate by blast radius instead of gating everything (slow) or nothing (reckless).

Q22. interrupt() + Command resume — show the pattern.
interrupt() pauses the graph inside a node and surfaces a payload to the caller. The human responds, and you resume with Command(resume=...), which makes interrupt() return that value and continue.

from langgraph.types import interrupt, Command

def approval(state: State):
    decision = interrupt({"action": state["proposed_action"]})
    return {"approved": decision == "yes"}

# first call pauses at the interrupt
app.invoke(inp, config=cfg)
# human reviewed; resume with their answer
app.invoke(Command(resume="yes"), config=cfg)
Testing: Whether you know the modern interrupt() / Command(resume=...) pair, not a deprecated static-breakpoint hack.

Q23. Let a human edit the plan or state mid-run, then resume.
Two paths. The interrupt() payload can carry the current plan; the human returns an edited version, and the node writes it back to state before continuing. Or you use update_state to patch the checkpoint directly between invocations, then resume — the graph picks up the edited state. Either way, the edit lands in the checkpoint, so the resumed run reasons over the human-corrected plan.

Testing: Whether you can both surface state for editing and write the edit back so resume honors it.

Q24. Approve / edit / reject patterns at an interrupt point.
One interrupt, three outcomes based on the human's response. Approve: proceed as proposed. Edit: the human returns a modified action, which you write to state and execute. Reject: route to an alternative node — regenerate, escalate, or end. The node reads the resume value and a conditional edge branches on it. This trio covers essentially every review workflow.

Testing: Whether you handle all three branches, especially reject, instead of a binary yes/no.

Q25. Why interrupt() needs a checkpointer, and common gotchas.
interrupt() works by saving state, returning control, and later restoring — that only works if there's a checkpointer to save to. No checkpointer, no pause point to resume from. Common gotchas: forgetting to pass the same thread_id on resume (it starts fresh); putting non-idempotent side effects before the interrupt in the same node, so they re-run when the node restarts on resume; and assuming the node resumes mid-line — it re-executes from the top, with interrupt() returning the resume value.

Testing: Whether you know the node re-runs from the top on resume — the single most common interrupt bug.

Multi-Agent Graphs & Streaming (6 Questions)
Coordination and output. Where over-engineering shows up first.

Q26. Supervisor vs swarm vs hierarchical multi-agent in LangGraph.
Supervisor: one router agent owns control and delegates to worker agents, collecting results — centralized, easy to reason about. Swarm: agents hand off to each other peer-to-peer with no central boss — flexible, harder to debug. Hierarchical: supervisors of supervisors, for large systems that decompose into teams. In LangGraph each agent is a node or subgraph, and handoffs are edges or Command(goto=...) jumps. Default to supervisor; reach for swarm or hierarchy only when one router becomes the bottleneck.

Testing: Whether you default to the simplest topology that works instead of building a swarm for a two-tool problem.

Q27. Shared-state vs message-passing handoffs between agents.
Shared-state: all agents read and write one graph state; a handoff just routes to the next agent, which sees everything. Message-passing: each agent gets a scoped message and returns a result, with less shared surface. Shared state is simpler and the LangGraph-native default, but couples agents and risks key collisions (use reducers). Message-passing isolates agents at the cost of plumbing. Most LangGraph multi-agent systems use shared state with careful field ownership.

Testing: Whether you understand the coupling tradeoff and would use reducers to avoid handoff write conflicts.

Q28. When is one agent plus a tool list better than multi-agent?
Almost always, until it isn't. One agent with a good tool list is simpler to build, cheaper to run, and far easier to debug. Go multi-agent only when you have a real reason: context windows blowing up from too many tools, genuinely separate domains needing different prompts or models, or independent subtasks you want to run in parallel. Multi-agent multiplies cost and failure modes; reach for it when a single agent measurably can't cope, not by default.

Testing: Whether you resist multi-agent hype. The strongest answer starts with "usually one agent is enough."

Q29. Streaming modes: values / updates / messages / custom.
stream_mode controls what the graph emits. values: the full state after each step. updates: only the delta each node returned — lighter, good for tracing which node did what. messages: LLM tokens as they generate, for chat UIs. custom: arbitrary data you emit from inside a node (progress, tool status). You can combine modes. Pick messages for token streaming to a user, updates for observability.

for chunk in app.stream(inp, config=cfg, stream_mode="updates"):
    print(chunk)
Testing: Whether you can match each mode to a use case rather than just listing names.

Q30. Stream tokens to a UI without leaking full state.
Use stream_mode="messages" to stream only the model tokens, not values (which would push the entire state object — internal scratchpad, tool results, secrets — to the client every step). For richer UI signals, add custom events you explicitly emit, so you control exactly what the frontend sees. The rule: stream what the user should see, never the whole state.

Testing: Whether you'd avoid shipping internal state to the browser — a security and UX instinct.

Q31. Map-reduce / fan-out with Send and parallel nodes.
Send lets one node dispatch dynamic parallel work: you return a list of Send("worker", payload) objects and the runtime fans out a worker invocation per payload, in parallel. Each worker writes to a state field with a reducer (so concurrent writes append instead of colliding), and a downstream node reduces the collected results. That's map-reduce: fan out with Send, gather with a reducer.

from langgraph.types import Send

def fan_out(state):
    return [Send("worker", {"item": x}) for x in state["items"]]
Testing: Whether you pair Send with a reducer on the result field — without it, parallel workers collide (see Q3).

Production & LangGraph vs LangChain (4 Questions)
The design and judgment round.

Q32. Design a stateful research agent for 10k requests/day on LangGraph.
10k/day is about 7 requests/minute average, with bursts. That's modest — the constraints are cost and state, not raw throughput. Architecture: a create_react_agent or custom StateGraph with search and synthesis tools; a PostgresSaver checkpointer for durability and cross-instance state; large tool results stored by reference, not in state (Q20); a step cap and token budget per request to bound runaway loops; and an interrupt gate only on actions with real blast radius. Run app instances behind a queue so bursts don't exhaust API rate limits. The math is easy; the discipline is keeping state lean and bounding cost per request.

Testing: Whether you size the actual load, pick durable persistence, and bound cost — instead of over-architecting for imaginary scale.

Q33. LangGraph vs LangChain: explicit typed state vs implicit, when each wins.
LangChain chains carry state implicitly — it threads through the chain and you mostly don't manage it. LangGraph makes state explicit and typed: you declare the schema and every node operates on it. Implicit wins for simple linear pipelines where the ceremony isn't worth it. Explicit wins the moment you need branching, loops, persistence, or HITL — you want one typed object you can inspect, checkpoint, and resume. And remember the 2026 reality: LangChain's agents now run on the LangGraph runtime, so it's a layering choice, not a rivalry.

Testing: Whether you frame it as layers (LangGraph under, LangChain over) rather than competitors, and know when explicit state pays off.

Q34. Migrate an AgentExecutor app to StateGraph / create_react_agent.
AgentExecutor is deprecated and in maintenance until December 2026, so new work shouldn't use it. The fast migration: replace the AgentExecutor with create_react_agent(model, tools), which gives an equivalent ReAct loop on the LangGraph runtime, plus checkpointing and streaming for free. If you had custom logic the executor couldn't express — branching, retries, approval gates — that's the cue to drop to a hand-built StateGraph. Keep your tools and prompts; swap the execution layer underneath.

Testing: Whether you know AgentExecutor is deprecated and can name the two migration targets and when each applies.

Q35. Debug an agent that passes in dev but loops or fails in prod.
Start from the checkpoints — they're your trace. Pull the thread's state history and find where it diverged. Common prod-only causes: an in-memory checkpointer that loses state across workers, so resumes start fresh; missing or wrong reducers causing parallel-write errors under real concurrency that never fired in single-threaded dev; unbounded loops because the step cap was a dev placeholder; and fat state hitting serialization or DB limits at production payload sizes. The discipline is replaying from the checkpoint, not re-running blind.

Testing: Whether you debug from checkpoint state and know which failure modes are concurrency- and scale-specific.


