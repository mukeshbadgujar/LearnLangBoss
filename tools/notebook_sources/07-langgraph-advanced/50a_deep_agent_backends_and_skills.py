# %% [markdown]
# # 50a - Deep Agent Backends, Skills and AGENTS.md
#
# | | |
# |---|---|
# | **Level** | Expert |
# | **Time** | 60 minutes |
# | **Prerequisites** | `50_deep_agents` |
# | **Checklist ID** | `50a_deep_agent_backends_and_skills` |
# | **Sourced from** | [emarco177/langchain-course `project/agent-harnesses`](https://github.com/emarco177/langchain-course/tree/project/agent-harnesses), [krishnaik06/Deep-agents-With-Langchain](https://github.com/krishnaik06/Deep-agents-With-Langchain) |
#
# ## Why this matters
#
# Notebook 50 builds a deep agent **from primitives** - planning middleware, a
# state-backed workspace, sub-agents, a harness prompt. That is the right way
# to learn. This lesson uses the **packaged** `deepagents` harness the way the
# Kris Naik and Eden Marco courses do, and covers three things notebook 50
# only sketched:
#
# 1. **Backends** - where files actually live (`StateBackend`, `FilesystemBackend`, `StoreBackend`)
# 2. **Skills** - progressive-disclosure procedure files (`SKILL.md`)
# 3. **AGENTS.md** - standing context loaded into the workspace at invoke time
#
# If `deepagents` is not installed, every cell degrades to an explanation of
# the same idea built from the primitives you already know.

# %%
from pathlib import Path
import sys

_root = Path.cwd()
while not (_root / "shared").is_dir() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from shared.notebook_setup import setup, require_package  # noqa: E402

ctx = setup("50a_deep_agent_backends_and_skills")

# %%
HAS_DEEPAGENTS = require_package(
    "deepagents", feature="packaged deep agent harness", pip="deepagents"
)

from shared.llm import get_chat_model

model = get_chat_model()

# %% [markdown]
# ## 1. The bare harness
#
# Eden Marco's `01_bare_agent.py`: one argument, a working agent. The harness
# already includes planning, file tools, an `execute` tool, and the `task`
# tool for sub-agents.

# %%
if HAS_DEEPAGENTS:
    from deepagents import create_deep_agent

    bare = create_deep_agent(model=model)
    result = bare.invoke({"messages": [{"role": "user", "content": "In one sentence, what is an LLM?"}]})
    print(result["messages"][-1].content)
    print("\nstate keys:", sorted(k for k in result if k != "messages"))
else:
    print("Without deepagents: see notebook 50's hand-built version - same four pillars.")

# %% [markdown]
# ## 2. Backends - where files live
#
# | Backend | Files live in | Survives process exit | Use when |
# |---|---|---|---|
# | **StateBackend** (default) | Graph state (`result["files"]`) | No (unless checkpointer) | Demos, single-threaded runs |
# | **FilesystemBackend** | Real disk under a root | Yes | Local tools that need real paths |
# | **StoreBackend** | LangGraph store (notebook 49) | Yes, cross-thread | Multi-user, durable memory |
#
# Kris Naik's `3-backends.ipynb` compares all three. The interface the agent
# sees (`ls`, `read_file`, `write_file`, `edit_file`) does not change.

# %%
if HAS_DEEPAGENTS:
    # Default StateBackend - files appear in result["files"], never on disk.
    state_agent = create_deep_agent(model=model)
    task = (
        "Using your file tools: write /notes/hybrid.md with two sentences on hybrid search, "
        "then reply with only that file's contents."
    )
    out = state_agent.invoke({"messages": [{"role": "user", "content": task}]})
    print("answer:", out["messages"][-1].content[:300])
    files = out.get("files") or {}
    print(f"files in state: {sorted(files) or '(none - model may have answered inline)'}")

# %% [markdown]
# ```python
# # FilesystemBackend - real disk under a sandbox root.
# from deepagents.backends import FilesystemBackend
# disk_agent = create_deep_agent(
#     model=model,
#     backend=FilesystemBackend(root_dir=str(ctx.artifact("deep_sandbox"))),
# )
#
# # StoreBackend - durable across threads via the LangGraph store.
# from deepagents.backends import StoreBackend
# from langgraph.store.memory import InMemoryStore
# store_agent = create_deep_agent(
#     model=model,
#     backend=StoreBackend(store=InMemoryStore()),
# )
# ```
#
# **Never point `FilesystemBackend` at your repo root.** Point it at a scratch
# directory. Path-based permission rules (when available) should deny `..`.

# %% [markdown]
# ## 3. Skills - progressive disclosure
#
# A skill is a folder the agent can open on demand:
#
# ```
# skills/
#   report-writer/
#     SKILL.md          # short: when to use, core workflow
#     instructions.md   # long: templates and standards
#     examples.md       # worked examples
# ```
#
# The agent reads `SKILL.md` first (cheap), then opens the deeper files only
# when the skill applies. That is progressive disclosure - the same idea as
# not stuffing every procedure into the system prompt.
#
# We write a minimal skill now, matching the Kris Naik course layout.

# %%
skills_root = ctx.artifact("skills", "report-writer")
skills_root.mkdir(parents=True, exist_ok=True)

(skills_root / "SKILL.md").write_text('''\
---
name: report-writer
description: >
  Write a structured markdown report AFTER answering any substantive question.
  Save it with write_file. Skip for greetings and clarifications.
license: MIT
---

# Report Writer

## When to Use
- After any substantive answer
- When the user asks for a report or saved summary

## Core Workflow
1. Answer the user first.
2. Read `instructions.md` for the template.
3. Write the report to `/reports/<topic>-report.md`.
4. Tell the user where it was saved.
''', encoding="utf-8")

(skills_root / "instructions.md").write_text('''\
# Report template

# {Title}
Date: {ISO date}

## Question
## Approach
## Key Findings
## Answer
## Sources / Tools Used

Save under `/reports/` as kebab-case + `-report.md`.
Keep it self-contained. Do not invent claims absent from your answer.
''', encoding="utf-8")

print(f"skill at {skills_root}")

# %%
if HAS_DEEPAGENTS:
    try:
        skilled = create_deep_agent(
            model=model,
            skills=[str(ctx.artifact("skills"))],
            system_prompt=(
                "You are a research assistant. After substantive answers, use the "
                "report-writer skill to save a structured report."
            ),
        )
        skilled_out = skilled.invoke({
            "messages": [{"role": "user",
                          "content": "In 3 sentences, why use hybrid search in a policy chatbot?"}]
        })
        print(skilled_out["messages"][-1].content[:400])
        print("files:", sorted((skilled_out.get("files") or {})))
    except TypeError as exc:
        # Older deepagents builds used different kw names.
        print(f"skills= not supported in this deepagents version: {exc}")
        print("Fall back to notebook 50's store-based skills pattern.")

# %% [markdown]
# Without `deepagents`, the same idea is notebook 50's store-backed
# `list_skills` / `load_skill` tools - progressive disclosure via the store
# instead of the filesystem.

# %% [markdown]
# ## 4. AGENTS.md - standing context
#
# Kris Naik's `projects/AGENTS.md` is loaded into the agent's workspace so it
# always knows its own architecture and operating rules. Think of it as a
# project-level system prompt the agent can also *re-read* with `read_file`.

# %%
agents_md = ctx.artifact("project", "AGENTS.md")
agents_md.parent.mkdir(parents=True, exist_ok=True)
agents_md.write_text('''\
# AGENTS.md — Northwind Research Desk

## Operating rules
1. Plan with write_todos before any task longer than 2 steps.
2. Offload bulky notes to files; keep the conversation lean.
3. Prefer hybrid search over dense-only for policy questions.
4. Never invent policy. If unsure, say so and cite what you checked.
5. After substantive answers, write a report under /reports/.

## Architecture reminder
You are a deep agent: planning + filesystem + subagents + a detailed prompt.
Delegate research to subagents; synthesise yourself.
''', encoding="utf-8")
print(f"wrote {agents_md}")

# %%
if HAS_DEEPAGENTS:
    # Seed the virtual filesystem with AGENTS.md at invoke time.
    seeded = create_deep_agent(
        model=model,
        system_prompt="Follow AGENTS.md in your workspace as standing orders.",
    )
    seeded_out = seeded.invoke({
        "messages": [{"role": "user", "content": "What operating rules do you follow? Cite AGENTS.md."}],
        "files": {
            "/AGENTS.md": {
                "content": agents_md.read_text(encoding="utf-8"),
                "encoding": "utf-8",
            }
        },
    })
    print(seeded_out["messages"][-1].content[:500])

# %% [markdown]
# ## 5. Putting the knobs together
#
# Eden Marco's harness chapter turns one knob at a time:
#
# | Example | Knob |
# |---|---|
# | `01_bare_agent` | nothing - the default harness |
# | `02_planning_tool` | (ships by default; observe `write_todos`) |
# | `03_subagents` | `subagents=[...]` |
# | `04_filesystem` | observe `result["files"]` |
# | `05_system_prompt` | `system_prompt=...` |
#
# Plus, from the Kris Naik course: `skills=...`, backend choice, and `AGENTS.md`.
#
# Notebook 50 taught you why each knob exists. This lesson teaches you where
# the packaged API puts them.

# %% [markdown]
# ## 6. When to use the package vs primitives
#
# | Situation | Prefer |
# |---|---|
# | Learning / interviews / custom middleware | Primitives (notebook 50) |
# | Shipping a research desk quickly | `deepagents` package |
# | Need a backend the package does not support | Primitives, or extend the package |
# | Need to unit-test planning in isolation | Primitives (middleware is testable) |
#
# ## Try it yourself
#
# 1. Add a `python` skill (copy the structure from the Kris Naik repo) and ask
#    the agent to write a typed function - confirm it opens `instructions.md`.
# 2. Point a `FilesystemBackend` at `ctx.artifact("sandbox")`, run a task that
#    writes three files, and confirm they exist on disk after the run.
# 3. Compare token usage of a bare agent vs a skills-loaded agent on the same
#    question - progressive disclosure should win on long procedures.
#
# ## Recap
#
# | Concept | Takeaway |
# |---|---|
# | Bare `create_deep_agent` | Planning + files + subagents already included |
# | StateBackend | Default; files in `result["files"]` |
# | FilesystemBackend | Real disk under a sandbox root |
# | StoreBackend | Durable cross-thread files via the store |
# | Skills | Progressive disclosure via `SKILL.md` + deeper files |
# | AGENTS.md | Standing project context, re-readable as a file |
# | Package vs primitives | Ship with the package; learn and customise with primitives |
#
# ## Next
#
# Track 07 extensions are complete. On to the capstones.
#
# -> [../08-capstones/51_policy_rag_chatbot.ipynb](../08-capstones/51_policy_rag_chatbot.ipynb)
