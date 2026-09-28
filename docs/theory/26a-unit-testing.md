# Unit-testing chains, graders, and graphs

## 30-second answer

Unit-test graders and routers in isolation with fixed fixtures and exact asserts (`relevant is True`, `datasource == "refuse"`). Test graph routing **without an LLM**: stub nodes, assert on `Command` destinations and reducer state. Put tests next to chains (`graph/chains/tests/`). Unit tests ≠ evaluation: exact properties in CI every PR; aggregate quality on a schedule (notebook 26).

## Tiny example

You lock three fixtures. Leave policy doc vs leave question → `relevant is True`. Pizza question vs policy doc → `relevant is False`. Jailbreak text → router `datasource == "refuse"`. A stubbed graph with no model calls still proves research → write → publish, and that hitting `MAX_REVISIONS` forces publish with `"budget"` in the trace.

## Why it exists

Capstone HTTP tests cover endpoints. Almost nobody tests the graders and routers that decide whether an answer is grounded. Those pieces silently rot when a prompt drifts or a model changes. This chapter rebuilds that pattern: structured graders first, then a whole graph without paying for a model.

## Runtime

**Components under test (same shape as notebook 48 / agentic RAG)**

1. `retrieval_grader` — `GradeDocuments(relevant: bool, reason: str)`.
2. `hallucination_grader` — `GradeHallucination(grounded: bool, ...)`. True in the world but absent from docs is **not** grounded.
3. `question_router` — `RouteQuery(datasource: Literal["vectorstore", "websearch", "refuse"])`.

**Pytest-style checks**

1. One assert per behaviour; fixtures, not live traffic.
2. Retrieval: relevant policy doc → True; pizza vs policy → False.
3. Hallucination: faithful paraphrase → grounded; fabricated pony benefits → not grounded.
4. Router: leave policy → `vectorstore`; weather → `websearch`; jailbreak → `refuse`.
5. Fail the session if any check fails (`assert FAIL == 0`).

**Graph without an LLM**

1. Small `StateGraph` with reducers (`Annotated[list, operator.add]`).
2. Supervisor returns `Command(goto=..., update={...})`.
3. Stub `research` / `write` / `publish` as pure lambdas — no model calls.
4. Assert happy path and budget path (`revisions >= MAX_REVISIONS` → publish with `"budget"` in trace).

**Layout**

```
graph/
  chains/
    retrieval_grader.py
    hallucination_grader.py
    router.py
    tests/
      test_chains.py
```

**Unit test vs evaluation**

| | Unit test | Evaluation (notebook 26) |
|---|---|---|
| Input | Fixed fixture | Dataset of many examples |
| Assert | Exact property | Aggregate score |
| Runs in CI | Every PR | Nightly / on release |
| Catches | Prompt drift, routing bugs | Quality regressions |

Do both.

```mermaid
flowchart TD
  fixture[FixedFixture] --> grader[StructuredGrader]
  grader --> assertExact[ExactAssert]
  stubNodes[StubbedNodes] --> graph[StateGraph]
  graph --> cmd[Command_goto]
  cmd --> stateAssert[StateAndTraceAssert]
  assertExact --> ci[CI_every_PR]
  stateAssert --> ci
  evalSet[EvalDataset] --> harness[Notebook26Harness]
  harness --> nightly[NightlyQuality]
```

*Picture: fixtures lock graders and stubbed graphs in CI; the eval harness runs on a schedule for quality.*

## Objects, fields, and merge rules

| Object | Role |
|---|---|
| `GradeDocuments` / `GradeHallucination` / `RouteQuery` | Contracts under test |
| `Command(goto, update)` | Supervisor routing + partial state merge |
| `MAX_REVISIONS` | Hard budget before forced publish |
| Stub node lambdas | Deterministic edges without LLM cost |

**Merge rules**

- Hallucination identity: world-true ∩ docs-absent ⇒ not grounded.
- Router refuse path is as important as the happy path.
- Graph reducers with `operator.add` **append/sum** — they do not overwrite. Assert cumulative `trace` / `revisions`.

## Control surface

| Knob | Effect |
|---|---|
| Fixture documents / questions | What behaviour is locked |
| Grader system prompts | Strictness of relevance / grounding |
| `MAX_REVISIONS` | When supervisor forces publish |
| Stub vs live model | Determinism and CI cost |
| Package layout | Tests next to chains |

## Failure anatomy

| Symptom | Cause | Fix |
|---|---|---|
| Grader flips on same fixture | Prompt drift / model change | Pin prompt; re-run suite |
| Hallucination grader passes inventions | Rubric too weak | Explicit “absent from docs = fail” |
| Jailbreak routes to vectorstore | Missing refuse fixtures | Assert `datasource == "refuse"` |
| Graph loops forever | No revision budget | Cap with `MAX_REVISIONS` |
| CI too slow / flaky | Live LLM in every test | Stub nodes |
| Quality still regresses | Only unit tests | Add notebook 26 on a schedule |
| Reducer surprises | Expected overwrite, got append | Assert cumulative lists |

## Keywords

- **Grader** — small structured chain that scores relevance or grounding.
- **Fixture** — fixed input/expected pair for exact asserts.
- **`Command`** — sets `goto` and a partial `update`.
- **Stubbed nodes** — pure functions replacing LLM nodes.
- **Unit test vs evaluation** — exact CI properties vs aggregate scores.
- **Refuse path** — router destination for unsafe / off-topic inputs.

## Minimal fragment

```python
yes = retrieval_grader.invoke({"question": "How many annual leave days?", "document": POLICY_DOC})
assert yes.relevant is True

hallucinated = hallucination_grader.invoke({
    "documents": POLICY_DOC,
    "generation": "Employees get unlimited mental health days and a company pony.",
})
assert hallucinated.grounded is False

out = graph.invoke({"findings": [], "draft": "", "revisions": 0, "published": False, "trace": []})
assert out["published"] is True
```

## Interview traps

**Shallow:** "Evaluation notebooks replace unit tests."

**Better:** Evaluation measures aggregate quality. Unit tests lock exact grader/router/graph properties on fixtures every PR. You need both.

**Shallow:** "You cannot test LangGraph without calling an LLM."

**Better:** Stub LLM nodes as lambdas. Assert on `Command` destinations, reducer merges, and budget paths. Zero tokens.

**Shallow:** "If the happy-path grader passes, the system is safe."

**Better:** Refuse / jailbreak routes and hallucination-negative fixtures catch silent rot.

## Lab

Hands-on: [26a_unit_testing_chains_and_graphs.ipynb](../../04-langchain-production/26a_unit_testing_chains_and_graphs.ipynb)
