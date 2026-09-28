"""
Build LangChain interview Q&A from sampleQuestonsdata.md using ORIGINAL answers only.
Does not invent or rewrite answers.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SAMPLE = ROOT / "sampleQuestonsdata.md"
OUT = ROOT / "interview" / "langchain-qa.md"

# First high-quality article: questions in order (exact wording as in sample).
ARTICLE1_QUESTIONS = [
    ("Basics", "What is LangChain?"),
    ("Basics", "Why would you use LangChain instead of calling an LLM API directly?"),
    ("Basics", "What are the main components of LangChain?"),
    ("Basics", "What problems does LangChain solve?"),
    ("Basics", "What types of applications can be built with LangChain?"),
    ("Core concepts", "What are chains?"),
    ("Core concepts", "What are tools?"),
    ("Core concepts", "What are agents?"),
    ("Core concepts", "What are Runnables?"),
    ("Core concepts", "What are output parsers?"),
    ("RAG", "How does LangChain implement RAG?"),
    ("RAG", "What is a retriever?"),
    ("RAG", "How do vector databases fit into LangChain?"),
    ("RAG", "What embedding models can LangChain use?"),
    ("RAG", "How would you improve retrieval quality?"),
    ("RAG", "How do you reduce hallucinations in a RAG application?"),
    ("Agents", "What is an AI agent?"),
    ("Agents", "How do LangChain agents choose tools?"),
    ("Agents", "What is the difference between a chain and an agent?"),
    ("Agents", "When should you avoid agents?"),
    ("Agents", "How do you evaluate an agent?"),
    ("Memory", "What is memory in LangChain?"),
    ("Memory", "What memory types are available?"),
    ("Memory", "How would you summarize conversation history?"),
    ("Memory", "What problems arise with long conversations?"),
    ("Memory", "How would you reduce context growth?"),
    ("Architecture", "Design a chatbot using LangChain"),
    ("Architecture", "Design a document question-answering system"),
    ("Architecture", "Design a customer support assistant"),
    ("Architecture", "How would you build a multi-step workflow?"),
    ("Architecture", "How would you add human approval?"),
    ("Architecture", "How would you debug complex chains?"),
    ("Production", "How do you deploy LangChain applications?"),
    ("Production", "How do you monitor LLM applications?"),
    ("Production", "How do you cache responses?"),
    ("Production", "How do you handle API failures?"),
    ("Production", "How do you control costs?"),
    ("Production", "How do you evaluate application quality?"),
    ("Advanced", "How would you build a multi-agent system?"),
    ("Advanced", "How would you implement retries and fallbacks?"),
    ("Advanced", "How would you manage context efficiently?"),
    ("Advanced", "How would you optimize latency?"),
    ("Advanced", "How would you evaluate tool selection?"),
    ("Advanced", "How would you build an enterprise RAG pipeline?"),
    ("Advanced", "What are common production bottlenecks?"),
    ("Comparisons", "LangChain vs. LlamaIndex"),
    ("Comparisons", "LangChain vs. Semantic Kernel"),
    ("Comparisons", "LangChain vs. Haystack"),
    ("Comparisons", "LangGraph vs. LangChain"),
]

# Section headers / fluff to strip when they appear as "answers"
SKIP_PREFIXES = (
    "Basic LangChain",
    "Questions About",
    "LangChain RAG",
    "LangChain Agents",
    "LangChain Memory",
    "LangChain Architecture",
    "LangChain Production",
    "Advanced LangChain",
    "LangChain Interview Tips",
    "Conclusion",
    "FAQs",
    "Every LangChain interview",
    "This is where interviewers",
    "RAG comes up",
    "Everyone knows agents",
    "Memory is what makes",
    "System design is a part",
    "As you can see",
    "Production questions are",
    "I'll now walk",
    "These questions are for senior",
    "The interviewer already knows",
    "Comparison questions",
    "Usually, the right answer",
    "Preparation for these",
    "LangChain interviews in",
    "While studying",
    "Using LangChain and",
    "I'll walk you",
    "If you're new",
)


def clean_answer(raw: str) -> str:
    """Keep original wording; only light cleanup for markdown readability."""
    # Drop "Powered By" junk from scraped pages
    raw = re.sub(r"(?m)^\s*Powered By\s*$", "", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw).strip()

    lines = raw.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Fix smashed imports: from x import y@tooldef -> break
        if "from " in stripped and ("@tool" in stripped or "class " in stripped or "def " in stripped):
            # try to split common smash patterns
            s = stripped
            s = re.sub(r"(@tool)(def )", r"\n\n\1\n\2", s)
            s = re.sub(r"(import \w+)(@tool)", r"\1\n\n\2", s)
            s = re.sub(r"(BaseModel)(from )", r"\1\n\n\2", s)
            s = re.sub(r"(PydanticOutputParser)(class )", r"\1\n\nclass ", s)
            s = re.sub(r"(InMemorySaver)(from )", r"\1\n\nfrom ", s)
            s = re.sub(r"(InMemoryCache)(set_llm)", r"\1\n\n\2", s)
            # put code in fence if it looks like code start
            if s.startswith("from ") or s.startswith("@tool") or s.startswith("class "):
                # collect following indented/code-ish lines already in block - just fence this line group
                code_lines = [ln for ln in s.splitlines() if ln.strip()]
                # also pull continuation lines that look like code
                j = i + 1
                while j < len(lines):
                    nxt = lines[j].strip()
                    if not nxt:
                        break
                    if nxt.startswith(
                        (
                            "from ",
                            "import ",
                            "def ",
                            "class ",
                            "@",
                            "agent ",
                            "parser ",
                            "return ",
                            "    ",
                            "severity:",
                            "issues:",
                            "model=",
                            "tools=",
                            "system_prompt=",
                            "middleware=",
                            "set_llm",
                            "model =",
                            "chain =",
                            "retriever =",
                            "result =",
                            "|",
                            "{",
                            "}",
                            ")",
                            "(",
                        )
                    ) or nxt.endswith((")", ",", "{", "}")) or nxt.startswith('"'):
                        code_lines.append(nxt)
                        j += 1
                    else:
                        break
                out.append("")
                out.append("```python")
                out.extend(code_lines)
                out.append("```")
                out.append("")
                i = j
                continue

        # Standalone code patterns starting a line
        if stripped.startswith(
            ("chain =", "retriever =", "agent =", "model =", "from langchain", "from langgraph", "from pydantic")
        ) or (
            stripped.startswith("from ") and "import" in stripped
        ):
            code_lines = [stripped]
            j = i + 1
            while j < len(lines):
                nxt = lines[j].rstrip()
                n = nxt.strip()
                if not n or n.startswith("Powered"):
                    if n.startswith("Powered"):
                        j += 1
                    break
                # end code when prose sentence (starts with capital word and has space, no =)
                if (
                    n
                    and not n.startswith(("|", "{", "}", ")", "(", "    ", "model=", "tools=", "system_", "middleware=", '"', "'"))
                    and not n.endswith((",", "{", "(", "=", "\\"))
                    and " = " not in n
                    and not n.startswith(("from ", "import ", "def ", "class ", "@", "return ", "agent.", "set_", "print"))
                    and re.match(r"^[A-Z]", n)
                    and len(n) > 40
                ):
                    break
                code_lines.append(n)
                j += 1
            out.append("")
            out.append("```python")
            out.extend(code_lines)
            out.append("```")
            out.append("")
            i = j
            continue

        # Bullet-like "Label: text" from sample — keep as markdown list when clear
        if re.match(r"^[A-Za-z][\w /-]*: .+", stripped) and len(stripped) < 200:
            label, _, rest = stripped.partition(":")
            # only convert known list styles
            if label in {
                "Models",
                "Prompts",
                "Tools",
                "Agents",
                "Retrievers",
                "Runnables",
                "Provider abstraction",
                "Tool calling",
                "Persistence",
                "Structured output",
                "Short-term memory",
                "Long-term memory",
                "Chunking",
                "Metadata filtering",
                "Hybrid search",
                "Reranking",
                "Query rewriting",
                "Trajectory",
                "Outcome",
                "Ingestion",
                "Retrieval",
                "Generation",
                "Knowledge",
                "Actions",
                "Control",
                "Supervisor",
                "Handoff",
                "Trimming",
                "Summarization",
                "Structured extraction",
                "Stream",
                "Parallelize",
                "Cache",
                "Context growth",
                "Retrieval quality",
                "Unbounded agent loops",
                "Provider rate limits",
                "Access control",
                "Incremental indexing",
                "Multiple sources",
                "Auditability",
                "Cost",
                "Latency",
                "Context overflow",
                "Attention dilution",
                "Descriptions",
                "Names",
                "Count",
                "Deterministic checks",
                "Reference comparison",
                "LLM-as-judge",
                "Human review",
                "Infrastructure metrics",
                "LLM-specific metrics",
                "Provider-side prompt caching",
                "Embedding caching",
            } or label.startswith(("Retrieve ", "Return ", "Chunk ")):
                out.append(f"- **{label}:**{rest}")
                i += 1
                continue

        out.append(line)
        i += 1

    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def extract_article1(text: str) -> list[tuple[str, str, str]]:
    """Extract Q→answer from first article using known question list."""
    # Work only on first triple-quoted article roughly
    end = text.find('"""', text.find("What is LangChain?"))
    # better: cut at second """ after line 25
    m = re.search(r"What is LangChain\?", text)
    if not m:
        raise SystemExit("Could not find start")
    start = m.start()
    # End before Basic Questions / second article
    end_m = re.search(r'\n"""\s*\n"""\s*\nBasic Questions', text)
    if not end_m:
        end_m = re.search(r"\n1\. What is LangChain, and why is it useful\?", text)
    chunk = text[start : end_m.start() if end_m else start + 50000]

    results: list[tuple[str, str, str]] = []
    for idx, (section, q) in enumerate(ARTICLE1_QUESTIONS):
        # find question as whole line-ish
        pos = chunk.find(q)
        if pos < 0:
            print("MISSING Q:", q)
            continue
        ans_start = pos + len(q)
        # next question start
        if idx + 1 < len(ARTICLE1_QUESTIONS):
            nq = ARTICLE1_QUESTIONS[idx + 1][1]
            ans_end = chunk.find(nq, ans_start)
            if ans_end < 0:
                ans_end = len(chunk)
        else:
            # stop before Interview Tips / Conclusion
            stop = re.search(r"\nLangChain Interview Tips\n|\nConclusion\n", chunk[ans_start:])
            ans_end = ans_start + stop.start() if stop else len(chunk)

        raw = chunk[ans_start:ans_end].strip()
        # strip leading section intros that appear before next Q's answer
        # (already between Q and next Q)
        ans = clean_answer(raw)
        # Strip trailing section banners / intros that sit between questions
        trail_headers = [
            "Questions About LangChain Core Concepts",
            "LangChain RAG Interview Questions",
            "LangChain Agents Interview Questions",
            "LangChain Memory Interview Questions",
            "LangChain Architecture Interview Questions",
            "LangChain Production Interview Questions",
            "Advanced LangChain Interview Questions",
            "Questions About LangChain vs Other AI Frameworks",
        ]
        for hdr in trail_headers:
            idx_h = ans.find(hdr)
            if idx_h >= 0:
                ans = ans[:idx_h].strip()
        results.append((section, q, ans))
    return results


def slug(n: int, q: str) -> str:
    s = re.sub(r"[^a-z0-9\s-]", "", q.lower())
    s = re.sub(r"\s+", "-", s).strip("-")[:70]
    return f"q{n}-{s}"


def main() -> None:
    text = SAMPLE.read_text(encoding="utf-8")
    items = extract_article1(text)

    lines: list[str] = []
    lines.append("# LangChain Interview Q&A")
    lines.append("")
    lines.append(
        "Questions and answers taken from [`docs/sampleQuestonsdata.md`](../sampleQuestonsdata.md) "
        "(primary article). Answers are kept in the original wording — not rewritten."
    )
    lines.append("")
    lines.append(f"**Total questions:** {len(items)}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Index")
    lines.append("")

    by_sec: dict[str, list[tuple[int, str]]] = {}
    numbered: list[tuple[int, str, str, str]] = []
    for i, (sec, q, ans) in enumerate(items, 1):
        numbered.append((i, sec, q, ans))
        by_sec.setdefault(sec, []).append((i, q))

    for sec, group in by_sec.items():
        lines.append(f"### {sec}")
        lines.append("")
        for n, q in group:
            lines.append(f"- [Q{n}. {q}](#{slug(n, q)})")
        lines.append("")

    lines.append("---")
    lines.append("")

    current = None
    for n, sec, q, ans in numbered:
        if sec != current:
            lines.append(f"## {sec}")
            lines.append("")
            current = sec
        lines.append(f'<a id="{slug(n, q)}"></a>')
        lines.append("")
        lines.append(f"### Q{n}. {q}")
        lines.append("")
        lines.append("**Answer.**")
        lines.append("")
        lines.append(ans)
        lines.append("")
        lines.append("---")
        lines.append("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"Wrote {OUT} with {len(items)} questions")
    # sanity: Q3 components
    for n, sec, q, ans in numbered:
        if "main components" in q.lower():
            print("--- Q3 preview ---")
            print(ans[:500])
            break


if __name__ == "__main__":
    main()
