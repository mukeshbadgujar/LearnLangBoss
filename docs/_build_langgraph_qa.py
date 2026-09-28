"""
Build LangGraph interview Q&A from sample_langgraph.md using ORIGINAL answers only.
Primary source: FAQ block "1 · LangGraph and Agentic AI Fundamentals" through Q50.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SAMPLE = ROOT / "sample_langgraph.md"
OUT = ROOT / "interview" / "langgraph-qa.md"

SECTIONS = [
    ("Fundamentals", range(1, 10)),
    ("State, schema and reducers", range(10, 18)),
    ("Persistence", range(18, 25)),
    ("Control flow", range(25, 33)),
    ("Multi-agent", range(33, 41)),
    ("HITL, streaming and production", range(41, 49)),
    ("Ecosystem comparisons", range(49, 51)),
]


def section_for(n: int) -> str:
    for name, r in SECTIONS:
        if n in r:
            return name
    return "Other"


def _looks_like_code_line(ns: str) -> bool:
    return bool(
        ns.startswith(
            (
                "from ",
                "import ",
                "def ",
                "class ",
                "return ",
                "assert ",
                "graph.",
                "builder.",
                "app.",
                "with ",
                "#",
                "for ",
                "if ",
                "else",
                "    ",
                ")",
                "}",
                "{",
                "@",
            )
        )
        or re.match(
            r"^(builder|graph|app|g|result|config|history|memory|checkpointer|encrypted|k8s_|db_|incident_|swarm|handoff_|triage|fan_out|tool_node|send_to_)\b",
            ns,
        )
        or ns.endswith((",", "\\"))
        or (" = " in ns and len(ns) < 100)
    )


def _is_prose_start(ns: str) -> bool:
    """True when a line is clearly explanatory prose after a code sample."""
    if ns.startswith(("Interview tip:", "Testing:")):
        return True
    prose_starts = (
        "Notice ",
        "MemorySaver ",
        "SqliteSaver ",
        "PostgresSaver ",
        "operator.add ",
        "create_supervisor",
        "create_handoff",
        "Calling ",
        "Requesting ",
        "Checkpointers ",
        "The Command",
        "triage decides",
        "Two layers",
        "Frame it",
        "Yes —",
        "Yes -",
        "A subgraph",
        "A checkpointer",
        "A thread",
        "A normal",
        "A node",
        "LangGraph's ",
        "LangGraph trades",
        "The honest ",
        "Without a ",
        "By default",
        "If a process",
    )
    if ns.startswith(prose_starts):
        return True
    # Long sentence starting with capital, not code
    if (
        re.match(r"^[A-Z“\"]", ns)
        and len(ns) > 50
        and not _looks_like_code_line(ns)
        and " = " not in ns
    ):
        return True
    return False


def clean_answer(raw: str) -> str:
    raw = raw.strip()
    # Drop "Interview tip:" / "Testing:" footers that are meta (keep if short tip after answer - user wants original; keep Testing lines as they are in sample for fundamentals later; for Q1-50 keep Interview tip)
    raw = re.sub(r"\n{3,}", "\n\n", raw)

    lines = raw.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        s = line.strip()

        # Start of code block detection
        if (
            s.startswith(("from ", "import ", "def ", "class ", "builder ", "graph ", "with ", "history ", "encrypted", "config =", "for ", "# "))
            or s.startswith("builder.")
            or s.startswith("graph.")
            or (s.startswith("k8s_agent") or s.startswith("db_agent") or s.startswith("incident_") or s.startswith("swarm ") or s.startswith("handoff_"))
            or s.startswith("triage decides") is False
            and (
                s.startswith(("return [", "return Command", "return Send", "result =", "app =", "g =", "memory =", "checkpointer", "to_replay", "earlier_", "past_config"))
            )
        ):
            # Special: don't treat prose "from the langgraph-supervisor" as code
            if s.startswith("from ") and "package" in s.lower():
                out.append(line)
                i += 1
                continue
            if not (
                s.startswith(("from ", "import ", "def ", "class ", "with ", "# Create", "# Development", "# Production", "# Get", "# Go back", "# Resume", "# ...", "# first", "# human", "# crash", "# Equivalent"))
                or re.match(r"^(builder|graph|app|g|result|config|history|memory|checkpointer|encrypted|k8s_|db_|incident_|swarm|handoff_|triage|fan_out|tool_node)\b", s)
                or s.startswith("    ")
                or s.startswith("return ")
                or s.startswith("assert ")
                or s.startswith("send_to_")
                or (s.startswith("for ") and ("stream" in s or "chunk" in s))
            ):
                out.append(line)
                i += 1
                continue

            code: list[str] = []
            j = i
            while j < len(lines):
                nxt = lines[j]
                ns = nxt.strip()
                if not ns:
                    # peek: blank then prose → end code before blank
                    if j + 1 < len(lines):
                        peek = lines[j + 1].strip()
                        if peek and _is_prose_start(peek):
                            break
                    code.append(nxt)
                    j += 1
                    continue
                if _is_prose_start(ns) and not _looks_like_code_line(ns):
                    break
                if ns.startswith(("Interview tip:", "Testing:", "Framework\t")):
                    break
                if "\t" in ns and ("Control level" in ns or (ns.startswith("LangGraph") and "Low-level" in ns)):
                    break
                code.append(nxt)
                j += 1

            # trim trailing blanks
            while code and not code[-1].strip():
                code.pop()
            if code:
                out.append("")
                out.append("```python")
                out.extend(c.rstrip() for c in code)
                out.append("```")
                out.append("")
            i = j
            continue

        # Markdown table for Q50
        if "\t" in s and ("Control level" in s or s.startswith(("LangGraph", "CrewAI", "AutoGen", "LangChain", "Framework"))):
            # convert TSV-ish to markdown table
            rows = []
            j = i
            while j < len(lines) and "\t" in lines[j]:
                rows.append([c.strip() for c in lines[j].split("\t")])
                j += 1
            if rows:
                out.append("")
                out.append("| " + " | ".join(rows[0]) + " |")
                out.append("| " + " | ".join("---" for _ in rows[0]) + " |")
                for r in rows[1:]:
                    # pad
                    while len(r) < len(rows[0]):
                        r.append("")
                    out.append("| " + " | ".join(r[: len(rows[0])]) + " |")
                out.append("")
            i = j
            continue

        # Bullet-like Label — text (em dash lists from sample)
        m = re.match(r"^(State|Nodes|Edges|Thread|Checkpoint|Run) — (.+)$", s)
        if m:
            out.append(f"- **{m.group(1)}:** {m.group(2)}")
            i += 1
            continue

        out.append(line)
        i += 1

    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def extract_faq(text: str) -> list[tuple[int, str, str, str]]:
    # Primary FAQ starts at Q1. What is LangGraph, and what problem...
    start = text.find("Q1. What is LangGraph, and what problem does it solve")
    if start < 0:
        raise SystemExit("FAQ start not found")
    # End after Q50 answer, before next big carousel / "LangGraph Advanced"
    end_markers = [
        "\nLangGraph Advanced Interview Questions",
        "\nFundamentals & State Graphs",
        "\nNaved Khan\nSenior Gen-AI",
    ]
    end = len(text)
    for m in end_markers:
        p = text.find(m, start + 100)
        if 0 < p < end:
            end = p
    # Prefer cut after Q50 table prose - find Q50 and take until blank section before line 715 content
    q50 = text.find("Q50. LangGraph vs. CrewAI", start)
    if q50 > 0:
        # take until double newline after long answer, or until "LangGraph works standalone" alternate section
        alt = text.find("\nLangGraph works standalone", q50)
        if alt > 0:
            end = min(end, alt)
        else:
            # take ~3k chars after Q50
            end = min(end, q50 + 3500)

    chunk = text[start:end]

    # Find all Qn. titles
    pattern = re.compile(r"^Q(\d+)\.\s+(.+)$", re.M)
    matches = list(pattern.finditer(chunk))
    items: list[tuple[int, str, str, str]] = []

    for i, m in enumerate(matches):
        num = int(m.group(1))
        title = m.group(2).strip()
        # join multi-line titles? titles seem one line
        ans_start = m.end()
        ans_end = matches[i + 1].start() if i + 1 < len(matches) else len(chunk)
        raw = chunk[ans_start:ans_end].strip()
        # strip section banners
        for hdr in (
            "2 · State, Schema and Reducers",
            "Beginner–Intermediate — Q10–Q17.",
            "Beginner — Q1–Q9.",
            "3 · Persistence: Checkpoints, Threads and Stores",
            "Intermediate — Q18–Q24.",
            "4 · Control Flow: Conditional Edges, Command and Send",
            "Intermediate–Advanced — Q25–Q32.",
            "5 · Multi-Agent Architectures",
            "Advanced — Q33–Q40.",
            "6 · Human-in-the-Loop, Streaming and Production",
            "Advanced — Q41–Q48.",
            "7 · LangGraph vs. The Ecosystem",
            "Q49–Q50.",
        ):
            if hdr in raw:
                raw = raw.split(hdr)[0].strip()
        ans = clean_answer(raw)
        items.append((num, section_for(num), title, ans))

    return items


def slug(n: int, q: str) -> str:
    s = re.sub(r"[^a-z0-9\s-]", "", q.lower())
    s = re.sub(r"\s+", "-", s).strip("-")[:70]
    return f"q{n}-{s}"


def main() -> None:
    text = SAMPLE.read_text(encoding="utf-8")
    items = extract_faq(text)
    # Sort by original Q number
    items.sort(key=lambda x: x[0])

    lines: list[str] = []
    lines.append("# LangGraph Interview Q&A")
    lines.append("")
    lines.append(
        "Questions and answers taken from [`docs/sample_langgraph.md`](../sample_langgraph.md) "
        "(primary FAQ: Fundamentals through ecosystem comparisons). "
        "Answers are kept in the original wording — not rewritten."
    )
    lines.append("")
    lines.append(f"**Total questions:** {len(items)}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Index")
    lines.append("")

    by_sec: dict[str, list[tuple[int, str]]] = {}
    for num, sec, q, _ in items:
        by_sec.setdefault(sec, []).append((num, q))

    # Preserve section order
    for sec, _, _ in [(s, None, None) for s, _ in SECTIONS]:
        if sec not in by_sec:
            continue
        lines.append(f"### {sec}")
        lines.append("")
        for num, q in by_sec[sec]:
            lines.append(f"- [Q{num}. {q}](#{slug(num, q)})")
        lines.append("")

    lines.append("---")
    lines.append("")

    current = None
    for num, sec, q, ans in items:
        if sec != current:
            lines.append(f"## {sec}")
            lines.append("")
            current = sec
        lines.append(f'<a id="{slug(num, q)}"></a>')
        lines.append("")
        lines.append(f"### Q{num}. {q}")
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
    # preview Q1 and Q3
    for num, sec, q, ans in items:
        if num in (1, 3, 18):
            print(f"\n=== Q{num} ===")
            print(q)
            print(ans[:280], "...")


if __name__ == "__main__":
    main()
