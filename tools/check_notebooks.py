"""Static health check for every generated notebook.

Verifies, without calling any LLM (so it needs no API keys):

1. the notebook is valid JSON with the expected structure
2. all code cells parse as Python (syntax check on the concatenated cells)
3. every ``import`` used in the notebook actually resolves in this environment
4. the lesson has the required teaching sections

Run: ``python tools/check_notebooks.py`` or ``python tools/check_notebooks.py 02-langchain-rag``
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRACK_DIRS = [
    "00-setup",
    "01-langchain-foundations",
    "02-langchain-rag",
    "03-langchain-agents",
    "04-langchain-production",
    "05-langgraph-beginner",
    "06-langgraph-intermediate",
    "07-langgraph-advanced",
    "08-capstones",
]

REQUIRED_SECTIONS = ["## Why this matters", "## Recap", "## Try it yourself", "**Checklist ID**"]

# Imports that are legitimately optional: they belong to paid or heavyweight
# extras and their cells are guarded with require(...) in the lessons.
OPTIONAL_MODULES = {
    "langchain_cohere",
    "langchain_pinecone",
    "langchain_tavily",
    "langchain_anthropic",
    "langgraph_checkpoint_postgres",
    "langgraph.checkpoint.postgres",
    "psycopg",
    "deepagents",
    "redis",
    "langchain_redis",
    "pinecone",
    "weaviate",
    "chromadb",
    "sentence_transformers",
    "torch",
    "transformers",
    "wikipedia",
    "duckduckgo_search",
    "ddgs",
    "numexpr",
    "unstructured",
    "grandalf",
    "matplotlib",
    "IPython",
    "uvicorn",
    "nest_asyncio",
    "litellm",
    "langchain_litellm",
    "langchain_mcp_adapters",
    "mcp",
    "mcp.server.fastmcp",
    "langchain_experimental",
    "langchain_experimental.tools",
    "langchain_experimental.utilities",
}


def notebook_code(nb: dict) -> str:
    cells = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        # Strip IPython magics and shell escapes, which are not valid Python.
        lines = [
            line
            for line in source.splitlines()
            if not line.lstrip().startswith(("%", "!", "?"))
        ]
        cells.append("\n".join(lines))
    return "\n\n".join(cells)


def notebook_markdown(nb: dict) -> str:
    return "\n".join(
        "".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "markdown"
    )


def collect_imports(tree: ast.AST) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                modules.add(node.module)
    return modules


def module_available(name: str) -> bool:
    if name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def resolves(module: str) -> bool:
    """A dotted import resolves if the full path imports or the root package exists."""
    if module_available(module):
        return True
    try:
        importlib.import_module(module)
        return True
    except Exception:  # noqa: BLE001 - any failure means unusable here
        return module_available(module.split(".")[0]) and module.split(".")[0] in OPTIONAL_MODULES


def check_notebook(path: Path) -> list[str]:
    problems: list[str] = []

    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"invalid JSON: {exc}"]

    if not nb.get("cells"):
        problems.append("notebook has no cells")

    code = notebook_code(nb)
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        # Top-level await is valid in notebooks but not in ast.parse by default.
        try:
            tree = ast.parse(code, mode="exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        except SyntaxError:
            return [f"syntax error line {exc.lineno}: {exc.msg}"]

    for module in sorted(collect_imports(tree)):
        root = module.split(".")[0]
        if root in {"shared", "app"}:
            continue
        if not resolves(module):
            severity = "optional" if root in OPTIONAL_MODULES or module in OPTIONAL_MODULES else "MISSING"
            if severity == "MISSING":
                problems.append(f"unresolved import: {module}")

    markdown = notebook_markdown(nb)
    for section in REQUIRED_SECTIONS:
        if section not in markdown:
            problems.append(f"missing required section: {section!r}")

    return problems


def main(argv: list[str]) -> int:
    track_filter = argv[0] if argv else None
    results: dict[str, list[str]] = {}
    counts = defaultdict(int)

    for track in TRACK_DIRS:
        if track_filter and track_filter not in track:
            continue
        track_dir = ROOT / track
        if not track_dir.exists():
            continue
        for path in sorted(track_dir.glob("*.ipynb")):
            problems = check_notebook(path)
            key = str(path.relative_to(ROOT))
            results[key] = problems
            counts["total"] += 1
            counts["failed" if problems else "passed"] += 1

    for name, problems in results.items():
        if problems:
            print(f"\nFAIL  {name}")
            for problem in problems:
                print(f"        - {problem}")
        else:
            print(f"ok    {name}")

    print(f"\n{counts['passed']}/{counts['total']} notebooks passed static checks.")
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
