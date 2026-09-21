"""Build .ipynb lesson files from readable percent-format Python sources.

Why this exists
---------------
Raw ``.ipynb`` JSON is painful to edit and review. Every lesson is therefore
authored as a plain ``.py`` file in ``tools/notebook_sources/<track>/<name>.py``
using the Jupyter "percent" convention, and this script renders the notebooks.

Source format::

    # %% [markdown]
    # # Heading
    # Normal markdown text.

    # %%
    print("a code cell")

Usage::

    python tools/nbgen.py              # rebuild every notebook
    python tools/nbgen.py 01-langchain-foundations
    python tools/nbgen.py --check      # fail if any notebook is out of date
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE_ROOT = ROOT / "tools" / "notebook_sources"

CELL_MARKER = "# %%"
MARKDOWN_MARKER = "# %% [markdown]"
RAW_MARKER = "# %% [raw]"


def _split_cells(text: str) -> list[tuple[str, str]]:
    """Return ``[(cell_type, source), ...]`` from percent-format text."""
    cells: list[tuple[str, str]] = []
    cell_type = "code"
    buffer: list[str] = []
    started = False

    def flush() -> None:
        if not started:
            return
        source = "\n".join(buffer).strip("\n")
        if source.strip():
            cells.append((cell_type, source))

    for line in text.splitlines():
        stripped = line.rstrip()
        if stripped.startswith(CELL_MARKER):
            flush()
            started = True
            buffer = []
            if stripped.startswith(MARKDOWN_MARKER):
                cell_type = "markdown"
            elif stripped.startswith(RAW_MARKER):
                cell_type = "raw"
            else:
                cell_type = "code"
            continue
        buffer.append(line)

    flush()
    return cells


def _uncomment_markdown(source: str) -> str:
    """Strip the leading ``# `` that keeps markdown cells valid Python."""
    out: list[str] = []
    for line in source.splitlines():
        if line.startswith("# "):
            out.append(line[2:])
        elif line.strip() == "#":
            out.append("")
        else:
            out.append(line)
    return "\n".join(out).strip("\n")


def _as_lines(source: str) -> list[str]:
    """nbformat stores sources as a list of lines that each keep their newline."""
    lines = source.split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def build_notebook(source_path: Path) -> dict:
    """Convert one percent-format source file into a notebook dict."""
    text = source_path.read_text(encoding="utf-8")
    cells = []
    for cell_type, source in _split_cells(text):
        body = _uncomment_markdown(source) if cell_type in {"markdown", "raw"} else source
        cell: dict = {"cell_type": cell_type, "metadata": {}, "source": _as_lines(body)}
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {
                "name": "python",
                "version": "3.10",
                "mimetype": "text/x-python",
                "file_extension": ".py",
                "pygments_lexer": "ipython3",
                "nbconvert_exporter": "python",
                "codemirror_mode": {"name": "ipython", "version": 3},
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def target_path(source_path: Path) -> Path:
    """Map ``tools/notebook_sources/<track>/<name>.py`` -> ``<track>/<name>.ipynb``."""
    relative = source_path.relative_to(SOURCE_ROOT)
    return ROOT / relative.with_suffix(".ipynb")


def iter_sources(track_filter: str | None = None):
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        if path.name.startswith("_"):
            continue
        if track_filter and track_filter not in str(path.relative_to(SOURCE_ROOT)):
            continue
        yield path


def main(argv: list[str]) -> int:
    check_only = "--check" in argv
    args = [a for a in argv if not a.startswith("--")]
    track_filter = args[0] if args else None

    if not SOURCE_ROOT.exists():
        print(f"No sources found at {SOURCE_ROOT}")
        return 1

    written, stale, unchanged = 0, [], 0
    for source_path in iter_sources(track_filter):
        notebook = build_notebook(source_path)
        payload = json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"
        out_path = target_path(source_path)

        current = out_path.read_text(encoding="utf-8") if out_path.exists() else None
        if current == payload:
            unchanged += 1
            continue
        if check_only:
            stale.append(out_path.relative_to(ROOT))
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload, encoding="utf-8")
        written += 1
        print(f"wrote {out_path.relative_to(ROOT)}  ({len(notebook['cells'])} cells)")

    if check_only:
        if stale:
            print("Out of date notebooks:")
            for path in stale:
                print(f"  {path}")
            return 1
        print(f"All {unchanged} notebooks are up to date.")
        return 0

    print(f"\nDone. {written} written, {unchanged} already current.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
