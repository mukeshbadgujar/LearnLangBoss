"""One-call bootstrap that every notebook runs in its first code cell.

It loads ``.env``, prints which provider is active, and hands back small
conveniences (repo paths, a lazy model factory) so the lesson cells stay focused
on the concept being taught rather than on plumbing.
"""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

from shared.llm import (
    REPO_ROOT,
    available_providers,
    describe_environment,
    enable_tracing,
    get_chat_model,
    get_embeddings,
    has_key,
    load_env,
)


@dataclass
class NotebookContext:
    """Everything a lesson notebook needs to know about its environment."""

    lesson: str
    repo_root: Path
    data_dir: Path
    artifacts_dir: Path
    providers: list[str] = field(default_factory=list)
    tracing: bool = False

    @property
    def ready(self) -> bool:
        """True when at least one chat provider key is configured."""
        return bool(self.providers)

    def model(self, *args, **kwargs):
        """Shortcut for :func:`shared.llm.get_chat_model`."""
        return get_chat_model(*args, **kwargs)

    def embeddings(self, *args, **kwargs):
        """Shortcut for :func:`shared.llm.get_embeddings`."""
        return get_embeddings(*args, **kwargs)

    def data(self, *parts: str) -> Path:
        """Absolute path to a file under ``shared/sample_data``."""
        return self.data_dir.joinpath(*parts)

    def artifact(self, *parts: str) -> Path:
        """Absolute path under ``.artifacts`` (git-ignored scratch space)."""
        path = self.artifacts_dir.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


def add_repo_to_path() -> Path:
    """Make ``import shared...`` work no matter which folder the notebook lives in."""
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return REPO_ROOT


def setup(lesson: str = "", *, quiet: bool = False, trace: bool | None = None) -> NotebookContext:
    """Prepare the notebook environment and report what is available.

    Args:
        lesson: Notebook id, used as the LangSmith run name/project suffix.
        quiet: Suppress the printed summary.
        trace: Force LangSmith tracing on/off. ``None`` keeps the ``.env`` setting.
    """
    add_repo_to_path()
    load_env()

    warnings.filterwarnings("ignore", category=UserWarning, module="langsmith")
    warnings.filterwarnings("ignore", message=".*TqdmWarning.*")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")  # Chroma telemetry off

    tracing = False
    want_trace = (os.environ.get("LANGSMITH_TRACING", "").lower() == "true") if trace is None else trace
    if want_trace:
        tracing = enable_tracing(os.environ.get("LANGSMITH_PROJECT") or "genai-mastery")
    elif trace is False:
        os.environ["LANGSMITH_TRACING"] = "false"

    ctx = NotebookContext(
        lesson=lesson,
        repo_root=REPO_ROOT,
        data_dir=REPO_ROOT / "shared" / "sample_data",
        artifacts_dir=REPO_ROOT / ".artifacts" / (lesson or "scratch"),
        providers=available_providers(),
        tracing=tracing,
    )

    if not quiet:
        header = f"Lesson: {lesson}" if lesson else "Notebook environment"
        print(header)
        print("-" * max(len(header), 46))
        print(describe_environment())
        if not ctx.ready:
            print(
                "\nACTION NEEDED: copy .env.example to .env and add a key.\n"
                "Free option: https://console.groq.com/keys -> GROQ_API_KEY"
            )

    return ctx


def require(*env_keys: str, feature: str = "this section") -> bool:
    """Guard for optional cells: returns False (with a hint) when a key is missing.

    Usage::

        if require("TAVILY_API_KEY", feature="live web search"):
            ...run the cell...
    """
    missing = [k for k in env_keys if not has_key(k)]
    if missing:
        print(f"[skipped] {feature} needs {', '.join(missing)} in your .env - the rest of the notebook still works.")
        return False
    return True


def require_package(*modules: str, feature: str = "this section", pip: str = "") -> bool:
    """Guard for cells that need an optional dependency from requirements.txt.

    The later lessons demo packages that are deliberately left commented out in
    ``requirements.txt`` (litellm, mcp, deepagents...). Those cells print the
    install command instead of raising, so the notebook still runs top to bottom.

    Usage::

        if require_package("litellm", feature="gateway routing", pip="litellm"):
            ...run the cell...
    """
    import importlib.util

    missing = []
    for name in modules:
        try:
            found = importlib.util.find_spec(name) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            found = False
        if not found:
            missing.append(name)

    if missing:
        install = pip or " ".join(m.replace("_", "-") for m in missing)
        print(
            f"[skipped] {feature} needs {', '.join(missing)}.\n"
            f"          Install with:  python -m pip install {install}\n"
            f"          The rest of the notebook still works."
        )
        return False
    return True


__all__ = ["NotebookContext", "add_repo_to_path", "require", "require_package", "setup"]
