"""Provider-agnostic model access for the whole curriculum.

Every notebook calls :func:`get_chat_model` instead of importing a provider class
directly. That way one ``.env`` file decides which provider runs, and the same
lesson code works on Groq, OpenRouter, OpenAI or Anthropic without edits.

Resolution order (first one with an API key wins):

    1. Groq        - fast, generous free tier
    2. OpenRouter  - one key, many models, OpenAI-compatible API
    3. OpenAI      - the reference implementation
    4. Anthropic   - optional extra

Set ``LLM_PROVIDER=openrouter`` in ``.env`` to pin one provider explicitly.

Embeddings resolve separately (see :func:`embedding_provider`) but follow the
same free-first policy: Groq's embeddings endpoint, then a local
sentence-transformers model, then OpenAI.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Order in which providers are tried when none is pinned.
PROVIDER_ORDER: tuple[str, ...] = ("groq", "openrouter", "openai", "anthropic")

#: Environment variable that holds each provider's API key.
API_KEY_ENV: dict[str, str] = {
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

#: Fallback model per provider when the matching ``*_MODEL`` env var is unset.
DEFAULT_MODEL: dict[str, str] = {
    "groq": "llama-3.3-70b-versatile",
    "openrouter": "openai/gpt-4o-mini",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-5",
}

MODEL_ENV: dict[str, str] = {
    "groq": "GROQ_MODEL",
    "openrouter": "OPENROUTER_MODEL",
    "openai": "OPENAI_MODEL",
    "anthropic": "ANTHROPIC_MODEL",
}

OPENROUTER_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

#: Groq speaks the OpenAI protocol, including for embeddings.
GROQ_OPENAI_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_DEFAULT_EMBEDDING_MODEL = "nomic-embed-text-v1_5"


class NoProviderConfigured(RuntimeError):
    """Raised when no chat provider has a usable API key."""


# --------------------------------------------------------------------------- #
# Environment loading
# --------------------------------------------------------------------------- #
def load_env(dotenv_path: str | Path | None = None, override: bool = False) -> Path | None:
    """Load ``.env`` from the repo root into ``os.environ``.

    Falls back to a tiny hand-rolled parser when ``python-dotenv`` is missing so
    the very first notebook works before dependencies are fully installed.
    """
    path = Path(dotenv_path) if dotenv_path else REPO_ROOT / ".env"
    if not path.exists():
        return None

    try:
        from dotenv import load_dotenv

        load_dotenv(path, override=override)
        return path
    except ImportError:
        pass

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and (override or not os.environ.get(key)):
            os.environ[key] = value
    return path


def _key_for(provider: str) -> str:
    return (os.environ.get(API_KEY_ENV.get(provider, ""), "") or "").strip()


def has_key(name: str) -> bool:
    """True when an arbitrary env var (``TAVILY_API_KEY`` etc.) is filled in."""
    return bool((os.environ.get(name, "") or "").strip())


def available_providers() -> list[str]:
    """Providers that currently have an API key, in resolution order."""
    return [p for p in PROVIDER_ORDER if _key_for(p)]


def active_provider(preferred: str | None = None) -> str:
    """Pick the provider to use, honouring ``LLM_PROVIDER`` then the default order."""
    pinned = (preferred or os.environ.get("LLM_PROVIDER", "") or "").strip().lower()
    if pinned:
        if pinned not in API_KEY_ENV:
            raise ValueError(
                f"Unknown provider {pinned!r}. Choose one of: {', '.join(API_KEY_ENV)}"
            )
        if not _key_for(pinned):
            raise NoProviderConfigured(
                f"Provider {pinned!r} was requested but {API_KEY_ENV[pinned]} is empty in your .env"
            )
        return pinned

    for provider in available_providers():
        return provider

    raise NoProviderConfigured(
        "No chat provider key found. Copy .env.example to .env and fill in at least "
        "GROQ_API_KEY (free tier: https://console.groq.com/keys)."
    )


def model_name_for(provider: str) -> str:
    """The model id this provider should use, from env or the built-in default."""
    return (os.environ.get(MODEL_ENV[provider], "") or "").strip() or DEFAULT_MODEL[provider]


# --------------------------------------------------------------------------- #
# Chat models
# --------------------------------------------------------------------------- #
def get_chat_model(
    model: str | None = None,
    *,
    provider: str | None = None,
    temperature: float = 0.0,
    **kwargs: Any,
):
    """Return a ready-to-use chat model for the active provider.

    Args:
        model: Override the model id. Defaults to the provider's ``*_MODEL`` env var.
        provider: Force a provider (``"groq"``, ``"openrouter"``, ``"openai"``,
            ``"anthropic"``). Defaults to the resolution order above.
        temperature: Sampling temperature. Lessons default to ``0.0`` so output
            is reproducible while you learn.
        **kwargs: Passed straight through to the underlying chat class
            (``max_tokens``, ``timeout``, ``model_kwargs``, ...).

    Returns:
        A ``BaseChatModel`` exposing ``invoke`` / ``batch`` / ``stream`` /
        ``bind_tools`` / ``with_structured_output``.
    """
    load_env()
    chosen = active_provider(provider)
    model_id = model or model_name_for(chosen)

    if chosen == "openrouter":
        # OpenRouter speaks the OpenAI protocol, so we reuse ChatOpenAI with a base_url.
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_id,
            temperature=temperature,
            api_key=_key_for("openrouter"),
            base_url=os.environ.get("OPENROUTER_BASE_URL") or OPENROUTER_DEFAULT_BASE_URL,
            **kwargs,
        )

    if chosen == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=model_id, temperature=temperature, **kwargs)

    if chosen == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model_id, temperature=temperature, **kwargs)

    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=model_id, temperature=temperature, **kwargs)


def get_vision_model(temperature: float = 0.0, **kwargs: Any):
    """Chat model chosen for image input (notebook 22).

    Text-only models on Groq will reject images, so ``VISION_MODEL`` in ``.env``
    lets you point at a multimodal model without changing every other lesson.
    """
    load_env()
    explicit = (os.environ.get("VISION_MODEL", "") or "").strip()
    if explicit:
        # Allow "provider:model" syntax, e.g. "openrouter:google/gemini-2.0-flash-001".
        if ":" in explicit and explicit.split(":", 1)[0] in API_KEY_ENV:
            provider, model_id = explicit.split(":", 1)
            return get_chat_model(model_id, provider=provider, temperature=temperature, **kwargs)
        return get_chat_model(explicit, temperature=temperature, **kwargs)

    vision_defaults = {
        "openai": "gpt-4o-mini",
        "openrouter": "openai/gpt-4o-mini",
        "anthropic": "claude-sonnet-4-5",
        "groq": "meta-llama/llama-4-scout-17b-16e-instruct",
    }
    provider = active_provider()
    return get_chat_model(vision_defaults[provider], provider=provider, temperature=temperature, **kwargs)


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
def _groq_embeddings(**kwargs: Any):
    """Groq's OpenAI-compatible embeddings endpoint (free tier, 768 dimensions)."""
    from langchain_openai import OpenAIEmbeddings

    model = os.environ.get("GROQ_EMBEDDING_MODEL") or GROQ_DEFAULT_EMBEDDING_MODEL
    return OpenAIEmbeddings(
        model=model,
        api_key=os.environ["GROQ_API_KEY"],
        base_url=GROQ_OPENAI_BASE_URL,
        # Groq is not OpenAI, so skip the tiktoken-based length check that would
        # otherwise tokenise for the wrong model and chunk incorrectly.
        check_embedding_ctx_length=False,
        **kwargs,
    )


def _huggingface_embeddings(**kwargs: Any):
    """Local sentence-transformers model: free, offline, no rate limits."""
    from langchain_huggingface import HuggingFaceEmbeddings

    model = os.environ.get("HUGGINGFACE_EMBEDDING_MODEL") or "sentence-transformers/all-MiniLM-L6-v2"
    return HuggingFaceEmbeddings(model_name=model, **kwargs)


def _openai_embeddings(**kwargs: Any):
    from langchain_openai import OpenAIEmbeddings

    model = os.environ.get("OPENAI_EMBEDDING_MODEL") or "text-embedding-3-small"
    return OpenAIEmbeddings(model=model, **kwargs)


def _fake_embeddings(**_: Any):
    from langchain_core.embeddings import DeterministicFakeEmbedding

    return DeterministicFakeEmbedding(size=384)


def embedding_provider(provider: str | None = None) -> str:
    """Resolve which embedding provider will actually be used.

    Order (first that works wins), mirroring the free-first policy used for chat:

        1. ``EMBEDDING_PROVIDER`` in ``.env``, if set to something other than ``auto``
        2. ``groq``        - free tier, no local install, same key as the chat model
        3. ``huggingface`` - local and offline, but needs torch (~2 GB)
        4. ``openai``      - paid, but the best quality baseline
        5. ``fake``        - deterministic nonsense so notebooks still run
    """
    load_env()
    pinned = (provider or os.environ.get("EMBEDDING_PROVIDER") or "auto").lower()
    if pinned not in ("", "auto"):
        return pinned

    if has_key("GROQ_API_KEY"):
        return "groq"
    if importlib.util.find_spec("sentence_transformers") is not None:
        return "huggingface"
    if has_key("OPENAI_API_KEY"):
        return "openai"
    return "fake"


def get_embeddings(provider: str | None = None, **kwargs: Any):
    """Return an embedding model, preferring free providers.

    With no configuration this picks Groq when ``GROQ_API_KEY`` is set - the same
    free key the chat model uses - so the RAG notebooks produce real similarity
    scores without a multi-gigabyte local install.

    Pin a provider with ``EMBEDDING_PROVIDER`` in ``.env``:
    ``groq``, ``huggingface``, ``openai`` or ``fake``.

    Note:
        Groq's free tier is rate limited. When indexing a large corpus, either
        set ``EMBEDDING_PROVIDER=huggingface`` (after installing
        ``langchain-huggingface[full]``) or embed in smaller batches.
    """
    load_env()
    chosen = embedding_provider(provider)

    builders = {
        "groq": _groq_embeddings,
        "huggingface": _huggingface_embeddings,
        "openai": _openai_embeddings,
        "fake": _fake_embeddings,
    }
    builder = builders.get(chosen)
    if builder is None:
        raise ValueError(
            f"Unknown EMBEDDING_PROVIDER {chosen!r}. Use one of: {', '.join(builders)}."
        )

    try:
        return builder(**kwargs)
    except Exception as exc:  # pragma: no cover - missing key, no torch, offline
        print(
            f"[shared.llm] Embedding provider {chosen!r} unavailable "
            f"({type(exc).__name__}: {exc}).\n"
            "            Falling back to DeterministicFakeEmbedding so the notebook still runs.\n"
            "            Similarity scores will be MEANINGLESS. Fix by setting GROQ_API_KEY "
            "(free), or installing 'langchain-huggingface[full]' for local embeddings."
        )
        return _fake_embeddings()


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def enable_tracing(project: str | None = None) -> bool:
    """Turn on LangSmith tracing if a key is present. Returns True when enabled."""
    load_env()
    if not has_key("LANGSMITH_API_KEY"):
        return False
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = project or os.environ.get("LANGSMITH_PROJECT") or "genai-mastery"
    return True


def describe_environment(extra_keys: Iterable[str] = ()) -> str:
    """Human-readable summary of what is configured. Printed at the top of notebooks."""
    load_env()
    lines: list[str] = []

    found = available_providers()
    if found:
        chosen = active_provider()
        lines.append(f"Chat provider : {chosen}  ->  {model_name_for(chosen)}")
        others = [p for p in found if p != chosen]
        if others:
            lines.append(f"Also available: {', '.join(others)}  (set LLM_PROVIDER to switch)")
    else:
        lines.append("Chat provider : NONE CONFIGURED - add GROQ_API_KEY to .env")

    embed = embedding_provider()
    embed_model = {
        "groq": os.environ.get("GROQ_EMBEDDING_MODEL", GROQ_DEFAULT_EMBEDDING_MODEL),
        "openai": os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        "huggingface": os.environ.get("HUGGINGFACE_EMBEDDING_MODEL",
                                      "sentence-transformers/all-MiniLM-L6-v2"),
        "fake": "DeterministicFakeEmbedding - SCORES ARE MEANINGLESS",
    }.get(embed, "?")
    lines.append(f"Embeddings    : {embed}  ->  {embed_model}")
    if embed == "fake":
        lines.append("                ^ set GROQ_API_KEY (free) or install "
                     "langchain-huggingface[full]")

    tracing = "on" if (os.environ.get("LANGSMITH_TRACING", "").lower() == "true" and has_key("LANGSMITH_API_KEY")) else "off"
    lines.append(f"LangSmith     : {tracing}")

    optional = ["TAVILY_API_KEY", "COHERE_API_KEY", "PINECONE_API_KEY", *extra_keys]
    present = [k.replace("_API_KEY", "").lower() for k in optional if has_key(k)]
    lines.append(f"Optional keys : {', '.join(present) if present else 'none'}")

    return "\n".join(lines)


__all__ = [
    "NoProviderConfigured",
    "REPO_ROOT",
    "active_provider",
    "available_providers",
    "describe_environment",
    "embedding_provider",
    "enable_tracing",
    "get_chat_model",
    "get_embeddings",
    "get_vision_model",
    "has_key",
    "load_env",
    "model_name_for",
]
