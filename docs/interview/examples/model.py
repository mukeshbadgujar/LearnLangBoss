"""The one place interview examples get a chat model.

Every example imports `chat_model` from this file. To switch providers, change
`.env` in the repo root. Do not edit the example files.

    LLM_PROVIDER=groq
    LLM_PROVIDER=openrouter
    LLM_PROVIDER=openai
    LLM_PROVIDER=anthropic

Leave `LLM_PROVIDER` empty to use the first provider that has an API key:
Groq, then OpenRouter, then OpenAI, then Anthropic.

Set the model id with the matching variable: `GROQ_MODEL`, `OPENROUTER_MODEL`,
`OPENAI_MODEL`, or `ANTHROPIC_MODEL`.

Search examples use `embedding_model()`. That follows `EMBEDDING_PROVIDER`
in the same `.env` file (`auto`, `groq`, `openai`, or `huggingface`).
"""

from __future__ import annotations

import sys
from pathlib import Path

# docs/interview/examples/model.py -> repo root is three folders up.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.llm import (  # noqa: E402
    active_provider,
    embedding_provider,
    get_chat_model,
    get_embeddings,
    model_name_for,
)


def chat_model(temperature: float = 0.0):
    """Return the chat model selected by `LLM_PROVIDER` in `.env`."""
    return get_chat_model(temperature=temperature)


def embedding_model():
    """Return the embedding model selected by `EMBEDDING_PROVIDER` in `.env`."""
    return get_embeddings()


def provider_in_use() -> str:
    """Short label of the chat provider and model id, for printing."""
    provider = active_provider()
    return f"{provider} / {model_name_for(provider)}"


def embedding_in_use() -> str:
    """Name of the embedding provider selected in `.env`."""
    return embedding_provider()
