"""Shared helpers for the GenAI Mastery curriculum.

Import these from any notebook after running the standard bootstrap cell:

    from shared.notebook_setup import setup
    ctx = setup("01_models_messages")
"""

from shared.llm import (  # noqa: F401
    active_provider,
    available_providers,
    describe_environment,
    enable_tracing,
    get_chat_model,
    get_embeddings,
    get_vision_model,
    has_key,
    load_env,
)

__all__ = [
    "active_provider",
    "available_providers",
    "describe_environment",
    "enable_tracing",
    "get_chat_model",
    "get_embeddings",
    "get_vision_model",
    "has_key",
    "load_env",
]
