"""Second-round API probe: exact signatures for the pieces the notebooks use."""

from __future__ import annotations

import inspect


def show(label, fn):
    try:
        print(f"\n=== {label} ===")
        print(fn())
    except Exception as exc:  # noqa: BLE001
        print(f"\n=== {label} ===\nFAILED: {type(exc).__name__}: {exc}")


def enum_parser():
    tries = [
        "langchain_classic.output_parsers",
        "langchain_classic.output_parsers.enum",
        "langchain_core.output_parsers.enum",
    ]
    out = []
    for module in tries:
        try:
            mod = __import__(module, fromlist=["*"])
            out.append(f"{module}: EnumOutputParser {'OK' if hasattr(mod, 'EnumOutputParser') else 'MISSING'}")
        except ImportError as exc:
            out.append(f"{module}: IMPORT FAILED ({exc})")
    return "\n".join(out)


def datetime_parser():
    out = []
    for module in ["langchain_classic.output_parsers", "langchain_core.output_parsers"]:
        try:
            mod = __import__(module, fromlist=["*"])
            out.append(f"{module}: DatetimeOutputParser {'OK' if hasattr(mod, 'DatetimeOutputParser') else 'MISSING'}")
        except ImportError as exc:
            out.append(f"{module}: FAILED {exc}")
    return "\n".join(out)


def middleware_hook_signatures():
    from langchain.agents.middleware import AgentMiddleware

    lines = []
    for hook in ["before_model", "after_model", "before_agent", "after_agent", "wrap_model_call", "wrap_tool_call"]:
        fn = getattr(AgentMiddleware, hook, None)
        lines.append(f"{hook}{inspect.signature(fn)}" if fn else f"{hook}: MISSING")
    return "\n".join(lines)


def decorator_signatures():
    from langchain.agents.middleware import before_model, dynamic_prompt, wrap_model_call, wrap_tool_call

    return "\n".join(
        [
            f"before_model{inspect.signature(before_model)}",
            f"wrap_model_call{inspect.signature(wrap_model_call)}",
            f"wrap_tool_call{inspect.signature(wrap_tool_call)}",
            f"dynamic_prompt{inspect.signature(dynamic_prompt)}",
        ]
    )


def hitl_signature():
    from langchain.agents.middleware import HumanInTheLoopMiddleware

    return f"HumanInTheLoopMiddleware{inspect.signature(HumanInTheLoopMiddleware.__init__)}"


def other_middleware_signatures():
    from langchain.agents.middleware import (
        ModelCallLimitMiddleware,
        ModelFallbackMiddleware,
        PIIMiddleware,
        ToolCallLimitMiddleware,
        ToolRetryMiddleware,
    )

    return "\n".join(
        [
            f"ModelFallbackMiddleware{inspect.signature(ModelFallbackMiddleware.__init__)}",
            f"ModelCallLimitMiddleware{inspect.signature(ModelCallLimitMiddleware.__init__)}",
            f"ToolCallLimitMiddleware{inspect.signature(ToolCallLimitMiddleware.__init__)}",
            f"ToolRetryMiddleware{inspect.signature(ToolRetryMiddleware.__init__)}",
            f"PIIMiddleware{inspect.signature(PIIMiddleware.__init__)}",
        ]
    )


def fake_embeddings():
    from langchain_core.embeddings import DeterministicFakeEmbedding, FakeEmbeddings  # noqa: F401

    return "DeterministicFakeEmbedding OK, FakeEmbeddings OK"


def agent_state():
    from langchain.agents import AgentState

    return f"AgentState = {AgentState}  annotations={getattr(AgentState, '__annotations__', {})}"


def tool_runtime():
    from langchain.tools import ToolRuntime

    return f"ToolRuntime = {ToolRuntime}"


def structured_output_strategies():
    from langchain.agents.structured_output import ProviderStrategy, ToolStrategy

    return f"ToolStrategy{inspect.signature(ToolStrategy.__init__)}\nProviderStrategy present: {ProviderStrategy}"


def caches():
    out = []
    for module, name in [
        ("langchain_core.caches", "InMemoryCache"),
        ("langchain_community.cache", "SQLiteCache"),
        ("langchain_core.globals", "set_llm_cache"),
    ]:
        try:
            mod = __import__(module, fromlist=["*"])
            out.append(f"{module}.{name}: {'OK' if hasattr(mod, name) else 'MISSING'}")
        except ImportError as exc:
            out.append(f"{module}: FAILED {exc}")
    return "\n".join(out)


def callbacks():
    out = []
    for module, names in [
        ("langchain_core.callbacks", ["BaseCallbackHandler", "StdOutCallbackHandler"]),
        ("langchain_community.callbacks", ["get_openai_callback"]),
        ("langchain_core.tracers.context", ["collect_runs"]),
    ]:
        try:
            mod = __import__(module, fromlist=["*"])
            for name in names:
                out.append(f"{module}.{name}: {'OK' if hasattr(mod, name) else 'MISSING'}")
        except ImportError as exc:
            out.append(f"{module}: FAILED {exc}")
    return "\n".join(out)


def retrievers():
    out = []
    for module, names in [
        ("langchain_classic.retrievers", ["MultiQueryRetriever", "ContextualCompressionRetriever", "EnsembleRetriever"]),
        ("langchain_classic.retrievers.document_compressors", ["LLMChainExtractor", "EmbeddingsFilter", "DocumentCompressorPipeline"]),
        ("langchain_community.retrievers", ["BM25Retriever"]),
        ("langchain_classic.chains", ["create_retrieval_chain", "create_history_aware_retriever"]),
        ("langchain_classic.chains.combine_documents", ["create_stuff_documents_chain"]),
    ]:
        try:
            mod = __import__(module, fromlist=["*"])
            for name in names:
                out.append(f"{module}.{name}: {'OK' if hasattr(mod, name) else 'MISSING'}")
        except ImportError as exc:
            out.append(f"{module}: FAILED {exc}")
    return "\n".join(out)


def loaders_splitters():
    out = []
    for module, names in [
        ("langchain_community.document_loaders", ["TextLoader", "CSVLoader", "PyPDFLoader", "WebBaseLoader", "DirectoryLoader", "UnstructuredMarkdownLoader"]),
        ("langchain_text_splitters", ["CharacterTextSplitter", "RecursiveCharacterTextSplitter", "TokenTextSplitter", "MarkdownHeaderTextSplitter"]),
        ("langchain_core.documents", ["Document"]),
    ]:
        try:
            mod = __import__(module, fromlist=["*"])
            for name in names:
                out.append(f"{module}.{name}: {'OK' if hasattr(mod, name) else 'MISSING'}")
        except ImportError as exc:
            out.append(f"{module}: FAILED {exc}")
    return "\n".join(out)


def embeddings_cache():
    out = []
    for module, names in [
        ("langchain_classic.embeddings", ["CacheBackedEmbeddings"]),
        ("langchain_core.stores", ["InMemoryByteStore"]),
        ("langchain_community.storage", ["LocalFileStore"]),
    ]:
        try:
            mod = __import__(module, fromlist=["*"])
            for name in names:
                out.append(f"{module}.{name}: {'OK' if hasattr(mod, name) else 'MISSING'}")
        except ImportError as exc:
            out.append(f"{module}: FAILED {exc}")
    return "\n".join(out)


def hub_and_tools():
    out = []
    for module, names in [
        ("langchain_classic", ["hub"]),
        ("langchain_community.tools", ["WikipediaQueryRun", "DuckDuckGoSearchRun"]),
        ("langchain_community.utilities", ["WikipediaAPIWrapper", "SQLDatabase"]),
        ("langchain_classic.agents.agent_toolkits", ["SQLDatabaseToolkit"]),
        ("langchain_community.agent_toolkits", ["SQLDatabaseToolkit"]),
    ]:
        try:
            mod = __import__(module, fromlist=["*"])
            for name in names:
                out.append(f"{module}.{name}: {'OK' if hasattr(mod, name) else 'MISSING'}")
        except ImportError as exc:
            out.append(f"{module}: FAILED {exc}")
    return "\n".join(out)


if __name__ == "__main__":
    show("EnumOutputParser location", enum_parser)
    show("DatetimeOutputParser location", datetime_parser)
    show("AgentMiddleware hook signatures", middleware_hook_signatures)
    show("middleware decorators", decorator_signatures)
    show("HumanInTheLoopMiddleware", hitl_signature)
    show("other middleware", other_middleware_signatures)
    show("fake embeddings", fake_embeddings)
    show("AgentState", agent_state)
    show("ToolRuntime", tool_runtime)
    show("structured output strategies", structured_output_strategies)
    show("caches", caches)
    show("callbacks", callbacks)
    show("retrievers and chains", retrievers)
    show("loaders and splitters", loaders_splitters)
    show("embeddings cache", embeddings_cache)
    show("hub and tools", hub_and_tools)
