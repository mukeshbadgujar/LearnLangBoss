"""Third probe: resolve the import paths that failed indirectly in probe 2."""

from __future__ import annotations


def check(label: str, code: str) -> None:
    try:
        exec(code, {})  # noqa: S102 - deliberate probe
        print(f"OK       {label}")
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED   {label}\n           {type(exc).__name__}: {str(exc)[:160]}")


checks = [
    ("create_retrieval_chain", "from langchain_classic.chains.retrieval import create_retrieval_chain"),
    ("create_history_aware_retriever", "from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever"),
    ("create_stuff_documents_chain", "from langchain_classic.chains.combine_documents import create_stuff_documents_chain"),
    ("hub (classic)", "from langchain_classic.hub import pull"),
    ("LocalFileStore core", "from langchain_core.stores import LocalFileStore"),
    ("LocalFileStore classic", "from langchain_classic.storage import LocalFileStore"),
    ("LocalFileStore community file_system", "from langchain_community.storage.file_system import LocalFileStore"),
    ("CacheBackedEmbeddings", "from langchain_classic.embeddings import CacheBackedEmbeddings"),
    ("InMemoryByteStore", "from langchain_core.stores import InMemoryByteStore"),
    ("MultiQueryRetriever direct", "from langchain_classic.retrievers.multi_query import MultiQueryRetriever"),
    ("EnsembleRetriever direct", "from langchain_classic.retrievers.ensemble import EnsembleRetriever"),
    ("ContextualCompression direct", "from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever"),
    ("BM25Retriever", "from langchain_community.retrievers import BM25Retriever"),
    ("EnumOutputParser", "from langchain_classic.output_parsers import EnumOutputParser"),
    ("OutputFixingParser", "from langchain_classic.output_parsers import OutputFixingParser"),
    ("RetryOutputParser", "from langchain_classic.output_parsers import RetryOutputParser"),
    ("DatetimeOutputParser", "from langchain_classic.output_parsers import DatetimeOutputParser"),
    ("SQLDatabase", "from langchain_community.utilities import SQLDatabase"),
    ("SQLDatabaseToolkit", "from langchain_community.agent_toolkits import SQLDatabaseToolkit"),
    ("Wikipedia tool", "from langchain_community.tools import WikipediaQueryRun\nfrom langchain_community.utilities import WikipediaAPIWrapper"),
    ("DuckDuckGo tool", "from langchain_community.tools import DuckDuckGoSearchRun"),
    ("InMemoryCache", "from langchain_core.caches import InMemoryCache\nfrom langchain_core.globals import set_llm_cache"),
    ("SQLiteCache", "from langchain_community.cache import SQLiteCache"),
    ("get_openai_callback", "from langchain_community.callbacks import get_openai_callback"),
    ("UsageMetadataCallbackHandler", "from langchain_core.callbacks import UsageMetadataCallbackHandler"),
    ("get_usage_metadata_callback", "from langchain_core.callbacks import get_usage_metadata_callback"),
    ("langgraph Send/Command", "from langgraph.types import Command, Send, interrupt, RetryPolicy"),
    ("langgraph StateSnapshot", "from langgraph.graph.state import CompiledStateGraph"),
    ("langgraph store", "from langgraph.store.memory import InMemoryStore"),
    ("langgraph SqliteSaver", "from langgraph.checkpoint.sqlite import SqliteSaver"),
    ("langgraph AsyncSqliteSaver", "from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver"),
    ("ToolNode/tools_condition", "from langgraph.prebuilt import ToolNode, tools_condition"),
    ("InjectedState", "from langgraph.prebuilt import InjectedState"),
    ("langchain tool decorator", "from langchain.tools import tool, ToolRuntime"),
    ("init_chat_model", "from langchain.chat_models import init_chat_model"),
    ("init_embeddings", "from langchain.embeddings import init_embeddings"),
    ("AgentState", "from langchain.agents import AgentState, create_agent"),
    ("ToolStrategy", "from langchain.agents.structured_output import ToolStrategy, ProviderStrategy"),
    ("MarkdownHeaderTextSplitter", "from langchain_text_splitters import MarkdownHeaderTextSplitter"),
    ("SemanticChunker", "from langchain_experimental.text_splitter import SemanticChunker"),
    ("langsmith Client", "from langsmith import Client"),
    ("langsmith evaluate", "from langsmith import evaluate"),
    ("langsmith traceable", "from langsmith import traceable"),
]

for label, code in checks:
    check(label, code)
