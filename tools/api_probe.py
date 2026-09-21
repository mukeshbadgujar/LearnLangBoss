"""Probe the installed LangChain/LangGraph versions for the exact API surface the
notebooks rely on. Run: python tools/api_probe.py

This is a development aid, not part of the curriculum.
"""

from __future__ import annotations

import inspect


def show(label: str, fn) -> None:
    try:
        print(f"\n=== {label} ===")
        print(fn())
    except Exception as exc:  # noqa: BLE001
        print(f"\n=== {label} ===\nFAILED: {type(exc).__name__}: {exc}")


def versions():
    import langchain, langchain_core, langgraph

    return "\n".join(
        [
            f"langchain      {langchain.__version__}",
            f"langchain_core {langchain_core.__version__}",
            f"langgraph      {getattr(langgraph, '__version__', 'n/a')}",
        ]
    )


def middleware_exports():
    import langchain.agents.middleware as mw

    names = sorted(n for n in dir(mw) if not n.startswith("_"))
    return "\n".join(names)


def agent_middleware_hooks():
    from langchain.agents.middleware import AgentMiddleware

    hooks = [n for n in dir(AgentMiddleware) if not n.startswith("_")]
    return "AgentMiddleware members: " + ", ".join(sorted(hooks))


def summarization_signature():
    from langchain.agents.middleware import SummarizationMiddleware

    return f"SummarizationMiddleware{inspect.signature(SummarizationMiddleware.__init__)}"


def create_agent_signature():
    from langchain.agents import create_agent

    return f"create_agent{inspect.signature(create_agent)}"


def structured_output_exports():
    import langchain.agents.structured_output as so

    return ", ".join(sorted(n for n in dir(so) if not n.startswith("_")))


def messages_exports():
    import langchain.messages as m

    wanted = ["AIMessage", "HumanMessage", "SystemMessage", "ToolMessage", "trim_messages", "RemoveMessage"]
    return "\n".join(f"{w:16} {'OK' if hasattr(m, w) else 'MISSING'}" for w in wanted)


def output_parser_locations():
    results = []
    for module, names in [
        ("langchain_core.output_parsers", ["StrOutputParser", "JsonOutputParser", "PydanticOutputParser",
                                           "CommaSeparatedListOutputParser", "EnumOutputParser", "XMLOutputParser"]),
        ("langchain.output_parsers", ["OutputFixingParser", "RetryOutputParser"]),
        ("langchain_classic.output_parsers", ["OutputFixingParser", "RetryOutputParser"]),
    ]:
        try:
            mod = __import__(module, fromlist=["*"])
            for name in names:
                results.append(f"{module}.{name:32} {'OK' if hasattr(mod, name) else 'MISSING'}")
        except ImportError as exc:
            results.append(f"{module}: IMPORT FAILED ({exc})")
    return "\n".join(results)


def example_selector_locations():
    results = []
    for module, names in [
        ("langchain_core.example_selectors",
         ["LengthBasedExampleSelector", "SemanticSimilarityExampleSelector",
          "MaxMarginalRelevanceExampleSelector", "BaseExampleSelector"]),
    ]:
        try:
            mod = __import__(module, fromlist=["*"])
            for name in names:
                results.append(f"{module}.{name:40} {'OK' if hasattr(mod, name) else 'MISSING'}")
        except ImportError as exc:
            results.append(f"{module}: IMPORT FAILED ({exc})")
    return "\n".join(results)


def prompts_exports():
    import langchain_core.prompts as p

    wanted = ["PromptTemplate", "ChatPromptTemplate", "MessagesPlaceholder", "FewShotPromptTemplate",
              "FewShotChatMessagePromptTemplate", "SystemMessagePromptTemplate", "HumanMessagePromptTemplate",
              "AIMessagePromptTemplate"]
    return "\n".join(f"{w:36} {'OK' if hasattr(p, w) else 'MISSING'}" for w in wanted)


def runnables_exports():
    import langchain_core.runnables as r

    wanted = ["RunnablePassthrough", "RunnableParallel", "RunnableLambda", "RunnableBranch",
              "RunnableConfig", "ConfigurableField", "chain"]
    return "\n".join(f"{w:24} {'OK' if hasattr(r, w) else 'MISSING'}" for w in wanted)


def langgraph_exports():
    import langgraph.graph as g
    from langgraph.checkpoint.memory import InMemorySaver

    wanted = ["StateGraph", "START", "END", "MessagesState", "add_messages"]
    lines = [f"langgraph.graph.{w:16} {'OK' if hasattr(g, w) else 'MISSING'}" for w in wanted]
    lines.append(f"InMemorySaver OK -> {InMemorySaver}")
    try:
        from langgraph.types import Command, Send, interrupt  # noqa: F401

        lines.append("langgraph.types: Command, Send, interrupt OK")
    except ImportError as exc:
        lines.append(f"langgraph.types FAILED: {exc}")
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: F401

        lines.append("SqliteSaver OK")
    except ImportError as exc:
        lines.append(f"SqliteSaver FAILED: {exc}")
    try:
        from langgraph.prebuilt import ToolNode, tools_condition  # noqa: F401

        lines.append("langgraph.prebuilt: ToolNode, tools_condition OK")
    except ImportError as exc:
        lines.append(f"langgraph.prebuilt FAILED: {exc}")
    return "\n".join(lines)


def tools_exports():
    import langchain.tools as t

    wanted = ["tool", "BaseTool", "InjectedState", "ToolRuntime"]
    return "\n".join(f"{w:20} {'OK' if hasattr(t, w) else 'MISSING'}" for w in wanted)


def vectorstore_exports():
    from langchain_core.vectorstores import InMemoryVectorStore  # noqa: F401

    return "InMemoryVectorStore OK"


if __name__ == "__main__":
    show("versions", versions)
    show("langchain.agents.middleware exports", middleware_exports)
    show("AgentMiddleware hooks", agent_middleware_hooks)
    show("SummarizationMiddleware signature", summarization_signature)
    show("create_agent signature", create_agent_signature)
    show("structured_output", structured_output_exports)
    show("langchain.messages", messages_exports)
    show("output parsers", output_parser_locations)
    show("example selectors", example_selector_locations)
    show("prompts", prompts_exports)
    show("runnables", runnables_exports)
    show("langgraph", langgraph_exports)
    show("tools", tools_exports)
    show("vector stores", vectorstore_exports)
