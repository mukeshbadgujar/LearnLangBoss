"""Q1. What is LangChain?

THE PROBLEM
The bookstore support bot calls one vendor's API directly. Moving from Groq
to OpenAI, OpenRouter, or Anthropic means rewriting every call that answers
a return question.

WHAT WE ARE GOING TO SOLVE
Keep one call, model.invoke. Which company answers is chosen in the repo
.env file, not in this example.

WHAT THIS EXAMPLE IS ABOUT
A shopper asks how long they have to return an unread print book. This file
asks whatever chat model .env selected, then prints that provider's name.

WHAT IT SOLVES
The desk code stays the same when you switch providers. Change one line in
.env and run this file again:

    LLM_PROVIDER=groq
    LLM_PROVIDER=openrouter
    LLM_PROVIDER=openai
    LLM_PROVIDER=anthropic

The model name for that provider is GROQ_MODEL, OPENROUTER_MODEL, OPENAI_MODEL,
or ANTHROPIC_MODEL. The invoke line below does not change.

KEYWORDS
- LangChain: A framework that wraps models, prompts, tools, and data behind the same calls.
- LLM: A large language model that reads text and writes text.
- Provider: The company that runs the model: Groq, OpenAI, OpenRouter, or Anthropic.
- LLM_PROVIDER: The .env setting that picks which of those four this file uses.
- Provider abstraction: App code calls invoke. It does not build the vendor's HTTP request.
- invoke: The standard call that sends input and waits for one result.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import chat_model, provider_in_use

# chat_model() reads LLM_PROVIDER from .env. See docs/interview/examples/model.py.
model = chat_model()


def main() -> None:
    print("Using:", provider_in_use())
    question = "How many days do I have to return an unread print book?"
    # .content is the plain text of the chat message. Same call for every provider.
    reply = model.invoke(question).content
    print(reply)


if __name__ == "__main__":
    # print(__doc__)
    main()
