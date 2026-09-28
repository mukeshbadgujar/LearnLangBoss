"""Q27. Design a chatbot using LangChain.

THE PROBLEM
A single prompt forgets that the shopper already gave their name and order id.

WHAT WE ARE GOING TO SOLVE
Build the smallest chatbot: system role, message history, one model call per turn.

WHAT THIS EXAMPLE IS ABOUT
Maya asks about returning order A100, then asks "what is my name?" on the next turn.

WHAT IT SOLVES
The second answer can use the name because the full history is sent again each turn.

KEYWORDS
- Chatbot: An app whose main loop is a conversation, turn after turn.
- System message: Instructions that set the role, such as bookstore desk.
- Session: One shopper's conversation, kept separate from everyone else's.
- Message history: The list of prior turns re-sent with each new question.

Run:
    python docs/interview/examples/langchain/q27_design_a_chatbot.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import chat_model

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

model = chat_model()


def main() -> None:
    # History is the chatbot's memory: every turn appends, then the whole list goes to the model.
    history = [
        SystemMessage(
            "You are the bookstore desk. Be brief. "
            "Unread print books return within 30 days."
        )
    ]
    turns = [
        "I am Maya. Can I return order A100 The Little Prince?",
        "What is my name?",
    ]
    for user_text in turns:
        history.append(HumanMessage(user_text))
        reply = model.invoke(history)
        history.append(AIMessage(reply.content))
        print(f"shopper: {user_text}")
        print(f"desk: {reply.content}\n")


if __name__ == "__main__":
    print(__doc__)
    main()
