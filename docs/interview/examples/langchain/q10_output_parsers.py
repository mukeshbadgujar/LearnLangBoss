"""Q10. What are output parsers?

THE PROBLEM
The model returns free text, but the returns desk needs a JSON object with
order_id, eligible, and reason for order A300.

WHAT WE ARE GOING TO SOLVE
Ask for JSON and parse it with JsonOutputParser into a Python dict.

WHAT THIS EXAMPLE IS ABOUT
Shopper says A300 Dune is unread and delivered. We parse eligibility against
the 30-day unread print-book return policy.

WHAT IT SOLVES
Raw model text becomes a dict the ticket system can store.

KEYWORDS
- Output parser: Converts model text into structured data.
- JsonOutputParser: Parses a JSON string into a Python dict/list.
- Structured output: Constraining replies to a schema apps can trust.
- Schema: The expected fields (here order_id, eligible, reason).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import chat_model

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

model = chat_model()


def main() -> None:
    parser = JsonOutputParser()
    prompt = ChatPromptTemplate.from_template(
        "Return ONLY JSON with keys order_id (str), eligible (bool), reason (str).\n"
        "Policy: unread print books can be returned within 30 days. "
        "Digital books are not refundable after download.\n"
        "Case: order A300 Dune paperback, delivered, unread.\n"
        "{format_instructions}"
    )
    chain = prompt | model | parser
    data = chain.invoke({"format_instructions": parser.get_format_instructions()})
    print(data)
    print("eligible is bool:", isinstance(data.get("eligible"), bool))


if __name__ == "__main__":
    print(__doc__)
    main()
