"""Q14. What embedding models can LangChain use?

THE PROBLEM
To search policy by meaning, text must become vectors. Mixing two embedding
models on the same store breaks search.

WHAT WE ARE GOING TO SOLVE
Embed one policy sentence with Groq's OpenAI-compatible embeddings API and
print vector length (dimension count).

WHAT THIS EXAMPLE IS ABOUT
We embed the 30-day return policy line used for print books at the bookstore.

WHAT IT SOLVES
A real embedding vector from one model. Index and query must use that same model.

KEYWORDS
- Embedding model: Turns text into a list of floats (a vector).
- Dimension: Length of that vector; affects storage and speed.
- Same-model rule: Index and query embeddings must come from one model.
- Reindexing: Rebuild the store if you change embedding models.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import embedding_in_use, embedding_model

embeddings = embedding_model()


def main() -> None:
    text = "Unread print books can be returned within 30 days of delivery."
    vector = embeddings.embed_query(text)
    print("embedding provider from .env:", embedding_in_use())
    print("dimensions:", len(vector))
    print("first 5 floats:", vector[:5])


if __name__ == "__main__":
    print(__doc__)
    main()
