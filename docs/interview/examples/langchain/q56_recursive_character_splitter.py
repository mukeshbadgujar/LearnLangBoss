"""Q56. What is RecursiveCharacterTextSplitter and why is it the industry standard?

THE PROBLEM
Fixed-length character cuts slice bookstore policy mid-sentence and hurt retrieval.

WHAT WE ARE GOING TO SOLVE
Split policy text with RecursiveCharacterTextSplitter's separator hierarchy.

WHAT THIS EXAMPLE IS ABOUT
A multi-paragraph returns-and-shipping policy is chunked on paragraphs before characters.

WHAT IT SOLVES
Chunks stay closer to semantic units than a naive CharacterTextSplitter cut.

KEYWORDS
- RecursiveCharacterTextSplitter: Tries separators from large to small (\\n\\n, \\n, space, "").
- Separator hierarchy: Prefer keeping paragraphs and sentences together.
- Chunk size: Target maximum characters (or tokens) per piece.
- Chunk overlap: Repeated border text so meaning is not lost at cuts.

Run:
    python docs/interview/examples/langchain/q56_recursive_character_splitter.py
"""

from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

POLICY = """
Unread print books may be returned within 30 days of delivery.

Digital books are not refundable after download.

Standard shipping takes 5 business days. Express shipping takes 2 days.

Order examples: A100 The Little Prince shipped $18. A200 Clean Code processing $42. A300 Dune delivered $16.
""".strip()


def main() -> None:
    recursive = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=20)
    naive = CharacterTextSplitter(separator="", chunk_size=120, chunk_overlap=20)

    r_chunks = recursive.split_text(POLICY)
    n_chunks = naive.split_text(POLICY)

    print(f"recursive chunks={len(r_chunks)}")
    for i, c in enumerate(r_chunks, 1):
        print(f"  R{i}: {c!r}")
    print(f"character chunks={len(n_chunks)}")
    for i, c in enumerate(n_chunks, 1):
        print(f"  C{i}: {c!r}")


if __name__ == "__main__":
    print(__doc__)
    main()
