"""Q57. What are the major text splitting strategy types?

THE PROBLEM
One splitter does not fit every bookstore document: FAQ pages, markdown manuals, and code samples differ.

WHAT WE ARE GOING TO SOLVE
Compare character-based strategies side by side on the same policy text.

WHAT THIS EXAMPLE IS ABOUT
Recursive vs plain CharacterTextSplitter on returns and shipping paragraphs.

WHAT IT SOLVES
You can name character, token, structure-aware, and semantic strategies and show two character variants.

KEYWORDS
- Character-based: Split by characters or recursive separator lists.
- Token-based: Split by tokenizer limits to protect context windows.
- Structure-aware: Split on markdown headers or code syntax boundaries.
- Semantic chunking: Split where embedding distance shifts between sentences.

Run:
    python docs/interview/examples/langchain/q57_text_splitting_strategies.py
"""

from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

TEXT = (
    "## Returns\n\n"
    "Unread print books may be returned within 30 days of delivery.\n\n"
    "Digital books are not refundable after download.\n\n"
    "## Shipping\n\n"
    "Standard shipping takes 5 business days. Express shipping takes 2 days.\n"
)


def main() -> None:
    strategies = {
        "recursive_character": RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10),
        "plain_character": CharacterTextSplitter(separator="\n\n", chunk_size=100, chunk_overlap=0),
    }
    for name, splitter in strategies.items():
        chunks = splitter.split_text(TEXT)
        print(f"\n{name}: {len(chunks)} chunks")
        for c in chunks:
            print(" -", c.replace("\n", " / "))

    print(
        "\nAlso know: token-based (tiktoken), structure-aware (MarkdownHeader), "
        "semantic (embedding distance)  -  not shown here."
    )


if __name__ == "__main__":
    print(__doc__)
    main()
