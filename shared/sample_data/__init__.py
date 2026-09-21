"""Sample corpus used across the curriculum.

Files that ship as text live next to this module. Binary-ish artifacts (a PDF and
a SQLite database) are generated on demand so the repository stays diff-friendly
and nothing has to be downloaded.
"""

from shared.sample_data.builders import (  # noqa: F401
    DATA_DIR,
    GENERATED_DIR,
    ensure_all,
    ensure_pdf,
    ensure_sqlite_db,
    list_files,
)

__all__ = [
    "DATA_DIR",
    "GENERATED_DIR",
    "ensure_all",
    "ensure_pdf",
    "ensure_sqlite_db",
    "list_files",
]
