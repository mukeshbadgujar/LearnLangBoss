"""Generate the binary sample assets (PDF + SQLite) with zero extra dependencies.

The PDF is written using raw PDF syntax rather than a library so that
``PyPDFLoader`` has something realistic to parse even on a machine with no
internet access and no reportlab installed.
"""

from __future__ import annotations

import csv
import sqlite3
import textwrap
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
GENERATED_DIR = DATA_DIR / "generated"

PDF_NAME = "northwind_security_policy.pdf"
DB_NAME = "northwind.db"

_PDF_PAGES: list[tuple[str, list[str]]] = [
    (
        "Northwind Analytics - Information Security Policy",
        [
            "Document owner: Office of the CISO",
            "Version 4.2, effective 1 February 2026",
            "",
            "1. Scope",
            "",
            "This policy applies to all employees, contractors and vendors who",
            "access Northwind systems, customer data or corporate networks.",
            "",
            "2. Access Control",
            "",
            "Access follows least privilege. Production access requires hardware",
            "2FA, completed security training and written manager approval.",
            "Access is reviewed every quarter and revoked automatically after",
            "45 days without use.",
            "",
            "Shared accounts are prohibited. Service accounts must be owned by a",
            "named team and rotated every 180 days.",
        ],
    ),
    (
        "3. Data Classification",
        [
            "Northwind classifies data into four tiers:",
            "",
            "  Public       - marketing material, published documentation",
            "  Internal     - roadmaps, internal metrics, design documents",
            "  Confidential - customer data, contracts, source code",
            "  Restricted   - credentials, encryption keys, audit evidence",
            "",
            "Confidential and Restricted data may never be copied to personal",
            "devices, personal cloud storage or unapproved AI tools.",
            "",
            "4. Incident Response",
            "",
            "Suspected incidents must be reported to security@northwind.example",
            "within 1 hour of discovery. The on-call security engineer",
            "acknowledges within 30 minutes and declares severity within 2 hours.",
            "",
            "Severity 1 incidents trigger customer notification within 72 hours,",
            "as required by contractual and regulatory obligations.",
        ],
    ),
    (
        "5. Vendor and AI Tool Usage",
        [
            "Any third-party service that processes Confidential data requires a",
            "security review and a signed data processing agreement before use.",
            "",
            "Large language model providers are treated as sub-processors.",
            "Employees may use approved LLM providers for Internal data. Sending",
            "Confidential or Restricted data to an LLM requires an approved",
            "zero-retention agreement and CISO sign-off.",
            "",
            "6. Device Security",
            "",
            "Full-disk encryption is mandatory. Screens lock after 5 minutes.",
            "Lost or stolen devices must be reported within 4 hours so the device",
            "can be wiped remotely. Replacement hardware is issued within 3",
            "business days.",
            "",
            "7. Exceptions",
            "",
            "Exceptions are time-bound, documented in the risk register and",
            "expire after 90 days unless renewed by the CISO.",
        ],
    ),
]


def _escape_pdf_text(value: str) -> str:
    return value.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def _page_content_stream(title: str, body_lines: list[str]) -> bytes:
    parts = ["BT", "/F1 14 Tf", "72 740 Td", "18 TL", f"({_escape_pdf_text(title)}) Tj", "T*", "/F1 10 Tf", "14 TL"]
    for line in body_lines:
        parts.append(f"({_escape_pdf_text(line)}) Tj" if line else "() Tj")
        parts.append("T*")
    parts.append("ET")
    return "\n".join(parts).encode("latin-1", errors="replace")


def _build_pdf_bytes() -> bytes:
    """Assemble a minimal but spec-valid multi-page PDF."""
    page_count = len(_PDF_PAGES)
    font_obj_num = 3 + page_count * 2  # catalog(1), pages(2), then page+content pairs

    objects: dict[int, bytes] = {}
    kids = " ".join(f"{3 + i * 2} 0 R" for i in range(page_count))
    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects[2] = f"<< /Type /Pages /Kids [{kids}] /Count {page_count} >>".encode("latin-1")

    for index, (title, body) in enumerate(_PDF_PAGES):
        page_num = 3 + index * 2
        content_num = page_num + 1
        objects[page_num] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_obj_num} 0 R >> >> "
            f"/Contents {content_num} 0 R >>"
        ).encode("latin-1")
        stream = _page_content_stream(title, body)
        objects[content_num] = b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"

    objects[font_obj_num] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: dict[int, int] = {}
    for num in sorted(objects):
        offsets[num] = len(out)
        out += f"{num} 0 obj\n".encode("latin-1") + objects[num] + b"\nendobj\n"

    xref_offset = len(out)
    max_num = max(objects)
    out += f"xref\n0 {max_num + 1}\n".encode("latin-1")
    out += b"0000000000 65535 f \n"
    for num in range(1, max_num + 1):
        out += f"{offsets[num]:010d} 00000 n \n".encode("latin-1")
    out += f"trailer\n<< /Size {max_num + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("latin-1")
    return bytes(out)


def ensure_pdf(force: bool = False) -> Path:
    """Write (once) and return the path to the sample security-policy PDF."""
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    path = GENERATED_DIR / PDF_NAME
    if force or not path.exists():
        path.write_bytes(_build_pdf_bytes())
    return path


def ensure_sqlite_db(force: bool = False) -> Path:
    """Create the SQLite database used by the SQL toolkit and evaluation lessons."""
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    path = GENERATED_DIR / DB_NAME
    if path.exists() and not force:
        return path
    if path.exists():
        path.unlink()

    with sqlite3.connect(path) as conn:
        conn.executescript(
            textwrap.dedent(
                """
                CREATE TABLE employees (
                    employee_id          TEXT PRIMARY KEY,
                    name                 TEXT NOT NULL,
                    department           TEXT NOT NULL,
                    level                TEXT NOT NULL,
                    location             TEXT NOT NULL,
                    manager              TEXT,
                    joined_on            TEXT NOT NULL,
                    annual_leave_balance INTEGER NOT NULL,
                    sick_leave_balance   INTEGER NOT NULL
                );

                CREATE TABLE tickets (
                    ticket_id        TEXT PRIMARY KEY,
                    customer         TEXT NOT NULL,
                    plan             TEXT NOT NULL,
                    category         TEXT NOT NULL,
                    priority         TEXT NOT NULL,
                    status           TEXT NOT NULL,
                    opened_on        TEXT NOT NULL,
                    summary          TEXT NOT NULL,
                    resolution_hours INTEGER
                );

                CREATE TABLE leave_requests (
                    request_id  TEXT PRIMARY KEY,
                    employee_id TEXT NOT NULL REFERENCES employees(employee_id),
                    leave_type  TEXT NOT NULL,
                    start_date  TEXT NOT NULL,
                    days        INTEGER NOT NULL,
                    status      TEXT NOT NULL
                );
                """
            )
        )

        with open(DATA_DIR / "employees.csv", newline="", encoding="utf-8") as handle:
            rows = [tuple(row.values()) for row in csv.DictReader(handle)]
        conn.executemany("INSERT INTO employees VALUES (?,?,?,?,?,?,?,?,?)", rows)

        with open(DATA_DIR / "support_tickets.csv", newline="", encoding="utf-8") as handle:
            ticket_rows = []
            for row in csv.DictReader(handle):
                hours = row["resolution_hours"]
                ticket_rows.append(
                    (
                        row["ticket_id"],
                        row["customer"],
                        row["plan"],
                        row["category"],
                        row["priority"],
                        row["status"],
                        row["opened_on"],
                        row["summary"],
                        int(hours) if hours else None,
                    )
                )
        conn.executemany("INSERT INTO tickets VALUES (?,?,?,?,?,?,?,?,?)", ticket_rows)

        conn.executemany(
            "INSERT INTO leave_requests VALUES (?,?,?,?,?,?)",
            [
                ("LR-001", "E-101", "annual", "2026-02-10", 5, "approved"),
                ("LR-002", "E-103", "sick", "2026-01-22", 2, "approved"),
                ("LR-003", "E-104", "annual", "2026-03-02", 10, "pending"),
                ("LR-004", "E-106", "casual", "2026-01-30", 1, "approved"),
                ("LR-005", "E-108", "annual", "2026-02-16", 3, "rejected"),
                ("LR-006", "E-102", "annual", "2026-04-06", 7, "pending"),
                ("LR-007", "E-109", "parental", "2026-03-16", 15, "approved"),
            ],
        )
        conn.commit()

    return path


def list_files() -> list[Path]:
    """All sample files, generating the binary ones if needed."""
    ensure_all()
    return sorted(p for p in DATA_DIR.rglob("*") if p.is_file() and p.suffix not in {".py", ".pyc"})


def ensure_all(force: bool = False) -> dict[str, Path]:
    """Create every generated asset and return a name -> path map."""
    return {"pdf": ensure_pdf(force), "sqlite": ensure_sqlite_db(force)}


if __name__ == "__main__":
    created = ensure_all(force=True)
    for name, path in created.items():
        print(f"{name:8} -> {path}  ({path.stat().st_size:,} bytes)")
