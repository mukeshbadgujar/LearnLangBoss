"""Smoke-test the generated sample assets. Run: python tools/verify_assets.py"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shared.sample_data import ensure_all  # noqa: E402

paths = ensure_all(force=True)

con = sqlite3.connect(paths["sqlite"])
tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
print("tables         :", tables)
for table in tables:
    print(f"  {table:15} rows = {con.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]}")
print("sample join    :", con.execute(
    "SELECT e.name, lr.leave_type, lr.days FROM leave_requests lr "
    "JOIN employees e ON e.employee_id = lr.employee_id LIMIT 3"
).fetchall())
con.close()

try:
    from pypdf import PdfReader

    reader = PdfReader(paths["pdf"])
    print("pdf pages      :", len(reader.pages))
    text = reader.pages[0].extract_text() or ""
    print("pdf page1 text :", repr(text[:180]))
    assert "Security Policy" in text, "PDF text extraction produced unexpected content"
    print("PDF extraction OK")
except ImportError:
    print("pypdf not installed - skipping PDF extraction check")
