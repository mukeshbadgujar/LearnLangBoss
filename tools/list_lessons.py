"""Print every lesson's folder, filename, checklist id and title.

Used to keep CHECKLIST.md and references/official-sources.md in sync with the
notebooks rather than hand-maintaining three lists.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "tools" / "notebook_sources"

rows = []
for path in sorted(SOURCES.rglob("*.py")):
    text = path.read_text(encoding="utf-8")
    checklist = re.search(r"\*\*Checklist ID\*\*\s*\|\s*`([\w_]+)`", text)
    # A trailing letter marks an extension lesson (17a sits between 17 and 18).
    title = re.search(r"^# # (\d+[a-z]?) [-–] (.+)$", text, re.MULTILINE)
    level = re.search(r"\*\*Level\*\*\s*\|\s*([^|\n]+)\|", text)
    rows.append({
        "folder": path.parent.name,
        "stem": path.stem,
        "checklist_id": checklist.group(1) if checklist else "MISSING",
        "number": title.group(1) if title else "??",
        "title": title.group(2).strip() if title else "?",
        "level": level.group(1).strip() if level else "?",
    })

missing = [r for r in rows if r["checklist_id"] == "MISSING"]
mismatched = [r for r in rows if r["checklist_id"] not in ("MISSING",) and r["checklist_id"] != r["stem"]]

if "--check" in sys.argv:
    for row in missing:
        print(f"MISSING checklist id: {row['folder']}/{row['stem']}")
    for row in mismatched:
        print(f"MISMATCH: {row['folder']}/{row['stem']} declares '{row['checklist_id']}'")
    print(f"\n{len(rows)} lessons, {len(missing)} missing ids, {len(mismatched)} mismatched")
    sys.exit(1 if (missing or mismatched) else 0)

current = None
for row in rows:
    if row["folder"] != current:
        current = row["folder"]
        print(f"\n## {current}")
    print(f"{row['number']:>3}  {row['level']:<26} {row['checklist_id']:<46} {row['title']}")

print(f"\n{len(rows)} lessons")
