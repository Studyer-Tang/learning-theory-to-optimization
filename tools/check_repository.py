"""Offline publication checks: files, local Markdown links, delimiters, privacy."""
from pathlib import Path
import re
import sys
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {".git", ".venv", "__pycache__"}
errors = []
files = [p for p in ROOT.rglob("*") if p.is_file()
         and not any(part in EXCLUDED for part in p.relative_to(ROOT).parts)]
markdown = [p for p in files if p.suffix == ".md"]

for path in markdown:
    content = path.read_text(encoding="utf-8")
    relative = path.relative_to(ROOT)
    if content.count("```") % 2:
        errors.append(f"{relative}: unmatched fenced code block")
    if re.search(r"/(?:Users|home)/[A-Za-z0-9_.-]+/", content):
        errors.append(f"{relative}: identifying local filesystem path")
    if re.search(r"(?im)^(?:user|assistant|system):", content):
        errors.append(f"{relative}: possible conversation transcript")
    stripped = re.sub(r"```.*?```", "", content, flags=re.S)
    if len(re.findall(r"(?<!\\)\$\$", stripped)) % 2:
        errors.append(f"{relative}: unmatched display-math delimiter")
    inline = re.sub(r"\$\$.*?\$\$", "", stripped, flags=re.S)
    if len(re.findall(r"(?<!\\)\$", inline)) % 2:
        errors.append(f"{relative}: unmatched inline-math delimiter")
    for target in re.findall(r"!?\[[^\]]*\]\(([^\s)]+)\)", content):
        if target.startswith(("https://", "http://", "mailto:", "#")):
            continue
        local = unquote(target.split("#")[0])
        if local and not (path.parent / local).exists():
            errors.append(f"{relative}: missing local target {local}")

pdfs = [p.relative_to(ROOT).as_posix() for p in files if p.suffix.lower() == ".pdf"]
if pdfs != ["reference/lecture_note2.pdf"]:
    errors.append(f"Unexpected PDF set: {pdfs}")
if not (ROOT / "reference/lecture_note2.pdf").read_bytes().startswith(b"%PDF-"):
    errors.append("Original source is not a PDF")

for name in ["README.md", "LICENSE", "SOURCES.md", "labs/RESULTS.md", "assets/overview.png"]:
    if not (ROOT / name).exists():
        errors.append(f"Missing publication file: {name}")

if errors:
    print("\n".join(errors))
    sys.exit(1)
print(f"PASS: {len(markdown)} Markdown files; local links, math delimiters, and publication scan.")
