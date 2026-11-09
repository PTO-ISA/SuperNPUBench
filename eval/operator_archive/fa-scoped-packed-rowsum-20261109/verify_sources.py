#!/usr/bin/env python3
"""Verify archived source bytes only; fail on missing or changed files."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
records = json.loads((root / "checksums.json").read_text())
for name, digest in records.items():
    path = root / name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise SystemExit(f"FAIL: {name}")
print(f"PASS: {len(records)} archived files; no functional claim")
