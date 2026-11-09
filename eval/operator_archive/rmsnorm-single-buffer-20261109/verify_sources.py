#!/usr/bin/env python3
"""Check the archived file identities without a compiler or simulator."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
records = json.loads((root / 'checksums.json').read_text())
actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
if actual != set(records) | {'checksums.json'}:
    raise SystemExit('FAIL: missing or unexpected archive files')
for name, digest in records.items():
    path = root / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise SystemExit(f'FAIL: {name}')
print(f'PASS: {len(records)} archived files; no numerical correctness claim')
