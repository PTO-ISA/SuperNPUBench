#!/usr/bin/env python3
"""Fail closed on payload mutation, symlinks, traversal and unexpected files."""
from pathlib import Path
import hashlib,json
r=Path(__file__).resolve().parent
items=json.loads((r/'files.json').read_text())['files']
expected={x['path'] for x in items}|{'files.json'}
if len(expected)!=len(items)+1:raise SystemExit('FAIL: duplicate/self inventory entry')
actual=set()
for p in r.rglob('*'):
 if p.is_symlink():raise SystemExit('FAIL: symlink '+str(p.relative_to(r)))
 if p.is_file():actual.add(p.relative_to(r).as_posix())
if actual!=expected:raise SystemExit('FAIL: missing/unexpected payload files '+str(sorted(actual^expected)))
for x in items:
 p=r/x['path']
 if not p.resolve().is_relative_to(r) or not p.is_file() or p.stat().st_size!=x['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=x['sha256']:raise SystemExit('FAIL: '+x['path'])
print('Payload exact file postimages verified; no simulator execution')
