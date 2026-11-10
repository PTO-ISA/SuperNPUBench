#!/usr/bin/env python3
"""Verify this Release payload and (for a runtime) its pinned host libraries."""
import hashlib
import argparse
import json
from pathlib import Path
import platform
import re
import subprocess

ROOT = Path(__file__).resolve().parent

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-only", action="store_true", help="verify file bytes only; no host/runtime execution acceptance")
    args = ap.parse_args()
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    if any(p.is_symlink() for p in ROOT.rglob('*')):
        raise SystemExit('FAIL: symlinks are not permitted in a self-contained payload')
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()}
    expected = {r['path'] for r in manifest['files']} | {'manifest.json'}
    if actual != expected:
        raise SystemExit('FAIL: missing/unexpected payload files')
    for r in manifest['files']:
        p = (ROOT / r['path']).resolve()
        if not p.is_relative_to(ROOT) or p.is_symlink():
            raise SystemExit('FAIL: payload path escapes root')
        if p.stat().st_size != r['bytes'] or digest(p) != r['sha256']:
            raise SystemExit('FAIL: payload identity: ' + r['path'])
    if manifest['kind'] == 'model_runtime' and not args.data_only:
        host = json.loads((ROOT / 'host_requirements.json').read_text())
        if platform.system() != 'Linux' or platform.machine() != 'x86_64':
            raise SystemExit('FAIL: this runtime requires Linux x86_64')
        libraries = subprocess.check_output([str(ROOT / 'lib/ld-linux-x86-64.so.2'),
            '--library-path', str(ROOT / 'lib'), '--list', str(ROOT / 'bin/gfsim')], text=True)
        if 'not found' in libraries:
            raise SystemExit('FAIL: missing host library: ' + libraries)
        resolved = {}
        for line in libraries.splitlines():
            m = re.search(r'^\s*(\S+)\s+=>\s+(/\S+)\s+\(', line)
            if m:
                resolved[m[1]] = Path(m[2])
            else:
                m = re.search(r'^\s*(/\S+)\s+\(', line)
                if m:
                    resolved[Path(m[1]).name] = Path(m[1])
        for r in host['libraries']:
            # glibc --list prints the ELF's original PT_INTERP name for itself.
            p = (ROOT / 'lib/ld-linux-x86-64.so.2' if r['soname'] == 'ld-linux-x86-64.so.2'
                 else resolved.get(r['soname']))
            if not p or not p.resolve().is_relative_to(ROOT) or digest(p) != r['sha256']:
                raise SystemExit('FAIL: loader did not resolve the shipped library: ' + r['soname'])
    print(json.dumps({'status': 'pass', 'kind': manifest['kind'], 'files': len(expected),
                      'scope': 'payload bytes only' if args.data_only else 'payload bytes and declared host dependencies; no numerical acceptance'}))

if __name__ == '__main__':
    main()
