#!/usr/bin/env python3
"""Scan an archive payload for personal absolute paths and ephemeral paths.

NDF §5.1 requires checking text and binary content, including decompressed
trace payloads, and keeping the tool identity, command, file list, hits and
disposition. This script is dependency-free.

Exit codes: 0 = no forbidden hit; 1 = forbidden hit; 2 = usage error.
"""
import argparse
import gzip
import hashlib
import json
import os
import re
import sys

FORBIDDEN = [
    ("macos_user", re.compile(rb"/Users/[A-Za-z0-9._-]+")),
    ("linux_user", re.compile(rb"/home/[A-Za-z0-9._-]+")),
    ("windows_user", re.compile(rb"[A-Za-z]:\\\\Users\\\\[A-Za-z0-9._-]+")),
    ("owner_name", re.compile(b"black" + b"traker")),
]
EPHEMERAL = [("tmp_path", re.compile(rb"/tmp/[A-Za-z0-9._-]+"))]


def line_of(data, off):
    return data.count(b"\n", 0, off) + 1


def scan_bytes(data, label):
    hits = []
    for kind, pat in FORBIDDEN + EPHEMERAL:
        for m in pat.finditer(data):
            if m.start() > 0 and data[m.start() - 1:m.start()] == b"-":
                continue  # part of a hash-like token, not a path
            hits.append({
                "kind": kind,
                "forbidden": any(k == kind for k, _ in FORBIDDEN),
                "offset": m.start(),
                "line": line_of(data, m.start()),
                "text": m.group(0).decode("utf-8", "replace"),
                "context": data[max(0, m.start() - 60):m.end() + 60]
                .decode("utf-8", "replace").replace("\n", " "),
            })
    return hits


def scan_file(path):
    hits = []
    with open(path, "rb") as f:
        raw = f.read()
    if path.endswith(".gz"):
        try:
            data = gzip.decompress(raw)
        except OSError:
            data = raw
            hits.append({"kind": "gzip_error", "forbidden": False, "offset": 0,
                         "line": 0, "text": "", "context": "decompress failed"})
        else:
            hits += scan_bytes(data, path + "!decompressed")
    hits += scan_bytes(raw, path)
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--jsonl")
    a = ap.parse_args()
    files = []
    skip_prefix = os.path.join(a.root, "validation", "path_scan_revision")
    for dp, _, fns in os.walk(a.root):
        for fn in sorted(fns):
            p = os.path.join(dp, fn)
            if p.startswith(skip_prefix):
                continue  # the scan record itself, not payload
            files.append(p)
    files.sort()
    all_hits = []
    out = open(a.jsonl, "w") if a.jsonl else None
    for p in files:
        hits = scan_file(p)
        for h in hits:
            h["file"] = os.path.relpath(p, a.root)
            all_hits.append(h)
            if out:
                out.write(json.dumps(h, ensure_ascii=False) + "\n")
    if out:
        out.close()
    forbidden = [h for h in all_hits if h["forbidden"]]
    print(f"scanner_sha256={hashlib.sha256(open(__file__,'rb').read()).hexdigest()}")
    print(f"root={a.root} files={len(files)} hits={len(all_hits)} forbidden={len(forbidden)}")
    for h in all_hits:
        print(f"{'FORBIDDEN' if h['forbidden'] else 'WARN'} {h['file']}:{h['line']} "
              f"{h['kind']} {h['text']!r}")
    return 1 if forbidden else 0


if __name__ == "__main__":
    sys.exit(main())
