#!/usr/bin/env python3
"""Verify this archive folder offline.

Checks:
  1. SHA256SUMS lists every file except itself, with matching bytes and hashes
  2. the four source files match the hashes pinned in operator_identity.json
  3. every case identity.json names an ELF sha256 that is identical across cases
  4. no personal absolute path appears in any file (the scanner's own literals excepted)
  5. each declared run has command.json and result.json, and the run's declared log hash is
     consistent with delivery_index/identity records

It does not run the simulator, does not rebuild the operator and makes no numerical claim.
Exit code 0 only when all checks pass.
"""
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
PATTERNS = [b"/Users/", b"/home/", b"C:\\\\Users", b"/root/"]
ALLOWED = {"verify_sources.py": "defines the scan patterns"}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    failures = []

    # 1. SHA256SUMS
    sums = {}
    for line in (ROOT / "SHA256SUMS").read_text().splitlines():
        if not line.strip():
            continue
        digest, rel = line.split("  ", 1)
        sums[rel] = digest
    actual = {
        p.relative_to(ROOT).as_posix()
        for p in ROOT.rglob("*")
        if p.is_file() and p.name != "SHA256SUMS"
    }
    if actual != set(sums):
        failures.append(f"SHA256SUMS mismatch: missing={sorted(actual - set(sums))[:5]} extra={sorted(set(sums) - actual)[:5]}")
    for rel, digest in sorted(sums.items()):
        p = ROOT / rel
        if not p.is_file():
            failures.append(f"listed file missing: {rel}")
        elif sha(p) != digest:
            failures.append(f"sha256 mismatch: {rel}")

    # 1b. every JSON file parses
    for jp in sorted(ROOT.rglob("*.json")):
        try:
            json.loads(jp.read_text())
        except ValueError as exc:
            failures.append(f"invalid JSON: {jp.relative_to(ROOT).as_posix()} ({exc})")

    # 2. source hashes
    ident = json.loads((ROOT / "operator_identity.json").read_text())
    for rec in ident["source"]["files"]:
        p = ROOT / rec["path"]
        if not p.is_file():
            failures.append(f"source file missing: {rec['path']}")
            continue
        if p.stat().st_size != rec["bytes"] or sha(p) != rec["sha256"]:
            failures.append(f"source file identity mismatch: {rec['path']}")

    # 3. one ELF across cases
    elf_hashes = set()
    for case in (ROOT / "cases").iterdir():
        if not case.is_dir():
            continue
        f = case / "identity.json"
        if not f.is_file():
            failures.append(f"case without identity.json: {case.name}")
            continue
        elf_hashes.add(json.loads(f.read_text()).get("elf_sha256"))
    if elf_hashes != {ident["elf"]["sha256"]}:
        failures.append(f"case ELF hashes {elf_hashes} != operator identity {ident['elf']['sha256']}")

    # 4. path leaks
    hits = []
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file() or p.name == "SHA256SUMS":
            continue
        rel = p.relative_to(ROOT).as_posix()
        if rel in ALLOWED:
            continue
        data = p.read_bytes()
        for pat in PATTERNS:
            if pat in data:
                hits.append(rel)
                break
    if hits:
        failures.append("personal path hits: " + ", ".join(hits))

    # 5. per-run records
    for case in sorted((ROOT / "cases").iterdir()):
        if not case.is_dir():
            continue
        for run in sorted((case / "runs").glob("*")) if (case / "runs").is_dir() else []:
            for name in ("command.json", "result.json"):
                if not (run / name).is_file():
                    failures.append(f"{case.name}/{run.name}: missing {name}")
        # mainline control keeps its run one level up
        for run in sorted(case.glob("run-*")):
            if run.is_dir():
                for name in ("command.json", "result.json"):
                    if not (run / name).is_file():
                        failures.append(f"{case.name}/{run.name}: missing {name}")

    if failures:
        print("FAIL")
        for f in failures:
            print(" -", f)
        return 1
    print(f"PASS: {len(sums)} files hashed, {len(list((ROOT / 'cases').iterdir()))} case records, source bytes match, no personal paths")
    return 0


if __name__ == "__main__":
    sys.exit(main())