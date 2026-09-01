#!/usr/bin/env python3
"""Create deterministic ELF hashes, sizes, disassemblies and histograms."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--objdump", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records = []
    args.out.mkdir(parents=True, exist_ok=True)
    for elf in sorted(args.root.rglob("*.elf")):
        proc = subprocess.run([args.objdump, "-dl", str(elf)], text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        diss = args.out / f"{elf.name}.diss"
        diss.write_text(proc.stdout)
        histogram = Counter()
        for line in proc.stdout.splitlines():
            match = re.match(r"^\s*[0-9a-f]+:\s+(?:[0-9a-f]+\s+)+([^\s]+)", line)
            if match:
                histogram[match.group(1)] += 1
        records.append({"elf": str(elf), "sha256": sha256(elf),
                        "size_bytes": elf.stat().st_size, "diss": str(diss),
                        "objdump_rc": proc.returncode,
                        "instruction_histogram": dict(histogram.most_common())})
    output = args.out / "static_report.json"
    output.write_text(json.dumps({"artifacts": records}, indent=2) + "\n")
    print(output)
    return 0 if all(r["objdump_rc"] == 0 for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
