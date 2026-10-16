#!/usr/bin/env python3
"""Observe one gfrun execution using bounded, contiguous memory dumps."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

MAX_DUMP_BYTES = 16 * 1024 * 1024
GFRUN_TIMEOUT_SECONDS = 120


def has_success_finisher(log: str) -> bool:
    """Accept the current marker and the spelling in frozen older gfrun builds."""
    return any(re.fullmatch(
        r"(?:Success|Suaccelss) to Reach the End of Benchmark! R2 = 0", line)
        for line in log.splitlines())


def plan_dump(dump_range: str, output: Path) -> list[dict[str, object]]:
    address_text, size_text = dump_range.split(":")
    address, size = int(address_text, 0), int(size_text, 0)
    if address < 0 or size <= 0 or address + size > (1 << 64) - 1:
        raise ValueError("dump range is empty or overflows")
    chunks = []
    offset = 0
    while offset < size:
        count = min(MAX_DUMP_BYTES, size - offset)
        path = output if size <= MAX_DUMP_BYTES else Path(f"{output}.part{len(chunks)}")
        chunks.append({"address": address + offset, "size": count,
                       "path": str(path)})
        offset += count
    return chunks


def dump_specs(chunks: list[dict[str, object]]) -> list[str]:
    return [f"{chunk['address']:#x}:{chunk['size']:#x}:{chunk['path']}"
            for chunk in chunks]


def assemble_dump(chunks: list[dict[str, object]], output: Path) -> None:
    # Preflight every file before publishing the complete observation. A short,
    # missing or extra-byte chunk is a harness failure, never a passing result.
    for chunk in chunks:
        path = Path(str(chunk["path"]))
        if path.stat().st_size != chunk["size"]:
            raise ValueError(f"wrong dump size: {path}")
    if len(chunks) == 1:
        if Path(str(chunks[0]["path"])) != output:
            raise ValueError("single dump path does not match output")
        return
    temporary = output.with_name(output.name + ".assembling")
    created = False
    try:
        with temporary.open("xb") as destination:
            created = True
            for chunk in chunks:
                with Path(str(chunk["path"])).open("rb") as source:
                    shutil.copyfileobj(source, destination)
        temporary.replace(output)
    finally:
        if created:
            temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provenance", required=True, type=Path)
    args = parser.parse_args()
    identity = json.loads(args.provenance.read_text())
    observation = identity["target"]["gfrun_memory_dump"]
    output = Path(observation["output"])
    chunks = plan_dump(observation["range"], output)
    if chunks != observation["chunks"]:
        raise ValueError("memory observation plan changed")
    command = identity["commands"]["gfrun"]
    start = command.index("--dump-memory") + 1
    if command[start:command.index("-f", start)] != dump_specs(chunks):
        raise ValueError("command does not match memory observation plan")
    for chunk in chunks:
        if Path(str(chunk["path"])).exists():
            raise FileExistsError(str(chunk["path"]))
    if output.exists():
        raise FileExistsError(str(output))
    (output.parent / "gfrun.argv").write_text(shlex.join(command) + "\n")
    started = time.monotonic()
    with (output.parent / "gfrun.log").open("w") as log:
        try:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=GFRUN_TIMEOUT_SECONDS)
            exit_code = result.returncode
            timed_out = False
        except subprocess.TimeoutExpired:
            exit_code = 124
            timed_out = True
    (output.parent / "gfrun.exit").write_text(str(exit_code) + "\n")
    (output.parent / "gfrun-run.json").write_text(json.dumps({
        "elapsed_seconds": time.monotonic() - started,
        "timeout_seconds": GFRUN_TIMEOUT_SECONDS,
        "timed_out": timed_out, "exit_code": exit_code,
    }, indent=2) + "\n")
    if exit_code != 0:
        print(f"gfrun exited with {exit_code}; see {output.parent / 'gfrun.log'}",
              file=sys.stderr)
        return exit_code if timed_out else 1
    assemble_dump(chunks, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
