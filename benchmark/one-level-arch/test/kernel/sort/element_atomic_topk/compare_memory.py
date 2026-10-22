#!/usr/bin/env python3
"""Derive dump bounds and compare model memory with independent goldens."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


def load_symbols(path: Path) -> dict[str, tuple[int, int]]:
    symbols: dict[str, tuple[int, int]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 6:
            continue
        try:
            address = int(fields[0], 16)
            size = int(fields[-2], 16)
        except ValueError:
            continue
        symbols[fields[-1]] = (address, size)
    return symbols


def matches_segment(item: dict[str, object], actual: bytes, expected: bytes) -> bool:
    comparison = item.get("comparison", "exact")
    if comparison == "exact":
        return actual == expected
    if comparison == "relaxed_atomic_pair_7_8":
        if len(actual) != len(expected) or len(actual) < 8:
            return False
        old0, old1 = struct.unpack_from("<II", actual)
        return {old0, old1} == {7, 8} and actual[8:] == expected[8:]
    raise ValueError(f"unknown comparison mode: {comparison}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", required=True, type=Path)
    parser.add_argument("--golden", required=True, type=Path)
    parser.add_argument("--print-range", action="store_true")
    parser.add_argument("--dump", type=Path)
    parser.add_argument("--dump-base", type=lambda value: int(value, 0))
    args = parser.parse_args()

    manifest = json.loads((args.golden / "manifest.json").read_text())
    symbols = load_symbols(args.symbols)
    segments = manifest["segments"]
    missing = [item["symbol"] for item in segments if item["symbol"] not in symbols]
    if missing:
        raise SystemExit(f"missing ELF symbols: {', '.join(missing)}")

    low = min(symbols[item["symbol"]][0] for item in segments)
    high = max(
        symbols[item["symbol"]][0] + symbols[item["symbol"]][1]
        for item in segments
    )
    if args.print_range:
        print(f"0x{low:x}:0x{high - low:x}")
        return 0

    if args.dump is None or args.dump_base is None:
        parser.error("comparison requires --dump and --dump-base")
    memory = args.dump.read_bytes()
    failures = 0
    for item in segments:
        symbol = item["symbol"]
        address, size = symbols[symbol]
        expected = (args.golden / item["file"]).read_bytes()
        if len(expected) != size:
            print(
                f"{symbol}: golden size {len(expected)} does not match ELF size {size}"
            )
            failures += 1
            continue
        offset = address - args.dump_base
        if offset < 0:
            print(
                f"{symbol}: address 0x{address:x} is below dump base "
                f"0x{args.dump_base:x}"
            )
            failures += 1
            continue
        if offset + size > len(memory):
            print(
                f"{symbol}: dump is too short for offset {offset} and "
                f"size {size} (dump length {len(memory)})"
            )
            failures += 1
            continue
        actual = memory[offset : offset + size]
        try:
            matched = matches_segment(item, actual, expected)
        except ValueError as error:
            print(f"{symbol}: {error}")
            failures += 1
            continue
        if not matched:
            first = next(
                (index for index, pair in enumerate(zip(actual, expected)) if pair[0] != pair[1]),
                min(len(actual), len(expected)),
            )
            print(f"{symbol}: mismatch at byte {first}")
            failures += 1
        else:
            print(f"{symbol}: PASS ({size} bytes)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
