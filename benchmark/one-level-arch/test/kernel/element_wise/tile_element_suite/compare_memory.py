#!/usr/bin/env python3
"""Compare exact segments and order-independent atomic old values."""

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
            symbols[fields[-1]] = (int(fields[0], 16), int(fields[-2], 16))
        except ValueError:
            pass
    return symbols


def u32(data: bytes) -> list[int]:
    if len(data) % 4:
        raise ValueError("u32 segment has a non-multiple-of-four size")
    return list(struct.unpack(f"<{len(data) // 4}I", data))


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

    def segment_bytes(symbol: str) -> bytes:
        address, size = symbols[symbol]
        offset = address - args.dump_base
        if offset < 0 or offset + size > len(memory):
            raise ValueError(f"dump does not cover {symbol}")
        return memory[offset : offset + size]

    failures = 0
    for item in segments:
        if item["kind"] != "exact":
            continue
        actual = segment_bytes(item["symbol"])
        expected = (args.golden / item["file"]).read_bytes()
        if actual != expected:
            first = next(
                (i for i, pair in enumerate(zip(actual, expected)) if pair[0] != pair[1]),
                min(len(actual), len(expected)),
            )
            print(f"{item['symbol']}: mismatch at byte {first}")
            failures += 1
        else:
            print(f"{item['symbol']}: PASS ({len(actual)} bytes)")

    for status in (item for item in segments if item["kind"] == "zero_status_range"):
        values = u32(segment_bytes(status["symbol"]))
        first = status["first"]
        last = status["last"]
        if any(values[index] != 0 for index in range(first, last)):
            print(
                f"{status['symbol']}: cases [{first}, {last}) contain failures"
            )
            failures += 1
        else:
            print(f"{status['symbol']}: PASS (cases [{first}, {last}))")

    for observation in (
        item for item in segments if item["kind"] == "old_value_permutation"
    ):
        observation_failures = 0
        input_values = u32(segment_bytes(observation["input_symbol"]))
        histogram = u32(segment_bytes(observation["histogram_symbol"]))
        old_values = u32(segment_bytes(observation["symbol"]))
        count = manifest["count"]
        selected_high = observation.get(
            "selected_high", manifest.get("selected_high")
        )
        addend = observation.get("addend", manifest.get("old_addend", 0))
        bucket_shift = observation.get("bucket_shift", 0)
        observed: list[list[int]] = [[] for _ in range(256)]
        for logical in range(count):
            value = input_values[logical]
            active = selected_high is None or value >> 8 == selected_high
            if active:
                bucket = (value >> bucket_shift) & 0xFF
                observed[bucket].append(old_values[logical])
            elif old_values[logical] != addend:
                print(
                    f"{observation['symbol']}: inactive element {logical} "
                    f"is not {addend}"
                )
                observation_failures += 1
        for logical in range(count, manifest["padded_count"]):
            if old_values[logical] != addend:
                print(
                    f"{observation['symbol']}: padded element {logical} "
                    f"is not {addend}"
                )
                observation_failures += 1
                break
        for bucket, values in enumerate(observed):
            values.sort()
            expected = list(range(addend, addend + histogram[bucket]))
            if values != expected:
                print(
                    f"{observation['symbol']}: bin {bucket} old values are not "
                    f"the permutation [{addend}, {addend + histogram[bucket]})"
                )
                observation_failures += 1
        if observation_failures == 0:
            print(
                f"{observation['symbol']}: PASS "
                "(order-independent per-bin permutation)"
            )
        failures += observation_failures
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
