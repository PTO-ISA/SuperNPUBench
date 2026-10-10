#!/usr/bin/env python3
"""Compare exact segments and order-independent atomic old values."""

from __future__ import annotations

import argparse
import json
import struct
from functools import lru_cache
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
            detail = ""
            if first + 4 <= len(actual) and first + 4 <= len(expected):
                word = first // 4
                actual_word = struct.unpack_from("<I", actual, word * 4)[0]
                expected_word = struct.unpack_from("<I", expected, word * 4)[0]
                detail = (
                    f", u32 element {word}: actual=0x{actual_word:08x} "
                    f"expected=0x{expected_word:08x}"
                )
            print(f"{item['symbol']}: mismatch at byte {first}{detail}")
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

    for observation in (
        item for item in segments if item["kind"] == "atomic_pair_trace_mod32"
    ):
        observation_failures = 0
        mask32 = 0xFFFFFFFF
        indices = list(struct.unpack(
            f"<{len(segment_bytes(observation['index_symbol'])) // 4}i",
            segment_bytes(observation["index_symbol"]),
        ))
        histogram = u32(segment_bytes(observation["histogram_symbol"]))
        first_stored = u32(segment_bytes(observation["symbol"]))
        second_stored = u32(segment_bytes(observation["second_symbol"]))
        combined_stored = u32(segment_bytes(observation["combined_symbol"]))
        initial = u32((args.golden / observation["initial_histogram_file"]).read_bytes())
        first_adds = u32((args.golden / observation["first_add_file"]).read_bytes())
        second_adds = u32((args.golden / observation["second_add_file"]).read_bytes())
        valid = manifest["valid_count"]
        count = manifest["count"]
        center = manifest["center"]
        first_old = [
            (value - observation["first_post_add"]) & mask32
            for value in first_stored
        ]
        second_old = [
            (value - observation["second_post_add"]) & mask32
            for value in second_stored
        ]
        combined = [
            (value - observation["combined_post_add"]) & mask32
            for value in combined_stored
        ]

        for element in range(valid):
            expected_combined = (
                first_old[element]
                ^ ((second_old[element] + observation["combined_bias"]) & mask32)
            )
            if combined[element] != expected_combined:
                print(
                    f"{observation['combined_symbol']}: element {element} does not "
                    "use both observed atomic old values"
                )
                observation_failures += 1
        for element in range(valid, count):
            if (first_stored[element] != observation["first_post_add"] or
                    second_stored[element] != observation["second_post_add"] or
                    combined_stored[element] != observation["combined_post_add"]):
                print(
                    f"{observation['symbol']}: inactive element {element} did not "
                    "retain its seeded values"
                )
                observation_failures += 1
                break

        for bin_index in range(len(histogram)):
            elements = [
                element for element in range(valid)
                if indices[element] + center == bin_index
            ]
            all_done = (1 << len(elements)) - 1

            @lru_cache(maxsize=None)
            def reachable(current: int, first_done: int, second_done: int) -> bool:
                if second_done == all_done:
                    return current == histogram[bin_index]
                for local, element in enumerate(elements):
                    bit = 1 << local
                    if not first_done & bit and first_old[element] == current:
                        if reachable(
                            (current + first_adds[element]) & mask32,
                            first_done | bit,
                            second_done,
                        ):
                            return True
                    if (first_done & bit and not second_done & bit and
                            second_old[element] == current):
                        if reachable(
                            (current + second_adds[element]) & mask32,
                            first_done,
                            second_done | bit,
                        ):
                            return True
                return False

            if not reachable(initial[bin_index], 0, 0):
                print(
                    f"{observation['symbol']}: bin {bin_index} old values cannot "
                    "form a modulo-2^32 atomic order that preserves each "
                    "element's first-before-second dependency"
                )
                observation_failures += 1
        if observation_failures == 0:
            print(
                f"{observation['symbol']}: PASS (order-independent paired "
                "atomic trace modulo 2^32)"
            )
        failures += observation_failures
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
