#!/usr/bin/env python3
"""Generate independent exact goldens and observation contracts."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

COUNT = 263
PADDED_COUNT = 384
RADIX = 256
TOPK_CASES = [
    (0, 0, 0), (0, 7, 0), (129, 0, 0), (129, 1, 0),
    (129, 129, 0), (129, 222, 0), (1, 1, 0), (31, 17, 0),
    (32, 17, 0), (33, 17, 0), (127, 37, 0), (128, 37, 0),
    (129, 37, 0), (10, 3, 2), (129, 1, 1), (129, 65, 1),
    (129, 129, 1),
]


def histogram_input() -> list[int]:
    result: list[int] = []
    for element in range(COUNT):
        value = (element * 73 + (element // 9) * 11) & 0xFF
        if (element & 3) < 2:
            value = 0x2A
        if element in (0, 128):
            value = 0
        if element in (1, 262):
            value = 255
        result.append(value)
    return result


def selected_input() -> list[int]:
    result: list[int] = []
    for element in range(COUNT):
        high = (element * 29 + 3) & 0xFF
        if element % 7 < 3:
            high = 0xA5
        if high == 0xA5 and element % 7 >= 3:
            high = 0x5A
        low = (element * 73 + (element // 9) * 11) & 0xFF
        if element % 5 < 3:
            low = 0x2A
        if element == 0:
            low = 0
        if element == 1:
            low = 255
        result.append((high << 8) | low)
    return result


def boundary_input(count: int, pattern: int) -> list[int]:
    ties = [
        0xFFFF, 0xF100, 0xF100, 0xF100, 0xF100,
        0xE999, 0xE999, 0x8000, 0x0100, 0x0000,
    ]
    result: list[int] = []
    for element in range(count):
        if pattern == 1:
            value = 0x9A7C
        elif pattern == 2:
            value = ties[element]
        else:
            value = (
                element * 40503
                + (element // 17) * 257
                + (element % 11) * 4099
            ) & 0xFFFF
            if element == 0:
                value = 0xFFFF
            if element == 1:
                value = 0
        result.append(value)
    return result


def write_u32(path: Path, values: list[int]) -> None:
    path.write_bytes(struct.pack(f"<{len(values)}I", *values))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=(
        "histogram_tile_element", "selected_radix_tile_element",
        "element_expression_chain",
        "topk_boundaries", "topk_boundaries_0", "topk_boundaries_1",
        "topk_boundaries_2",
    ))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.case == "element_expression_chain":
        mask32 = 0xFFFFFFFF
        active = []
        for element in range(COUNT):
            value = (element * 2654435761 + 0x01234567) & mask32
            value ^= ((element // 7) * 0x00010101) & mask32
            active.append(value)
        input_values = active + [0xDEADBEEF] * (PADDED_COUNT - COUNT)
        loaded_values = active + [0] * (PADDED_COUNT - COUNT)

        arithmetic = []
        bitwise = []
        unary = []
        for value in loaded_values:
            biased = (value + 19) & mask32
            multiplied = (biased * 13) & mask32
            reduced = (multiplied - 7) & mask32
            divided = reduced // 5
            arithmetic.append(divided % 251)

            shifted_left = (value << 3) & mask32
            shifted_right = shifted_left >> 2
            masked = shifted_right & 0xFFFFFFF0
            merged = masked | 0x00000105
            mixed = (merged ^ 0x0055AA11) & mask32
            bitwise.append((mixed ^ shifted_left) & mask32)

            negated = (-value) & mask32
            unary.append((~negated) & mask32)

        status = [
            COUNT,
            0,
            sum(arithmetic) & mask32,
            sum(bitwise) & mask32,
            sum(unary) & mask32,
            PADDED_COUNT,
            10,
            0x45585052,
        ]
        files = {
            "element_expression_chain_input": ("input_u32.bin", input_values),
            "element_expression_chain_arithmetic": (
                "arithmetic_u32.bin", arithmetic
            ),
            "element_expression_chain_bitwise": ("bitwise_u32.bin", bitwise),
            "element_expression_chain_unary": ("unary_u32.bin", unary),
            "element_expression_chain_status": ("status_u32.bin", status),
        }
        for filename, values in files.values():
            write_u32(args.out / filename, values)
        manifest = {
            "case": args.case,
            "count": COUNT,
            "padded_count": PADDED_COUNT,
            "segments": [
                {"symbol": symbol, "kind": "exact", "file": filename}
                for symbol, (filename, _values) in files.items()
            ],
        }
        (args.out / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        return 0

    if args.case == "histogram_tile_element":
        active = histogram_input()
        selected_high = None
        addend = 1
        input_symbol = "histogram_tile_element_input"
        histogram_symbol = "histogram_tile_element_histogram"
        old_symbol = "histogram_tile_element_old_plus_one"
        status_symbol = "histogram_tile_element_status"
        marker = 0x48495354
    elif args.case == "selected_radix_tile_element":
        active = selected_input()
        selected_high = 0xA5
        addend = 7
        input_symbol = "selected_radix_tile_element_input"
        histogram_symbol = "selected_radix_tile_element_histogram"
        old_symbol = "selected_radix_tile_element_old_plus_seven"
        status_symbol = "selected_radix_tile_element_status"
        marker = addend

    else:
        if args.case == "topk_boundaries":
            first, last = 0, len(TOPK_CASES)
        else:
            batch = int(args.case.rsplit("_", 1)[1])
            first, last = ((0, 6), (6, 13), (13, 17))[batch]
        count, requested_k, pattern = TOPK_CASES[last - 1]
        active = boundary_input(count, pattern)
        actual_k = min(count, requested_k)
        sorted_values = sorted(active, reverse=True)
        high_histogram = [0] * RADIX
        low_histogram = [0] * RADIX
        for value in active:
            high_histogram[value >> 8] += 1
        selected_high = None
        if actual_k:
            selected_high = sorted_values[actual_k - 1] >> 8
            for value in active:
                if value >> 8 == selected_high:
                    low_histogram[value & 0xFF] += 1
        sentinel = 0xDEADBEEF
        files = {
            "topk_boundaries_input": "input_u32.bin",
            "topk_boundaries_high_hist": "high_hist_u32.bin",
            "topk_boundaries_low_hist": "low_hist_u32.bin",
            "topk_boundaries_output": "output_u32.bin",
        }
        write_u32(
            args.out / files["topk_boundaries_input"],
            active + [sentinel] * (129 - count),
        )
        write_u32(args.out / files["topk_boundaries_high_hist"], high_histogram)
        write_u32(args.out / files["topk_boundaries_low_hist"], low_histogram)
        write_u32(
            args.out / files["topk_boundaries_output"],
            sorted_values[:actual_k] + [sentinel] * (133 - actual_k),
        )
        manifest = {
            "case": args.case,
            "count": count,
            "padded_count": ((count + 127) // 128) * 128,
            "segments": [
                {"symbol": symbol, "kind": "exact", "file": filename}
                for symbol, filename in files.items()
            ] + [
                {
                    "symbol": "topk_boundaries_status",
                    "kind": "zero_status_range",
                    "first": first,
                    "last": last,
                },
                {
                    "symbol": "topk_boundaries_old_high",
                    "kind": "old_value_permutation",
                    "input_symbol": "topk_boundaries_input",
                    "histogram_symbol": "topk_boundaries_high_hist",
                    "bucket_shift": 8,
                    "selected_high": None,
                    "addend": 0,
                },
                {
                    "symbol": "topk_boundaries_old_low",
                    "kind": "old_value_permutation",
                    "input_symbol": "topk_boundaries_input",
                    "histogram_symbol": "topk_boundaries_low_hist",
                    "bucket_shift": 0,
                    "selected_high": 256 if selected_high is None
                    else selected_high,
                    "addend": 0,
                },
            ],
        }
        (args.out / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        return 0

    histogram = [0] * RADIX
    selected_count = 0
    for value in active:
        if selected_high is None or value >> 8 == selected_high:
            histogram[value & 0xFF] += 1
            selected_count += 1

    padded = active + [0xFFFFFFFF] * (PADDED_COUNT - COUNT)
    checksum = sum(bin_index * count for bin_index, count in enumerate(histogram))
    status = [
        COUNT if selected_high is None else selected_count,
        0,
        selected_count,
        histogram[0],
        histogram[255],
        checksum,
        addend if selected_high is None else selected_high,
        marker,
    ]

    input_file = "input_u32.bin"
    histogram_file = "histogram_u32.bin"
    status_file = "status_u32.bin"
    write_u32(args.out / input_file, padded)
    write_u32(args.out / histogram_file, histogram)
    write_u32(args.out / status_file, status)

    manifest = {
        "case": args.case,
        "count": COUNT,
        "padded_count": PADDED_COUNT,
        "selected_high": selected_high,
        "old_addend": addend,
        "segments": [
            {"symbol": input_symbol, "kind": "exact", "file": input_file},
            {
                "symbol": histogram_symbol,
                "kind": "exact",
                "file": histogram_file,
            },
            {
                "symbol": old_symbol,
                "kind": "old_value_permutation",
                "input_symbol": input_symbol,
                "histogram_symbol": histogram_symbol,
            },
            {"symbol": status_symbol, "kind": "exact", "file": status_file},
        ],
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
