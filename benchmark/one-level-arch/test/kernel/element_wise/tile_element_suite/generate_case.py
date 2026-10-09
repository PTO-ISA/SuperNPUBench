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
        "element_expression_chain", "signed_element_expression", "indexed_gather_tile_element",
        "generic_predicated_cfg_i32",
        "topk_boundaries", "topk_boundaries_0", "topk_boundaries_1",
        "topk_boundaries_2",
    ))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.case == "generic_predicated_cfg_i32":
        elements = 32
        valid_elements = 29
        guard_elements = 8
        guard_value = 0x5A5A6B6B
        mask32 = 0xFFFFFFFF

        def signed32(value: int) -> int:
            value &= mask32
            return value if value < 0x80000000 else value - 0x100000000

        def trunc_div(dividend: int, divisor: int) -> int:
            quotient = abs(dividend) // abs(divisor)
            return -quotient if (dividend < 0) != (divisor < 0) else quotient

        input_values = [element * 7919 + 12345 - 131071
                        for element in range(elements)]
        input_values[0] = -100
        input_values[3] = 100
        control = [element % 3 - 1 for element in range(elements)]
        divisor = [
            (-5 if element & 1 else 3) if control[element] < 0 else 0
            for element in range(elements)
        ]
        output = []
        for element in range(elements):
            if element >= valid_elements:
                output.append(guard_value)
                continue
            value = input_values[element]
            selector = control[element]
            if selector < 0:
                result = trunc_div(value + 21, divisor[element])
            elif selector == 0:
                result = value * 3 - 17
            else:
                result = signed32((value & mask32) ^ 0x13579BDF) + 9
            output.append(signed32(result))
        guarded_output = ([guard_value] * guard_elements + output +
                          [guard_value] * guard_elements)
        direct = []
        for element in range(elements):
            if element < valid_elements and control[element] < 0:
                direct.append(trunc_div(
                    input_values[element] + 21, divisor[element]
                ))
            else:
                direct.append(guard_value)
        guarded_direct = ([guard_value] * guard_elements + direct +
                          [guard_value] * guard_elements)
        checksum = sum(value & mask32 for value in output) & mask32
        direct_checksum = sum(value & mask32 for value in direct) & mask32
        status = [
            elements, valid_elements, 0, 0, checksum, checksum,
            direct_checksum, output[0] & mask32, output[3] & mask32,
            direct[0] & mask32, 3, 3,
            sum(1 for element in range(valid_elements)
                if control[element] < 0),
            0, 0, 0x47434647,
        ]
        signed_files = {
            "generic_predicated_cfg_i32_input": ("input_s32.bin", input_values),
            "generic_predicated_cfg_i32_control": ("control_s32.bin", control),
            "generic_predicated_cfg_i32_divisor": ("divisor_s32.bin", divisor),
            "generic_predicated_cfg_i32_nested": (
                "nested_s32.bin", guarded_output
            ),
            "generic_predicated_cfg_i32_reversed": (
                "reversed_s32.bin", guarded_output
            ),
            "generic_predicated_cfg_i32_direct": (
                "direct_s32.bin", guarded_direct
            ),
        }
        for filename, values in signed_files.values():
            (args.out / filename).write_bytes(
                struct.pack(f"<{len(values)}i", *values)
            )
        write_u32(args.out / "status_u32.bin", status)
        manifest = {
            "case": args.case,
            "count": elements,
            "valid_count": valid_elements,
            "segments": [
                {"symbol": symbol, "kind": "exact", "file": filename}
                for symbol, (filename, _values) in signed_files.items()
            ] + [{
                "symbol": "generic_predicated_cfg_i32_status",
                "kind": "exact",
                "file": "status_u32.bin",
            }],
        }
        (args.out / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        return 0

    if args.case == "signed_element_expression":
        active = [(element * 7919 + 12345) % 65535 - 32767
                  for element in range(COUNT)]
        active[0:2] = [-7, 7]
        input_values = active + [-559038737] * (PADDED_COUNT - COUNT)
        loaded = active + [0] * (PADDED_COUNT - COUNT)
        guard = [-1234567] * 16
        # C++ truncates division toward zero. The remainder keeps the
        # dividend sign; Python's signed % would implement the wrong rule.
        quotient = []
        remainder = []
        for value in loaded:
            numerator = (value + 19) * 3 - 7
            quotient.append((1 if numerator >= 0 else -1) * (abs(numerator) // 3))
            remainder.append((1 if value >= 0 else -1) * (abs(value) % 3))
        assert remainder[0:2] == [-1, 1]
        shifted = [-(((value >> 2) ^ 0x555) & 0xFF) for value in loaded]
        arrays = {
            "signed_element_input": ("input_s32.bin", input_values),
            "signed_element_quotient": ("quotient_s32.bin", guard + quotient + guard),
            "signed_element_remainder": ("remainder_s32.bin", guard + remainder + guard),
            "signed_element_shift": ("shift_s32.bin", guard + shifted + guard),
        }
        for filename, values in arrays.values():
            (args.out / filename).write_bytes(struct.pack(f"<{len(values)}i", *values))
        status = [COUNT, 0, 0, PADDED_COUNT, 0xFFFFFFFF, 1, 3, 0x53333245]
        write_u32(args.out / "status_u32.bin", status)
        manifest = {"case": args.case, "count": COUNT,
                    "padded_count": PADDED_COUNT, "segments": [
            {"symbol": symbol, "kind": "exact", "file": filename}
            for symbol, (filename, _) in arrays.items()]
                    + [{"symbol": "signed_element_status", "kind": "exact",
                        "file": "status_u32.bin"}]}
        (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return 0

    if args.case == "indexed_gather_tile_element":
        mask32 = 0xFFFFFFFF
        table = [
            0x80000000 | ((element * 0x01020305 + 0x10203040) & 0x7FFFFFFF)
            for element in range(257)
        ]
        indices = []
        for element in range(PADDED_COUNT):
            if element >= COUNT:
                indices.append(mask32)
                continue
            index = (element * 73 + (element // 5) * 19) % 257
            if element % 11 < 4:
                index = 42
            if element == 0:
                index = 0
            if element == COUNT - 1:
                index = 256
            indices.append(index)

        gathered = [
            (table[indices[element]] + 3) & mask32
            if element < COUNT else 3
            for element in range(PADDED_COUNT)
        ]
        guard = [0x6A09E667] * 16
        output = guard + gathered + guard
        status = [
            COUNT,
            0,
            sum(gathered[:COUNT]) & mask32,
            sum(gathered) & mask32,
            0,
            len(table),
            PADDED_COUNT,
            0x47415448,
            257,
            0,
            0,
            0,
        ]
        files = {
            "indexed_gather_tile_element_table": ("table_u32.bin", table),
            "indexed_gather_tile_element_indices": (
                "indices_u32.bin", indices
            ),
            "indexed_gather_tile_element_output": ("output_u32.bin", output),
            "indexed_gather_tile_element_status": ("status_u32.bin", status),
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
