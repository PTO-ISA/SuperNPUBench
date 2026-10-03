#!/usr/bin/env python3
"""Generate independent binary goldens for element_atomic_topk."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

COUNT = 777
PADDED_COUNT = 896
TOP_K = 37
RADIX = 256


def make_input() -> list[int]:
    values = [
        (index * 40503 + (index // 17) * 257) & 0xFFFF
        for index in range(COUNT)
    ]
    values[250:270] = [0xF123] * 20
    values[510:526] = [0xF123] * 16
    values[0] = 0xFFFF
    values[256] = 0xFFFF
    values[512] = 0xFFFF
    return values + [0xFFFFFFFF] * (PADDED_COUNT - COUNT)


def histogram(values: list[int], shift: int) -> list[int]:
    result = [0] * RADIX
    for value in values:
        result[(value >> shift) & 0xFF] += 1
    return result


def descending_bucket(counts: list[int], requested: int) -> tuple[int, int]:
    greater = 0
    for bucket in range(RADIX - 1, -1, -1):
        if greater + counts[bucket] >= requested:
            return bucket, greater
        greater += counts[bucket]
    raise ValueError("requested rank exceeds histogram population")


def write_u32(path: Path, values: list[int]) -> None:
    path.write_bytes(struct.pack(f"<{len(values)}I", *values))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    padded_input = make_input()
    active = padded_input[:COUNT]
    high_hist = histogram(active, 8)
    selected_high, greater_high = descending_bucket(high_hist, TOP_K)
    selected_values = [value for value in active if value >> 8 == selected_high]
    low_hist = histogram(selected_values, 0)
    selected_low, greater_low = descending_bucket(
        low_hist, TOP_K - greater_high
    )
    cutoff = (selected_high << 8) | selected_low
    expected_topk = sorted(active, reverse=True)[:TOP_K]
    strictly_greater = sum(value > cutoff for value in active)
    equal_needed = TOP_K - strictly_greater

    assert sum(high_hist) == COUNT
    assert sum(low_hist) == high_hist[selected_high]
    assert strictly_greater + equal_needed == TOP_K
    assert expected_topk[-1] == cutoff

    files = {
        "element_atomic_topk_input": "input_u32.bin",
        "element_atomic_topk_high_hist": "high_hist_u32.bin",
        "element_atomic_topk_low_hist": "low_hist_u32.bin",
        "element_atomic_topk_output": "topk_u32.bin",
        "element_atomic_topk_status": "status_u32.bin",
    }
    write_u32(args.out / files["element_atomic_topk_input"], padded_input)
    write_u32(args.out / files["element_atomic_topk_high_hist"], high_hist)
    write_u32(args.out / files["element_atomic_topk_low_hist"], low_hist)
    write_u32(args.out / files["element_atomic_topk_output"], expected_topk)
    write_u32(
        args.out / files["element_atomic_topk_status"],
        [
            TOP_K,
            0,
            cutoff,
            selected_high,
            COUNT,
            sum(low_hist),
            high_hist[255],
            high_hist[0],
        ],
    )

    manifest = {
        "case": "element_atomic_topk_u16_domain_u32_carrier",
        "count": COUNT,
        "padded_count": PADDED_COUNT,
        "top_k": TOP_K,
        "cutoff": cutoff,
        "strictly_greater": strictly_greater,
        "equal_needed": equal_needed,
        "segments": [
            {"symbol": symbol, "file": filename, "dtype": "u32"}
            for symbol, filename in files.items()
        ],
        "non_deterministic_observations": [
            "element_atomic_topk_old_high",
            "element_atomic_topk_old_low",
        ],
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
