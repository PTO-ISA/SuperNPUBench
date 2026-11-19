#!/usr/bin/env python3
"""Generate FP32 unsorted-segment-sum inputs and check the four-PE ELF."""

import argparse
import math
import struct
import subprocess
import sys
from pathlib import Path

ROWS = 64
FEATURES = 64
SEGMENTS = 12
SCRIPT_DIR = Path(__file__).resolve().parent
ONE_LEVEL_ROOT = SCRIPT_DIR.parents[4]
ELF_NAME = (
    "kernel_reduction_unsorted_segment_sum_"
    "unsorted_segment_sum_PE4.elf"
)
ELF = ONE_LEVEL_ROOT / "output/kernel/reduction/unsorted_segment_sum/elf" / ELF_NAME
CASE_DIR = ONE_LEVEL_ROOT / "compare" / ELF.stem


def fixture():
    # Several rows collide in segments 0..6. Segments 7..11 are empty.
    # -1 and SEGMENTS exercise invalid-ID filtering.
    ids = [
        -1 if i % 13 == 0 else SEGMENTS if i % 17 == 0 else (i * 3) % 7
        for i in range(ROWS)
    ]
    data = [float(((i * 11 + j * 7) % 17) - 8)
            for i in range(ROWS) for j in range(FEATURES)]
    return data, ids


def reference(data, ids):
    out = [0.0] * (SEGMENTS * FEATURES)
    for row, segment in enumerate(ids):
        if not 0 <= segment < SEGMENTS:
            continue
        for col in range(FEATURES):
            out[segment * FEATURES + col] += data[row * FEATURES + col]
    return out


def self_test():
    data, ids = fixture()
    assert len(data) == ROWS * FEATURES and len(ids) == ROWS
    assert -1 in ids and SEGMENTS in ids
    assert len(set(ids)) < len(ids)
    expected = reference(data, ids)
    assert all(value == 0.0 for value in expected[7 * FEATURES:])
    assert any(value != 0.0 for value in expected[:7 * FEATURES])

    # A single row maps to one segment, while invalid IDs contribute nothing.
    isolated_ids = [-1] * ROWS
    isolated_ids[1] = 2
    isolated = reference(data, isolated_ids)
    assert isolated[2 * FEATURES:3 * FEATURES] == data[FEATURES:2 * FEATURES]
    assert all(value == 0.0 for value in isolated[:2 * FEATURES])
    return data, ids, expected


def write_fixture(data, ids, expected):
    CASE_DIR.mkdir(parents=True, exist_ok=True)
    (CASE_DIR / "output.bin").unlink(missing_ok=True)
    (CASE_DIR / "data.bin").write_bytes(struct.pack(f"<{len(data)}f", *data))
    (CASE_DIR / "segment_ids.bin").write_bytes(
        struct.pack(f"<{len(ids)}i", *ids)
    )
    (CASE_DIR / "golden.bin").write_bytes(
        struct.pack(f"<{len(expected)}f", *expected)
    )


def run(gfrun, timeout, expected):
    if not ELF.is_file():
        raise RuntimeError(f"missing ELF: {ELF}")
    command = [str(gfrun), "-s", "softcore.multiThreadNum=4", "-f", str(ELF)]
    proc = subprocess.run(command, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True,
                          timeout=timeout, check=False)
    (CASE_DIR / "gfrun.log").write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"gfrun exited {proc.returncode}; see {CASE_DIR / 'gfrun.log'}")
    output_path = CASE_DIR / "output.bin"
    if not output_path.is_file():
        raise RuntimeError(f"missing output: {output_path}")
    raw = output_path.read_bytes()
    if len(raw) != len(expected) * 4:
        raise RuntimeError(f"output has {len(raw)} bytes; expected {len(expected) * 4}")
    actual = struct.unpack(f"<{len(expected)}f", raw)
    for index, (got, want) in enumerate(zip(actual, expected)):
        if not math.isclose(got, want, rel_tol=1e-5, abs_tol=1e-5):
            raise RuntimeError(
                f"segment={index // FEATURES} feature={index % FEATURES}: "
                f"got {got}, expected {want}"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gfrun", type=Path, help="run and check this gfrun binary")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--self-test", action="store_true",
                        help="validate the host reference without running an ELF")
    args = parser.parse_args()
    data, ids, expected = self_test()
    if args.self_test:
        print("PASS host reference: collisions, invalid IDs, empty segments")
        return 0
    if args.gfrun is None:
        parser.error("provide --gfrun or --self-test")
    try:
        write_fixture(data, ids, expected)
        run(args.gfrun, args.timeout, expected)
    except (RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"FAIL unsorted_segment_sum: {error}", file=sys.stderr)
        return 1
    print(f"PASS unsorted_segment_sum: {SEGMENTS * FEATURES} values; {CASE_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
