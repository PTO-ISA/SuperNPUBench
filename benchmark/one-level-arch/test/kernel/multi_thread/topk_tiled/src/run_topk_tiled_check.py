#!/usr/bin/env python3
"""Host-side check for the topk radix-select kernel.

The host owns the input: generate input.bin/starts.bin/ends.bin, run
gfrun -s softcore.multiThreadNum=4 -f <elf>, then compare. Guards:
  1. input_readback.bin must equal input.bin (input reached the device);
  2. errors.bin must be all zero (no candidate overflow / bad range);
  3. the comparison must not be vacuous (golden not all zero);
  4. per row, the sorted output index set must equal the golden top-K set
     exactly (input values are distinct, so the set is unique).
"""

import argparse
import math
import random
import struct
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
# src -> topk -> multi_thread -> kernel -> test -> one-level-arch
ONE_LEVEL_ROOT = SCRIPT_DIR.parents[4]
ELF_DIR = ONE_LEVEL_ROOT / "output/kernel/multi_thread/topk_tiled/elf"
COMPARE_ROOT = ONE_LEVEL_ROOT / "compare"

ROWS = 4
COLS = 8192
TOPK = 512
PAD = 32  # must match kLane tail-TLOAD padding in topk_driver.h
STARTS = [0, 123, 7, 31]
ENDS = [COLS, 6000, 1531, COLS - 1]

ELF_NAME = "kernel_multi_thread_topk_tiled_topk_tiled_PE4.elf"
EXPORTS = ("input_readback.bin", "output.bin", "errors.bin")


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def make_input(seed):
    rng = random.Random(seed)
    seen = set()
    values = []

    def push(v):
        f = f32(v)
        if f == 0.0 or f in seen or math.isinf(f) or math.isnan(f):
            return False
        seen.add(f)
        values.append(f)
        return True

    total = ROWS * COLS
    # Wide-range values: random sign, exponent in 2^-6..2^6, random mantissa.
    while len(values) < total - 600:
        sign = 1.0 if rng.random() < 0.5 else -1.0
        push(sign * (1.0 + rng.random()) * (2.0 ** rng.randint(-6, 6)))
    # Tight cluster near 1.0: many values share an FP16 high byte, forcing
    # the candidate ping-pong and the FP32 byte passes to do real work.
    while len(values) < total:
        push(1.0 + rng.random() * 1e-3)
    rng.shuffle(values)
    values.extend([0.0] * PAD)  # tail-TLOAD padding, never part of a range
    return values


def golden_topk(values, row):
    lo, hi = STARTS[row], ENDS[row]
    base = row * COLS
    window = [(values[base + i], i) for i in range(lo, hi)]
    window.sort(key=lambda p: (-p[0], p[1]))
    return sorted(i for _, i in window[:TOPK])


def run_check(gfrun, timeout, seed):
    elf = ELF_DIR / ELF_NAME
    if not elf.exists():
        return "SKIP", f"missing {elf}"
    case_dir = COMPARE_ROOT / elf.stem
    case_dir.mkdir(parents=True, exist_ok=True)
    for stem in EXPORTS + ("golden.bin",):
        (case_dir / stem).unlink(missing_ok=True)

    values = make_input(seed)
    (case_dir / "input.bin").write_bytes(
        struct.pack(f"<{len(values)}f", *values))
    (case_dir / "starts.bin").write_bytes(struct.pack(f"<{ROWS}i", *STARTS))
    (case_dir / "ends.bin").write_bytes(struct.pack(f"<{ROWS}i", *ENDS))

    command = [str(gfrun), "-s", "softcore.multiThreadNum=4", "-f", str(elf)]
    try:
        proc = subprocess.run(command, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True,
                              timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        (case_dir / "gfrun.log").write_text(exc.stdout or "", encoding="utf-8")
        return "TIMEOUT", f"after {timeout}s: {case_dir / 'gfrun.log'}"
    (case_dir / "gfrun.log").write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        return "FAIL", f"gfrun rc={proc.returncode}: {case_dir / 'gfrun.log'}"

    missing = [stem for stem in EXPORTS if not (case_dir / stem).exists()]
    if missing:
        return "FAIL", f"missing exports {missing}: {case_dir / 'gfrun.log'}"

    # Guard 1: the device must have seen the input we wrote.
    if (case_dir / "input_readback.bin").read_bytes() != (case_dir / "input.bin").read_bytes():
        return "FAIL", "input never reached the device (readback differs)"

    # Guard 2: device-reported errors (candidate overflow / degenerate range).
    errors = struct.unpack(f"<{ROWS}i", (case_dir / "errors.bin").read_bytes())
    if any(errors):
        return "FAIL", f"device error flags set: {errors}"

    output = struct.unpack(f"<{ROWS * TOPK}i", (case_dir / "output.bin").read_bytes())

    golden_flat = []
    for row in range(ROWS):
        want = golden_topk(values, row)
        got = sorted(output[row * TOPK:(row + 1) * TOPK])
        golden_flat.extend(want)

        # Guard 3: the comparison must not be vacuous.
        if not any(want):
            return "FAIL", f"row {row}: golden is all zero, comparison is vacuous"
        if got == [-1] * TOPK:
            return "FAIL", f"row {row}: output never written (all -1 sentinel)"

        if got != want:
            first = next(k for k, (a, b) in enumerate(zip(got, want)) if a != b)
            return "FAIL", (f"row {row}: index set mismatch, first diff at "
                            f"sorted position {first}: got {got[first]} want {want[first]} "
                            f"({sum(a != b for a, b in zip(got, want))} of {TOPK} differ)")

    (case_dir / "golden.bin").write_bytes(struct.pack(f"<{len(golden_flat)}i", *golden_flat))
    return "PASS", f"rows={ROWS} cols={COLS} topk={TOPK} output {ROWS * TOPK * 4}/{ROWS * TOPK * 4} bytes exact"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gfrun", required=True, type=Path, help="path to gfrun")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    status, detail = run_check(args.gfrun, args.timeout, args.seed)
    print(f"{status} topk {detail}")
    failed = status not in ("PASS", "SKIP")
    print(f"summary: {0 if failed else 1} ok, {1 if failed else 0} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
