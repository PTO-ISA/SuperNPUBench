#!/usr/bin/env python3
"""Host-side check for the 256-bin suffix cumsum operator.

Generate hist.bin (per-PE 256 bins + 128 zero pad), run gfrun, and compare the
exported result.bin to the scalar suffix sum:
    H[i] = sum_{j=i..255} bins[j].

Guards:
  1. the comparison is not vacuous (golden not all zero);
  2. the 128-word pad after the sentinel is untouched;
  3. every bin matches exactly.
"""

import argparse
import random
import struct
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
# src -> histogram_cumsum_m32 -> multi_thread -> kernel -> test -> one-level-arch
ONE_LEVEL_ROOT = SCRIPT_DIR.parents[4]
ELF = (ONE_LEVEL_ROOT / "output/kernel/multi_thread/histogram_cumsum_m32/elf"
       / "kernel_multi_thread_histogram_cumsum_m32_PE4.elf")
COMPARE_ROOT = ONE_LEVEL_ROOT / "compare"

ROWS = 4
BINS = 256
PAD = 128
WORDS = BINS + PAD


def make_hist(seed):
    rng = random.Random(seed)
    rows = []
    for _ in range(ROWS):
        bins = [rng.randint(0, 40) for _ in range(BINS)]
        rows.append(bins + [0] * PAD)
    return rows


def golden_suffix(bins):
    out = [0] * BINS
    acc = 0
    for i in range(BINS - 1, -1, -1):
        acc += bins[i]
        out[i] = acc
    return out


def run_check(gfrun, timeout, seed):
    if not ELF.exists():
        return "SKIP", f"missing {ELF}"
    case_dir = COMPARE_ROOT / ELF.stem
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "result.bin").unlink(missing_ok=True)

    rows = make_hist(seed)
    flat = [v for row in rows for v in row]
    (case_dir / "hist.bin").write_bytes(struct.pack(f"<{len(flat)}i", *flat))

    command = [str(gfrun), "-s", "softcore.multiThreadNum=4", "-f", str(ELF)]
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

    if not (case_dir / "result.bin").exists():
        return "FAIL", f"missing result.bin: {case_dir / 'gfrun.log'}"
    result = struct.unpack(f"<{ROWS * WORDS}i", (case_dir / "result.bin").read_bytes())

    for row in range(ROWS):
        want = golden_suffix(rows[row][:BINS])
        got = list(result[row * WORDS:(row + 1) * WORDS])
        if not any(want):
            return "FAIL", f"row {row}: golden is all zero, comparison is vacuous"
        if any(got[BINS:]):
            return "FAIL", f"row {row}: zero pad was written: {got[BINS:]}"
        for i in range(BINS):
            if got[i] != want[i]:
                return "FAIL", (f"row {row}: bin {i}: got {got[i]} want {want[i]} "
                                f"({sum(a != b for a, b in zip(got[:BINS], want))} "
                                f"of {BINS} differ)")
    return "PASS", f"{ROWS} rows exact ({BINS} bins each)"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gfrun", required=True, type=Path, help="path to gfrun")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()

    status, detail = run_check(args.gfrun, args.timeout, args.seed)
    print(f"{status} histogram_cumsum_m32 {detail}")
    return 1 if status not in ("PASS", "SKIP") else 0


if __name__ == "__main__":
    sys.exit(main())
