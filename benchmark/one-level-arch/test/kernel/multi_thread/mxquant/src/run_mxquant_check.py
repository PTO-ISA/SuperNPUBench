#!/usr/bin/env python3
"""Generate inputs, run the multi_thread/mxquant ELFs, and compare exactly.

Each ELF must be built with ``res_check=on``.  This script writes input.bin into
the CHK_DIR encoded in the ELF, runs gfrun with four PEs, and compares the two
kernel outputs against the reference the device computed from that same input:

    output.bin         E4M3 payload from the tile kernel
    scale_output.bin   E8M0 group-32 scales from the tile kernel
    golden_output.bin  payload from mxquant_scalar_ref.h
    golden_scales.bin  scales from mxquant_scalar_ref.h
    input_readback.bin the input the device actually saw

MX quantization is bit-defined, so the comparison is byte-exact.

Two guards keep a silent pass from being possible:
  * input_readback.bin must equal the input.bin written here, which is what
    catches the input never reaching the device;
  * the golden must not be uniformly zero, which is what an all-zero input
    would produce and which would make any comparison vacuous.

Usage:
    python3 run_mxquant_check.py --gfrun /path/to/gfrun
    python3 run_mxquant_check.py --gfrun /path/to/gfrun mxquant
"""

from __future__ import annotations

import argparse
import struct
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
# src -> mxquant -> multi_thread -> kernel -> test -> one-level-arch
ONE_LEVEL_ROOT = SCRIPT_DIR.parents[4]
ELF_DIR = ONE_LEVEL_ROOT / "output/kernel/multi_thread/mxquant/elf"
COMPARE_ROOT = ONE_LEVEL_ROOT / "compare"

ROWS = 128
BLOCK = 32
# variant name -> column count
VARIANTS = {
    "mxquant": 64,
    "mxquant_1024_assembly": 1024,
    "mxquant_1024_streaming": 1024,
}

EXPORTS = ("input_readback.bin", "output.bin", "scale_output.bin",
           "golden_output.bin", "golden_scales.bin")


def make_input(rows: int, cols: int, seed: int) -> bytes:
    """BF16 tensor spanning a wide exponent range within one group of 32.

    Biased exponent 118..133 with a varying mantissa, so each group of 32 has a
    non-trivial amax and the E8M0 scale differs across groups.
    """
    state = seed & 0xFFFFFFFF
    out = bytearray()
    for _ in range(rows * cols):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        sign = (state >> 8) & 1
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        exponent = 118 + ((state >> 8) % 16)
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        mantissa = (state >> 8) & 0x7F
        bits = (sign << 15) | (exponent << 7) | mantissa
        out += struct.pack("<H", bits)
    return bytes(out)


def check_variant(name: str, gfrun: Path, timeout: int, seed: int) -> tuple[str, str]:
    cols = VARIANTS[name]
    elf = ELF_DIR / f"kernel_multi_thread_mxquant_{name}_PE4.elf"
    if not elf.is_file():
        return "SKIP", f"missing ELF: {elf}"

    case_dir = COMPARE_ROOT / elf.stem
    case_dir.mkdir(parents=True, exist_ok=True)
    # Clear stale exports so a crashed run cannot look like a pass.
    for stem in EXPORTS:
        (case_dir / stem).unlink(missing_ok=True)

    payload_bytes = ROWS * cols
    scale_bytes = ROWS * (cols // BLOCK)
    source = make_input(ROWS, cols, seed)
    (case_dir / "input.bin").write_bytes(source)

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

    missing = [stem for stem in EXPORTS if not (case_dir / stem).is_file()]
    if missing:
        return "FAIL", f"exports missing ({', '.join(missing)}) in {case_dir}"

    readback = (case_dir / "input_readback.bin").read_bytes()
    if readback != source:
        differing = sum(a != b for a, b in zip(readback, source))
        return "FAIL", (f"input never reached the device: {differing} of "
                        f"{len(source)} bytes differ from input.bin")

    results = []
    for label, actual_name, golden_name, expected in (
        ("payload", "output.bin", "golden_output.bin", payload_bytes),
        ("scale", "scale_output.bin", "golden_scales.bin", scale_bytes),
    ):
        actual = (case_dir / actual_name).read_bytes()
        golden = (case_dir / golden_name).read_bytes()
        if len(actual) != expected or len(golden) != expected:
            return "FAIL", (f"{label}: size actual={len(actual)} "
                            f"golden={len(golden)} expected={expected}")
        if not any(golden):
            return "FAIL", f"{label}: golden is all zero, comparison is vacuous"
        bad = sum(a != b for a, b in zip(actual, golden))
        if bad:
            first = next(i for i, (a, b) in enumerate(zip(actual, golden))
                         if a != b)
            return "FAIL", (f"{label}: {bad}/{expected} bytes differ, first at "
                            f"{first} (got 0x{actual[first]:02x} "
                            f"want 0x{golden[first]:02x})")
        results.append(f"{label} {expected}/{expected}")
    return "PASS", ", ".join(results)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gfrun", type=Path, required=True,
                        help="path to the gfrun binary")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("variants", nargs="*",
                        help=f"default: {' '.join(VARIANTS)}")
    args = parser.parse_args()

    selected = args.variants or list(VARIANTS)
    unknown = [name for name in selected if name not in VARIANTS]
    if unknown:
        parser.error(f"unknown variant(s): {', '.join(unknown)}")

    failures = 0
    for name in selected:
        status, detail = check_variant(name, args.gfrun, args.timeout, args.seed)
        if status not in ("PASS", "SKIP"):
            failures += 1
        print(f"{status:7} {name:24} {detail}", flush=True)
    print(f"summary: {len(selected) - failures} ok, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
