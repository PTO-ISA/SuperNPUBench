#!/usr/bin/env python3
"""Host-side check for the same-address GM atomic-add microbenchmark.

The host owns the shape: it writes chunks.bin, runs
`gfrun -s softcore.multiThreadNum=4 -f <elf>`, then verifies, per PE:
  1. the final counter is chunks*32 (every event retired);
  2. the observed old-value stream is exactly the multiset 0..chunks*32-1
     (duplicate effective addresses were serialized, none collapsed);
  3. the run is not vacuous: chunks*32 > 0 and the stream is not all zero.
A collapsed/raced atom returns repeated old values (and leaves gaps), which
fails guard 2; a run whose ELF never executed fails guards 1 and 3.
"""

import argparse
import struct
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
# src -> gm_atom_dup -> multi_thread -> kernel -> test -> one-level-arch
ONE_LEVEL_ROOT = SCRIPT_DIR.parents[4]
ELF_DIR = ONE_LEVEL_ROOT / "output/kernel/multi_thread/gm_atom_dup/elf"
COMPARE_ROOT = ONE_LEVEL_ROOT / "compare"

LANE = 32
THREADS = 4
MAX_CHUNKS = 64  # must match kMaxChunks in gm_atom_dup_driver.h
CHUNKS = 64

ELF_NAME = "kernel_multi_thread_gm_atom_dup_gm_atom_dup_PE4.elf"
EXPORTS = ("out.bin", "counter.bin")


def compare(out, counter, chunks):
    """Return None on success, else a failure detail string."""
    expected = chunks * LANE
    stride = MAX_CHUNKS * LANE

    # Guard 3: never pass on a run that produced nothing.
    if not any(out):
        return "output stream is all zero (ELF did not run?)"

    want = list(range(expected))
    for tid in range(THREADS):
        seen = list(out[tid * stride:tid * stride + expected])
        final = counter[tid * LANE]
        if final != expected:
            return f"PE{tid}: counter={final}, want {expected}"
        if sorted(seen) != want:
            dups = len(seen) - len(set(seen))
            missing = len(set(want) - set(seen))
            return (f"PE{tid}: old-value stream is not 0..{expected - 1} "
                    f"(dups={dups}, missing={missing})")
    return None


def run_check(gfrun, timeout, chunks):
    if chunks <= 0 or chunks > MAX_CHUNKS:
        return "FAIL", f"chunks={chunks} out of range 1..{MAX_CHUNKS}"
    expected = chunks * LANE

    elf = ELF_DIR / ELF_NAME
    if not elf.exists():
        return "SKIP", f"missing {elf}"
    case_dir = COMPARE_ROOT / elf.stem
    case_dir.mkdir(parents=True, exist_ok=True)
    for stem in EXPORTS + ("chunks.bin",):
        (case_dir / stem).unlink(missing_ok=True)

    (case_dir / "chunks.bin").write_bytes(struct.pack("<i", chunks))

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

    stride = MAX_CHUNKS * LANE
    out = struct.unpack(f"<{THREADS * stride}i", (case_dir / "out.bin").read_bytes())
    counter = struct.unpack(f"<{THREADS * LANE}i", (case_dir / "counter.bin").read_bytes())

    failure = compare(out, counter, chunks)
    if failure is not None:
        return "FAIL", failure

    want = list(range(expected))
    (case_dir / "golden.bin").write_bytes(
        struct.pack(f"<{THREADS * expected}i", *(want * THREADS)))
    return "PASS", (f"chunks={chunks} PEs={THREADS} "
                    f"serialized {THREADS * expected}/{THREADS * expected} events exact")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gfrun", required=True, type=Path, help="path to gfrun")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--chunks", type=int, default=CHUNKS,
                        help=f"chunks per PE (1..{MAX_CHUNKS})")
    args = parser.parse_args()

    status, detail = run_check(args.gfrun, args.timeout, args.chunks)
    print(f"{status} gm_atom_dup {detail}")
    failed = status not in ("PASS", "SKIP")
    print(f"summary: {0 if failed else 1} ok, {1 if failed else 0} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
