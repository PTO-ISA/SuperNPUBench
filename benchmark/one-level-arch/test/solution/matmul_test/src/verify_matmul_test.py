#!/usr/bin/env python3
"""Numerical (precision) and workload/performance verification for the
matmul_test kernel (benchmark/one-level-arch/test/kernel/matmul).

matmul_test contract:
  * fp16 (__half) inputs A[M,K], B[K,N]; fp32 accumulate; fp16 output C[M,N]
  * dynamic shapes; kernel template params are only tile sizes (tM/tN/tK)
  * built with res_check=on the kernel reads src0.bin (A) / src1.bin (B) from
    CHK_DIR and writes res.bin (C); all files are raw fp16 (little-endian 2B).

Golden semantics: the kernel loads fp16 inputs and accumulates in fp32, then
casts the fp32 result to fp16 on the way out.  We therefore compute
    golden_f32 = A.f32 @ B.f32          (numpy float32 matmul)
    golden_f16 = golden_f32.astype(fp16)  (final cast, matching TCVT)
and compare res.bin (fp16) against golden_f16 (as fp16 rather than widening
to fp32 first, to honour bit-level EXACT matches expected of fp16 ops).

Usage:
  python3 verify_matmul_test.py -d <single elf> \
      --gfrun /path/to/SuperScalarModel/bin/gfrun --gfrun-args "-t 1 -f" \
      [--seed 123] [--input-scale 0.1] [--atol 0.05] [--rtol 0.05]
  python3 verify_matmul_test.py -l elf_list.txt --gfrun ...   # batch

If --gfsim is given, the ELF is also run through gfsim for cycle counts
(Total Cycles etc.); note Test/ISSUE-003: at the pinned model commit gfsim
aborts on matmul ELFs, so this is best-effort.

The script also reports an "uncomputed" metric: cells that stay 0 while the
golden is non-zero.  A large uncomputed count points at the model-side partial
tile write limitation documented in Test/ISSUE-001 (functional PASS but not all
tile rows reach global memory).
"""

import argparse
import os
import re
import signal
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ONE_LEVEL_ROOT = SCRIPT_DIR.parents[3]
COMPARE_ROOT = ONE_LEVEL_ROOT / "compare"

DEFAULT_GFRUN = "/Users/blacktraker/Programming/gitproj/DV4/SuperScalarModel/bin/gfrun"
DEFAULT_GFRUN_ARGS = "-t 1 -f"

# fp16 accumulate error budget; fp16 representation is the dominant error.
DEFAULT_ATOL = 0.05
DEFAULT_RTOL = 0.05


def parse_shape(elf: Path):
    """Extract M/N/K/tM/tN/tK from 'matmul_test_M{N}_N{N}_K{N}_tM{tM}_tN{tN}_tK{tK}.elf'."""
    name = elf.stem
    m = re.search(
        r"_M(?P<M>\d+)_N(?P<N>\d+)_K(?P<K>\d+)"
        r"_tM(?P<tM>\d+)_tN(?P<tN>\d+)_tK(?P<tK>\d+)$",
        name,
    )
    if not m:
        raise ValueError(f"cannot parse matmul_test shape from ELF name: {name}")
    shape = {k: int(v) for k, v in m.groupdict().items()}
    shape["name"] = name
    return shape


def prepare_case(elf: Path, shape: dict, args):
    """Generate fp16 inputs / golden, pre-allocate res.bin under COMPARE_ROOT."""
    case_dir = COMPARE_ROOT / shape["name"]
    case_dir.mkdir(parents=True, exist_ok=True)

    M, K = shape["M"], shape["K"]
    Kb, N = shape["K"], shape["N"]
    rng = np.random.default_rng(args.seed)
    if args.ones:
        a = np.ones((M, K), dtype=np.float16)
        b = np.ones((Kb, N), dtype=np.float16)
    else:
        a = (rng.standard_normal((M, K)) * args.input_scale).astype(np.float16)
        b = (rng.standard_normal((Kb, N)) * args.input_scale).astype(np.float16)
        # keep magnitudes small enough that fp16 accumulation of K products
        # cannot overflow (K <= 1024 typical in the suite)
    a.clip(-1.0, 1.0, out=a)
    b.clip(-1.0, 1.0, out=b)

    golden_f16 = (a.astype(np.float32) @ b.astype(np.float32)).astype(np.float16)

    # raw fp16 little-endian files
    a.tofile(case_dir / "src0.bin")
    b.tofile(case_dir / "src1.bin")
    golden_f16.tofile(case_dir / "golden.bin")
    np.zeros((M, N), dtype=np.float16).tofile(case_dir / "res.bin")
    return case_dir, golden_f16


def run_one(binary, args_list):
    return subprocess.run(
        args_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        timeout=args_list[-1] if False else 120,
    )


def run_gfrun(elf: Path, args) -> tuple:
    command = [args.gfrun, *shlex.split(args.gfrun_args), str(elf)]
    if args.multi_thread:
        # 4-PE cooperative kernels (e.g. matmul_test_mt) need 4 threads
        command.insert(1, "-s")
        command.insert(2, "softcore.multiThreadNum=4")
    proc = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, start_new_session=True,
    )
    try:
        out, _ = proc.communicate(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        out, _ = proc.communicate()
        return "timeout", -1, out
    status = "pass" if proc.returncode == 0 else "fail"
    return status, proc.returncode, out


def collect_gfrun_stats(out: str):
    blk = re.search(r"Total Block number = (\d+)", out)
    inst = re.search(r"Total Inst number = (\d+)", out)
    return {
        "blocks": int(blk.group(1)) if blk else None,
        "insts": int(inst.group(1)) if inst else None,
        "reached_end": "Reach the End of Benchmark" in out,
    }


def compare_result(case_dir: Path, golden_f16):
    result = np.fromfile(case_dir / "res.bin", dtype=np.float16)
    golden = golden_f16.flatten()
    if result.size != golden.size:
        return None, {"reason": f"res.bin elements {result.size} != golden {golden.size}"}
    result = result.reshape(golden_f16.shape)
    golden = golden_f16

    rf = result.astype(np.float32)
    gf = golden.astype(np.float32)
    diff = rf - gf
    abs_diff = np.abs(diff)
    max_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)

    # "uncomputed": output stayed 0 while golden is non-zero (partial tile write)
    nonzero_mask = gf != 0.0
    uncomputed = int(((rf == 0.0) & nonzero_mask).sum())
    covered = int(nonzero_mask.sum())
    passed = bool(np.allclose(rf, gf, atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL))

    metrics = {
        "mse": float(np.mean(diff * diff)),
        "max_abs": float(abs_diff[max_idx]),
        "max_index": tuple(int(i) for i in max_idx),
        "actual_at_max": float(rf[max_idx]),
        "golden_at_max": float(gf[max_idx]),
        "uncomputed_cells": int(uncomputed),
        "nonzero_golden_cells": int(covered),
        "exact_matches": int((rf == gf).sum()),
        "total_cells": int(rf.size),
    }
    report = case_dir / "verify_matmul_test.log"
    with report.open("w", encoding="utf-8") as f:
        f.write(f"status: {'PASS' if passed else 'FAIL'}\n")
        f.write(f"atol: {DEFAULT_ATOL} rtol: {DEFAULT_RTOL}\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
        f.write("\nactual head:\n")
        f.write(np.array2string(rf[:4, :8], precision=6))
        f.write("\n\ngolden head:\n")
        f.write(np.array2string(gf[:4, :8], precision=6))
        f.write("\n")
    metrics["report"] = str(report)
    return passed, metrics


def check_one(elf_text: str, args):
    elf = Path(elf_text).expanduser().resolve()
    if not elf.is_file():
        return False, {"elf": str(elf), "reason": "ELF does not exist"}
    try:
        shape = parse_shape(elf)
        case_dir, golden = prepare_case(elf, shape, args)
    except (ValueError, RuntimeError) as exc:
        return False, {"elf": str(elf), "reason": str(exc)}

    status, rc, out = run_gfrun(elf, args)
    stats = collect_gfrun_stats(out)
    if status != "pass":
        (case_dir / "gfrun.log").write_text(out, encoding="utf-8")
        return False, {"elf": str(elf), "shape": shape,
                       "run_status": status, "returncode": rc, "stats": stats}

    passed, metrics = compare_result(case_dir, golden)
    return passed, {"elf": str(elf), "shape": shape, "run_status": status,
                    "stats": stats, "compare_status": "pass" if passed else "fail",
                    "metrics": metrics}


def collect_elfs(args):
    if args.elf:
        return [args.elf]
    with open(args.list, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]


def main():
    ap = argparse.ArgumentParser(
        description="matmul_test fp16 GEMM precision/performance verification")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("-d", "--elf", help="single matmul_test ELF")
    src.add_argument("-l", "--list", help="text file containing ELF paths")
    ap.add_argument("--ones", action="store_true",
                    help="deterministic all-one inputs (golden == K)")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--input-scale", type=float, default=0.1)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--multi-thread", action="store_true",
                    help="append -s softcore.multiThreadNum=4 for 4-PE kernels "
                         "(matmul_test_mt)")
    ap.add_argument("--gfrun", default=DEFAULT_GFRUN)
    ap.add_argument("--gfrun-args", default=DEFAULT_GFRUN_ARGS)
    ap.add_argument("--gfsim", default=None,
                    help="optional gfsim binary for cycle counts (see ISSUE-003)")
    args = ap.parse_args()

    elfs = collect_elfs(args)
    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(check_one, e, args) for e in elfs]
        for fut in as_completed(futures):
            passed, details = fut.result()
            results.append(passed)
            line = f"{'PASS' if passed else 'FAIL'}: {details.get('elf', '?')}"
            if "shape" in details:
                s = details["shape"]
                line += (f" M{s['M']} N{s['N']} K{s['K']} "
                         f"tM{s['tM']} tN{s['tN']} tK{s['tK']}")
            st = details.get("stats", {})
            if st.get("blocks") is not None:
                line += f" blocks={st['blocks']} insts={st['insts']} end={st.get('reached_end')}"
            mt = details.get("metrics")
            if mt:
                line += (f" max_abs={mt['max_abs']:.4f} mse={mt['mse']:.4e} "
                         f"uncomputed={mt['uncomputed_cells']}/{mt['nonzero_golden_cells']}")
            print(line)
            if isinstance(details.get("reason"), str):
                print(f"    reason: {details['reason']}")

    ok = sum(results)
    print(f"summary: pass={ok}, fail={len(results) - ok}")
    return 0 if results and all(results) else 1


if __name__ == "__main__":
    sys.exit(main())