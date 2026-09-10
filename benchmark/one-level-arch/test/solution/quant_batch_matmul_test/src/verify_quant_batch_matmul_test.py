#!/usr/bin/env python3
"""Numerical (precision) and workload verification for the
quant_batch_matmul_test kernel.

quant_batch_matmul_test contract:
  * fp4 e2m1x2 (__fp4_e2m1x2) inputs A[M,K], B[K,N]
  * MX e8m0 scaling per 32-element group (smatrix_wfactor=32)
  * fp32 accumulate, fp32 output C[M,N]
  * Multi-block: NBLOCKS PEs each process blockM = M / NBLOCKS rows
  * Built with res_check=on reads src0.bin / src1.bin (fp4 packed) +
    src0_mx.bin / src1_mx.bin (e8m0) from CHK_DIR and writes res.bin (fp32)

Golden semantics:
  * Quantize float32 inputs to fp4 with per-32-group e8m0 scaling
  * Reconstruct float32 from quantized data (dequant)
  * Compute C = A_dec @ B_dec in float32

Usage:
  python3 verify_quant_batch_matmul_test.py -d <single elf> \\
      --gfrun /path/to/SuperScalarModel/bin/gfrun --gfrun-args "-t 1 -f" \\
      [--seed 42] [--input-scale 0.5] [--ones]
  python3 verify_quant_batch_matmul_test.py -l elf_list.txt --gfrun ...

The script reports mse, max_abs, mean_abs, and a PASS/FAIL verdict based on
atol/rtol against the fp32 golden.
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

DEFAULT_GFRUN = "/home/j00833640/v300/SuperScalarModel/bin/gfrun"
DEFAULT_GFRUN_ARGS = "-t 1 -f"

# fp4 accumulates in fp32; expect near-perfect matches modulo fp4 quant noise.
DEFAULT_ATOL = 2e-2
DEFAULT_RTOL = 2e-2

MX_BLOCK = 32  # smatrix_wfactor: scale group size

# fp4_e2m1x2 code -> float32
_FP4 = np.array([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=np.float32)


def _e8m0_decode(codes):
    return np.where(codes == 0, 2.0 ** -127,
                    2.0 ** (codes.astype(np.float32) - 127))


def _e8m0_encode(vals):
    out = np.zeros(vals.shape, dtype=np.uint8)
    ok = vals > 2.0 ** -127
    out[ok] = (np.log2(vals[ok]) + 127).astype(np.int32).clip(1, 255)
    return out


def _quant_fp4(vals):
    vals = np.asarray(vals, dtype=np.float32)
    codes = np.zeros(len(vals), dtype=np.uint8)
    best_err = np.full(len(vals), np.inf, dtype=np.float32)
    for c in range(16):
        err = np.abs(vals - _FP4[c])
        mask = err < best_err
        codes[mask] = c
        best_err[mask] = err[mask]
    return codes


def _pack_row(lo, hi):
    return lo | (hi << 4)


def _unpack_row(packed):
    lo = packed & 0xF
    hi = (packed >> 4) & 0xF
    out = np.empty(len(packed) * 2, dtype=np.float32)
    out[0::2] = _FP4[lo]
    out[1::2] = _FP4[hi]
    return out


def parse_shape(elf):
    name = elf.stem
    m = re.search(
        r"_M(?P<M>\d+)_N(?P<N>\d+)_K(?P<K>\d+)"
        r"_tM(?P<tM>\d+)_tN(?P<tN>\d+)_tK(?P<tK>\d+)$",
        name,
    )
    if not m:
        raise ValueError(f"cannot parse shape from ELF name: {name}")
    shape = {k: int(v) for k, v in m.groupdict().items()}
    shape["Kv"] = shape["K"] // 2  # fp4 packed carriers (= K logical / 2)
    shape["name"] = name
    return shape


def prepare_case(elf, shape, args):
    case_dir = COMPARE_ROOT / shape["name"]
    case_dir.mkdir(parents=True, exist_ok=True)
    M, N, K = shape["M"], shape["N"], shape["K"]
    Kv, Kb = shape["Kv"], K // MX_BLOCK

    rng = np.random.default_rng(args.seed)
    if args.ones:
        A_f32 = np.ones((M, K), dtype=np.float32)
        B_f32 = np.ones((K, N), dtype=np.float32)
    else:
        A_f32 = rng.standard_normal((M, K)).astype(np.float32) * args.input_scale
        B_f32 = rng.standard_normal((K, N)).astype(np.float32) * args.input_scale
        np.clip(A_f32, -6.0, 6.0, out=A_f32)
        np.clip(B_f32, -6.0, 6.0, out=B_f32)

    A_packed = np.zeros((M, Kv), dtype=np.uint8)
    A_scales = np.zeros((M, Kb), dtype=np.uint8)
    for m in range(M):
        row = A_f32[m]
        for kb in range(Kb):
            block = row[kb * MX_BLOCK : (kb + 1) * MX_BLOCK]
            mx = np.max(np.abs(block))
            A_scales[m, kb] = 0 if mx < 1e-8 else _e8m0_encode(np.array([6.0 / mx]))[0]
            sv = float(_e8m0_decode(np.array([A_scales[m, kb]]))[0])
            q = block / sv
            codes = _quant_fp4(q)
            h = (kb + 1) * MX_BLOCK // 2
            A_packed[m, kb * (MX_BLOCK // 2) : h] = _pack_row(codes[0::2], codes[1::2])

    B_packed = np.zeros((Kv, N), dtype=np.uint8)
    B_scales = np.zeros((Kb, N), dtype=np.uint8)
    for kb in range(Kb):
        bs = slice(kb * MX_BLOCK, (kb + 1) * MX_BLOCK)
        for n in range(N):
            block = B_f32[bs, n]
            mx = np.max(np.abs(block))
            B_scales[kb, n] = 0 if mx < 1e-8 else _e8m0_encode(np.array([6.0 / mx]))[0]
            sv = float(_e8m0_decode(np.array([B_scales[kb, n]]))[0])
            q = block / sv
            codes = _quant_fp4(q)
            for kk in range(MX_BLOCK // 2):
                k = kb * MX_BLOCK + kk * 2
                B_packed[k // 2, n] = (codes[kk * 2] & 0xF) | ((codes[kk * 2 + 1] & 0xF) << 4)

    A_packed.tofile(case_dir / "src0.bin")
    B_packed.tofile(case_dir / "src1.bin")
    A_scales.tofile(case_dir / "src0_mx.bin")
    B_scales.tofile(case_dir / "src1_mx.bin")

    A_dec = np.zeros((M, K), dtype=np.float32)
    for m in range(M):
        f32_row = _unpack_row(A_packed[m])
        A_dec[m] = f32_row
        for kb in range(Kb):
            sl = slice(kb * MX_BLOCK, (kb + 1) * MX_BLOCK)
            A_dec[m, sl] *= float(_e8m0_decode(np.array([A_scales[m, kb]]))[0])

    B_dec = np.zeros((K, N), dtype=np.float32)
    for n in range(N):
        for k in range(K):
            packed = int(B_packed[k // 2, n])
            code = (packed & 0xF) if (k % 2 == 0) else ((packed >> 4) & 0xF)
            B_dec[k, n] = _FP4[code] * float(
                _e8m0_decode(np.array([B_scales[k // MX_BLOCK, n]]))[0])

    golden = A_dec @ B_dec
    golden.astype(np.float32).tofile(case_dir / "golden.bin")
    np.zeros(M * N, dtype=np.float32).tofile(case_dir / "res.bin")
    return case_dir, golden


def run_gfrun(elf, args):
    command = [args.gfrun, *shlex.split(args.gfrun_args), str(elf)]
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


def collect_gfrun_stats(out):
    blk = re.search(r"Total Block number = (\d+)", out)
    inst = re.search(r"Total Inst number = (\d+)", out)
    return {
        "blocks": int(blk.group(1)) if blk else None,
        "insts": int(inst.group(1)) if inst else None,
        "reached_end": "Reach the End of Benchmark" in out,
    }


def compare_result(case_dir, golden):
    rp = case_dir / "res.bin"
    if not rp.exists():
        return False, {"reason": "res.bin not created"}
    result = np.fromfile(rp, dtype=np.float32)
    if result.size != golden.size:
        return False, {"reason": f"size mismatch: {result.size} vs {golden.size}"}
    result = result.reshape(golden.shape)
    diff = result - golden
    abs_diff = np.abs(diff)
    mi = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
    nonzero_mask = golden != 0.0
    uncomputed = int(((result == 0.0) & nonzero_mask).sum())
    covered = int(nonzero_mask.sum())

    passed = bool(np.allclose(result, golden, atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL))
    metrics = {
        "mse": float(np.mean(diff * diff)),
        "max_abs": float(abs_diff[mi]),
        "max_index": str(mi),
        "actual_at_max": float(result[mi]),
        "golden_at_max": float(golden[mi]),
        "mean_abs": float(np.mean(abs_diff)),
        "uncomputed_cells": int(uncomputed),
        "nonzero_golden_cells": int(covered),
        "exact_matches": int((result == golden).sum()),
        "total_cells": int(result.size),
    }

    report = case_dir / "verify_quant_batch_matmul_test.log"
    with report.open("w") as f:
        f.write(f"status: {'PASS' if passed else 'FAIL'}\n")
        f.write(f"atol: {DEFAULT_ATOL} rtol: {DEFAULT_RTOL}\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
        f.write("\nactual head:\n")
        f.write(np.array2string(result[:4, :8], precision=6, suppress_small=True))
        f.write("\n\ngolden head:\n")
        f.write(np.array2string(golden[:4, :8], precision=6, suppress_small=True))
        f.write("\n")
    metrics["report"] = str(report)
    return passed, metrics


def check_one(elf_text, args):
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
    with open(args.list, "r") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]


def main():
    ap = argparse.ArgumentParser(
        description="quant_batch_matmul fp4 MX matmul precision/performance verification")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("-d", "--elf", help="single quant_batch_matmul_test ELF")
    src.add_argument("-l", "--list", help="text file containing ELF paths")
    ap.add_argument("--ones", action="store_true",
                    help="deterministic all-one inputs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--input-scale", type=float, default=0.5)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--gfrun", default=DEFAULT_GFRUN)
    ap.add_argument("--gfrun-args", default=DEFAULT_GFRUN_ARGS)
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
                line += f" blocks={st['blocks']} insts={st['insts']}"
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
