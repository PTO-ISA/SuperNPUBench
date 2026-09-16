#!/usr/bin/env python3
"""Numerical (precision) verification for the
quant_batch_matmul_test_hif4 (HiF4X2 / hifloat4) 4-PE cooperative kernel.

quant_batch_matmul_test_hif4 contract:
  * fp4 hif4x2 (__fp4_hif4x2) inputs A[M,K], B[K,N]
  * U32 (uint32_t) scaling per 64-element group (HiF4 Matrix-MX)
  * fp32 accumulate, fp32 output C[M,N]
  * 4-PE cooperative: SharedTile A/B, each PE writes [tM/4, tN] rows
  * Built with res_check=on reads src0.bin / src1.bin (fp4 packed) +
    src0_mx.bin / src1_mx.bin (uint32_t U32 scale) from CHK_DIR and
    writes res.bin (fp32)

HiF4 (E1M2) codebook (4-bit nibble, bit[3]=sign, bits[2:0]=magnitude):
  {0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75} with sign

U32 scale word layout (per 64 logical K elements):
  [7:0]   = E6M2:  6 exp bits [7:2] (bias 48) + 2 mantissa bits [1:0]
  [15:8]  = E1_8:  8 single-bit exponent adjustments (1 bit per 8 elements)
  [31:16] = E1_16: 16 single-bit exponent adjustments (1 bit per 4 elements)

  Full decode: value = ((4 + mant) / 4.0) * 2^(exp - 48 + E1_8_bit + E1_16_bit)

Usage:
  python3 verify_hif4.py -d <single elf> \\
      --gfrun /path/to/SuperScalarModel/bin/gfrun \\
      --gfrun-args "-t 1 -s softcore.multiThreadNum=4 -f" \\
      [--seed 42] [--input-scale 0.5] [--ones]
"""

import argparse
import math
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

DEFAULT_GFRUN = "/home/jtt/v300/SuperScalarModel/bin/gfrun"
DEFAULT_GFRUN_ARGS = "-t 1 -s softcore.multiThreadNum=4 -f"

DEFAULT_ATOL = 5e-2
DEFAULT_RTOL = 5e-2

PACKED_FACTOR = 2      # __fp4_hif4x2: 2 logical elements per carrier byte
SCALE_GROUP = 64       # HiF4 U32 scale: 64 logical K elements per U32 word


# ---------------------------------------------------------------------------
# HiF4 (E1M2) codebook — matches CubeEngine.cpp HIF4_VALUES lookup table
# used by DataFormatCvt (the cooperative path's FP4→FP32 conversion).
#   bit[3]=sign, bits[2:0]=magnitude index into {0,0.25,0.5,0.75,1,1.25,1.5,1.75}
# ---------------------------------------------------------------------------

_HIF4 = np.array([
    0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75,
    -0.0, -0.25, -0.5, -0.75, -1.0, -1.25, -1.5, -1.75,
], dtype=np.float32)


# ---------------------------------------------------------------------------
# E6M2 decode (matches CubeCalculate::E6m2ToF32Bits / MatrixScaleToFP32)
# ---------------------------------------------------------------------------

def _e6m2_decode_byte(base):
    """Decode E6M2 byte to float32."""
    base = int(base) & 0xFF
    if base == 0xFF:
        return float('nan')
    exp = ((base >> 2) & 0x3F) - 48
    mant = base & 0x3
    sig = 1.0 + ((mant >> 1) & 1) * 0.5 + (mant & 1) * 0.25
    return math.ldexp(sig, exp)


def _e6m2_encode(val):
    """Encode a positive float to the nearest E6M2 byte (brute-force, 256 candidates)."""
    if val <= 0.0 or math.isnan(val) or math.isinf(val):
        return 0
    best_byte = 0
    best_err = abs(val)
    for b in range(256):
        decoded = _e6m2_decode_byte(b)
        if math.isnan(decoded):
            continue
        err = abs(val - decoded)
        if err < best_err:
            best_err = err
            best_byte = b
    return best_byte


# ---------------------------------------------------------------------------
# U32 scale word decode (matches CubeEngine.cpp MatrixScaleToFP32 for HIF4)
# ---------------------------------------------------------------------------

def _u32_scale_decode(raw, lane):
    """Decode a single U32 scale word for a given lane (0..63 within group).

    Matches MatrixScaleToFP32(DataType::HIF4, raw, lane):
      base = raw & 0xFF (E6M2)
      e1_8_bit = (raw >> (8 + lane//8)) & 1
      e1_16_bit = (raw >> (16 + lane//4)) & 1
      value = E6M2_value * 2^(e1_8_bit + e1_16_bit)
    """
    raw = int(raw) & 0xFFFFFFFF
    base = raw & 0xFF
    if base == 0xFF:
        return float('nan')
    e6m2_val = _e6m2_decode_byte(base)
    e1_8_bit = (raw >> (8 + lane // 8)) & 1
    e1_16_bit = (raw >> (16 + lane // 4)) & 1
    return e6m2_val * math.ldexp(1.0, e1_8_bit + e1_16_bit)


# ---------------------------------------------------------------------------
# HiF4 quantize / pack
# ---------------------------------------------------------------------------

def _quant_hif4(vals):
    """Quantize float32 array to nearest HiF4 4-bit codes (0..15)."""
    vals = np.asarray(vals, dtype=np.float32)
    codes = np.zeros(len(vals), dtype=np.uint8)
    best_err = np.full(len(vals), np.inf, dtype=np.float32)
    for c in range(16):
        err = np.abs(vals - _HIF4[c])
        mask = err < best_err
        codes[mask] = c
        best_err[mask] = err[mask]
    return codes


def _pack_row(lo_codes, hi_codes):
    """Pack two 4-bit codes into one byte (low nibble = even K, high = odd K)."""
    return (lo_codes & 0xF) | ((hi_codes & 0xF) << 4)


def _unpack_row(packed):
    """Unpack bytes to HiF4 float32 values (2 per byte)."""
    lo = packed & 0xF
    hi = (packed >> 4) & 0xF
    out = np.empty(len(packed) * 2, dtype=np.float32)
    out[0::2] = _HIF4[lo]
    out[1::2] = _HIF4[hi]
    return out


# ---------------------------------------------------------------------------
# Shape parsing
# ---------------------------------------------------------------------------

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
    shape["Kv"] = shape["K"] // PACKED_FACTOR  # packed carriers
    shape["name"] = name
    return shape


# ---------------------------------------------------------------------------
# Prepare test case: generate inputs, quantize, write binaries, compute golden
# ---------------------------------------------------------------------------

def prepare_case(elf, shape, args):
    case_dir = COMPARE_ROOT / shape["name"]
    case_dir.mkdir(parents=True, exist_ok=True)
    M, N, K = shape["M"], shape["N"], shape["K"]
    Kv = shape["Kv"]
    Kblocks = K // SCALE_GROUP  # = K / 64

    rng = np.random.default_rng(args.seed)
    if args.ones:
        A_f32 = np.ones((M, K), dtype=np.float32)
        B_f32 = np.ones((K, N), dtype=np.float32)
    else:
        A_f32 = rng.standard_normal((M, K)).astype(np.float32) * args.input_scale
        B_f32 = rng.standard_normal((K, N)).astype(np.float32) * args.input_scale
        np.clip(A_f32, -1.75, 1.75, out=A_f32)
        np.clip(B_f32, -1.75, 1.75, out=B_f32)

    # --- Quantize A [M, K] ---
    # A scale: [M, Kblocks] uint32 (1 word per group: E6M2 in byte0, E1_8 in byte1)
    # A packed: [M, Kv] uint8
    A_packed = np.zeros((M, Kv), dtype=np.uint8)
    A_scales = np.zeros((M, Kblocks), dtype=np.uint32)

    for m in range(M):
        row = A_f32[m]
        for kb in range(Kblocks):
            block = row[kb * SCALE_GROUP:(kb + 1) * SCALE_GROUP]
            mx = np.max(np.abs(block))
            scale_val = mx / 3.5 if mx > 1e-8 else 1.0
            e6m2_byte = _e6m2_encode(scale_val)
            # word = E6M2 byte (bits[7:0]), E1_8 = 0 (bits[15:8])
            A_scales[m, kb] = np.uint32(e6m2_byte)
            # Decode back to get actual scale (may differ from scale_val)
            actual_scale = _e6m2_decode_byte(e6m2_byte)
            q = block / actual_scale if actual_scale > 0 else np.zeros_like(block)
            codes = _quant_hif4(q)
            h = (kb + 1) * SCALE_GROUP // 2
            A_packed[m, kb * (SCALE_GROUP // 2):h] = _pack_row(codes[0::2], codes[1::2])

    # --- Quantize B [K, N] ---
    # B stored as [N, Kv] row-major (PTO TransB=0); B packed: [N, Kv] uint8
    # B scale: [N, Kblocks] uint32 (1 word per group)
    B_packed = np.zeros((N, Kv), dtype=np.uint8)
    B_scales = np.zeros((N, Kblocks), dtype=np.uint32)

    for n in range(N):
        for kb in range(Kblocks):
            ks = slice(kb * SCALE_GROUP, (kb + 1) * SCALE_GROUP)
            block = B_f32[ks, n]
            mx = np.max(np.abs(block))
            scale_val = mx / 3.5 if mx > 1e-8 else 1.0
            e6m2_byte = _e6m2_encode(scale_val)
            B_scales[n, kb] = np.uint32(e6m2_byte)
            actual_scale = _e6m2_decode_byte(e6m2_byte)
            q = block / actual_scale if actual_scale > 0 else np.zeros_like(block)
            codes = _quant_hif4(q)
            # Pack into [N, Kv]: low nibble = even K, high nibble = odd K
            for kk in range(SCALE_GROUP // 2):
                B_packed[n, kb * (SCALE_GROUP // 2) + kk] = \
                    (codes[kk * 2] & 0xF) | ((codes[kk * 2 + 1] & 0xF) << 4)

    # --- Write binaries ---
    A_packed.tofile(case_dir / "src0.bin")
    B_packed.tofile(case_dir / "src1.bin")
    A_scales.tofile(case_dir / "src0_mx.bin")
    B_scales.tofile(case_dir / "src1_mx.bin")

    # --- Compute golden: dequant A and B, then matmul ---
    # The gfrun cooperative path (ExecuteOrSuspendCollective) uses
    # MatrixScaleToFP32 with 1 U32 word per 64-element group (lane = inner % 64).
    A_dec = np.zeros((M, K), dtype=np.float32)
    for m in range(M):
        for kb in range(Kblocks):
            for lane in range(SCALE_GROUP):
                packed = int(A_packed[m, kb * (SCALE_GROUP // 2) + lane // 2])
                code = (packed & 0xF) if (lane % 2 == 0) else ((packed >> 4) & 0xF)
                sv = _u32_scale_decode(int(A_scales[m, kb]), lane)
                A_dec[m, kb * SCALE_GROUP + lane] = _HIF4[code] * sv

    B_dec = np.zeros((K, N), dtype=np.float32)
    for n in range(N):
        for kb in range(Kblocks):
            for lane in range(SCALE_GROUP):
                packed = int(B_packed[n, kb * (SCALE_GROUP // 2) + lane // 2])
                code = (packed & 0xF) if (lane % 2 == 0) else ((packed >> 4) & 0xF)
                sv = _u32_scale_decode(int(B_scales[n, kb]), lane)
                B_dec[kb * SCALE_GROUP + lane, n] = _HIF4[code] * sv

    golden = A_dec @ B_dec
    golden.astype(np.float32).tofile(case_dir / "golden.bin")
    np.zeros(M * N, dtype=np.float32).tofile(case_dir / "res.bin")
    return case_dir, golden


# ---------------------------------------------------------------------------
# Run gfrun and compare
# ---------------------------------------------------------------------------

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

    report = case_dir / "verify_hif4.log"
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
        description="quant_batch_matmul_test_hif4 HiF4X2 4-PE matmul precision verification")
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
