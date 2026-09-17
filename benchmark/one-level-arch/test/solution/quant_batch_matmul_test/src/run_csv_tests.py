#!/usr/bin/env python3
"""CSV-driven test framework for mxfp4_mt and hif4 4-PE cooperative kernels.

Reads a CSV table of test cases, for each case:
  1. Compiles the kernel ELF (make ... res_check=on)
  2. Generates random input data + golden reference (numpy fp32)
  3. Runs gfrun with 4-PE cooperative mode
  4. Compares gfrun output vs golden (atol/rtol)
  5. Reports PASS/FAIL per case and summary

CSV columns: name,dtype,M,N,K,tM,tN,tK,B
  dtype: "mxfp4" → quant_batch_matmul_test_mxfp4_mt
         "hif4"  → quant_batch_matmul_test_hif4

Usage:
  python3 run_csv_tests.py [--csv testcases.csv] [--seed 42]
      [--input-scale 0.5] [--timeout 300] [--workers 1]
      [--gfrun /path/to/gfrun] [--compiler-dir /path/to/toolchain/bin]
      [--keep-elfs]  # don't clean ELF objects between cases
"""

import argparse
import csv
import math
import os
import re
import shlex
import signal
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
TEST_DIR = SCRIPT_DIR.parent  # quant_batch_matmul_test/
ONE_LEVEL_ROOT = TEST_DIR.parents[2]  # one-level-arch/
COMPARE_ROOT = ONE_LEVEL_ROOT / "compare"  # one-level-arch/compare/
OUTPUT_ROOT = ONE_LEVEL_ROOT / "output"  # one-level-arch/output/

DEFAULT_GFRUN = "/home/jtt/v300/SuperScalarModel/bin/gfrun"
DEFAULT_COMPILER_DIR = "/home/jtt/v300/linx-toolchain-build/output/linx_blockisa_llvm_musl/bin"
DEFAULT_GFRUN_ARGS = "-t 1 -s softcore.multiThreadNum=4 -f"
DEFAULT_ATOL = 5e-2
DEFAULT_RTOL = 5e-2

# ---------------------------------------------------------------------------
# FP4 codebooks (matching gfrun CubeEngine DataFormatCvt lookup tables)
# ---------------------------------------------------------------------------

# E2M1 (mxfp4): bit[3]=sign, bits[2:0] index into {0,0.5,1,1.5,2,3,4,6}
_FP4_E2M1 = np.array([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=np.float32)

# HiF4 (E1M2): bit[3]=sign, bits[2:0] index into {0,0.25,0.5,0.75,1,1.25,1.5,1.75}
_FP4_HIF4 = np.array([
    0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75,
    -0.0, -0.25, -0.5, -0.75, -1.0, -1.25, -1.5, -1.75,
], dtype=np.float32)

PACKED_FACTOR = 2  # __fp4_e2m1x2 / __fp4_hif4x2: 2 logical elements per byte

# Scale group sizes (logical K elements)
MXFP4_SCALE_GROUP = 32   # 32 logical K per E8M0 byte
HIF4_SCALE_GROUP = 64    # 64 logical K per U32 word


# ---------------------------------------------------------------------------
# E8M0 encode/decode (for mxfp4)
# ---------------------------------------------------------------------------

def _e8m0_decode(codes):
    return np.where(codes == 0, 2.0 ** -127,
                    2.0 ** (codes.astype(np.float32) - 127))

def _e8m0_encode(vals):
    out = np.zeros(vals.shape, dtype=np.uint8)
    ok = vals > 2.0 ** -127
    out[ok] = (np.log2(vals[ok]) + 127).astype(np.int32).clip(1, 255)
    return out


# ---------------------------------------------------------------------------
# E6M2 encode/decode (for hif4 U32 scale)
# ---------------------------------------------------------------------------

def _e6m2_decode_byte(base):
    base = int(base) & 0xFF
    if base == 0xFF:
        return float('nan')
    exp = ((base >> 2) & 0x3F) - 48
    mant = base & 0x3
    sig = 1.0 + ((mant >> 1) & 1) * 0.5 + (mant & 1) * 0.25
    return math.ldexp(sig, exp)

def _e6m2_encode(val):
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
# U32 scale decode (for hif4, matches MatrixScaleToFP32)
# ---------------------------------------------------------------------------

def _u32_scale_decode(raw, lane):
    raw = int(raw) & 0xFFFFFFFF
    base = raw & 0xFF
    if base == 0xFF:
        return float('nan')
    e6m2_val = _e6m2_decode_byte(base)
    e1_8_bit = (raw >> (8 + lane // 8)) & 1
    e1_16_bit = (raw >> (16 + lane // 4)) & 1
    return e6m2_val * math.ldexp(1.0, e1_8_bit + e1_16_bit)


# ---------------------------------------------------------------------------
# Quantize / pack helpers
# ---------------------------------------------------------------------------

def _quant_fp4(vals, codebook):
    vals = np.asarray(vals, dtype=np.float32)
    codes = np.zeros(len(vals), dtype=np.uint8)
    best_err = np.full(len(vals), np.inf, dtype=np.float32)
    for c in range(16):
        err = np.abs(vals - codebook[c])
        mask = err < best_err
        codes[mask] = c
        best_err[mask] = err[mask]
    return codes

def _pack_row(lo_codes, hi_codes):
    return (lo_codes & 0xF) | ((hi_codes & 0xF) << 4)


# ---------------------------------------------------------------------------
# Build ELF
# ---------------------------------------------------------------------------

def build_elf(case, args):
    testcase = "quant_batch_matmul_test_mxfp4_mt" if case["dtype"] == "mxfp4" else "quant_batch_matmul_test_hif4"
    # ELF_HEAD = solution_quant_batch_matmul_test, TARGET = ELF_HEAD_TESTCASE_B...
    # ELF_HEAD already contains "quant_batch_matmul_test", and so does TESTCASE,
    # producing a double prefix in the actual ELF name.
    elf_name = f"solution_quant_batch_matmul_test_{testcase}_B{case['B']}_M{case['M']}_N{case['N']}_K{case['K']}_tM{case['tM']}_tN{case['tN']}_tK{case['tK']}.elf"
    elf_path = OUTPUT_ROOT / "solution" / "quant_batch_matmul_test" / "elf" / elf_name

    cmd = [
        "make", f"TESTCASE={testcase}",
        f"M={case['M']}", f"N={case['N']}", f"K={case['K']}",
        f"tM={case['tM']}", f"tN={case['tN']}", f"tK={case['tK']}",
        f"B={case['B']}", "NBLOCKS=32", "PLAT=linx",
        f"COMPILER_DIR={args.compiler_dir}",
        "res_check=on",
    ]
    proc = subprocess.run(cmd, cwd=str(TEST_DIR), capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        return None, f"compile failed: {proc.stderr[-500:]}"
    if not elf_path.exists():
        # Try to find the ELF
        for root, dirs, files in os.walk(str(TEST_DIR / "output")):
            for f in files:
                if f == elf_name:
                    elf_path = Path(root) / f
                    return elf_path, None
        return None, f"ELF not found after build: {elf_name}"
    return elf_path, None


# ---------------------------------------------------------------------------
# Generate input data + golden
# ---------------------------------------------------------------------------

def prepare_case(case, elf_path, args):
    name = case["name"]
    dtype = case["dtype"]
    M, N, K = int(case["M"]), int(case["N"]), int(case["K"])
    B = int(case["B"])
    Kv = K // PACKED_FACTOR

    case_dir = COMPARE_ROOT / f"csv_{name}"
    case_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    A_f32 = rng.standard_normal((B, M, K)).astype(np.float32) * args.input_scale
    B_f32 = rng.standard_normal((B, K, N)).astype(np.float32) * args.input_scale

    is_hif4 = (dtype == "hif4")
    codebook = _FP4_HIF4 if is_hif4 else _FP4_E2M1
    scale_group = HIF4_SCALE_GROUP if is_hif4 else MXFP4_SCALE_GROUP
    max_val = 1.75 if is_hif4 else 6.0

    np.clip(A_f32, -max_val, max_val, out=A_f32)
    np.clip(B_f32, -max_val, max_val, out=B_f32)

    A_packed_all = []
    B_packed_all = []
    A_scales_all = []
    B_scales_all = []
    A_dec_all = []
    B_dec_all = []

    for b in range(B):
        A_f = A_f32[b]  # [M, K]
        B_f = B_f32[b]  # [K, N]

        Kb = K // scale_group
        A_packed = np.zeros((M, Kv), dtype=np.uint8)
        A_scales = np.zeros((M, Kb), dtype=np.uint32 if is_hif4 else np.uint8)
        A_dec = np.zeros((M, K), dtype=np.float32)

        for m in range(M):
            for kb in range(Kb):
                block = A_f[m, kb*scale_group:(kb+1)*scale_group]
                mx = np.max(np.abs(block))
                if is_hif4:
                    sv_raw = mx / 3.5 if mx > 1e-8 else 1.0
                    e6m2_byte = _e6m2_encode(sv_raw)
                    A_scales[m, kb] = np.uint32(e6m2_byte)
                    actual_sv = _e6m2_decode_byte(e6m2_byte)
                else:
                    sv_raw = max_val / mx if mx > 1e-8 else 1.0
                    A_scales[m, kb] = _e8m0_encode(np.array([sv_raw]))[0]
                    actual_sv = float(_e8m0_decode(np.array([A_scales[m, kb]]))[0])
                q = block / actual_sv if actual_sv > 0 else np.zeros_like(block)
                codes = _quant_fp4(q, codebook)
                h = (kb+1) * scale_group // 2
                A_packed[m, kb*(scale_group//2):h] = _pack_row(codes[0::2], codes[1::2])
                # golden dequant
                for lane in range(scale_group):
                    packed = int(A_packed[m, kb*(scale_group//2) + lane//2])
                    code = (packed & 0xF) if (lane%2==0) else ((packed>>4)&0xF)
                    if is_hif4:
                        sv = _u32_scale_decode(int(A_scales[m, kb]), lane)
                    else:
                        sv = actual_sv
                    A_dec[m, kb*scale_group + lane] = codebook[code] * sv

        # B: stored as [N, Kv] (TransB=0), B scale [N, Kb]
        B_packed = np.zeros((N, Kv), dtype=np.uint8)
        B_scales = np.zeros((N, Kb), dtype=np.uint32 if is_hif4 else np.uint8)
        B_dec = np.zeros((K, N), dtype=np.float32)

        for n in range(N):
            for kb in range(Kb):
                block = B_f[kb*scale_group:(kb+1)*scale_group, n]
                mx = np.max(np.abs(block))
                if is_hif4:
                    sv_raw = mx / 3.5 if mx > 1e-8 else 1.0
                    e6m2_byte = _e6m2_encode(sv_raw)
                    B_scales[n, kb] = np.uint32(e6m2_byte)
                    actual_sv = _e6m2_decode_byte(e6m2_byte)
                else:
                    sv_raw = max_val / mx if mx > 1e-8 else 1.0
                    B_scales[n, kb] = _e8m0_encode(np.array([sv_raw]))[0]
                    actual_sv = float(_e8m0_decode(np.array([B_scales[n, kb]]))[0])
                q = block / actual_sv if actual_sv > 0 else np.zeros_like(block)
                codes = _quant_fp4(q, codebook)
                for kk in range(scale_group // 2):
                    B_packed[n, kb*(scale_group//2)+kk] = \
                        (codes[kk*2]&0xF) | ((codes[kk*2+1]&0xF)<<4)
                # golden dequant
                for lane in range(scale_group):
                    packed = int(B_packed[n, kb*(scale_group//2)+lane//2])
                    code = (packed&0xF) if (lane%2==0) else ((packed>>4)&0xF)
                    if is_hif4:
                        sv = _u32_scale_decode(int(B_scales[n, kb]), lane)
                    else:
                        sv = actual_sv
                    B_dec[kb*scale_group+lane, n] = codebook[code] * sv

        golden = A_dec @ B_dec

        A_packed_all.append(A_packed)
        B_packed_all.append(B_packed)
        A_scales_all.append(A_scales)
        B_scales_all.append(B_scales)
        A_dec_all.append(A_dec)
        B_dec_all.append(B_dec)

    golden = np.stack([A_dec_all[b] @ B_dec_all[b] for b in range(B)])  # [B, M, N]

    # Write binaries (batch-flattened)
    np.concatenate(A_packed_all).tofile(case_dir / "src0.bin")
    np.concatenate(B_packed_all).tofile(case_dir / "src1.bin")
    np.concatenate(A_scales_all).tofile(case_dir / "src0_mx.bin")
    np.concatenate(B_scales_all).tofile(case_dir / "src1_mx.bin")
    golden.astype(np.float32).tofile(case_dir / "golden.bin")
    np.zeros(B * M * N, dtype=np.float32).tofile(case_dir / "res.bin")

    # Write CHK_DIR path marker
    (case_dir / "CHK_DIR.txt").write_text(str(case_dir))
    return case_dir, golden


# ---------------------------------------------------------------------------
# Run gfrun + compare
# ---------------------------------------------------------------------------

def run_gfrun(elf_path, case_dir, args):
    elf_name = elf_path.stem
    # CHK_DIR is baked into the ELF as: COMPARE_ROOT/elf_name
    actual_chk_dir = COMPARE_ROOT / elf_name
    actual_chk_dir.mkdir(parents=True, exist_ok=True)

    # Copy our generated files to the expected CHK_DIR
    for fname in ["src0.bin", "src1.bin", "src0_mx.bin", "src1_mx.bin", "res.bin"]:
        src = case_dir / fname
        dst = actual_chk_dir / fname
        if src.exists():
            import shutil
            shutil.copy2(str(src), str(dst))

    command = [args.gfrun, *shlex.split(args.gfrun_args), str(elf_path)]
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


def compare_result(elf_path, golden):
    elf_name = elf_path.stem
    chk_dir = COMPARE_ROOT / elf_name
    rp = chk_dir / "res.bin"
    if not rp.exists():
        return False, {"reason": "res.bin not created"}
    result = np.fromfile(rp, dtype=np.float32)
    golden_flat = golden.reshape(-1)
    if result.size != golden_flat.size:
        return False, {"reason": f"size mismatch: {result.size} vs {golden_flat.size}"}
    result = result.reshape(golden.shape)
    diff = result - golden
    abs_diff = np.abs(diff)
    passed = bool(np.allclose(result, golden, atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL))
    metrics = {
        "max_abs": float(abs_diff.max()),
        "mse": float(np.mean(diff * diff)),
        "exact": int((result == golden).sum()),
        "total": int(result.size),
    }
    return passed, metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="CSV-driven mxfp4/hif4 kernel test framework")
    ap.add_argument("--csv", default=str(TEST_DIR / "testcases.csv"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--input-scale", type=float, default=0.5)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--gfrun", default=DEFAULT_GFRUN)
    ap.add_argument("--gfrun-args", default=DEFAULT_GFRUN_ARGS)
    ap.add_argument("--compiler-dir", default=DEFAULT_COMPILER_DIR)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--keep-elfs", action="store_true")
    args = ap.parse_args()

    with open(args.csv) as f:
        cases = list(csv.DictReader(f))

    print(f"Loaded {len(cases)} test cases from {args.csv}")
    print(f"{'CASE':<30} {'DTYPE':<8} {'M':>5} {'N':>5} {'K':>5} {'tM':>4} {'tN':>4} {'tK':>4} {'B':>3}  STATUS  max_abs   mse")
    print("-" * 110)

    results = []
    for case in cases:
        name = case["name"]
        dtype = case["dtype"]
        M, N, K = case["M"], case["N"], case["K"]
        tM, tN, tK = case["tM"], case["tN"], case["tK"]
        B = case["B"]

        # Build
        elf_path, err = build_elf(case, args)
        if err:
            print(f"{name:<30} {dtype:<8} {M:>5} {N:>5} {K:>5} {tM:>4} {tN:>4} {tK:>4} {B:>3}  BUILD_ERR {err}")
            results.append(False)
            continue

        # Prepare
        try:
            case_dir, golden = prepare_case(case, elf_path, args)
        except Exception as e:
            print(f"{name:<30} {dtype:<8} {M:>5} {N:>5} {K:>5} {tM:>4} {tN:>4} {tK:>4} {B:>3}  PREP_ERR  {e}")
            results.append(False)
            continue

        # Run
        status, rc, out = run_gfrun(elf_path, case_dir, args)
        if status != "pass":
            print(f"{name:<30} {dtype:<8} {M:>5} {N:>5} {K:>5} {tM:>4} {tN:>4} {tK:>4} {B:>3}  RUN_FAIL  (rc={rc})")
            results.append(False)
            continue

        # Compare
        passed, metrics = compare_result(elf_path, golden)
        if isinstance(metrics, dict) and "max_abs" in metrics:
            print(f"{name:<30} {dtype:<8} {M:>5} {N:>5} {K:>5} {tM:>4} {tN:>4} {tK:>4} {B:>3}  "
                  f"{'PASS' if passed else 'FAIL':>7}  {metrics['max_abs']:.4f}  {metrics['mse']:.4e}")
        else:
            print(f"{name:<30} {dtype:<8} {M:>5} {N:>5} {K:>5} {tM:>4} {tN:>4} {tK:>4} {B:>3}  "
                  f"{'PASS' if passed else 'FAIL':>7}  {metrics}")
        results.append(passed)

    print("-" * 110)
    print(f"summary: pass={sum(results)}, fail={len(results)-sum(results)}, total={len(results)}")
    return 0 if results and all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
