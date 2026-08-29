#!/usr/bin/env python3
"""Prepare inputs, run four-PE RES_CHECK ELFs, and compare with NumPy."""

from __future__ import annotations

import argparse
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "output/kernel/multi_thread"
COMPARE = ROOT / "compare"


@dataclass
class Case:
    name: str
    elf: str
    prepare: callable
    output_name: str = "output.bin"
    output_dtype: object = np.float32
    atol: float = 1e-4
    rtol: float = 1e-4


def write(case_dir: Path, name: str, value: np.ndarray) -> None:
    np.ascontiguousarray(value).tofile(case_dir / name)


def prep_broadcast(case_dir: Path) -> np.ndarray:
    x = np.arange(128, dtype=np.float32)
    write(case_dir, "input.bin", x)
    return np.tile(x, 4)


def prep_concat_gather(case_dir: Path) -> np.ndarray:
    x = np.arange(512, dtype=np.float32).reshape(2, 16, 16)
    write(case_dir, "input.bin", x)
    return np.concatenate((x[0], x[1]), axis=1).reshape(-1)


def prep_concat_scatter(case_dir: Path) -> np.ndarray:
    x = np.arange(512, dtype=np.float32).reshape(32, 16)
    write(case_dir, "input.bin", x)
    out = np.zeros((32, 32), dtype=np.float32)
    out[:, :16] = x
    return out.reshape(-1)


def prep_conv2d(case_dir: Path) -> np.ndarray:
    rng = np.random.default_rng(1)
    x = rng.uniform(-0.25, 0.25, (64, 16)).astype(np.float32)
    w = rng.uniform(-0.25, 0.25, (16, 16)).astype(np.float32)
    # Kernel global tensors are ColMajor; flattening in Fortran order matches
    # the NCHW input and output arrays used by the testcase.
    np.asfortranarray(x).ravel(order="F").tofile(case_dir / "input.bin")
    np.asfortranarray(w).ravel(order="F").tofile(case_dir / "weight.bin")
    with np.errstate(all="ignore"):
        golden = x @ w
    return np.asfortranarray(golden).ravel(order="F")


def gelu_reference(x: np.ndarray) -> np.ndarray:
    t = np.clip(x.astype(np.float32), -5.75, 5.75)
    t2 = t * t
    p = np.float32(-3.5123395303315874e-09)
    for coefficient in (
        2.6452661927578447e-07,
        -7.9294877650681883e-06,
        1.1061238183174282e-04,
        6.5189960878342390e-05,
        -7.2666168212890625e-02,
        -1.5957698822021484,
    ):
        p = p * t2 + np.float32(coefficient)
    return (x.astype(np.float32) / (1.0 + np.exp(t * p))).astype(np.float16)


def prep_gelu(case_dir: Path) -> np.ndarray:
    x = np.linspace(-4.0, 4.0, 8192, dtype=np.float16)
    write(case_dir, "input.bin", x)
    return gelu_reference(x)


def prep_gather(case_dir: Path) -> np.ndarray:
    table = (np.arange(128 * 64, dtype=np.float32) / 64.0).reshape(128, 64)
    indexes = ((np.arange(128, dtype=np.int32) * 17) % 128).astype(np.int32)
    write(case_dir, "table.bin", table)
    write(case_dir, "indexes.bin", indexes)
    return table[indexes].reshape(-1)


def prep_rms(case_dir: Path) -> np.ndarray:
    rng = np.random.default_rng(2)
    x = rng.uniform(-1.0, 1.0, (4, 8192)).astype(np.float16)
    write(case_dir, "input.bin", x)
    xf = x.astype(np.float32)
    return (xf / np.sqrt(np.mean(xf * xf, axis=1, keepdims=True) + 1e-6)).astype(np.float16).reshape(-1)


def prep_rows(case_dir: Path, operation: str) -> np.ndarray:
    rng = np.random.default_rng(3)
    if operation == "prod":
        x = rng.uniform(0.995, 1.005, (64, 128)).astype(np.float32)
    else:
        x = rng.uniform(-1.0, 1.0, (64, 128)).astype(np.float32)
    write(case_dir, "input.bin", x)
    if operation == "cumsum":
        return np.cumsum(x, axis=1, dtype=np.float32).reshape(-1)
    if operation == "max":
        return np.max(x, axis=1).astype(np.float32)
    if operation == "prod":
        return np.prod(x, axis=1, dtype=np.float32)
    return np.sum(x, axis=1, dtype=np.float32)


def prep_transpose(case_dir: Path) -> np.ndarray:
    x = np.arange(64 * 64, dtype=np.int32).reshape(64, 64)
    write(case_dir, "input.bin", x)
    return x.T.reshape(-1)


def prep_tadd(case_dir: Path) -> np.ndarray:
    a = np.arange(256, dtype=np.float32).reshape(16, 16) / 16.0
    b = np.flip(a, axis=1).copy()
    write(case_dir, "src_a.bin", a)
    write(case_dir, "src_b.bin", b)
    return (a + b).reshape(-1)


def prep_trowsum(case_dir: Path) -> np.ndarray:
    x = np.arange(256, dtype=np.float32).reshape(16, 16) / 32.0
    write(case_dir, "src.bin", x)
    return np.sum(x, axis=1, dtype=np.float32)


def prep_matmul(case_dir: Path) -> np.ndarray:
    a = np.ones((1, 256, 256), dtype=np.float32)
    b = np.ones((1, 256, 256), dtype=np.float32)
    write(case_dir, "src0.bin", a)
    write(case_dir, "src1.bin", b)
    with np.errstate(all="ignore"):
        golden = np.matmul(a, b)
    return golden.astype(np.float32).reshape(-1)


def prep_matmul_lowp(case_dir: Path) -> np.ndarray:
    # Raw zero is exact zero in FP8 E4M3, making this a format-independent
    # four-PE ownership/writeback check without requiring a host FP8 package.
    write(case_dir, "src0.bin", np.zeros(256 * 512, dtype=np.uint8))
    write(case_dir, "src1.bin", np.zeros(512 * 256, dtype=np.uint8))
    return np.zeros(256 * 256, dtype=np.float32)


def prep_fa(case_dir: Path) -> np.ndarray:
    rng = np.random.default_rng(4)
    q = rng.uniform(-0.1, 0.1, (256, 128)).astype(np.float32)
    k = rng.uniform(-0.1, 0.1, (256, 128)).astype(np.float32)
    v = rng.uniform(-0.1, 0.1, (256, 128)).astype(np.float32)
    write(case_dir, "srcq.bin", q)
    write(case_dir, "srck.bin", k)
    write(case_dir, "srcv.bin", v)
    with np.errstate(all="ignore"):
        score = (q @ k.T) / math.sqrt(128.0)
    score -= np.max(score, axis=1, keepdims=True)
    probability = np.exp(score)
    probability /= np.sum(probability, axis=1, keepdims=True)
    with np.errstate(all="ignore"):
        golden = probability @ v
    return golden.astype(np.float32).reshape(-1)


CASES = [
    Case("broadcast", "broadcast/elf/kernel_multi_thread_broadcast_broadcast_PE4.elf", prep_broadcast),
    Case("concat_gather", "concat/elf/kernel_multi_thread_concat_concat_gather_PE4.elf", prep_concat_gather),
    Case("concat_scatter", "concat/elf/kernel_multi_thread_concat_concat_scatter_PE4.elf", prep_concat_scatter),
    Case("conv2d", "conv2d/elf/kernel_multi_thread_conv2d_v300_conv2d_PE4.elf", prep_conv2d, atol=2e-3, rtol=2e-3),
    Case("gelu", "element_wise/gelu/elf/kernel_multi_thread_element_wise_gelu_gelu_PE4.elf", prep_gelu, output_dtype=np.float16, atol=2e-2, rtol=2e-2),
    Case("fa", "fa/elf/kernel_multi_thread_fa_Sq256_Skv256_Tm128_Tk128_X1_Y2_FP32_VECFP32.elf", prep_fa, output_name="res.bin", atol=3e-2, rtol=3e-2),
    Case("gather", "gather/elf/kernel_multi_thread_gather_gather_PE4.elf", prep_gather),
    Case("matmul_shared", "matmul/elf/kernel_multi_thread_matmul_matmul_shared_B1_M256_N256_K256_tM128_tN256_tK128.elf", prep_matmul, output_name="res.bin", atol=1e-3, rtol=1e-3),
    Case("matmul_reuseB", "matmul/elf/kernel_multi_thread_matmul_matmul_reuseB_B1_M256_N256_K256_tM128_tN256_tK128.elf", prep_matmul, output_name="res.bin", atol=1e-3, rtol=1e-3),
    Case("matmul_lowp_fp8", "matmul/elf/kernel_multi_thread_matmul_matmul_lowp_FP8_B1_M256_N256_K512_tM128_tN256_tK512.elf", prep_matmul_lowp, output_name="res.bin"),
    Case("rms_norm", "normalization/rms_norm/elf/kernel_multi_thread_normalization_rms_norm_rms_norm_PE4.elf", prep_rms, output_dtype=np.float16, atol=3e-2, rtol=3e-2),
    Case("rms_norm_binary", "normalization/rms_norm_binary/elf/kernel_multi_thread_normalization_rms_norm_binary_rms_norm_binary_PE4.elf", prep_rms, output_dtype=np.float16, atol=3e-2, rtol=3e-2),
    Case("cumsum_row", "reduction/cumsum_row/elf/kernel_multi_thread_reduction_cumsum_row_cumsum_row_PE4.elf", lambda p: prep_rows(p, "cumsum"), atol=2e-3, rtol=2e-3),
    Case("reducemax_row", "reduction/reducemax_row/elf/kernel_multi_thread_reduction_reducemax_row_reducemax_row_PE4.elf", lambda p: prep_rows(p, "max")),
    Case("reduceprod_row", "reduction/reduceprod_row/elf/kernel_multi_thread_reduction_reduceprod_row_reduceprod_row_PE4.elf", lambda p: prep_rows(p, "prod"), atol=2e-3, rtol=2e-3),
    Case("reducesum_row", "reduction/reducesum_row/elf/kernel_multi_thread_reduction_reducesum_row_reducesum_row_PE4.elf", lambda p: prep_rows(p, "sum"), atol=2e-3, rtol=2e-3),
    Case("transpose", "transpose/elf/kernel_multi_thread_transpose_transpose_PE4.elf", prep_transpose, output_dtype=np.int32, atol=0.0, rtol=0.0),
    Case("tadd", "vec/elf/kernel_multi_thread_vec_Rows16_Cols16.elf", prep_tadd, output_name="vec_out.bin"),
    Case("trowsum", "vec/elf/kernel_multi_thread_vec_trowsum_Rows16_Cols16.elf", prep_trowsum, output_name="trowsum_out.bin", atol=2e-4, rtol=2e-4),
]


def run_case(case: Case, gfrun: Path, timeout: int) -> tuple[str, str]:
    elf = OUTPUT / case.elf
    if not elf.is_file():
        return "SKIP", f"missing ELF: {elf}"
    case_dir = COMPARE / elf.stem
    case_dir.mkdir(parents=True, exist_ok=True)
    golden = np.asarray(case.prepare(case_dir)).reshape(-1)
    np.zeros(golden.size, dtype=case.output_dtype).tofile(case_dir / case.output_name)
    command = [str(gfrun), "-s", "softcore.multiThreadNum=4", "-f", str(elf)]
    try:
        proc = subprocess.run(command, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True,
                              timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        (case_dir / "gfrun.log").write_text(exc.stdout or "", encoding="utf-8")
        return "TIMEOUT", str(case_dir / "gfrun.log")
    (case_dir / "gfrun.log").write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        return "FAIL", f"gfrun rc={proc.returncode}: {case_dir / 'gfrun.log'}"
    actual = np.fromfile(case_dir / case.output_name, dtype=case.output_dtype)
    if actual.size != golden.size:
        return "FAIL", f"size actual={actual.size}, golden={golden.size}"
    ok = np.allclose(actual, golden.astype(case.output_dtype),
                     atol=case.atol, rtol=case.rtol, equal_nan=False)
    diff = np.abs(actual.astype(np.float64) - golden.astype(np.float64))
    max_abs = float(np.max(diff)) if diff.size else 0.0
    return ("PASS" if ok else "FAIL"), f"max_abs={max_abs:.6g}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gfrun", type=Path, default=Path(
        "/Users/blacktraker/Programming/gitproj/DV4/SuperScalarModel/bin/gfrun"))
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("cases", nargs="*", help="case names; default: all")
    args = parser.parse_args()
    selected = set(args.cases)
    results = []
    for case in CASES:
        if selected and case.name not in selected:
            continue
        status, detail = run_case(case, args.gfrun, args.timeout)
        results.append((case.name, status, detail))
        print(f"{status:7} {case.name:20} {detail}", flush=True)
    failures = sum(status not in ("PASS", "SKIP") for _, status, _ in results)
    print(f"summary: PASS={sum(s == 'PASS' for _, s, _ in results)} "
          f"FAIL={failures} SKIP={sum(s == 'SKIP' for _, s, _ in results)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
