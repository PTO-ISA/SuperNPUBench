#!/usr/bin/env python3
"""Validate fa_lowp MXFP4 output against a decoded-payload PyTorch golden.

The ELF must be built with ``res_check=on``.  This script generates exact
E2M1x2 payload bytes and non-uniform group-32 E8M0 scales for Q/K/V, decodes
those same bytes to FP32 for the host reference, runs four-PE gfrun, and
compares the BF16 result with ``softmax(Q @ K.T / sqrt(QD)) @ V``.
"""

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError as exc:
    raise SystemExit(
        "gfrun_fa_mxfp4.py requires PyTorch: pip install torch"
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
ONE_LEVEL_ROOT = SCRIPT_DIR.parents[3]
COMPARE_ROOT = ONE_LEVEL_ROOT / "compare"
DEFAULT_GFRUN_ROOT = Path(
    "/Users/blacktraker/Programming/gitproj/DV4/SuperScalarModel-asl"
)
E2M1_VALUES = np.array(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=np.float32,
)
INPUT_CODES = np.array([1, 2, 3, 9, 10, 11], dtype=np.uint8)


def extract_case(elf: Path) -> dict:
    match = re.search(
        r"(?:fa_lowp(?:_recip|_ltile)?|fa_mxfp4_opt)_Sq(?P<Sq>\d+)_Skv(?P<Skv>\d+)"
        r"_Tm(?P<Tm>\d+)_Tk(?P<Tk>\d+)"
        r"(?:_qD(?P<QD>\d+)_vD(?P<VD>\d+))?"
        r"_X(?P<X>\d+)_Y(?P<Y>\d+)_CubeMXFP4_VectorBF16$",
        elf.stem,
    )
    if not match:
        raise ValueError(
            "expected an fa_lowp/fa_lowp_recip/fa_lowp_ltile/fa_mxfp4_opt "
            "CubeMXFP4_VectorBF16 ELF; "
            f"got {elf.name}"
        )
    case = {
        key: int(value)
        for key, value in match.groupdict().items()
        if value is not None
    }
    case.setdefault("QD", 128)
    case.setdefault("VD", 128)
    case["name"] = elf.stem
    return case


def pack_e2m1x2(codes: np.ndarray, axis: int) -> np.ndarray:
    """Pack adjacent logical reduction values, even index in low nibble."""
    if codes.shape[axis] % 2:
        raise ValueError("E2M1x2 packing axis must be even")
    lo_index = [slice(None)] * codes.ndim
    hi_index = [slice(None)] * codes.ndim
    lo_index[axis] = slice(0, None, 2)
    hi_index[axis] = slice(1, None, 2)
    lo = codes[tuple(lo_index)]
    hi = codes[tuple(hi_index)]
    return np.ascontiguousarray(lo | (hi << 4), dtype=np.uint8)


def decode_e2m1_codes(codes: np.ndarray) -> np.ndarray:
    return E2M1_VALUES[codes.astype(np.int64)]


def decode_e8m0(codes: np.ndarray) -> np.ndarray:
    if np.any(codes == 0xFF):
        raise ValueError("E8M0 code 0xff is NaN")
    return np.ldexp(
        np.ones(codes.shape, dtype=np.float32),
        codes.astype(np.int16) - 127,
    )


def make_scales(shape: tuple[int, ...], coefficients: tuple[int, ...]) -> np.ndarray:
    """Generate finite, non-uniform power-of-two scales in [2^-4, 2^-1]."""
    indices = np.indices(shape, dtype=np.int64)
    selector = np.zeros(shape, dtype=np.int64)
    for index, coefficient in zip(indices, coefficients):
        selector += coefficient * index
    return (0x7B + selector % 4).astype(np.uint8)


def bf16_bits_to_float(bits: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(bits, dtype=np.uint16)
    return (
        torch.from_numpy(contiguous)
        .view(torch.bfloat16)
        .to(torch.float32)
        .numpy()
    )


def prepare_case(case: dict, args) -> tuple[Path, np.ndarray]:
    if case["QD"] % 32 or case["Skv"] % 32:
        raise ValueError("MXFP4 requires QD and Skv divisible by 32")

    case_dir = args.compare_root.expanduser().resolve() / case["name"]
    case_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    q_codes = rng.choice(INPUT_CODES, size=(case["Sq"], case["QD"]))
    k_codes = rng.choice(INPUT_CODES, size=(case["Skv"], case["QD"]))
    v_codes = rng.choice(INPUT_CODES, size=(case["Skv"], case["VD"]))

    q_scale = make_scales((case["Sq"], case["QD"] // 32), (3, 1))
    k_scale = make_scales((case["Skv"], case["QD"] // 32), (1, 3))
    # V is the transposed B operand of PV. ScaleB remains [N,K/group], so the
    # global scale layout is [VD,Skv/32], independent of transpose_b().
    v_scale = make_scales((case["VD"], case["Skv"] // 32), (1, 3))

    q = decode_e2m1_codes(q_codes) * np.repeat(
        decode_e8m0(q_scale), 32, axis=1
    )
    k = decode_e2m1_codes(k_codes) * np.repeat(
        decode_e8m0(k_scale), 32, axis=1
    )
    v = decode_e2m1_codes(v_codes) * np.repeat(
        decode_e8m0(v_scale).T, 32, axis=0
    )

    q_t = torch.from_numpy(np.ascontiguousarray(q, dtype=np.float32))
    k_t = torch.from_numpy(np.ascontiguousarray(k, dtype=np.float32))
    v_t = torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32))
    scale = torch.sqrt(torch.tensor(float(case["QD"]), dtype=torch.float32))
    golden = torch.softmax(torch.matmul(q_t, k_t.T) / scale, dim=-1) @ v_t
    golden_np = golden.numpy().astype(np.float32, copy=False)

    pack_e2m1x2(q_codes, axis=1).tofile(case_dir / "srcq.bin")
    pack_e2m1x2(k_codes, axis=1).tofile(case_dir / "srck.bin")
    # PV uses physical Shared-B [K,N] with TransB. Packed-X2 RowMajor
    # carriers pair adjacent physical columns, hence V is stored as [K,N/2].
    pack_e2m1x2(v_codes, axis=1).tofile(case_dir / "srcv.bin")
    q_scale.tofile(case_dir / "srcq_scale.bin")
    k_scale.tofile(case_dir / "srck_scale.bin")
    v_scale.tofile(case_dir / "srcv_scale.bin")
    golden_np.tofile(case_dir / "golden.bin")
    np.zeros(golden_np.shape, dtype=np.uint16).tofile(case_dir / "res.bin")
    return case_dir, golden_np


def validate_prepared_case(case: dict, case_dir: Path, golden: np.ndarray) -> dict:
    expected_sizes = {
        "srcq.bin": case["Sq"] * case["QD"] // 2,
        "srck.bin": case["Skv"] * case["QD"] // 2,
        "srcv.bin": case["Skv"] * case["VD"] // 2,
        "srcq_scale.bin": case["Sq"] * case["QD"] // 32,
        "srck_scale.bin": case["Skv"] * case["QD"] // 32,
        "srcv_scale.bin": case["VD"] * case["Skv"] // 32,
        "golden.bin": case["Sq"] * case["VD"] * 4,
        "res.bin": case["Sq"] * case["VD"] * 2,
    }
    actual_sizes = {
        name: (case_dir / name).stat().st_size for name in expected_sizes
    }
    mismatches = {
        name: {"expected": expected_sizes[name], "actual": actual_sizes[name]}
        for name in expected_sizes
        if actual_sizes[name] != expected_sizes[name]
    }
    if mismatches:
        raise ValueError(f"prepared file size mismatch: {mismatches}")
    if golden.shape != (case["Sq"], case["VD"]):
        raise ValueError(f"unexpected golden shape: {golden.shape}")
    if not np.isfinite(golden).all():
        raise ValueError("golden contains non-finite values")
    return {
        "case_dir": str(case_dir),
        "file_sizes": actual_sizes,
        "golden_shape": list(golden.shape),
        "golden_min": float(golden.min()),
        "golden_max": float(golden.max()),
        "golden_all_finite": True,
    }


def run_gfrun(elf: Path, case_dir: Path, args) -> tuple[str, int]:
    gfrun_root = args.gfrun_root.expanduser().resolve()
    gfrun = gfrun_root / "bin" / "gfrun"
    if not gfrun.is_file():
        raise ValueError(f"gfrun does not exist: {gfrun}")

    command = [
        str(gfrun),
        "-s",
        "softcore.multiThreadNum=4",
        "-f",
        str(elf),
    ]
    print("command:", shlex.join(command))
    process = subprocess.Popen(
        command,
        cwd=gfrun_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        output, _ = process.communicate()
        (case_dir / "gfrun.log").write_text(output, encoding="utf-8")
        return "TIMEOUT", -1

    (case_dir / "gfrun.log").write_text(output, encoding="utf-8")
    reached_end = "Reach the End of Benchmark" in output
    r2_zero = re.search(r"R2\s*=\s*0\b", output) is not None
    passed = process.returncode == 0 and reached_end and r2_zero
    return ("PASS" if passed else "FAIL"), process.returncode


def compare_result(
    case_dir: Path, golden: np.ndarray, args
) -> tuple[bool, dict]:
    result_path = case_dir / "res.bin"
    if not result_path.is_file():
        return False, {"reason": "res.bin was not created"}

    result_bits = np.fromfile(result_path, dtype=np.uint16)
    if result_bits.size != golden.size:
        return False, {
            "reason": f"shape mismatch: result={result_bits.size}, golden={golden.size}"
        }
    result = bf16_bits_to_float(result_bits).reshape(golden.shape)
    result64 = result.astype(np.float64)
    golden64 = golden.astype(np.float64)
    diff = result64 - golden64
    abs_diff = np.abs(diff)
    tolerance = args.atol + args.rtol * np.abs(golden64)
    max_index = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
    metrics = {
        "mse": float(np.mean(diff * diff)),
        "max_abs": float(abs_diff[max_index]),
        "max_index": [int(index) for index in max_index],
        "actual_at_max": float(result[max_index]),
        "golden_at_max": float(golden[max_index]),
        "mismatches": int(np.count_nonzero(abs_diff > tolerance)),
        "elements": int(result.size),
        "all_finite": bool(np.isfinite(result).all()),
        "all_zero": bool(np.all(result == 0)),
    }
    passed = (
        bool(np.allclose(result, golden, atol=args.atol, rtol=args.rtol))
        and metrics["all_finite"]
        and not metrics["all_zero"]
    )

    report = case_dir / "golden_compare.log"
    with report.open("w", encoding="utf-8") as stream:
        stream.write(f"status: {'PASS' if passed else 'FAIL'}\n")
        stream.write(f"atol: {args.atol}\nrtol: {args.rtol}\n")
        stream.write(json.dumps(metrics, indent=2, sort_keys=True))
        stream.write("\n\nactual head:\n")
        stream.write(np.array2string(result[:4, :8], precision=9))
        stream.write("\n\ngolden head:\n")
        stream.write(np.array2string(golden[:4, :8], precision=9))
        stream.write("\n")
    metrics["report"] = str(report)
    return passed, metrics


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare fa_lowp MXFP4 gfrun output with PyTorch"
    )
    parser.add_argument("-d", "--elf", required=True, help="RES_CHECK fa_lowp ELF")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="generate and validate input/golden files without invoking gfrun",
    )
    parser.add_argument(
        "--compare-root",
        type=Path,
        default=COMPARE_ROOT,
        help="directory matching the CHK_DIR root embedded in the ELF",
    )
    parser.add_argument(
        "--gfrun-root",
        type=Path,
        default=DEFAULT_GFRUN_ROOT,
        help="SuperScalarModel-asl checkout root",
    )
    args = parser.parse_args()

    elf = Path(args.elf).expanduser().resolve()
    if not elf.is_file():
        parser.error(f"ELF does not exist: {elf}")
    try:
        case = extract_case(elf)
        case_dir, golden = prepare_case(case, args)
        preparation = validate_prepared_case(case, case_dir, golden)
        if args.prepare_only:
            print("PASS: MXFP4 inputs and decoded-payload golden prepared")
            print(json.dumps(preparation, indent=2, sort_keys=True))
            return 0
        run_status, returncode = run_gfrun(elf, case_dir, args)
    except ValueError as exc:
        parser.error(str(exc))

    if run_status != "PASS":
        print(
            f"FAIL: gfrun status={run_status}, returncode={returncode}, "
            f"log={case_dir / 'gfrun.log'}"
        )
        return 1

    passed, metrics = compare_result(case_dir, golden, args)
    print(f"{'PASS' if passed else 'FAIL'}: case={case}")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
