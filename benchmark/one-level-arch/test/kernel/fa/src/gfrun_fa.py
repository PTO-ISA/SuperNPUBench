#!/usr/bin/env python3
"""Run a GMMA FlashAttention ELF and compare it with a PyTorch golden.

The ELF must be built with ``res_check=on``.  The script parses the FA shape
and Cube dtype (FP32/BF16/FP16/FP8) from the ELF name, writes exact DUT
payloads into the embedded ``CHK_DIR``, runs the four-PE functional model, and
compares ``res.bin`` with

    softmax(Q @ K.T / sqrt(QD)) @ V

Example:

  python3 src/gfrun_fa.py -d /tmp/fa/kernel/fa/elf/kernel_fa_...elf
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
    raise SystemExit("gfrun_fa.py requires PyTorch: pip install torch") from exc


SCRIPT_DIR = Path(__file__).resolve().parent
ONE_LEVEL_ROOT = SCRIPT_DIR.parents[3]
COMPARE_ROOT = ONE_LEVEL_ROOT / "compare"
DEFAULT_GFRUN_ROOT = Path(
    "/Users/blacktraker/Programming/gitproj/DV4/SuperScalarModel-asl"
)
SUPPORTED_CUBE_DTYPES = ("FP32", "BF16", "FP16", "FP8")
DEFAULT_TOLERANCES = {
    "FP32": (1e-5, 1e-5),
    "BF16": (1e-4, 1e-4),
    "FP16": (1e-5, 1e-5),
    "FP8": (1e-3, 1e-3),
}


def extract_case(elf: Path) -> dict:
    """Extract the supported static FA configuration encoded in an ELF name."""
    match = re.search(
        r"fa_2d_unroll_gmma_Sq(?P<Sq>\d+)_Skv(?P<Skv>\d+)"
        r"_Tm(?P<Tm>\d+)_Tk(?P<Tk>\d+)"
        r"(?:_qD(?P<QD>\d+)_vD(?P<VD>\d+))?"
        r"_X(?P<X>\d+)_Y(?P<Y>\d+)"
        r"_Cube(?P<Cube>FP32|BF16|FP16|FP8)_VectorFP32$",
        elf.stem,
    )
    if not match:
        raise ValueError(
            "expected an fa_2d_unroll_gmma ELF with VectorFP32 and Cube dtype "
            f"in {SUPPORTED_CUBE_DTYPES}; "
            f"got {elf.name}"
        )
    fields = match.groupdict()
    case = {"Cube": fields.pop("Cube")}
    case.update(
        {key: int(value) for key, value in fields.items() if value is not None}
    )
    case.setdefault("QD", 128)
    case.setdefault("VD", 128)
    case["name"] = elf.stem
    return case


def encode_matrix(
    data: np.ndarray, cube_dtype: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact DUT payload and its FP32-decoded values."""
    if cube_dtype == "FP32":
        payload = np.ascontiguousarray(data, dtype=np.float32)
        return payload, payload
    if cube_dtype == "FP16":
        payload = np.ascontiguousarray(data, dtype=np.float16)
        return payload, payload.astype(np.float32)

    tensor = torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32))
    if cube_dtype == "BF16":
        quantized = tensor.to(torch.bfloat16)
        payload = quantized.view(torch.uint16).numpy().copy()
        return payload, quantized.float().numpy()
    if cube_dtype == "FP8":
        quantized = tensor.to(torch.float8_e4m3fn)
        payload = quantized.view(torch.uint8).numpy().copy()
        return payload, quantized.float().numpy()
    raise ValueError(f"unsupported Cube dtype: {cube_dtype}")


def prepare_case(case: dict, args) -> tuple[Path, np.ndarray]:
    """Write exact dtype payloads and a PyTorch FP32 golden."""
    case_dir = COMPARE_ROOT / case["name"]
    case_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    def make_input(shape):
        data = rng.normal(0.0, args.input_scale, shape)
        return np.clip(data, -1.0, 1.0).astype(np.float32)

    q_payload, q = encode_matrix(
        make_input((case["Sq"], case["QD"])), case["Cube"]
    )
    k_payload, k = encode_matrix(
        make_input((case["Skv"], case["QD"])), case["Cube"]
    )
    v_payload, v = encode_matrix(
        make_input((case["Skv"], case["VD"])), case["Cube"]
    )

    q_t = torch.from_numpy(q)
    k_t = torch.from_numpy(k)
    v_t = torch.from_numpy(v)
    scale = torch.sqrt(torch.tensor(float(case["QD"]), dtype=torch.float32))
    golden = (torch.softmax(torch.matmul(q_t, k_t.T) / scale, dim=-1) @ v_t)
    golden_np = golden.numpy().astype(np.float32, copy=False)

    q_payload.tofile(case_dir / "srcq.bin")
    k_payload.tofile(case_dir / "srck.bin")
    v_payload.tofile(case_dir / "srcv.bin")
    golden_np.tofile(case_dir / "golden.bin")
    np.zeros_like(golden_np).tofile(case_dir / "res.bin")
    return case_dir, golden_np


def run_gfrun(elf: Path, case_dir: Path, args) -> tuple[str, int, str]:
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
        return "TIMEOUT", -1, output

    (case_dir / "gfrun.log").write_text(output, encoding="utf-8")
    reached_end = "Reach the End of Benchmark" in output
    r2_zero = re.search(r"R2\s*=\s*0\b", output) is not None
    passed = process.returncode == 0 and reached_end and r2_zero
    return ("PASS" if passed else "FAIL"), process.returncode, output


def compare_result(
    case_dir: Path, golden: np.ndarray, args
) -> tuple[bool, dict]:
    result_path = case_dir / "res.bin"
    if not result_path.is_file():
        return False, {"reason": "res.bin was not created"}

    result = np.fromfile(result_path, dtype=np.float32)
    if result.size != golden.size:
        return False, {
            "reason": f"shape mismatch: result={result.size}, golden={golden.size}"
        }
    result = result.reshape(golden.shape)

    result64 = result.astype(np.float64)
    golden64 = golden.astype(np.float64)
    diff = result64 - golden64
    abs_diff = np.abs(diff)
    tolerance = args.atol + args.rtol * np.abs(golden64)
    mismatch_mask = abs_diff > tolerance
    max_index = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
    passed = bool(np.allclose(result, golden, atol=args.atol, rtol=args.rtol))
    metrics = {
        "mse": float(np.mean(diff * diff)),
        "max_abs": float(abs_diff[max_index]),
        "max_index": [int(index) for index in max_index],
        "actual_at_max": float(result[max_index]),
        "golden_at_max": float(golden[max_index]),
        "mismatches": int(np.count_nonzero(mismatch_mask)),
        "elements": int(result.size),
        "all_finite": bool(np.isfinite(result).all()),
        "all_zero": bool(np.all(result == 0)),
    }
    passed = passed and metrics["all_finite"] and not metrics["all_zero"]

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
        description="Compare GMMA FlashAttention gfrun output with PyTorch"
    )
    parser.add_argument("-d", "--elf", required=True, help="RES_CHECK FA ELF")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--input-scale", type=float, default=0.1)
    parser.add_argument("--atol", type=float, help="absolute tolerance")
    parser.add_argument("--rtol", type=float, help="relative tolerance")
    parser.add_argument("--timeout", type=int, default=300)
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
        default_atol, default_rtol = DEFAULT_TOLERANCES[case["Cube"]]
        if args.atol is None:
            args.atol = default_atol
        if args.rtol is None:
            args.rtol = default_rtol
        case_dir, golden = prepare_case(case, args)
        run_status, returncode, _ = run_gfrun(elf, case_dir, args)
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
