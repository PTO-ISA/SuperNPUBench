#!/usr/bin/env python3
"""Run a multi-thread matmul ELF and compare its output with PyTorch.

The ELF must be built with ``res_check=on``.  The script creates the binary
inputs in the CHK_DIR encoded in the ELF, runs gfrun with four PEs, and checks
the generated ``res.bin`` against ``torch.matmul``.

Example:

  python3 golden_cmp.py --ones -d /absolute/path/to/matmul_reuseA_...elf
"""

import argparse
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
    raise SystemExit("golden_cmp.py requires PyTorch: pip install torch") from exc


DEFAULT_GFRUN = "/Users/blacktraker/Programming/gitproj/DV4/SuperScalarModel/bin/gfrun"
DEFAULT_GFRUN_ARGS = "-s softcore.multiThreadNum=4 -f"
SCRIPT_DIR = Path(__file__).resolve().parent
ONE_LEVEL_ROOT = SCRIPT_DIR.parents[4]
COMPARE_ROOT = ONE_LEVEL_ROOT / "compare"


def extract_shape(elf: Path) -> dict:
    """Extract B/M/N/K/tM/tN/tK from a multi-thread matmul ELF name."""
    match = re.search(
        r"matmul_(?:shared|reuseA)_B(?P<B>\d+)_M(?P<M>\d+)_N(?P<N>\d+)_K(?P<K>\d+)"
        r"_tM(?P<tM>\d+)_tN(?P<tN>\d+)_tK(?P<tK>\d+)$",
        elf.stem,
    )
    if not match:
        raise ValueError(f"cannot parse kernel/matmul ELF name: {elf.stem}")
    shape = {key: int(value) for key, value in match.groupdict().items()}
    shape["name"] = elf.stem
    return shape


def prepare_case(shape: dict, args) -> tuple[Path, np.ndarray]:
    """Write fp32 inputs and a PyTorch fp32 golden result."""
    case_dir = COMPARE_ROOT / shape["name"]
    case_dir.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    a_shape = (shape["B"], shape["M"], shape["K"])
    b_shape = (shape["B"], shape["K"], shape["N"])
    if args.ones:
        a = torch.ones(a_shape, dtype=torch.float32)
        b = torch.ones(b_shape, dtype=torch.float32)
    else:
        a = torch.randn(a_shape, generator=generator) * args.input_scale
        b = torch.randn(b_shape, generator=generator) * args.input_scale
        a.clamp_(-1.0, 1.0)
        b.clamp_(-1.0, 1.0)

    golden = torch.matmul(a, b).float().numpy()
    a.numpy().tofile(case_dir / "src0.bin")
    b.numpy().tofile(case_dir / "src1.bin")
    golden.tofile(case_dir / "golden.bin")
    np.zeros_like(golden).tofile(case_dir / "res.bin")
    return case_dir, golden


def run_gfrun(elf: Path, case_dir: Path, args) -> tuple[str, int, str]:
    command = [args.gfrun, *shlex.split(args.gfrun_args), str(elf)]
    print("command:", shlex.join(command))
    process = subprocess.Popen(
        command,
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
        return "timeout", -1, output

    (case_dir / "gfrun.log").write_text(output, encoding="utf-8")
    status = "pass" if process.returncode == 0 else "fail"
    return status, process.returncode, output


def compare_result(case_dir: Path, golden: np.ndarray, args) -> tuple[bool, dict]:
    result_path = case_dir / "res.bin"
    if not result_path.exists():
        return False, {"reason": "res.bin was not created"}

    result = np.fromfile(result_path, dtype=np.float32)
    if result.size != golden.size:
        return False, {
            "reason": f"shape mismatch: result={result.size}, golden={golden.size}"
        }

    result = result.reshape(golden.shape)
    diff = result - golden
    abs_diff = np.abs(diff)
    max_index = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
    passed = bool(np.allclose(result, golden, atol=args.atol, rtol=args.rtol))
    metrics = {
        "mse": float(np.mean(diff * diff)),
        "max_abs": float(abs_diff[max_index]),
        "max_index": tuple(int(index) for index in max_index),
        "actual_at_max": float(result[max_index]),
        "golden_at_max": float(golden[max_index]),
    }

    report = case_dir / "golden_compare.log"
    with report.open("w", encoding="utf-8") as stream:
        stream.write(f"status: {'PASS' if passed else 'FAIL'}\n")
        stream.write(f"atol: {args.atol}\nrtol: {args.rtol}\n")
        for key, value in metrics.items():
            stream.write(f"{key}: {value}\n")
        stream.write("\nactual head:\n")
        stream.write(np.array2string(result[0, :4, :8], precision=8))
        stream.write("\n\ngolden head:\n")
        stream.write(np.array2string(golden[0, :4, :8], precision=8))
        stream.write("\n")
    metrics["report"] = str(report)
    return passed, metrics


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare multi-thread matmul gfrun output against PyTorch"
    )
    parser.add_argument("-d", "--elf", required=True, help="multi-thread matmul ELF")
    parser.add_argument("--ones", action="store_true", help="use all-one inputs")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--input-scale", type=float, default=0.1)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--gfrun", default=DEFAULT_GFRUN)
    parser.add_argument("--gfrun-args", default=DEFAULT_GFRUN_ARGS)
    args = parser.parse_args()

    elf = Path(args.elf).expanduser().resolve()
    if not elf.is_file():
        parser.error(f"ELF does not exist: {elf}")

    try:
        shape = extract_shape(elf)
        case_dir, golden = prepare_case(shape, args)
    except ValueError as exc:
        parser.error(str(exc))

    run_status, returncode, _ = run_gfrun(elf, case_dir, args)
    if run_status != "pass":
        print(
            f"FAIL: gfrun status={run_status}, returncode={returncode}, "
            f"log={case_dir / 'gfrun.log'}"
        )
        return 1

    passed, metrics = compare_result(case_dir, golden, args)
    print(f"{'PASS' if passed else 'FAIL'}: shape={shape}, metrics={metrics}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
