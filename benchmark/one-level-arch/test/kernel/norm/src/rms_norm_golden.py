#!/usr/bin/env python3
"""Generate and compare RMSNorm golden data."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def generate_data(
    data_dir: Path, rows: int, cols: int, epsilon: float, seed: int
) -> None:
    rng = np.random.default_rng(seed)
    input_data = rng.uniform(-1.0, 1.0, size=(rows, cols)).astype(np.float32)
    weight = rng.uniform(0.5, 1.5, size=(cols,)).astype(np.float32)

    mean_square = np.mean(input_data * input_data, axis=1, keepdims=True)
    inverse_rms = np.float32(1.0) / np.sqrt(
        mean_square + np.float32(epsilon)
    )
    golden = (input_data * inverse_rms * weight[None, :]).astype(np.float32)

    data_dir.mkdir(parents=True, exist_ok=True)
    # Never let a previous simulator run satisfy a new comparison.
    (data_dir / "output.bin").unlink(missing_ok=True)
    (data_dir / "gfrun.log").unlink(missing_ok=True)
    input_data.tofile(data_dir / "input.bin")
    weight.tofile(data_dir / "weight.bin")
    golden.tofile(data_dir / "golden.bin")
    (data_dir / "metadata.json").write_text(
        json.dumps(
            {
                "rows": rows,
                "cols": cols,
                "epsilon": epsilon,
                "seed": seed,
                "dtype": "float32",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Generated RMSNorm input and golden data in {data_dir}")


def compare_results(
    data_dir: Path, rows: int, cols: int, atol: float, rtol: float
) -> bool:
    output_path = data_dir / "output.bin"
    golden_path = data_dir / "golden.bin"
    expected_elements = rows * cols

    if not output_path.is_file():
        raise FileNotFoundError(f"operator output does not exist: {output_path}")
    if not golden_path.is_file():
        raise FileNotFoundError(f"golden output does not exist: {golden_path}")

    output = np.fromfile(output_path, dtype=np.float32)
    golden = np.fromfile(golden_path, dtype=np.float32)
    if output.size != expected_elements:
        raise ValueError(
            f"output contains {output.size} elements; expected {expected_elements}"
        )
    if golden.size != expected_elements:
        raise ValueError(
            f"golden contains {golden.size} elements; expected {expected_elements}"
        )

    output = output.reshape(rows, cols)
    golden = golden.reshape(rows, cols)
    finite = np.isfinite(output).all()
    difference = output - golden
    absolute_difference = np.abs(difference)
    denominator = np.maximum(np.abs(golden), np.finfo(np.float32).tiny)
    relative_difference = absolute_difference / denominator
    max_index = np.unravel_index(
        int(np.argmax(absolute_difference)), absolute_difference.shape
    )

    max_abs = float(absolute_difference[max_index])
    max_rel = float(np.max(relative_difference))
    mse = float(np.mean(difference * difference))
    passed = bool(
        finite and np.allclose(output, golden, atol=atol, rtol=rtol)
    )

    print("RMSNorm golden comparison")
    print(f"  result   : {'PASS' if passed else 'FAIL'}")
    print(f"  shape    : ({rows}, {cols})")
    print(f"  tolerance: atol={atol:g}, rtol={rtol:g}")
    print(f"  max_abs  : {max_abs:.9g}")
    print(f"  max_rel  : {max_rel:.9g}")
    print(f"  mse      : {mse:.9g}")
    print(
        "  max error: "
        f"index={max_index}, output={float(output[max_index]):.9g}, "
        f"golden={float(golden[max_index]):.9g}"
    )
    if not finite:
        print("  error    : output contains NaN or infinity")
    return passed


def run_gfrun(
    gfrun: Path, elf: Path, data_dir: Path, timeout: float
) -> None:
    # Do not pass "-t 1": it enables per-instruction basic tracing and can
    # turn a short numerical check into a multi-gigabyte log.
    command = [str(gfrun), "-f", str(elf)]
    print("Running:", " ".join(command))
    try:
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        partial_log = error.stdout or ""
        if isinstance(partial_log, bytes):
            partial_log = partial_log.decode(errors="replace")
        (data_dir / "gfrun.log").write_text(
            partial_log, encoding="utf-8"
        )
        raise RuntimeError(
            f"gfrun timed out after {timeout:g} seconds; "
            f"see {data_dir / 'gfrun.log'}"
        ) from error
    log = completed.stdout
    (data_dir / "gfrun.log").write_text(log, encoding="utf-8")
    if log:
        print(log, end="" if log.endswith("\n") else "\n")
    if completed.returncode != 0:
        raise RuntimeError(
            f"gfrun exited with status {completed.returncode}; "
            f"see {data_dir / 'gfrun.log'}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and compare RMSNorm golden output"
    )
    parser.add_argument(
        "action", choices=("generate", "compare", "check"),
        help="generate data, compare existing output, or run both with gfrun",
    )
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--rows", default=16, type=int)
    parser.add_argument("--cols", default=256, type=int)
    parser.add_argument("--epsilon", default=1.0e-6, type=float)
    parser.add_argument("--seed", default=2026, type=int)
    parser.add_argument("--atol", default=2.0e-5, type=float)
    parser.add_argument("--rtol", default=2.0e-4, type=float)
    parser.add_argument("--timeout", default=120.0, type=float)
    parser.add_argument("--elf", type=Path)
    parser.add_argument(
        "--gfrun",
        default=Path(
            "/Users/blacktraker/Programming/gitproj/DV4/"
            "SuperScalarModel/bin/gfrun"
        ),
        type=Path,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.rows <= 0 or args.cols <= 0:
        raise ValueError("rows and cols must be positive")

    if args.action in ("generate", "check"):
        generate_data(
            args.data_dir, args.rows, args.cols, args.epsilon, args.seed
        )

    if args.action == "check":
        if args.elf is None:
            raise ValueError("--elf is required for the check action")
        run_gfrun(args.gfrun, args.elf, args.data_dir, args.timeout)

    if args.action in ("compare", "check"):
        return 0 if compare_results(
            args.data_dir, args.rows, args.cols, args.atol, args.rtol
        ) else 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(2)
