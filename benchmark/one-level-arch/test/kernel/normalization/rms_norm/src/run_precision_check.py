#!/usr/bin/env python3
"""End-to-end rms_norm precision check: gen bins -> make res_check=on -> gfrun -> compare.

Usage (from SuperScalar root or anywhere):
  python3 .../run_precision_check.py
  python3 .../run_precision_check.py --skip-compile
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CASE_DIR = SCRIPT_DIR.parent  # .../rms_norm
ONE_LEVEL = CASE_DIR.parents[3]  # .../one-level-arch
SUPERSCALAR = ONE_LEVEL.parents[2]  # .../SuperScalar (workspace)
ELF_NAME = "kernel_normalization_rms_norm_rms_norm_DType__half_gA512_gR8192_PE4.elf"
ELF_PATH = ONE_LEVEL / "output" / "kernel" / "normalization" / "rms_norm" / "elf" / ELF_NAME
CMP_DIR = ONE_LEVEL / "compare" / ELF_NAME.replace(".elf", "")
DEFAULT_COMPILER = (
    SUPERSCALAR / "linx-toolchain-build" / "output" / "linx_blockisa_llvm_musl" / "bin"
)
DEFAULT_GFRUN = SUPERSCALAR / "SuperScalarModel" / "bin" / "gfrun"


def run(cmd: list[str], *, cwd: Path | None = None, env: dict | None = None) -> None:
    print("+", " ".join(cmd))
    if cwd:
        print(f"  (cwd={cwd})")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compiler-dir",
        type=Path,
        default=Path(os.environ.get("COMPILER_DIR", DEFAULT_COMPILER)),
    )
    parser.add_argument("--gfrun", type=Path, default=DEFAULT_GFRUN)
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--skip-gen", action="store_true")
    parser.add_argument("--skip-sim", action="store_true")
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--mse-tol", type=float, default=1e-3)
    args = parser.parse_args()

    if not args.skip_gen:
        run(
            [
                sys.executable,
                str(SCRIPT_DIR / "gen_rms_norm_data.py"),
                "-o",
                str(CMP_DIR),
                "--also-src-data",
            ]
        )

    if not args.skip_compile:
        if not args.compiler_dir.is_dir():
            print(f"FAIL: COMPILER_DIR not found: {args.compiler_dir}")
            return 2
        env = os.environ.copy()
        env["COMPILER_DIR"] = str(args.compiler_dir)
        run(
            [
                "make",
                "TESTCASE=rms_norm",
                f"COMPILER_DIR={args.compiler_dir}",
                "DType=__half",
                "res_check=on",
            ],
            cwd=CASE_DIR,
            env=env,
        )

    if not ELF_PATH.is_file():
        print(f"FAIL: ELF not found: {ELF_PATH}")
        return 2

    if not args.skip_sim:
        if not args.gfrun.is_file():
            print(f"FAIL: gfrun not found: {args.gfrun}")
            return 2
        # Ensure compare dir exists for host I/O under RES_CHECK.
        CMP_DIR.mkdir(parents=True, exist_ok=True)
        run(
            [str(args.gfrun), "-f", str(ELF_PATH), "-t", "1"],
            cwd=args.gfrun.parent.parent,  # SuperScalarModel
        )

    rc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_DIR / "rms_norm_data_compare.py"),
            "--cmp-dir",
            str(CMP_DIR),
            "--atol",
            str(args.atol),
            "--rtol",
            str(args.rtol),
            "--mse-tol",
            str(args.mse_tol),
        ],
        check=False,
    ).returncode
    return rc


if __name__ == "__main__":
    sys.exit(main())
