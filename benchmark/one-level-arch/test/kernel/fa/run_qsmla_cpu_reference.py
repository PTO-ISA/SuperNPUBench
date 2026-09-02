#!/usr/bin/env python3
"""Build and run the unified QSMLA CPU golden generator by case name.

Purpose
=======
This is the case-driven front end. It reads shape/window/scale metadata from
``qsmla_cases.py`` and uses the unified Python reference to generate HIF8
inputs, descales, and FP32 golden output. FP16 is a compatibility path enabled
explicitly with ``--dtype FP16``.

Common usage
============
List registered cases::

    python3 run_qsmla_cpu_reference.py --list

Generate one reference-feasible case::

    python3 run_qsmla_cpu_reference.py --case baseline_swa \
        --output-root /tmp/qsmla-reference

Generate every reference-feasible registered case::

    python3 run_qsmla_cpu_reference.py --all-reference \
        --output-root /tmp/qsmla-reference
"""

import argparse
import subprocess
import sys
from pathlib import Path

from qsmla_cases import QSMLA_CASES, generate_case, validate_case


HERE = Path(__file__).resolve().parent
def print_cases() -> None:
    print("NAME                              B    S1    S2   N1  N2    D    K   K1  MODE             Q/KV LAYOUT       LOGICAL/STORAGE -> STAGE0  ENABLE")
    for case in QSMLA_CASES:
        print(
            f"{case.name:32} {case.b:2} {case.s1:5} {case.s2:5} "
            f"{case.n1:4} {case.n2:3} {case.d:4} {case.k:4} "
            f"{('-' if case.k1 is None else str(case.k1)):>4}  {case.mode:16} "
            f"{case.q_layout}/{case.kv_layout:15} "
            f"{case.logical_dtype}/{case.source_storage_dtype} -> "
            f"{case.stage0_compute_dtype:5} {case.enable_stage}"
        )


def run_references(cases, output_root: Path, dtype: str) -> None:
    for case in cases:
        error = validate_case(case)
        if error is not None:
            raise ValueError(f"invalid QSMLA case {case.name}: {error}")
        if not (case.reference_feasible or case.mode_generation_feasible):
            raise ValueError(
                f"case {case.name} is a compile-shape target without generated inputs")

    for case in cases:
        output = generate_case(case, output_root, dtype=dtype)
        print(output)


def parse_args():
    parser = argparse.ArgumentParser(
        description="List QSMLA cases and generate deterministic HIF8 inputs plus CPU golden outputs")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", action="store_true", help="list all registered QSMLA cases")
    action.add_argument("--case", metavar="NAME", help="generate one reference-feasible CPU golden")
    action.add_argument("--all-reference", action="store_true", help="generate every reference-feasible CPU golden")
    parser.add_argument("--output-root", type=Path, default=Path("qsmla_cpu_reference_output"))
    parser.add_argument(
        "--dtype", choices=("HIF8", "FP16"), default="HIF8",
        help="HIF8 is primary; select FP16 explicitly for compatibility",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list:
        print_cases()
        return 0
    by_name = {case.name: case for case in QSMLA_CASES}
    if args.case:
        case = by_name.get(args.case)
        if case is None:
            print(f"unknown QSMLA case: {args.case}", file=sys.stderr)
            return 2
        selected = (case,)
    else:
        selected = tuple(case for case in QSMLA_CASES if case.reference_feasible)
    try:
        run_references(selected, args.output_root, args.dtype)
    except (ValueError, subprocess.CalledProcessError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
