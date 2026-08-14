#!/usr/bin/env python3

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from qsmla_stage0_cases import QSMLA_STAGE0_CASES, QsmlaCase, validate_case


HERE = Path(__file__).resolve().parent
REFERENCE_SOURCE = HERE / "src" / "qsmla_cpu_ref.cpp"


def print_cases() -> None:
    print("NAME                              B    S1    S2   N1  N2    D    K   K1  MODE             Q/KV LAYOUT       LOGICAL/STORAGE -> STAGE0  ENABLE")
    for case in QSMLA_STAGE0_CASES:
        print(
            f"{case.name:32} {case.b:2} {case.s1:5} {case.s2:5} "
            f"{case.n1:4} {case.n2:3} {case.d:4} {case.k:4} "
            f"{('-' if case.k1 is None else str(case.k1)):>4}  {case.mode:16} "
            f"{case.q_layout}/{case.kv_layout:15} "
            f"{case.logical_dtype}/{case.source_storage_dtype} -> {case.stage0_compute_dtype:5} {case.enable_stage}"
        )


def compile_defines(case: QsmlaCase, output_root: Path):
    values = {
        "QB": case.b, "QS1": case.s1, "QS2": case.s2,
        "QN1": case.n1, "QN2": case.n2, "QD": case.d, "QK": case.k,
        "QK1": 0 if case.k1 is None else case.k1,
        "QTM": case.tm, "QTK": case.tk, "QTD": case.td,
        "QWIN_LEFT": case.win_left, "QWIN_RIGHT": case.win_right,
        "QSOFTMAX_SCALE": f"{case.softmax_scale:.9g}f",
        "QCASE_NAME": f'\"{case.name}\"',
        "QOUTPUT_ROOT": f'\"{output_root.resolve()}\"',
        "QLAYOUT_BSND": 1 if case.q_layout == "BSND" else 0,
        "QKV_LAYOUT_BSND": 1 if case.kv_layout == "BSND" else 0,
    }
    return [f"-D{name}={value}" for name, value in values.items()]


def run_reference(case: QsmlaCase, output_root: Path) -> None:
    error = validate_case(case)
    if error is not None:
        raise ValueError(f"invalid QSMLA case {case.name}: {error}")
    if not case.reference_feasible:
        raise ValueError(f"case {case.name} is a compile-shape target, not a Stage 0 reference target")

    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="qsmla-stage0-build-") as build_dir:
        binary = Path(build_dir) / case.name
        command = [
            "g++", "-std=c++17", "-O2", str(REFERENCE_SOURCE), "-o", str(binary),
            *compile_defines(case, output_root),
        ]
        print("compile:", " ".join(command))
        subprocess.run(command, check=True)
        subprocess.run([str(binary)], check=True)


def parse_args():
    parser = argparse.ArgumentParser(description="List and generate QSMLA Stage 0 FP16 SWA references")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", action="store_true", help="list the fixed Stage 0 shape matrix")
    action.add_argument("--case", metavar="NAME", help="generate one reference-feasible case")
    action.add_argument("--all-reference", action="store_true", help="generate every reference-feasible case")
    parser.add_argument("--output-root", type=Path, default=Path("qsmla_stage0_output"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list:
        print_cases()
        return 0

    by_name = {case.name: case for case in QSMLA_STAGE0_CASES}
    if args.case:
        case = by_name.get(args.case)
        if case is None:
            print(f"unknown QSMLA case: {args.case}", file=sys.stderr)
            return 2
        try:
            run_reference(case, args.output_root)
        except (ValueError, subprocess.CalledProcessError) as error:
            print(error, file=sys.stderr)
            return 1
        return 0

    try:
        for case in QSMLA_STAGE0_CASES:
            if case.reference_feasible:
                run_reference(case, args.output_root)
    except (ValueError, subprocess.CalledProcessError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
