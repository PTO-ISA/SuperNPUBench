#!/usr/bin/env python3
"""DynamicMxQuant precision check pipeline.

Steps:
  1. Generate input + golden data
  2. Compile with res_check=on
  3. Run via QEMU
  4. Compare output vs golden

Usage:
  python run_precision_check.py --compiler-dir /path/to/compiler/bin
  python run_precision_check.py --compiler-dir /path/to/compiler/bin --algo OCP --type TAIL_OCP
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TEST_DIR = SCRIPT_DIR.parent
BENCH_ROOT = TEST_DIR.parents[3]
COMPARE_ROOT = BENCH_ROOT / "compare"

QEMU = os.environ.get("QEMU", "/remote/lms60/c00622284/qemu/LinxBlockModel/build/qemu-linx")
QEMU_ARGS = ["-blk_optimize", "force_tb_chained", "-s", "4096M"]

CONFIGS = {
    "TAIL_OCP":              {"M": 8, "K": 32, "algo": "OCP",           "kernel": "tail"},
    "TAIL_CUBLAS":           {"M": 8, "K": 32, "algo": "CUBLAS",        "kernel": "tail"},
    "TAIL_DYNAMIC_RANGE":    {"M": 8, "K": 32, "algo": "DYNAMIC_RANGE", "kernel": "tail"},
    "NONTAIL_OCP":           {"M": 8, "K": 32, "algo": "OCP",           "kernel": "nontail"},
    "NONTAIL_CUBLAS":        {"M": 8, "K": 32, "algo": "CUBLAS",        "kernel": "nontail"},
    "NONTAIL_DYNAMIC_RANGE": {"M": 8, "K": 32, "algo": "DYNAMIC_RANGE", "kernel": "nontail"},
}


def run(cmd, cwd=None, check=True):
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"  STDERR: {result.stderr}", file=sys.stderr)
        raise RuntimeError(f"command failed: {' '.join(str(c) for c in cmd)}")
    return result


def gen_data(type_name: str, cfg: dict):
    elf_name = f"kernel_quant_dynamic_mx_quant_dynamic_mx_quant_{type_name}"
    cmp_dir = COMPARE_ROOT / elf_name
    cmp_dir.mkdir(parents=True, exist_ok=True)
    gen_script = SCRIPT_DIR / "gen_dynamic_mx_quant_data.py"
    run([
        sys.executable, str(gen_script),
        "--M", str(cfg["M"]),
        "--K", str(cfg["K"]),
        "--algo", cfg["algo"],
        "-o", str(cmp_dir),
    ])
    return cmp_dir


def compile_elf(type_name: str, compiler_dir: str):
    env = os.environ.copy()
    env["COMPILER_DIR"] = compiler_dir
    run([
        "make",
        f"TESTCASE=dynamic_mx_quant",
        f"TYPE={type_name}",
        "res_check=on",
        "diss",
    ], cwd=str(TEST_DIR), check=True)


def find_elf(type_name: str) -> Path:
    elf_dir = BENCH_ROOT / "output" / "kernel" / "quant" / "dynamic_mx_quant" / "elf" / "kernel_quant_dynamic_mx_quant"
    pattern = f"*_{type_name}.elf"
    matches = list(elf_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no ELF matching {pattern} in {elf_dir}")
    return matches[0]


def run_qemu(elf_path: Path):
    run([QEMU] + QEMU_ARGS + [str(elf_path)])


def compare_results(type_name: str):
    cmp_script = SCRIPT_DIR / "dynamic_mx_quant_data_compare.py"
    elf_path = find_elf(type_name)
    result = run([
        sys.executable, str(cmp_script),
        "-d", str(elf_path),
        "--cmp-root", str(COMPARE_ROOT),
    ], check=False)
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler-dir", required=True,
                        help="path to linx_blockisa_llvm_musl bin directory")
    parser.add_argument("--type", dest="types", action="append",
                        choices=list(CONFIGS.keys()),
                        help="specific config(s) to test (default: all)")
    parser.add_argument("--algo", choices=["OCP", "CUBLAS", "DYNAMIC_RANGE"],
                        help="filter by algorithm")
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--skip-gen", action="store_true")
    parser.add_argument("--qemu", default=QEMU, help="QEMU path")
    args = parser.parse_args()

    global QEMU
    QEMU = args.qemu

    types_to_run = args.types or list(CONFIGS.keys())
    if args.algo:
        types_to_run = [t for t in types_to_run if CONFIGS[t]["algo"] == args.algo]

    results = []
    for type_name in types_to_run:
        cfg = CONFIGS[type_name]
        print(f"\n{'='*60}")
        print(f"Testing: {type_name} (M={cfg['M']}, K={cfg['K']}, algo={cfg['algo']})")
        print(f"{'='*60}")

        try:
            if not args.skip_gen:
                print("[1/4] Generating data...")
                gen_data(type_name, cfg)

            if not args.skip_compile:
                print("[2/4] Compiling with res_check=on...")
                compile_elf(type_name, args.compiler_dir)

            print("[3/4] Running QEMU...")
            elf_path = find_elf(type_name)
            run_qemu(elf_path)

            print("[4/4] Comparing results...")
            output = compare_results(type_name)
            results.append((type_name, output))
            print(output)

        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            results.append((type_name, f"ERROR: {e}"))

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for type_name, output in results:
        print(f"  {type_name}: {output.strip()}")


if __name__ == "__main__":
    main()
