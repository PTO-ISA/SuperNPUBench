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

# M/K are the full matrix dims of the driver (rows, cols). For tail the block
# reduces along cols; for nontail it reduces along rows (see driver .cpp dims).
# `driver` is the specialized kernel/driver basename; the ELF is
# dynamic_mx_quant_<driver>.elf and its compare dir shares that basename.
# Only DEBUGGED configs are registered (op-aligned to AscendC, all known issues in
# RECORD.md). UNDEBUGGED kernels (tail/nontail OCP-FP8, tail/nontail DynRange-FP4)
# had their drivers + TYPE blocks removed — see README.md「状态总览」.
# `blocked` = True would skip a config end-to-end; nothing is alignment-blocked now
# (fp4 emit is verified, RECORD 问题2). NOTE: end-to-end QEMU is still globally
# unreliable due to toolchain<->emulator skew; that is a separate documented caveat.
CONFIGS = {
    "TAIL_CUBLAS_FP8":      {"M": 8,  "K": 32, "algo": "CUBLAS",        "kernel": "tail",    "dtype": "FP8", "driver": "tail_cublas_fp8",     "blocked": False, "scale_layout": "compact"},
    "TAIL_OCP_FP4":         {"M": 8,  "K": 64, "algo": "OCP",           "kernel": "tail",    "dtype": "FP4", "driver": "tail_ocp_fp4",        "blocked": False, "scale_layout": "compact"},
    "NONTAIL_CUBLAS_FP8":   {"M": 32, "K": 32, "algo": "CUBLAS",        "kernel": "nontail", "dtype": "FP8", "driver": "nontail_cublas_fp8",  "blocked": False, "scale_layout": "compact"},
    # 方案A split-reduce bigbs: BS=128 (>=96) has no legal plain TileN -> auto-routes
    # to _bigbs. M=Axis=128 (=1 reduce block), K=Post=32. Golden is BS-parametric
    # (--block-size 128), same generator, no BS branch.
    "NONTAIL_CUBLAS_FP8_BIGBS": {"M": 128, "K": 32, "block_size": 128, "algo": "CUBLAS", "kernel": "nontail", "dtype": "FP8", "driver": "nontail_cublas_fp8_bigbs", "blocked": False, "scale_layout": "compact"},
    "NONTAIL_OCP_FP4":      {"M": 32, "K": 64, "algo": "OCP",           "kernel": "nontail", "dtype": "FP4", "driver": "nontail_ocp_fp4",     "blocked": False, "scale_layout": "compact"},
}


def run(cmd, cwd=None, check=True):
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"  STDERR: {result.stderr}", file=sys.stderr)
        raise RuntimeError(f"command failed: {' '.join(str(c) for c in cmd)}")
    return result


def gen_data(type_name: str, cfg: dict):
    # Compare dir shares the ELF basename: dynamic_mx_quant_<driver>.
    elf_name = f"dynamic_mx_quant_{cfg['driver']}"
    cmp_dir = COMPARE_ROOT / elf_name
    cmp_dir.mkdir(parents=True, exist_ok=True)
    gen_script = SCRIPT_DIR / "gen_dynamic_mx_quant_data.py"
    run([
        sys.executable, str(gen_script),
        "--M", str(cfg["M"]),
        "--K", str(cfg["K"]),
        "--block-size", str(cfg.get("block_size", 32)),
        "--algo", cfg["algo"],
        "--kernel", cfg["kernel"],
        "--dtype", cfg["dtype"],
        "--scale-layout", cfg.get("scale_layout", "broadcast"),
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


def find_elf(cfg: dict) -> Path:
    elf_dir = BENCH_ROOT / "output" / "kernel" / "quant" / "dynamic_mx_quant" / "elf" / "kernel_quant_dynamic_mx_quant"
    pattern = f"dynamic_mx_quant_{cfg['driver']}.elf"
    matches = list(elf_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no ELF matching {pattern} in {elf_dir}")
    return matches[0]


def run_qemu(elf_path: Path):
    run([QEMU] + QEMU_ARGS + [str(elf_path)])


def compare_results(cfg: dict):
    cmp_script = SCRIPT_DIR / "dynamic_mx_quant_data_compare.py"
    elf_path = find_elf(cfg)
    result = run([
        sys.executable, str(cmp_script),
        "-d", str(elf_path),
        "--dtype", cfg["dtype"],
        "--scale-layout", cfg.get("scale_layout", "broadcast"),
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
        print(f"Testing: {type_name} (M={cfg['M']}, K={cfg['K']}, algo={cfg['algo']}, dtype={cfg['dtype']})")
        print(f"{'='*60}")

        if cfg.get("blocked"):
            msg = "SKIPPED (FP4 toolchain-blocked: pto_tile.hpp:649 Cols*4%256)"
            print(f"  {msg}")
            results.append((type_name, msg))
            continue

        try:
            if not args.skip_gen:
                print("[1/4] Generating data...")
                gen_data(type_name, cfg)

            if not args.skip_compile:
                print("[2/4] Compiling with res_check=on...")
                compile_elf(type_name, args.compiler_dir)

            print("[3/4] Running QEMU...")
            elf_path = find_elf(cfg)
            run_qemu(elf_path)

            print("[4/4] Comparing results...")
            output = compare_results(cfg)
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
