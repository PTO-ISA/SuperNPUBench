#!/usr/bin/env python3
"""DynamicHiF4Quant 端到端数值验证:gen golden -> make res_check=on -> gfrun(单-PE)
-> hif4_compare.py 重建 MSE。

kernel 非 SPMD(单线程循环全 M),故 gfrun 用 softcore.multiThreadNum=1。

用法:
  python3 run_hif4_precision.py                 # 默认 M=32 N=128 seed=42
  python3 run_hif4_precision.py --M 32 --N 256
  python3 run_hif4_precision.py --skip-gen --skip-compile   # 只重跑 gfrun+比对
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CASE_DIR = SCRIPT_DIR.parent                       # .../quant/dynamic_hi_f4_quant
ONE_LEVEL = CASE_DIR.parents[3]                    # .../one-level-arch
WORKSPACE = ONE_LEVEL.parents[2]                   # .../SuperNPU
ELF_DIR = (ONE_LEVEL / "output" / "solution" / "quant" / "dynamic_hi_f4_quant"
           / "elf" / "solution_quant_dynamic_hi_f4_quant")
OBJ_DIR = ONE_LEVEL / "output" / "solution" / "quant" / "dynamic_hi_f4_quant" / "src"
CMP_ROOT = ONE_LEVEL / "compare"
GEN = SCRIPT_DIR / "gen_hif4_golden.py"
COMPARE = SCRIPT_DIR / "hif4_compare.py"
DEFAULT_COMPILER = WORKSPACE / "linx-toolchain-build" / "output" / "linx_blockisa_llvm_musl" / "bin"
DEFAULT_GFRUN = WORKSPACE / "SuperScalarModel" / "bin" / "gfrun"
ELF_NAME = "dynamic_hi_f4_quant_hif4_res_check"


def run(cmd, *, cwd=None, env=None, timeout=None):
    print("+", " ".join(str(c) for c in cmd), f"(cwd={cwd})" if cwd else "")
    subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None,
                   env=env, timeout=timeout, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--M", type=int, default=32)
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--compiler-dir", type=Path,
                    default=Path(os.environ.get("COMPILER_DIR", DEFAULT_COMPILER)))
    ap.add_argument("--gfrun", type=Path, default=DEFAULT_GFRUN)
    ap.add_argument("--skip-gen", action="store_true")
    ap.add_argument("--skip-compile", action="store_true")
    ap.add_argument("--skip-sim", action="store_true")
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()

    elf_path = ELF_DIR / f"{ELF_NAME}.elf"
    cmp_dir = CMP_ROOT / ELF_NAME
    env = os.environ.copy()
    env["COMPILER_DIR"] = str(args.compiler_dir)
    gfrun_env = env.copy()
    gfrun_env["GFRUN_FORCE_DIRECTBOOT_ABI"] = "1"

    print(f"===== hif4 数值验证  M={args.M} N={args.N} seed={args.seed} =====")

    if not args.skip_gen:
        cmp_dir.mkdir(parents=True, exist_ok=True)
        run([sys.executable, GEN, "--M", args.M, "--N", args.N,
             "--seed", args.seed, "-o", cmp_dir])

    if not args.skip_compile:
        # 清同名 .o/.elf:res_check 只依赖 .cpp,复用旧 .o 会假 pass。
        for p in (OBJ_DIR / "hif4_res_check.o", elf_path):
            if p.exists():
                p.unlink()
        run(["make", "TESTCASE=dynamic_hi_f4_quant", "TYPE=HIF4_RES_CHECK",
             f"COMPILER_DIR={args.compiler_dir}", "res_check=on",
             f"PM={args.M}", f"PN={args.N}"], cwd=CASE_DIR, env=env)

    if not elf_path.is_file():
        print(f"FAIL: ELF 未生成: {elf_path}")
        return 2

    if not args.skip_sim:
        if not args.gfrun.is_file():
            print(f"FAIL: gfrun 不存在: {args.gfrun}")
            return 2
        cmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            run([args.gfrun, "-f", elf_path, "-s", "softcore.multiThreadNum=1"],
                cwd=args.gfrun.parent.parent, env=gfrun_env, timeout=args.timeout)
        except subprocess.TimeoutExpired:
            print(f"FAIL: gfrun 超时 (>{args.timeout}s)")
            return 1
        except subprocess.CalledProcessError as e:
            print(f"FAIL: gfrun 非零退出 ({e.returncode})")
            return 1

    cp = subprocess.run(
        [sys.executable, str(COMPARE), "--cmp-dir", str(cmp_dir),
         "--M", str(args.M), "--N", str(args.N)],
        capture_output=True, text=True)
    print(cp.stdout.strip() or cp.stderr.strip())
    return 0 if "status=pass" in cp.stdout else 1


if __name__ == "__main__":
    sys.exit(main())
