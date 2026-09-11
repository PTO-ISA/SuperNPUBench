#!/usr/bin/env python3
"""DynamicMxQuant solution 端到端精度看护：gen -> make res_check=on -> gfrun -> 逐字节比对。

遍历全部已注册的 driver 用例（须与 compile.all 的 CONFIGS 保持同步；本文件多一列 PE =
gfrun 的 softcore.multiThreadNum）。每个用例：现场生成 golden -> 清 .o/.elf（避免
res_check/非-res_check 复用同名 .o 造成假 pass）-> res_check=on 编译 -> gfrun 4-PE（或
单 PE）-> dynamic_mx_quant_data_compare.py 逐字节比对 output + scale。末尾汇总 pass/fail。

用法（COMPILER_DIR / gfrun 有默认值，指向同一 SuperNPU 工作区）：
  python3 run_precision_check.py
  python3 run_precision_check.py --only tail_ocp_fp4_dyn nontail_ocp_fp4_splitN_dyn
  python3 run_precision_check.py --skip-gen --skip-compile        # 只重跑 gfrun + 比对
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CASE_DIR = SCRIPT_DIR.parent                 # .../quant/dynamic_mx_quant
ONE_LEVEL = CASE_DIR.parents[3]              # .../one-level-arch
WORKSPACE = ONE_LEVEL.parents[2]            # .../SuperNPU
ELF_DIR = ONE_LEVEL / "output" / "solution" / "quant" / "dynamic_mx_quant" / "elf"
OBJ_DIR = ONE_LEVEL / "output" / "solution" / "quant" / "dynamic_mx_quant" / "src"
CMP_ROOT = ONE_LEVEL / "compare"
GEN = SCRIPT_DIR / "gen_dynamic_mx_quant_data.py"
COMPARE = SCRIPT_DIR / "dynamic_mx_quant_data_compare.py"
DEFAULT_COMPILER = WORKSPACE / "linx-toolchain-build" / "output" / "linx_blockisa_llvm_musl" / "bin"
DEFAULT_GFRUN = WORKSPACE / "SuperScalarModel" / "bin" / "gfrun"
SEED = 42

# TYPE, driver, M(Axis), K(Post), BS, in-dtype, algo, kernel, dtype, PE
# ★ 须与 compile.all 的 CONFIGS 保持同步（PE 列为本 runner 独有 = gfrun multiThreadNum）。
#   big-BS 用例（bs128）numKb=1 无块行并行度，但仍统一走 kPeNum=4 kernel（PE0 全包、
#   PE1..3 空转），故 PE=4；输出与单 PE 逐字节一致。
CONFIGS = [
    ("TAIL_OCP_FP8",                 "tail_ocp_fp8",                 512, 256,  32, "fp16", "OCP",    "tail",    "FP8", 4),
    ("TAIL_OCP_FP8_DYN",             "tail_ocp_fp8_dyn",             512, 256,  32, "fp16", "OCP",    "tail",    "FP8", 4),
    ("TAIL_OCP_FP4",                 "tail_ocp_fp4",                 512, 256,  32, "bf16", "OCP",    "tail",    "FP4", 4),
    # 大 shape 基准 [15360,1536]（~47MB，太大，默认不跑）；手动:
    #   make TESTCASE=dynamic_mx_quant TYPE=TAIL_OCP_FP4_BENCH res_check=on  然后 gfrun + compare
    # ("TAIL_OCP_FP4_BENCH",         "tail_ocp_fp4_bench",         15360,1536,  32, "bf16", "OCP",    "tail",    "FP4", 4),
    ("TAIL_CUBLAS_FP8_4PE",          "tail_cublas_fp8_4pe",          512, 256,  32, "fp16", "CUBLAS", "tail",    "FP8", 4),
    ("NONTAIL_CUBLAS_FP8_4PE",       "nontail_cublas_fp8_4pe",       512, 256,  32, "fp16", "CUBLAS", "nontail", "FP8", 4),
    ("NONTAIL_CUBLAS_FP8_BS128",     "nontail_cublas_fp8_bs128",     128,  32, 128, "bf16", "CUBLAS", "nontail", "FP8", 4),
    ("NONTAIL_OCP_FP4_4PE",          "nontail_ocp_fp4_4pe",          512,  64,  32, "fp16", "OCP",    "nontail", "FP4", 4),
    ("NONTAIL_OCP_FP4_BS128",        "nontail_ocp_fp4_bs128",        128,  64, 128, "bf16", "OCP",    "nontail", "FP4", 4),
    # --- 运行期动态 shape（_dyn）家族 ---
    ("TAIL_OCP_FP4_DYN",             "tail_ocp_fp4_dyn",             512, 256,  32, "bf16", "OCP",    "tail",    "FP4", 4),
    ("TAIL_CUBLAS_FP8_DYN",          "tail_cublas_fp8_dyn",          512, 256,  32, "fp16", "CUBLAS", "tail",    "FP8", 4),
    ("NONTAIL_CUBLAS_FP8_DYN",       "nontail_cublas_fp8_dyn",       512, 256,  32, "fp16", "CUBLAS", "nontail", "FP8", 4),
    ("NONTAIL_CUBLAS_FP8_SPLITN_DYN","nontail_cublas_fp8_splitN_dyn",512, 256,  32, "fp16", "CUBLAS", "nontail", "FP8", 4),
    ("NONTAIL_OCP_FP4_DYN",          "nontail_ocp_fp4_dyn",          512,  64,  32, "fp16", "OCP",    "nontail", "FP4", 4),
    ("NONTAIL_OCP_FP4_SPLITN_DYN",   "nontail_ocp_fp4_splitN_dyn",   512, 256,  32, "fp16", "OCP",    "nontail", "FP4", 4),
]


def run(cmd, *, cwd=None, env=None, timeout=None):
    print("+", " ".join(str(c) for c in cmd), f"(cwd={cwd})" if cwd else "")
    subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None,
                   env=env, timeout=timeout, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compiler-dir", type=Path,
                    default=Path(os.environ.get("COMPILER_DIR", DEFAULT_COMPILER)))
    ap.add_argument("--gfrun", type=Path, default=DEFAULT_GFRUN)
    ap.add_argument("--only", nargs="+", default=None,
                    help="仅跑这些 driver 名（如 tail_ocp_fp4_dyn）")
    ap.add_argument("--skip-gen", action="store_true")
    ap.add_argument("--skip-compile", action="store_true")
    ap.add_argument("--skip-sim", action="store_true")
    ap.add_argument("--timeout", type=int, default=600, help="每个 gfrun 的超时秒数")
    args = ap.parse_args()

    configs = CONFIGS
    if args.only:
        want = set(args.only)
        configs = [c for c in CONFIGS if c[1] in want]
        missing = want - {c[1] for c in configs}
        if missing:
            print(f"FAIL: 未知 driver: {sorted(missing)}", file=sys.stderr)
            return 2
    if not args.skip_compile and not args.compiler_dir.is_dir():
        print(f"FAIL: COMPILER_DIR 不存在: {args.compiler_dir}", file=sys.stderr)
        return 2

    env = os.environ.copy()
    env["COMPILER_DIR"] = str(args.compiler_dir)
    gfrun_env = env.copy()
    gfrun_env["GFRUN_FORCE_DIRECTBOOT_ABI"] = "1"  # 新 musl 启动 ppoll 前置（见 RECORD）

    results = []
    for TYPE, driver, M, K, BS, indt, algo, kernel, dtype, pe in configs:
        elf_name = f"dynamic_mx_quant_{driver}"
        elf_path = ELF_DIR / f"{elf_name}.elf"
        cmp_dir = CMP_ROOT / elf_name
        print(f"\n===== {TYPE}  ({driver}  M={M} K={K} BS={BS} in={indt} "
              f"{algo}/{kernel}/{dtype}  PE={pe}) =====")

        if not args.skip_gen:
            cmp_dir.mkdir(parents=True, exist_ok=True)
            run([sys.executable, GEN, "--M", M, "--K", K, "--block-size", BS,
                 "--algo", algo, "--kernel", kernel, "--dtype", dtype,
                 "--in-dtype", indt, "--scale-layout", "compact", "--seed", SEED,
                 "-o", cmp_dir])

        if not args.skip_compile:
            # 清同名 .o/.elf：res_check 只依赖 .cpp，复用非-res_check 旧 .o 会假 pass。
            for p in (OBJ_DIR / f"{driver}.o", elf_path):
                if p.exists():
                    p.unlink()
            run(["make", "TESTCASE=dynamic_mx_quant", f"TYPE={TYPE}",
                 f"COMPILER_DIR={args.compiler_dir}", "res_check=on"],
                cwd=CASE_DIR, env=env)

        if not elf_path.is_file():
            print(f"FAIL: ELF 未生成: {elf_path}")
            results.append((driver, "NO-ELF", "NO-ELF"))
            continue

        if not args.skip_sim:
            if not args.gfrun.is_file():
                print(f"FAIL: gfrun 不存在: {args.gfrun}")
                return 2
            cmp_dir.mkdir(parents=True, exist_ok=True)
            try:
                run([args.gfrun, "-f", elf_path, "-s", f"softcore.multiThreadNum={pe}"],
                    cwd=args.gfrun.parent.parent, env=gfrun_env, timeout=args.timeout)
            except subprocess.TimeoutExpired:
                print(f"FAIL: gfrun 超时 (>{args.timeout}s)")
                results.append((driver, "TIMEOUT", "TIMEOUT"))
                continue
            except subprocess.CalledProcessError as e:
                print(f"FAIL: gfrun 非零退出 ({e.returncode})")
                results.append((driver, "GFRUN-ERR", "GFRUN-ERR"))
                continue

        # 比对：捕获 stdout 解析 output=/scale= 状态。
        cp = subprocess.run(
            [sys.executable, str(COMPARE), "-d", str(elf_path), "--dtype", dtype,
             "--scale-layout", "compact", "--cmp-root", str(CMP_ROOT)],
            capture_output=True, text=True)
        line = cp.stdout.strip().splitlines()[-1] if cp.stdout.strip() else ""
        print(line or cp.stderr.strip())
        out_st = "pass" if "output=pass" in line else "FAIL"
        sc_st = "pass" if "scale=pass" in line else "FAIL"
        results.append((driver, out_st, sc_st))

    print("\n" + "=" * 72)
    print(f"DynamicMxQuant 精度看护汇总（{len(results)} 用例）")
    print("=" * 72)
    n_fail = 0
    for driver, out_st, sc_st in results:
        ok = out_st == "pass" and sc_st == "pass"
        n_fail += 0 if ok else 1
        mark = "OK  " if ok else "FAIL"
        print(f"  [{mark}] {driver:<32} output={out_st}  scale={sc_st}")
    print("=" * 72)
    print(f"{len(results) - n_fail}/{len(results)} 通过" + ("" if n_fail == 0 else f"，{n_fail} 失败"))
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
