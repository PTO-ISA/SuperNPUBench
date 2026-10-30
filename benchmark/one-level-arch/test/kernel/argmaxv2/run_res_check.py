#!/usr/bin/env python3
"""ArgMaxV2 host-golden res_check（res_check_guide.md 第 3 层 per-operator
脚本范式，对齐 matmul src/golden_cmp.py / mxquant src/run_mxquant_check.py）。

两个代表例（每路径一例）：
  argmaxv2          fp32 dim=1 [7,3333]   切 R 路径（末段 tailB=5 + 哨兵 +
                                      屏障合并全链；[64,8] 条件块形）
  argmaxv2_dynamic  fp32 dim=1 [133,1000] 动态版（运行期 tiling、ceil 分区 +
                                      尾数）

金标准 np.argmax 零容差（atol=rtol=0），输入定向注入边界模式：平局取首个 /
+inf 后 NaN 仍胜 / ±0.0 相等 / 负位模式 NaN 视为最大 / 全下界行。

用法（外部工具路径经环境变量注入，同本目录其他 runner）：
  export SUPERSCALAR_MODEL_DIR=<SuperScalarModel 仓库根>
  export COMPILER_DIR=<Linx 编译器 bin 目录>
  python3 run_res_check.py [case ...]     # 缺省全量

产物：arch/output/kernel/argmaxv2/elf/<ELF>.elf（res_check=on 构建）、
arch/compare/<ELF-stem>/{input.bin,output.bin,gfrun.log}。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def _req_env(name: str, hint: str) -> Path:
    """仓库约定（compile.all）：外部工具路径经环境变量注入，禁止硬编码。"""
    v = os.environ.get(name)
    if not v:
        sys.exit(f"error: 环境变量 {name} 未设置（{hint}）")
    return Path(v)


GFRUN = _req_env("SUPERSCALAR_MODEL_DIR",
                 "SuperScalarModel 仓库根（含 bin/gfrun）") / "bin/gfrun"
COMPILER_DIR = _req_env("COMPILER_DIR",
                        "Linx 编译器 bin 目录（linx_blockisa_llvm_musl/bin）")

HERE = Path(__file__).resolve().parent
ARCH = HERE.parent.parent.parent                # .../one-level-arch
ELF_DIR = ARCH / "output/kernel/argmaxv2/elf"
COMPARE = ARCH / "compare"
GFRUN_TIMEOUT = 120  # seconds

# (TESTCASE, ELF stem) —— ELF 名与 Makefile 的 TARGET 一致。
CASES = [
    ("argmaxv2", "kernel_argmaxv2_argmaxv2_fp32_dim1_PE4"),
    ("argmaxv2_dynamic", "kernel_argmaxv2_argmaxv2_dynamic_fp32_dim1_PE4"),
]


def _seed_rows(x: np.ndarray) -> np.ndarray:
    """行方向（dim=1）注入边界模式：平局/+inf 后 NaN/±0/负 NaN/全下界。"""
    rows, cols = x.shape
    x[0, :] = -0.5
    x[0, 5] = 2.0
    x[0, cols // 2 + 5] = 2.0            # 平局取首个
    x[1, 40] = np.inf
    x[1, 41] = float("nan")              # +inf 之后 NaN 仍胜（首个 NaN）
    x[2, :] = 0.0
    x[2, 3] = -0.0                       # ±0.0 相等，取首个
    neg_nan = (np.array([0xFFC00000], dtype=np.uint32).view(np.float32)[0]
               if x.dtype == np.float32 else
               np.array([0xFE00], dtype=np.uint16).view(np.float16)[0])
    x[3, :] = 0.25
    x[3, 7] = neg_nan                    # 负位模式 NaN 同样视为最大
    x[4, :] = np.finfo(x.dtype).min      # 全下界 → 索引 0
    return x


def _prep_input(case_dir: Path, testcase: str) -> np.ndarray:
    """生成 input.bin 并返回 np.argmax 金标准（参考实现，语义逐位一致）。"""
    if testcase == "argmaxv2":
        rng = np.random.default_rng(11)   # 切 R 代表例 [7,3333]
        x = _seed_rows(rng.uniform(-1.0, 1.0, (7, 3333)).astype(np.float32))
    else:
        rng = np.random.default_rng(12)   # 动态版代表例 [133,1000]
        x = _seed_rows(rng.uniform(-1.0, 1.0, (133, 1000)).astype(np.float32))
    np.ascontiguousarray(x).tofile(case_dir / "input.bin")
    return np.argmax(x, axis=1).astype(np.int32)


def run_case(testcase: str, elf_stem: str) -> tuple[str, str]:
    # 1) res_check=on 构建（自包含，与 solution 树同款注入方式）
    cmd = ["make", f"TESTCASE={testcase}", "res_check=on",
           f"COMPILER_DIR={COMPILER_DIR}"]
    proc = subprocess.run(cmd, cwd=HERE, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, timeout=3600)
    elf = ELF_DIR / f"{elf_stem}.elf"
    if proc.returncode != 0 or not elf.is_file():
        return "BUILD_FAIL", f"rc={proc.returncode}; log tail: {proc.stdout[-400:]}"

    # 2) host 侧预置输入 + 全零占位输出（防上次残留 + 尺寸可判）
    case_dir = COMPARE / elf_stem
    case_dir.mkdir(parents=True, exist_ok=True)
    golden = _prep_input(case_dir, testcase)
    np.zeros(golden.size, dtype=np.int32).tofile(case_dir / "output.bin")

    # 3) 四 PE hosted 执行
    t0 = time.time()
    proc = subprocess.run(
        [str(GFRUN), "-s", "softcore.multiThreadNum=4", "-f", str(elf)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        timeout=GFRUN_TIMEOUT, check=False)
    (case_dir / "gfrun.log").write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        return "FAIL", f"gfrun rc={proc.returncode}: {case_dir / 'gfrun.log'}"

    # 4) 零容差比对（整数索引，atol=rtol=0）
    actual = np.fromfile(case_dir / "output.bin", dtype=np.int32)
    if actual.size != golden.size:
        return "FAIL", f"size actual={actual.size}, golden={golden.size}"
    if not np.array_equal(actual, golden):
        bad = int(np.count_nonzero(actual != golden))
        return "FAIL", f"mismatch count={bad}"
    return "PASS", f"max_abs=0 ({time.time() - t0:.1f}s)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="*",
                    help="case 名；缺省全量（argmaxv2 argmaxv2_dynamic）")
    args = ap.parse_args()
    selected = set(args.cases)
    unknown = sorted(selected - {c for c, _ in CASES})
    if unknown:
        ap.error(f"unknown case(s): {' '.join(unknown)}")
    rows = []
    for testcase, stem in CASES:
        if selected and testcase not in selected:
            continue
        status, detail = run_case(testcase, stem)
        rows.append((testcase, status, detail))
        print(f"{status:10} {testcase:20} {detail}", flush=True)
    failed = sum(s != "PASS" for _, s, _ in rows)
    print(f"summary: PASS={len(rows) - failed} FAIL={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
