#!/usr/bin/env python3
"""Prepare inputs, run four-PE RES_CHECK ELFs, and compare with NumPy — solution 树版。

与 test/kernel/multi_thread/res_check_all.py **对称**：同一套 host-golden 范式
（kernel 只做二进制 I/O：读 CHK_DIR/input.bin、写 CHK_DIR/output.bin；host 侧用 numpy
造 golden 并比对），同一套 CLI 与 stdout 输出格式，供上层统一接入。

差别（solution 树特性）：
  1. ROOT/OUTPUT 指向 output/solution（非 output/kernel/multi_thread）。
  2. **自包含编译**：main 跑 CASES 前，对 CASES 涉及算子目录注入 `res_check=on` 编译。
  3. Case 支持两类算子（见下）：
     - **单输出 + numpy golden**（如 rms_norm）：给 `prepare` 写 input.bin + 返回 golden，
       run_case 默认 `np.allclose(output.bin, golden)`。
     - **多输出 / 专用 golden**（如 dynamic_mx_quant：output + scale_output，FP8/FP4 解码 +
       scale 逐字节）：给 `verify` 钩子自定义比对；golden 由算子自带的 gen 脚本经 compile.all
       生成（`prepare` 可为 None），`four_pe` 标记是否 4-PE。

============================ 算子侧如何接入（维护 CASES）============================
往下方 `CASES` 加 `Case(...)`，无需改任何上层脚本。两种范式：

【A. 单输出 numpy golden】（kernel 读 input.bin、写 output.bin）
    def prep_rms_norm(case_dir: Path) -> np.ndarray:
        x = np.random.randn(512, 8192).astype(np.float16)
        write(case_dir, "input.bin", x)
        g = x / np.sqrt((x.astype(np.float32) ** 2).mean(-1, keepdims=True) + 1e-6)
        return g.astype(np.float16).reshape(-1)
    Case("rms_norm", "normalization/rms_norm/elf/<ELF>.elf", prepare=prep_rms_norm,
         output_dtype=np.float16, atol=2e-2, rtol=2e-2, four_pe=True)

【B. 多输出 / 专用 golden】（golden 由算子自带 gen 脚本经 compile.all 生成；verify 自定义比对）
    Case("op", "sub/elf/<ELF>.elf", verify=make_verify_xxx(...), four_pe=True/False)
    —— prepare 缺省（golden/input 已由 compile.all 生成到 case_dir=COMPARE/<ELF stem>）；
       verify(case_dir, elf) 返回 (status, detail)。dynamic_mx_quant 即此范式（见下）。

`Case.elf` 相对 output/solution。上层 run_precision 只按 `{status:7} {name:20} {detail}`
逐行取结果，不感知任何算子。
"""

from __future__ import annotations

import argparse
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]        # …/test/solution/common → one-level-arch
OUTPUT = ROOT / "output/solution"
COMPARE = ROOT / "compare"


@dataclass
class Case:
    name: str
    elf: str                        # 相对 output/solution，如 normalization/rms_norm/elf/x.elf
    prepare: object = None          # 可选：写 input + 返回 golden(默认 np.allclose 路径)；自带 gen 的算子留 None
    verify: object = None           # 可选：自定义比对 (case_dir, elf)->(status,detail)；多输出/专用 golden 用
    output_name: str = "output.bin"
    output_dtype: object = np.float32
    atol: float = 1e-4
    rtol: float = 1e-4
    four_pe: bool = False           # 是否 4-PE 跑(gfrun -s softcore.multiThreadNum=4)


def write(case_dir: Path, name: str, value: np.ndarray) -> None:
    np.ascontiguousarray(value).tofile(case_dir / name)


# ---- dynamic_mx_quant 样板：多输出（output+scale_output）+ 专用 compare（FP8/FP4 解码 MSE<0.1
#      + scale 逐字节）。golden 由算子自带 gen 脚本经 compile.all 生成；verify 调其 compare 脚本。----
_DMXQ_DIR = "quant/dynamic_mx_quant"
_DMXQ_COMPARE = ROOT / "test/solution" / _DMXQ_DIR / "src/dynamic_mx_quant_data_compare.py"


def make_verify_dmxq(dtype: str, scale_layout: str = "compact"):
    """dmxq 双输出比对：调算子自带 dynamic_mx_quant_data_compare.py（-d <elf> 按 basename 取
    COMPARE/<stem>/ 下 output/golden/scale_output/scale_golden.bin），解析其 stdout 的
    `output=pass` / `scale=pass` 判定。"""
    def _verify(case_dir, elf):
        p = subprocess.run(["python3", str(_DMXQ_COMPARE), "-d", str(elf),
                            "--dtype", dtype, "--scale-layout", scale_layout,
                            "--cmp-root", str(COMPARE)],
                           capture_output=True, text=True, timeout=120)
        out = (p.stdout or "") + (p.stderr or "")
        ok = ("output=pass" in out) and ("scale=pass" in out)
        line = next((l.strip() for l in out.splitlines()
                     if "output=" in l and "scale=" in l), "")
        if not line:
            tail = out.strip().splitlines()
            line = tail[-1] if tail else "no compare output"
        return ("PASS" if ok else "FAIL"), line[:120]
    return _verify


# driver : 输出 dtype : scale 布局 : 是否 4-PE（对齐 dynamic_mx_quant/compile.all CONFIGS）
_DMXQ = [
    ("tail_ocp_fp8",            "FP8", "compact", False),
    ("tail_ocp_fp8_dyn",       "FP8", "compact", False),
    ("tail_ocp_fp4",           "FP4", "compact", False),
    ("tail_cublas_fp8_4pe",    "FP8", "compact", True),
    ("nontail_cublas_fp8_4pe", "FP8", "compact", True),
    ("nontail_cublas_fp8_bigbs", "FP8", "compact", False),
    ("nontail_ocp_fp4_4pe",    "FP4", "compact", True),
    ("nontail_ocp_fp4_bigbs",  "FP4", "compact", False),
]

# ================================ CASES（算子侧维护）================================
# 样板：dynamic_mx_quant 8 个 driver（B 范式，多输出 + verify 钩子；golden 由 compile.all 的
# gen 脚本生成）。注：在缺 TileOP #63/#100 的发布版工具链上 kernel 编不过 → ELF 缺失 → SKIP。
CASES: list[Case] = [
    Case(f"dmxq_{drv}", f"{_DMXQ_DIR}/elf/dynamic_mx_quant_{drv}.elf",
         verify=make_verify_dmxq(dt, sl), four_pe=fp)
    for drv, dt, sl, fp in _DMXQ
]


def compile_units(cases: list[Case], compiler_dir: Path, timeout: int) -> None:
    """自包含：对 CASES 涉及的算子目录跑 compile.all（注入 res_check=on），产出带 I/O 桩的
    ELF；自带 gen 脚本的算子（如 dynamic_mx_quant）其 compile.all 会同时生成 golden。
    CASES 为空时不编译任何东西（返回 0 case）。"""
    units = sorted({str(Path(c.elf).parent.parent) for c in cases})   # <op>/elf/x.elf → <op>
    env = dict(os.environ, COMPILER_DIR=str(Path(compiler_dir).resolve()), baremetal="off")
    for rel in units:
        unit_dir = ROOT / "test/solution" / rel
        ca = unit_dir / "compile.all"
        if not ca.is_file():
            continue
        script = ca.read_text()
        if "res_check=on" not in script:
            script = script.replace("make ", "make res_check=on ")   # 只动没自带 res_check 的
        subprocess.run(["bash", "-c", script], cwd=unit_dir, env=env,
                       timeout=timeout * 30, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)


def run_case(case: Case, gfrun: Path, timeout: int) -> tuple[str, str]:
    """跑一个 case：prepare（可选，写 input/返回 golden）→ gfrun（kernel 读 input 写 output）
    → verify（自定义，多输出）或默认 np.allclose（单输出）。"""
    elf = OUTPUT / case.elf
    if not elf.is_file():
        return "SKIP", f"missing ELF: {elf}"
    case_dir = COMPARE / elf.stem
    case_dir.mkdir(parents=True, exist_ok=True)
    golden = None
    if case.prepare is not None:
        golden = case.prepare(case_dir)
        if golden is not None:
            golden = np.asarray(golden).reshape(-1)
            np.zeros(golden.size, dtype=case.output_dtype).tofile(case_dir / case.output_name)
    command = [str(gfrun), "-f", str(elf)]
    if case.four_pe:
        command += ["-s", "softcore.multiThreadNum=4"]
    try:
        proc = subprocess.run(command, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True,
                              timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        (case_dir / "gfrun.log").write_text(exc.stdout or "", encoding="utf-8")
        return "TIMEOUT", str(case_dir / "gfrun.log")
    (case_dir / "gfrun.log").write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        return "FAIL", f"gfrun rc={proc.returncode}: {case_dir / 'gfrun.log'}"
    # 多输出 / 专用 golden：算子自定义比对
    if case.verify is not None:
        return case.verify(case_dir, elf)
    # 单输出：默认 np.allclose
    if golden is None:
        return "FAIL", "no golden（prepare 未返回且无 verify 钩子）"
    actual = np.fromfile(case_dir / case.output_name, dtype=case.output_dtype)
    if actual.size != golden.size:
        return "FAIL", f"size actual={actual.size}, golden={golden.size}"
    ok = np.allclose(actual, golden.astype(case.output_dtype),
                     atol=case.atol, rtol=case.rtol, equal_nan=False)
    diff = np.abs(actual.astype(np.float64) - golden.astype(np.float64))
    max_abs = float(np.max(diff)) if diff.size else 0.0
    return ("PASS" if ok else "FAIL"), f"max_abs={max_abs:.6g}"


def main() -> int:
    parser = argparse.ArgumentParser(description="solution 树 host-golden 精度入口")
    parser.add_argument("--gfrun", type=Path, required=True)
    parser.add_argument("--compiler-dir", type=Path, default=None,
                        help="Linx 工具链 bin 目录；给了则自包含 res_check=on 编译，"
                             "不给则依赖外部已编好（上层 run_precision 统一编译前置的场景）")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("cases", nargs="*", help="case names; default: all")
    args = parser.parse_args()

    selected = set(args.cases)
    active = [c for c in CASES if not selected or c.name in selected]
    if args.compiler_dir:
        compile_units(active, args.compiler_dir, args.timeout)

    results = []
    for case in active:
        status, detail = run_case(case, args.gfrun, args.timeout)
        results.append((case.name, status, detail))
        print(f"{status:7} {case.name:20} {detail}", flush=True)
    failures = sum(status not in ("PASS", "SKIP") for _, status, _ in results)
    print(f"summary: PASS={sum(s == 'PASS' for _, s, _ in results)} "
          f"FAIL={failures} SKIP={sum(s == 'SKIP' for _, s, _ in results)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
