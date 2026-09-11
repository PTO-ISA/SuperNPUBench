#!/usr/bin/env python3
"""Prepare inputs, run four-PE RES_CHECK ELFs, and compare with NumPy — solution 树版。

与 test/kernel/multi_thread/res_check_all.py **对称**：同一套 host-golden 范式
（kernel 只做二进制 I/O：读 CHK_DIR/input.bin、写 CHK_DIR/output.bin；host 侧用 numpy
造 golden 并逐元素 np.allclose 比对），同一套 CLI 与 stdout 输出格式，供上层统一接入。

差别（仅两处，均为 solution 树特性）：
  1. ROOT/OUTPUT 指向 output/solution（非 output/kernel/multi_thread）。
  2. **自包含编译**：main 跑 CASES 前，对 CASES 涉及的算子目录注入 `res_check=on` 编译，
     故本入口只需 --compiler-dir 即可独立跑（multi_thread 版依赖外部先编）。

============================ 算子侧如何接入（维护 CASES）============================
本文件是 solution 精度的**统一入口骨架**；具体算子的 golden / 用例由**算子侧维护**：
往下方 `CASES` 加一行 `Case(...)` + 写一个 `prep_<op>(case_dir)` 即可，无需改任何上层脚本。

前提：该算子 kernel 已具备合规 host-golden I/O 桩（`#ifdef RES_CHECK` 内
`readBinaryFile(CHK_DIR "/input.bin", ...)` 读入 + `writeBinaryFile(CHK_DIR "/output.bin", ...)`
写出），且 compile.all 能以 `res_check=on` 编出该 ELF。范式：

    def prep_rms_norm(case_dir: Path) -> np.ndarray:
        x = np.random.randn(512, 8192).astype(np.float16)
        write(case_dir, "input.bin", x)              # 喂给 kernel 的 input.bin
        golden = x / np.sqrt((x.astype(np.float32) ** 2).mean(-1, keepdims=True) + 1e-6)
        return golden.astype(np.float16).reshape(-1)  # 返回展平 golden

    CASES = [
        Case("rms_norm", "normalization/rms_norm/elf/<确切ELF名>.elf",
             prep_rms_norm, output_dtype=np.float16, atol=2e-2, rtol=2e-2),
    ]

`Case.elf` 是相对 output/solution 的路径（编译产物落点）。上层 run_precision 只按
`{status:7} {name:20} {detail}` 逐行取结果、summary 汇总，不感知任何算子。
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
    elf: str                       # 相对 output/solution，如 normalization/rms_norm/elf/x.elf
    prepare: callable
    output_name: str = "output.bin"
    output_dtype: object = np.float32
    atol: float = 1e-4
    rtol: float = 1e-4


def write(case_dir: Path, name: str, value: np.ndarray) -> None:
    np.ascontiguousarray(value).tofile(case_dir / name)


# ================================ CASES（算子侧维护）================================
# 本轮框架收敛只建入口骨架，不预置任何 golden；算子侧按上方范式往此表追加。
CASES: list[Case] = []


def compile_units(cases: list[Case], compiler_dir: Path, timeout: int) -> None:
    """自包含：对 CASES 涉及的算子目录注入 res_check=on 编译，产出带 I/O 桩的 ELF。

    与可用性阶段的普通 ELF 隔离由上层 run_precision 的前缀 rm -rf output/compare 保证。
    CASES 为空时不编译任何东西（返回 0 case）。"""
    units = sorted({str(Path(c.elf).parent.parent) for c in cases})   # <op>/elf/x.elf → <op>
    env = dict(os.environ, COMPILER_DIR=str(Path(compiler_dir).resolve()), baremetal="off")
    for rel in units:
        unit_dir = ROOT / "test/solution" / rel
        ca = unit_dir / "compile.all"
        if not ca.is_file():
            continue
        script = ca.read_text().replace("make ", "make res_check=on ")   # 只动 make 行
        subprocess.run(["bash", "-c", script], cwd=unit_dir, env=env,
                       timeout=timeout * 3, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)


def run_case(case: Case, gfrun: Path, timeout: int) -> tuple[str, str]:
    """与 multi_thread 版逐字一致的判定口径：prep 写 input.bin+golden → 4-PE gfrun
    （kernel 读 input.bin/写 output.bin）→ 读回 output.bin → np.allclose。"""
    elf = OUTPUT / case.elf
    if not elf.is_file():
        return "SKIP", f"missing ELF: {elf}"
    case_dir = COMPARE / elf.stem
    case_dir.mkdir(parents=True, exist_ok=True)
    golden = np.asarray(case.prepare(case_dir)).reshape(-1)
    np.zeros(golden.size, dtype=case.output_dtype).tofile(case_dir / case.output_name)
    command = [str(gfrun), "-s", "softcore.multiThreadNum=4", "-f", str(elf)]
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
