#!/usr/bin/env python3
"""Build and run the 21 SuperNPUBench solution regressions from LLVM #129.

Every case is built separately into a new OBJ_ROOT.  A case passes only when
make succeeds, exactly one fresh ELF exists, gfrun exits successfully within
120 seconds, and the architectural finisher reports R2 = 0.  Existing
independent numerical checkers are used for the cases that do not validate
their output in the benchmark itself.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from typing import Sequence


ONE_LEVEL = Path(__file__).resolve().parents[1]
SOLUTION = ONE_LEVEL / "test" / "solution"
PASS_FINISHER = re.compile(
    r"(?:Suaccelss|Success) to Reach the End of Benchmark!\s+R2\s*=\s*0(?:\s|$)"
)
ANY_FINISHER = re.compile(r"Reach the End of Benchmark!\s+R2\s*=\s*([^\s]+)")


@dataclass(frozen=True)
class Case:
    name: str
    unit: str
    make_vars: tuple[str, ...]
    pe_count: int
    oracle: str = "embedded"
    oracle_args: tuple[str, ...] = field(default_factory=tuple)


CASES = (
    Case("group_token_vec", "group_token_vec", ("TESTCASE=group_token_vec",), 1),
    Case("group_token_vec_mt", "group_token_vec", ("TESTCASE=group_token_vec_mt",), 4),
    Case("group_token_vec_mt_dyn", "group_token_vec", ("TESTCASE=group_token_vec_mt_dyn",), 4),
    Case("dmxq_nontail_cublas_fp8", "quant/dynamic_mx_quant",
         ("TESTCASE=dynamic_mx_quant", "TYPE=NONTAIL_CUBLAS_FP8"), 4,
         "dmxq", ("512", "256", "32", "fp16", "CUBLAS", "nontail", "FP8")),
    Case("gather_f32_tail", "gather_v2",
         ("TESTCASE=gather_v2", "DType=float", "IType=uint32_t", "RANK=1",
          "GATHER_DIM=0", "INPUT_ELEMENTS=19", "OUTPUT_ELEMENTS=37",
          "INDEX_ELEMENTS=37", "INPUT_SHAPE=19", "INPUT_SHAPE_NAME=19",
          "OUTPUT_SHAPE=37", "OUTPUT_SHAPE_NAME=37", "TILE_ELEMENTS=32",
          "ABS_TOL=0.001f", "REL_TOL=0.001f"), 4),
    Case("gather_i32_columns", "gather_v2",
         ("TESTCASE=gather_v2", "DType=int32_t", "IType=int32_t", "RANK=2",
          "GATHER_DIM=1", "INPUT_ELEMENTS=55", "OUTPUT_ELEMENTS=35",
          "INDEX_ELEMENTS=7", "INPUT_SHAPE=5,11", "INPUT_SHAPE_NAME=5_11",
          "OUTPUT_SHAPE=5,7", "OUTPUT_SHAPE_NAME=5_7", "TILE_ELEMENTS=32",
          "ABS_TOL=0.0f", "REL_TOL=0.0f"), 4),
    Case("gather_f16_rows_tail", "gather_v2",
         ("TESTCASE=gather_v2", "DType=__half", "IType=uint32_t", "RANK=2",
          "GATHER_DIM=0", "INPUT_ELEMENTS=187", "OUTPUT_ELEMENTS=136",
          "INDEX_ELEMENTS=8", "INPUT_SHAPE=11,17", "INPUT_SHAPE_NAME=11_17",
          "OUTPUT_SHAPE=8,17", "OUTPUT_SHAPE_NAME=8_17", "TILE_ELEMENTS=64",
          "ABS_TOL=0.001f", "REL_TOL=0.001f"), 4),
    Case("group_token_old", "group_token_old", ("TESTCASE=group_token_old",), 1),
    Case("group_token_old_mt", "group_token_old", ("TESTCASE=group_token_old_mt",), 4),
    Case("mega_moe_sim_mt", "mega_moe",
         ("TESTCASE=mega_moe_sim_mt", "BS=16", "H=32", "HIDDEN=64"), 4),
    Case("mega_moe_sim_mt_dyn", "mega_moe",
         ("TESTCASE=mega_moe_sim_mt_dyn", "BS=16", "H=32", "HIDDEN=64"), 4),
    Case("moe_combine_mt", "moe_combine", ("TESTCASE=moe_combine_mt",), 4),
    Case("moe_combine_mt_dyn", "moe_combine", ("TESTCASE=moe_combine_mt_dyn",), 4),
    Case("moe_dispatch_mt", "moe_dispatch", ("TESTCASE=moe_dispatch_mt",), 4),
    Case("moe_dispatch_mt_dyn", "moe_dispatch", ("TESTCASE=moe_dispatch_mt_dyn",), 4),
    Case("group_norm_grad_1d_dynamic", "normalization/group_norm_grad",
         ("TESTCASE=group_norm_grad_1d_dynamic", "DType=__half", "N_BATCH=256",
          "C_CH=512", "G_GRP=8", "PE_NUM=4"), 4, "group_norm_grad_1d"),
    Case("view_copy_i32_transpose", "view_copy",
         ("TESTCASE=view_copy", "DType=int32_t", "RANK=3", "ELEMENTS=64",
          "TILE_ELEMENTS=32", "SHAPE=2,4,8", "SHAPE_NAME=2_4_8",
          "INPUT_GLOBAL_STRIDE=32,1,4", "INPUT_GLOBAL_STRIDE_NAME=32_1_4",
          "OUTPUT_GLOBAL_STRIDE=32,8,1", "OUTPUT_GLOBAL_STRIDE_NAME=32_8_1",
          "INPUT_OFFSET_BYTES=32", "OUTPUT_OFFSET_BYTES=64",
          "ABS_TOL=0.0f", "REL_TOL=0.0f"), 4),
    Case("view_copy_f16_transpose", "view_copy",
         ("TESTCASE=view_copy", "DType=__half", "RANK=3", "ELEMENTS=128",
          "TILE_ELEMENTS=64", "SHAPE=4,4,8", "SHAPE_NAME=4_4_8",
          "INPUT_GLOBAL_STRIDE=32,8,1", "INPUT_GLOBAL_STRIDE_NAME=32_8_1",
          "OUTPUT_GLOBAL_STRIDE=32,1,4", "OUTPUT_GLOBAL_STRIDE_NAME=32_1_4",
          "INPUT_OFFSET_BYTES=64", "OUTPUT_OFFSET_BYTES=128",
          "ABS_TOL=0.001f", "REL_TOL=0.001f"), 4),
    Case("view_copy_f16_tail", "view_copy",
         ("TESTCASE=view_copy", "DType=__half", "RANK=3", "ELEMENTS=96",
          "TILE_ELEMENTS=64", "SHAPE=4,4,6", "SHAPE_NAME=4_4_6",
          "INPUT_GLOBAL_STRIDE=24,1,4", "INPUT_GLOBAL_STRIDE_NAME=24_1_4",
          "OUTPUT_GLOBAL_STRIDE=24,6,1", "OUTPUT_GLOBAL_STRIDE_NAME=24_6_1",
          "INPUT_OFFSET_BYTES=64", "OUTPUT_OFFSET_BYTES=96",
          "ABS_TOL=0.001f", "REL_TOL=0.001f"), 4),
    Case("conv2d_img2col", "conv2d", ("TESTCASE=conv2d_img2col",), 4, "conv2d"),
    Case("conv2d_img2col_dyn", "conv2d", ("TESTCASE=conv2d_img2col_dyn",), 4, "conv2d_dyn"),
)


# Explicit opt-in coverage; never changes the original 21-case denominator.
SUPPLEMENTAL_CASES = (
    Case("group_token_histogram_element", "group_token_old",
         ("TESTCASE=group_token_histogram_element",), 1),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_gfrun(returncode: int, output: str, timed_out: bool = False) -> tuple[bool, str]:
    if timed_out:
        return False, "timeout"
    if re.search(r"ASSERTION FAILED|(^|[^A-Za-z])assert(?:ion)?([^A-Za-z]|$)",
                 output, re.IGNORECASE):
        return False, "assertion"
    if returncode != 0:
        return False, f"gfrun_exit_{returncode}"
    if PASS_FINISHER.search(output):
        return True, "r2_zero"
    match = ANY_FINISHER.search(output)
    if match:
        return False, f"nonzero_finisher_{match.group(1)}"
    return False, "missing_finisher"


def find_fresh_elf(obj_root: Path) -> Path:
    elfs = sorted(obj_root.rglob("*.elf"))
    if len(elfs) != 1:
        raise RuntimeError(f"expected exactly one fresh ELF under {obj_root}, found {len(elfs)}")
    if not elfs[0].is_file() or elfs[0].stat().st_size == 0:
        raise RuntimeError(f"fresh ELF is missing or empty: {elfs[0]}")
    return elfs[0].resolve()


def run_logged(command: Sequence[str], *, cwd: Path, env: dict[str, str], log: Path,
               timeout: int | None = None,
               max_log_bytes: int = 8 * 1024 * 1024) -> tuple[int, bool]:
    started = time.monotonic()
    head_limit = min(1024 * 1024, max_log_bytes // 2)
    tail_limit = max_log_bytes - head_limit
    head = bytearray()
    tail: deque[bytes] = deque()
    tail_size = 0
    total_size = 0
    process = subprocess.Popen([str(value) for value in command], cwd=cwd, env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               start_new_session=True)

    def drain() -> None:
        nonlocal tail_size, total_size
        assert process.stdout is not None
        for chunk in iter(lambda: process.stdout.read(64 * 1024), b""):
            total_size += len(chunk)
            if len(head) < head_limit:
                take = min(head_limit - len(head), len(chunk))
                head.extend(chunk[:take])
                chunk = chunk[take:]
            if not chunk or tail_limit == 0:
                continue
            tail.append(chunk)
            tail_size += len(chunk)
            while tail_size > tail_limit and tail:
                excess = tail_size - tail_limit
                first = tail[0]
                if len(first) <= excess:
                    tail.popleft()
                    tail_size -= len(first)
                else:
                    tail[0] = first[excess:]
                    tail_size -= excess

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    try:
        rc, timed_out = process.wait(timeout=timeout), False
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        rc, timed_out = 124, True
    except BaseException:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        reader.join()
        assert process.stdout is not None
        process.stdout.close()
        raise
    reader.join()
    assert process.stdout is not None
    process.stdout.close()
    with log.open("wb") as stream:
        stream.write(("+ " + shlex.join([str(value) for value in command]) + "\n").encode())
        stream.write(head)
        if total_size > max_log_bytes:
            omitted = total_size - len(head) - tail_size
            stream.write(f"\n... {omitted} output bytes omitted by bounded log ...\n".encode())
        for chunk in tail:
            stream.write(chunk)
        stream.write((f"\nrunner_exit={rc} timed_out={str(timed_out).lower()} "
                      f"elapsed_seconds={time.monotonic() - started:.3f}\n").encode())
    return rc, timed_out


def external_flags(args: argparse.Namespace, *, hosted: bool) -> tuple[str, str]:
    target_flags = [f"--target={args.target_triple}", f"--sysroot={args.sysroot}",
                    f"-resource-dir={args.resource_dir}"]
    compile_flags = target_flags + [f"-I{args.api_include}"]
    link_flags = target_flags + [f"-fuse-ld={args.linker}"]
    if not hosted:
        startup = ONE_LEVEL / "test" / "common" / "_start.s"
        link_flags += ["-nostartfiles", str(startup)]
    return shlex.join(compile_flags), shlex.join(link_flags)


def external_tool_vars(args: argparse.Namespace) -> list[str]:
    flags = shlex.join([f"--target={args.target_triple}", f"--sysroot={args.sysroot}",
                        f"-resource-dir={args.resource_dir}"])
    return [f"CXX={args.compiler_dir / 'clang++'} {flags}",
            f"CC={args.compiler_dir / 'clang'} {flags}",
            f"AS={args.compiler_dir / 'clang'} {flags}",
            f"LINK={args.compiler_dir / 'clang++'}"]


def prepare_oracle(case: Case, case_dir: Path,
                   elf_stem: str) -> tuple[Path | None, list[str] | None]:
    src = SOLUTION / case.unit / "src"
    if case.oracle == "dmxq":
        m, k, block, in_dtype, algo, kernel, dtype = case.oracle_args
        check_dir = case_dir / "compare" / elf_stem
        check_dir.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(src / "gen_dynamic_mx_quant_data.py"),
                   "--M", m, "--K", k, "--block-size", block, "--algo", algo,
                   "--kernel", kernel, "--dtype", dtype, "--in-dtype", in_dtype,
                   "--scale-layout", "compact", "--seed", "42", "-o", str(check_dir)]
        return check_dir, command
    if case.oracle == "group_norm_grad_1d":
        check_dir = case_dir / "compare" / elf_stem
        check_dir.mkdir(parents=True, exist_ok=True)
        return check_dir, [sys.executable, str(src / "gen_group_norm_grad_1d_data.py"),
                           "--out-dir", str(check_dir)]
    return None, None


def oracle_check_command(case: Case, case_dir: Path, elf: Path,
                         gfrun: Path) -> list[str] | None:
    src = SOLUTION / case.unit / "src"
    check_root = case_dir / "compare"
    check_dir = check_root / elf.stem
    if case.oracle == "dmxq":
        return [sys.executable, str(src / "dynamic_mx_quant_data_compare.py"),
                "-d", str(elf), "--dtype", case.oracle_args[-1],
                "--scale-layout", "compact", "--cmp-root", str(check_root)]
    if case.oracle == "group_norm_grad_1d":
        return [sys.executable, str(src / "group_norm_grad_1d_data_compare.py"),
                "--cmp-dir", str(check_dir)]
    if case.oracle in ("conv2d", "conv2d_dyn"):
        script = "golden_conv2d_img2col.py" if case.oracle == "conv2d" else "golden_conv2d_img2col_dyn.py"
        return [sys.executable, str(src / script), "--elf", str(elf), "--gfrun", str(gfrun)]
    return None


ORACLE_STEMS = {
    "dmxq": "dynamic_mx_quant_nontail_cublas_fp8",
    "group_norm_grad_1d":
        "solution_normalization_group_norm_grad_group_norm_grad_1d_dynamic_DType__half_N256_C512_G8_PE4",
    "conv2d": "solution_conv2d_conv2d_img2col_PE4",
    "conv2d_dyn": "solution_conv2d_conv2d_img2col_dyn_PE4",
}


def run_external_oracle(case: Case, args: argparse.Namespace,
                        base_env: dict[str, str], case_dir: Path) -> tuple[bool, str]:
    oracle_dir = case_dir / "oracle"
    obj_root = oracle_dir / "obj"
    oracle_dir.mkdir()
    obj_root.mkdir()
    compile_flags, link_flags = external_flags(args, hosted=True)
    stem = ORACLE_STEMS[case.oracle]
    if case.oracle.startswith("conv2d"):
        check_dir = ONE_LEVEL / "compare" / stem
        check_dir.mkdir(parents=True, exist_ok=True)
    else:
        check_dir = oracle_dir / "compare" / stem
        check_dir.mkdir(parents=True, exist_ok=True)
    make_vars = [*case.make_vars, f"COMPILER_DIR={args.compiler_dir}",
                 *external_tool_vars(args),
                 f"OBJ_ROOT={obj_root}", f"TARGET_TRIPLE={args.target_triple}",
                 f"SYSROOT={args.sysroot}", f"RESOURCE_DIR={args.resource_dir}",
                 f"API_INCLUDE={args.api_include}", f"CFLAGS={compile_flags}",
                 f"CC_LINK={link_flags}", f"DUMP={args.objdump}",
                 "res_check=on", f"CHK_DIR={check_dir}"]
    _, prepare_command = prepare_oracle(case, oracle_dir, stem)
    if prepare_command:
        rc, timed_out = run_logged(prepare_command, cwd=SOLUTION / case.unit,
                                   env=base_env, log=oracle_dir / "prepare.log",
                                   timeout=args.timeout)
        if rc != 0 or timed_out:
            return False, "oracle_prepare_failed"
    rc, timed_out = run_logged(["make", *make_vars, "diss"],
                               cwd=SOLUTION / case.unit, env=base_env,
                               log=oracle_dir / "compile.log",
                               timeout=args.compile_timeout)
    if rc != 0 or timed_out:
        return False, "oracle_compile_failed"
    try:
        elf = find_fresh_elf(obj_root)
    except RuntimeError:
        return False, "oracle_missing_fresh_elf"

    checker = oracle_check_command(case, oracle_dir, elf, args.gfrun)
    if case.oracle.startswith("conv2d"):
        assert checker is not None
        rc, timed_out = run_logged(checker, cwd=SOLUTION / case.unit,
                                   env=base_env, log=oracle_dir / "check.log",
                                   timeout=args.timeout)
        if rc != 0 or timed_out:
            return False, "oracle_check_failed"

    command = [str(args.gfrun), "--pto059-execution-mask-dev", "-s",
               f"softcore.multiThreadNum={case.pe_count}", "-f", str(elf)]
    rc, timed_out = run_logged(command, cwd=args.model_root, env=base_env,
                               log=oracle_dir / "gfrun.log", timeout=args.timeout)
    output = (oracle_dir / "gfrun.log").read_text(errors="replace")
    passed, status = classify_gfrun(rc, output, timed_out)
    if not passed:
        return False, "oracle_" + status

    if checker and not case.oracle.startswith("conv2d"):
        rc, timed_out = run_logged(checker, cwd=SOLUTION / case.unit,
                                   env=base_env, log=oracle_dir / "check.log",
                                   timeout=args.timeout)
        if rc != 0 or timed_out:
            return False, "oracle_check_failed"
    return True, "oracle_pass"


def check_histogram_lowering(case: Case, args: argparse.Namespace,
                             base_env: dict[str, str], case_dir: Path,
                             make_vars: list[str], elf: Path) -> tuple[bool, str]:
    """Numerical success cannot hide scalarization of the migrated helpers."""
    if case.name not in ("group_token_old", "group_token_vec_mt_dyn"):
        return True, "not_applicable"
    import check_moe_element_histograms as checker

    dynamic = case.name == "group_token_vec_mt_dyn"
    source = ONE_LEVEL / "kernels/solution" / (
        "group_token_vec/group_token_vec_mt_dyn_histogram.hpp" if dynamic else
        "group_token_old/group_token_old.hpp")
    ir = case_dir / "element-histogram.ll"
    target = "histogram-ir" if dynamic else "ir"
    rc, timed_out = run_logged(
        ["make", *make_vars, f"IR_TARGET={ir}", target],
        cwd=SOLUTION / case.unit, env=base_env,
        log=case_dir / "element-ir-compile.log", timeout=args.compile_timeout)
    if rc != 0 or timed_out or not ir.is_file():
        return False, "element_ir_compile_failed"
    disassembly = case_dir / "element-histogram.dis"
    rc, timed_out = run_logged(
        [str(args.objdump), "-d", str(elf)], cwd=case_dir, env=base_env,
        log=disassembly, timeout=args.compile_timeout)
    if rc != 0 or timed_out:
        return False, "element_disassembly_failed"
    try:
        checker.validate_artifact(source.read_text(), ir.read_text(),
                                  disassembly.read_text(),
                                  checker.DYNAMIC if dynamic else checker.OLD)
    except checker.CheckError as error:
        (case_dir / "element-lowering-check.log").write_text(str(error) + "\n")
        return False, "element_lowering_failed"
    (case_dir / "element-lowering-check.log").write_text(
        "PASS: exact helper source, masked IR and native Tile data path\n")
    return True, "masked_tile_pass"


def run_case(case: Case, args: argparse.Namespace, base_env: dict[str, str]) -> dict[str, object]:
    case_dir = args.output / "cases" / case.name
    obj_root = case_dir / "obj"
    case_dir.mkdir(parents=True)
    obj_root.mkdir()
    result: dict[str, object] = {"case": case.name, "unit": case.unit,
                                "pe_count": case.pe_count, "oracle": case.oracle,
                                "passed": False}
    compile_flags, link_flags = external_flags(args, hosted=False)
    make_vars = [*case.make_vars, f"COMPILER_DIR={args.compiler_dir}",
                 *external_tool_vars(args),
                 f"OBJ_ROOT={obj_root}", f"TARGET_TRIPLE={args.target_triple}",
                 f"SYSROOT={args.sysroot}", f"RESOURCE_DIR={args.resource_dir}",
                 f"API_INCLUDE={args.api_include}", f"CFLAGS={compile_flags}",
                 f"CC_LINK={link_flags}", f"DUMP={args.objdump}"]

    make_command = ["make", *make_vars, "diss"]
    rc, timed_out = run_logged(make_command, cwd=SOLUTION / case.unit, env=base_env,
                               log=case_dir / "compile.log", timeout=args.compile_timeout)
    result["compile_exit"] = rc
    if rc != 0 or timed_out:
        result["status"] = "compile_timeout" if timed_out else "compile_failed"
        return result
    try:
        elf = find_fresh_elf(obj_root)
    except RuntimeError as error:
        result.update(status="missing_fresh_elf", error=str(error))
        return result
    result.update(elf=str(elf), elf_sha256=sha256(elf))

    lowering_passed, lowering_status = check_histogram_lowering(
        case, args, base_env, case_dir, make_vars, elf)
    result["element_lowering_status"] = lowering_status
    if not lowering_passed:
        result["status"] = lowering_status
        return result

    command = [str(args.gfrun), "--pto059-execution-mask-dev", "-s",
               f"softcore.multiThreadNum={case.pe_count}", "-f", str(elf)]
    rc, timed_out = run_logged(command, cwd=args.model_root, env=base_env,
                               log=case_dir / "gfrun.log", timeout=args.timeout)
    output = (case_dir / "gfrun.log").read_text(errors="replace")
    passed, status = classify_gfrun(rc, output, timed_out)
    result.update(gfrun_exit=rc, status=status)
    if not passed:
        return result

    if case.oracle != "embedded":
        oracle_passed, oracle_status = run_external_oracle(
            case, args, base_env, case_dir)
        result["oracle_status"] = oracle_status
        if not oracle_passed:
            result["status"] = oracle_status
            return result
    result.update(passed=True, status="pass")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler-dir", required=True, type=Path)
    parser.add_argument("--linker", type=Path,
                        help="ld.lld path; defaults to COMPILER_DIR/ld.lld")
    parser.add_argument("--objdump", type=Path,
                        help="llvm-objdump path; defaults to COMPILER_DIR/llvm-objdump")
    parser.add_argument("--gfrun", required=True, type=Path)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--sysroot", required=True, type=Path)
    parser.add_argument("--resource-dir", required=True, type=Path)
    parser.add_argument("--api-include", type=Path,
                        help="TileOP API include root; defaults to RESOURCE_DIR/include/tileop-api")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-triple", default="linx64v5-unknown-linux-musl")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--compile-timeout", type=int, default=1200)
    parser.add_argument("--only", action="append", default=[])
    args = parser.parse_args(argv)
    args.compiler_dir = args.compiler_dir.resolve()
    # Keep the ld.lld basename when it is a symlink to the generic `lld`
    # multicall binary; resolving it makes the driver select the wrong mode.
    args.linker = Path(os.path.abspath(
        args.linker or args.compiler_dir / "ld.lld"))
    args.objdump = (args.objdump or args.compiler_dir / "llvm-objdump").resolve()
    args.gfrun = args.gfrun.resolve()
    args.model_root = args.model_root.resolve()
    args.sysroot = args.sysroot.resolve()
    args.resource_dir = args.resource_dir.resolve()
    args.api_include = (args.api_include or args.resource_dir / "include" / "tileop-api").resolve()
    args.output = args.output.resolve()
    if not 1 <= args.timeout <= 120:
        parser.error("--timeout must be between 1 and 120 seconds")
    if args.output.exists():
        parser.error(f"--output must not already exist: {args.output}")
    required = [args.compiler_dir / "clang++", args.compiler_dir / "clang",
                args.linker, args.objdump, args.gfrun]
    for path in required:
        if not path.is_file():
            parser.error(f"required executable is missing: {path}")
    for path in (args.model_root, args.sysroot, args.resource_dir, args.api_include):
        if not path.is_dir():
            parser.error(f"required directory is missing: {path}")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    selected = list(CASES)
    if args.only:
        wanted = set(args.only)
        selected = [case for case in (*CASES, *SUPPLEMENTAL_CASES) if case.name in wanted]
        missing = sorted(wanted - {case.name for case in selected})
        if missing:
            print(f"unknown cases: {', '.join(missing)}", file=sys.stderr)
            return 2
    args.output.mkdir(parents=True)
    env = os.environ.copy()
    env["GFRUN_FORCE_DIRECTBOOT_ABI"] = "1"
    rows = []
    for case in selected:
        print(f"[{len(rows) + 1}/{len(selected)}] {case.name}", flush=True)
        row = run_case(case, args, env)
        rows.append(row)
        print(f"  {'PASS' if row['passed'] else 'FAIL'}: {row['status']}", flush=True)
    report = {
        "schema_version": 1,
        "suite": "llvm-129-solution-gfrun-regressions",
        "case_count": len(rows),
        "passed_count": sum(bool(row["passed"]) for row in rows),
        "timeout_seconds": args.timeout,
        "compiler": {"path": str(args.compiler_dir),
                     "clang_sha256": sha256(args.compiler_dir / "clang"),
                     "linker": str(args.linker), "linker_sha256": sha256(args.linker),
                     "objdump": str(args.objdump), "objdump_sha256": sha256(args.objdump)},
        "gfrun": {"path": str(args.gfrun), "sha256": sha256(args.gfrun)},
        "sysroot": str(args.sysroot),
        "resource_dir": str(args.resource_dir),
        "api_include": str(args.api_include),
        "cases": rows,
    }
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["passed_count"] == report["case_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
