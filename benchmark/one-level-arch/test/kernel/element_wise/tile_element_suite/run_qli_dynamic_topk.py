#!/usr/bin/env python3
"""Compile and run guarded runtime-shape QLI TopK profiles on gfrun."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import qli_dynamic_topk_fixture as fixture
from qli_dynamic_topk_native import check_native
from compare_memory import load_symbols
from run_qli_element import OPTIONS


HERE = Path(__file__).resolve().parent
BENCH = HERE.parents[3]
sys.path.insert(0, str(BENCH / "verification"))
from gfrun_dump import (assemble_dump, dump_specs, has_success_finisher,
                        plan_dump)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def kernel_source(case, info):
    return f'''#include <common/pto_tileop.hpp>
#include "benchmark.h"
#include "solution/qli/qli_pto_opt_dynamic.hpp"
extern "C" uint32_t qli_dynamic_workspace[];
volatile int qli_runtime_sq = {case.sq};
volatile int qli_runtime_skv = {case.skv};
volatile int qli_runtime_topk = {case.topk};
volatile int qli_runtime_num_pes = {case.num_pes};
int main() {{
  float *scores = reinterpret_cast<float *>(qli_dynamic_workspace);
  int32_t *indices = reinterpret_cast<int32_t *>(
      qli_dynamic_workspace + {info["index_start"]});
  uint32_t *keys = qli_dynamic_workspace + {info["key_start"]};
  BENCHSTART;
  qli_topk_radix_dynamic(scores, indices, qli_runtime_sq, qli_runtime_skv,
                         qli_runtime_topk, qli_runtime_num_pes, keys);
  BENCHEND;
  return 0;
}}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("compiler-bin", "sysroot", "resource-dir", "api-include",
                   "gfrun", "out"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--case", choices=[case.name for case in fixture.CASES],
                        action="append")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    tools = args.compiler_bin.resolve()
    selected = [case for case in fixture.CASES
                if args.case is None or case.name in args.case]

    sources = [Path(__file__).resolve(), Path(fixture.__file__).resolve(),
               HERE / "qli_dynamic_topk_native.py",
               HERE / "test_qli_dynamic_topk_fixture.py",
               HERE / "test_qli_dynamic_topk_native.py",
               HERE / "check_disassembly.py", HERE / "compare_memory.py",
               HERE / "run_qli_element.py", BENCH / "verification/gfrun_dump.py",
               BENCH / "kernels/solution/qli/qli_pto_opt_dynamic.hpp",
               BENCH / "kernels/solution/qli/qli_element.hpp",
               BENCH / "test/common/_start.s", args.gfrun.resolve()]
    sources += [tools / name for name in
                ("clang++", "llc", "ld.lld", "llvm-objdump", "opt")]
    for name in ("evidence.json", "SHA256SUMS"):
        evidence = tools / name
        if evidence.exists():
            sources.append(evidence)
    for evidence in (tools.parent / "evidence.json",
                     args.gfrun.resolve().parent / "evidence.json"):
        if evidence.exists():
            sources.append(evidence)
    sources += list(args.api_include.rglob("*.hpp"))
    sources += list((args.resource_dir / "include").rglob("*.hpp"))
    before = {str(path.resolve()): sha(path) for path in sources}
    compiler_version = subprocess.run(
        [tools / "clang++", "--version"], check=True,
        capture_output=True, text=True).stdout
    head = re.search(r"\b[0-9a-f]{40}\b", compiler_version)
    provenance = {
        "hashes": before,
        "compiler_head": head.group(0) if head else None,
        "compiler_version": compiler_version.splitlines()[0],
        "model": "gfrun", "pe_count": 1, "timeout_seconds": 120,
        "scope": "runtime-shape QLI TopK full fixture-owned memory",
        "cases": [case.name for case in selected],
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    commands, reports = [], []

    def run(command, log, timeout=600):
        command = list(map(str, command))
        commands.append(command)
        (out / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        with log.open("w") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                           check=True, timeout=timeout)

    target = ["--target=linx64v5-unknown-linux-musl",
              "--sysroot=" + str(args.sysroot),
              "-resource-dir=" + str(args.resource_dir)]
    for index, case in enumerate(selected):
        directory = out / f"{index:02d}-{case.name}"
        directory.mkdir()
        fixture_dir = directory / "fixture"
        _initial, _expected, _rows, info = fixture.generate(fixture_dir, case)
        fixture_hashes = {str(path): sha(path) for path in fixture_dir.iterdir()
                          if path.is_file()}
        (directory / "kernel.cpp").write_text(kernel_source(case, info))
        report = {"case": case.name, "status": "FAIL"}
        try:
            flags = ["-c", "-mlxbc", "-fenable-matrix", "-O2", "-std=c++20",
                     "-fno-vectorize", "-fno-slp-vectorize",
                     "-Werror=unknown-pragmas", "-D__linx",
                     "-DENABLE_TENSOR_INSTR", "-I" + str(args.api_include),
                     "-I" + str(BENCH / "kernels"),
                     "-I" + str(BENCH / "test/common"),
                     "-I" + str(BENCH / "test/common/src")]
            for option in OPTIONS:
                flags += ["-mllvm", "-" + option]
            run([tools / "clang++", *target, *flags, "-S", "-emit-llvm",
                 directory / "kernel.cpp", "-o", directory / "kernel.ll"],
                directory / "frontend.log")
            run([tools / "llc", "-mtriple=linx64v5-unknown-linux-musl", "-O2",
                 "-filetype=obj", *["-" + option for option in OPTIONS],
                 directory / "kernel.ll", "-o", directory / "kernel.o"],
                directory / "backend.log")
            run([tools / "clang++", *target, "-nostartfiles", "-fuse-ld=lld",
                 BENCH / "test/common/_start.s", fixture_dir / "data.s",
                 directory / "kernel.o", "-o", directory / "kernel.elf"],
                directory / "link.log")
            run([tools / "llvm-objdump", "-d",
                 "--disassembler-options=no-tile-macros", directory / "kernel.elf"],
                directory / "kernel.diss")
            native = check_native((directory / "kernel.diss").read_text(),
                                  (directory / "kernel.ll").read_text())
            (directory / "native.json").write_text(json.dumps(native, indent=2) + "\n")
            run([tools / "llvm-objdump", "-t", directory / "kernel.elf"],
                directory / "symbols.txt")
            symbols = load_symbols(directory / "symbols.txt")
            workspace = symbols["qli_dynamic_workspace"]
            before_guard = symbols["qli_dynamic_workspace_before"]
            after_guard = symbols["qli_dynamic_workspace_after"]
            expected_bytes = (fixture_dir / "workspace.expected.bin").read_bytes()
            if workspace[1] != len(expected_bytes):
                raise ValueError("linked dynamic workspace extent changed")
            if (before_guard[0] + before_guard[1] != workspace[0] or
                    workspace[0] + workspace[1] != after_guard[0]):
                raise ValueError("dynamic workspace guards are not adjacent")
            observations = {
                "workspace_before": before_guard,
                "workspace": workspace,
                "workspace_after": after_guard,
            }
            plans = {name: plan_dump(f"{address:#x}:{size:#x}",
                                     directory / (name + ".dump"))
                     for name, (address, size) in observations.items()}
            specs = [item for chunks in plans.values() for item in dump_specs(chunks)]
            command = [args.gfrun, "--pto059-execution-mask-dev", "-s",
                       "softcore.multiThreadNum=1", "--dump-force",
                       "--dump-memory", *specs, "-f", directory / "kernel.elf"]
            started = time.monotonic()
            timed_out, exit_code = False, None
            try:
                run(command, directory / "gfrun.log", timeout=120)
                exit_code = 0
            except subprocess.TimeoutExpired:
                timed_out, exit_code = True, 124
                raise
            except subprocess.CalledProcessError as error:
                exit_code = error.returncode
                raise
            finally:
                (directory / "gfrun-run.json").write_text(json.dumps({
                    "elapsed_seconds": time.monotonic() - started,
                    "timeout_seconds": 120, "timed_out": timed_out,
                    "exit_code": exit_code,
                }, indent=2) + "\n")
            if not has_success_finisher((directory / "gfrun.log").read_text()):
                raise ValueError("missing successful merged-model finisher")
            for name, chunks in plans.items():
                assemble_dump(chunks, directory / (name + ".dump"))
            if (directory / "workspace_before.dump").read_bytes() != b"\xA5" * 64:
                raise ValueError("dynamic workspace leading guard changed")
            if (directory / "workspace_after.dump").read_bytes() != b"\x3C" * 64:
                raise ValueError("dynamic workspace trailing guard changed")
            fixture.compare((directory / "workspace.dump").read_bytes(),
                            expected_bytes, case)
            if fixture_hashes != {path: sha(path) for path in fixture_hashes}:
                raise ValueError("dynamic QLI fixture artifact changed during run")
            report.update(status="PASS", segments=len(observations),
                          observed_bytes=sum(size for _, size in observations.values()),
                          elf_sha256=sha(directory / "kernel.elf"), native=native)
        except subprocess.TimeoutExpired as error:
            report.update(status="TIMEOUT", error=str(error))
        except (subprocess.CalledProcessError, AssertionError, ValueError,
                RuntimeError, KeyError) as error:
            report["error"] = str(error)
        reports.append(report)
        (directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        (out / "summary.json").write_text(json.dumps({
            "cases": reports,
            "passed": sum(row["status"] == "PASS" for row in reports),
            "total": len(reports),
        }, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    if before != {path: sha(path) for path in before}:
        raise ValueError("dynamic QLI source/tool/API/model changed during run")
    return 0 if all(report["status"] == "PASS" for report in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
