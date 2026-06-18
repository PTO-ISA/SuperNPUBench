# SuperNPUBench

SuperNPUBench is a TileOP API, kernel, model, and benchmark workspace for
LinxISA/PTO-style tile programming experiments. The repository is organized
around header-only TileOP APIs, reusable kernels, model-level examples, and
make-driven test suites.

## Repository Map

| Path | Purpose |
| --- | --- |
| [`include/common`](include/common) | Shared TileOP API surface, data types, tensor helpers, layouts, and compile-time utilities. |
| [`include/cpu_sim`](include/cpu_sim) | CPU simulation backend used by tests built with `PLAT=cpu`. |
| [`include/aarch64`](include/aarch64) | Host/Arm-oriented backend headers and TileOP API variants. |
| [`include/accelerator`](include/accelerator) | Accelerator-facing headers split by versioned targets such as `v220` and `v310`. |
| [`include/jcore`](include/jcore) | Linx/JCore backend headers used by tests built with `PLAT=linx`. |
| [`kernels`](kernels) | Reusable kernel implementations, grouped by domain such as element-wise, memory, reduction, and matmul. |
| [`models`](models) | Model-level examples and compositions, currently including DeepSeekV3-oriented code. |
| [`test/common`](test/common) | Shared make harness, platform selection, compiler flags, and simulator targets. |
| [`test/tileop_api`](test/tileop_api) | Focused TileOP API compile and behavior tests. Start here for single-operation API work. |
| [`test/py_api`](test/py_api) | Python extension and golden-comparison flow for TileOP API checks. |
| [`test/accelerator`](test/accelerator) | Accelerator-oriented benchmark and validation suites. |
| [`test/kernel`](test/kernel) | Kernel benchmark and validation suites. |
| [`test/other`](test/other) | Additional model, microbenchmark, TileOP, vector, and script-driven suites. |
| [`test/script`](test/script) | Recursive compile/run helper for batch workflows. |
| `output/` | Generated build products. Treat this as local output, not source. |

## Quick Navigation

- To understand the public API, start with [`include/common`](include/common).
- To inspect a backend implementation of an operation, compare the matching
  headers under [`include/cpu_sim`](include/cpu_sim),
  [`include/jcore`](include/jcore), and [`include/aarch64`](include/aarch64).
- To add or update reusable compute code, use the matching domain under
  [`kernels`](kernels).
- To compile one small API test, use [`test/tileop_api`](test/tileop_api).
- To validate a Python-facing API path, use [`test/py_api`](test/py_api) and
  [`test/py_api/golden_cmp`](test/py_api/golden_cmp).
- To run larger suites, use the `compile.all` files in the relevant
  [`test`](test) subdirectories.

## Header Installation

The top-level `Makefile` installs the TileOP header tree into the active
Clang resource directory under `include/tileop-api`.

```sh
make -n CLANG_PREFIX=/usr
make install CLANG_PREFIX=/path/to/clang/prefix
make uninstall CLANG_PREFIX=/path/to/clang/prefix
```

`CLANG_PREFIX` should point to a prefix containing `bin/clang`. The dry run is
useful because the install location is derived from
`clang -print-resource-dir`.

## Building Tests

Most test directories include [`test/common/Makefile.common`](test/common/Makefile.common).
The common harness is controlled mainly by `TESTCASE`, `PLAT`,
`COMPILER_DIR`, and `QEMU`.

```sh
cd test/tileop_api
make clean
make TESTCASE=TAdd PLAT=cpu COMPILER_DIR=/usr/bin
make TESTCASE=TAdd PLAT=linx COMPILER_DIR=/path/to/linx/compiler/bin
make TESTCASE=TAdd PLAT=linx QEMU=/path/to/qemu-linx sim
```

| Variable | Meaning |
| --- | --- |
| `TESTCASE` | Source basename to build, for example `TAdd` in `test/tileop_api/src/TAdd.cpp`. |
| `PLAT=cpu` | Builds against the CPU simulation backend and defines `__cpu_sim__`. |
| `PLAT=linx` | Builds for the Linx target and defines `__linx`. |
| `PLAT=arm_sme` | Builds Arm SME-oriented cases and defines `__ARM_FEATURE_SME`. |
| `COMPILER_DIR` | Directory containing `clang`, `clang++`, `llvm-objdump`, and related tools. |
| `QEMU` | Simulator binary used by `make sim` for Linx-targeted test execution. |

Common targets:

```sh
make TESTCASE=TAdd all
make TESTCASE=TAdd diss
make TESTCASE=TAdd sim
make TESTCASE=TAdd debug
make clean
make clean_all
```

Generated objects, executables, disassembly, and logs are written below
`output/`.

## Batch Suites

Many suites provide a local `compile.all` file. Run these from the suite
directory so relative paths and local make variables resolve as intended.

Examples:

```sh
cd test/tileop_api && bash compile.all
cd test/py_api && bash compile.all
cd test/accelerator/vec_simt && bash compile.all
cd test/kernel/gemm/matmul && bash compile.all
```

For recursive compile/run automation, see
[`test/script/README.md`](test/script/README.md).

## Python API And Golden Comparison

The Python API flow builds a shared object and compares selected cases against
Python golden logic.

```sh
cd test/py_api
make clean
make TESTCASE=tileop_py
python3 golden_cmp/golden_cmp.py -i tadd
```

For adding new golden cases, see
[`test/py_api/golden_cmp/README.md`](test/py_api/golden_cmp/README.md).

## Adding Work

Use the existing directory shape when adding code:

1. Add API-facing definitions or declarations in [`include/common`](include/common).
2. Add backend behavior in the relevant backend directory, usually
   [`include/cpu_sim`](include/cpu_sim), [`include/jcore`](include/jcore), or
   [`include/aarch64`](include/aarch64).
3. Add reusable compute kernels under the matching [`kernels`](kernels)
   domain.
4. Add focused tests under [`test/tileop_api`](test/tileop_api) or the
   relevant suite under [`test/kernel`](test/kernel),
   [`test/accelerator`](test/accelerator), or [`test/other`](test/other).
5. Add the case to the local `compile.all` file when it should be part of the
   batch suite.

New make-driven test directories should keep the local `Makefile` small and
include [`test/common/Makefile.common`](test/common/Makefile.common) for shared
platform flags, output paths, simulator targets, and cleanup behavior.

## Generated Files

Do not commit generated files from `output/`, object files, executable test
artifacts, local logs, or disassembly files. Keep source changes in `include/`,
`kernels/`, `models/`, and `test/`.
