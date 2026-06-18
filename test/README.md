# Test Navigation

The `test` tree contains focused API tests, kernel and accelerator suites,
Python golden-comparison tests, and batch scripts. Most make-driven suites
reuse [`common/Makefile.common`](common/Makefile.common), so the same
`TESTCASE`, `PLAT`, `COMPILER_DIR`, and `QEMU` variables work across many
directories.

## Directory Map

| Path | Use it for |
| --- | --- |
| [`common`](common) | Shared make rules, platform flags, output layout, and simulator targets. |
| [`tileop_api`](tileop_api) | Small TileOP API tests. This is the best first stop for validating an individual API operation. |
| [`py_api`](py_api) | Python extension build and golden-comparison tests. |
| [`accelerator`](accelerator) | Accelerator-oriented suites such as cube, vector, DMA, fusion, and versioned target tests. |
| [`kernel`](kernel) | Kernel suites for control, element-wise, fusion, GEMM, memory, reduction, sort, and related cases. |
| [`other`](other) | Additional model, microbenchmark, TileOP, vector, and script-driven suites. |
| [`script`](script) | Recursive compile/run helper for larger batch workflows. |

## Common Build Pattern

```sh
cd test/tileop_api
make clean
make TESTCASE=TAdd PLAT=cpu COMPILER_DIR=/usr/bin
make TESTCASE=TAdd PLAT=linx COMPILER_DIR=/path/to/linx/compiler/bin
make TESTCASE=TAdd PLAT=linx QEMU=/path/to/qemu-linx sim
```

Platform values:

| Platform | Backend |
| --- | --- |
| `PLAT=cpu` | CPU simulation backend with `__cpu_sim__`. |
| `PLAT=linx` | Linx target backend with `__linx`. |
| `PLAT=arm_sme` | Arm SME-oriented backend with `__ARM_FEATURE_SME`. |

Common targets:

```sh
make TESTCASE=<case> all
make TESTCASE=<case> diss
make TESTCASE=<case> sim
make TESTCASE=<case> debug
make clean
make clean_all
```

Build products are written below the repository-level `output/` directory.

## Batch Runs

Several suites include a local `compile.all` file. Run it from the suite
directory:

```sh
cd test/tileop_api && bash compile.all
cd test/py_api && bash compile.all
cd test/kernel/gemm/matmul && bash compile.all
cd test/accelerator/vec_simt && bash compile.all
```

For recursive compile/run automation, see [`script/README.md`](script/README.md).

## Python Golden Comparison

```sh
cd test/py_api
make clean
make TESTCASE=tileop_py
python3 golden_cmp/golden_cmp.py -i tadd
```

For adding golden-comparison cases, see
[`py_api/golden_cmp/README.md`](py_api/golden_cmp/README.md).

## Adding A Test Case

For an existing make-driven suite:

1. Add the source file under that suite's `src/` directory.
2. Set `SRC_FILE`, `TARGET`, and any suite-specific variables in the local
   `Makefile`.
3. Include [`common/Makefile.common`](common/Makefile.common).
4. Add the case to the local `compile.all` file if it belongs in batch runs.

Minimal local makefile shape:

```make
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/$(TESTCASE).cpp
TARGET = $(ELF_HEAD)_$(TESTCASE).elf
include ../common/Makefile.common
```

Adjust the relative include path when the suite is nested more deeply.

For a new suite, create a directory with `src/`, a small local `Makefile`, and
an optional `compile.all` batch entrypoint.

Back to the repository overview: [`../README.md`](../README.md).
