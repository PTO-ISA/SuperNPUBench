# Test Navigation

The `test` tree contains kernel and accelerator test suites. Most make-driven
suites reuse [`common/Makefile.common`](common/Makefile.common), so the same
`TESTCASE`, `PLAT`, `COMPILER_DIR`, and `QEMU` variables work across many
directories.

## Directory Map

| Path | Use it for |
| --- | --- |
| [`common`](common) | Shared make rules, platform flags, output layout, and simulator targets. |
| [`accelerator`](accelerator) | Accelerator-oriented suites: fusion tests and shared headers. |
| [`kernel`](kernel) | Kernel suites for various operations (broadcast, concat, control, element_wise, fa, gather, matmul, reduction, sort, transpose). |

## Kernel Test Suites

The [`kernel`](kernel) directory contains focused test suites:

| Directory | Operations |
| --- | --- |
| [`kernel/broadcast`](kernel/broadcast) | Broadcast operation tests with various shapes. |
| [`kernel/concat`](kernel/concat) | Concat-gather and concat-scatter tests. |
| [`kernel/control`](kernel/control) | Control flow tests (hash table lookups). |
| [`kernel/element_wise`](kernel/element_wise) | Element-wise operation tests (GELU). |
| [`kernel/fa`](kernel/fa) | Flash attention tests. |
| [`kernel/gather`](kernel/gather) | Gather operation tests. |
| [`kernel/matmul`](kernel/matmul) | Matrix multiplication tests (HIF4_HIF4, A16W4). |
| [`kernel/reduction`](kernel/reduction) | Reduction operation tests (max, sum). |
| [`kernel/sort`](kernel/sort) | Sorting tests (topk). |
| [`kernel/transpose`](kernel/transpose) | Transpose operation tests. |

## Common Build Pattern

```sh
cd test/kernel/matmul
make clean
make TESTCASE=matmul TYPE=HIF4_HIF4 M=256 N=2048 K=2048 tM=64 tN=64 tK=128 \
    COMPILER_DIR=/path/to/linx/compiler/bin
make TESTCASE=matmul TYPE=HIF4_HIF4 M=256 N=2048 K=2048 tM=64 tN=64 tK=128 \
    COMPILER_DIR=/path/to/linx/compiler/bin QEMU=/path/to/qemu-linx sim
```

Platform values:

| Platform | Backend |
| --- | --- |
| `PLAT=linx` | Linx target backend with `__linx`. |

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
cd test/kernel/matmul && bash compile.all
cd test/kernel/broadcast && bash compile.all
cd test/accelerator/fusion && bash compile.all
```

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
include ../../common/Makefile.common
```

Adjust the relative include path when the suite is nested more deeply.

For a new suite, create a directory with `src/`, a small local `Makefile`, and
an optional `compile.all` batch entrypoint.

Back to the repository overview: [`../README.md`](../README.md).
