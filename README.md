# SuperNPUBench

SuperNPUBench is a kernel and benchmark workspace for LinxISA/PTO-style tile
programming experiments. The repository is organized around reusable kernels
and make-driven test suites.

## Repository Map

| Path | Purpose |
| --- | --- |
| [`kernels`](kernels) | Reusable kernel implementations, grouped by operation type. |
| [`test/common`](test/common) | Shared make harness, platform selection, compiler flags, and simulator targets. |
| [`test/accelerator`](test/accelerator) | Accelerator-oriented benchmark and validation suites (fusion, include). |
| [`test/kernel`](test/kernel) | Kernel benchmark and validation suites for various operations. |

### Kernel Implementations

The [`kernels`](kernels) directory contains header-only implementations:

| Directory | Operations |
| --- | --- |
| [`kernels/broadcast`](kernels/broadcast) | Broadcast operations with various shapes and configurations. |
| [`kernels/concat`](kernels/concat) | Concat-gather and concat-scatter operations. |
| [`kernels/element_wise`](kernels/element_wise) | Element-wise operations (GELU, etc.). |
| [`kernels/fa`](kernels/fa) | Flash attention implementations. |
| [`kernels/gather`](kernels/gather) | Gather operations. |
| [`kernels/matmul`](kernels/matmul) | Matrix multiplication variants. |
| [`kernels/reduction`](kernels/reduction) | Reduction operations (max, sum). |
| [`kernels/transpose`](kernels/transpose) | Transpose operations. |

### Test Suites

The [`test/kernel`](test/kernel) directory contains test suites for:

- `broadcast` - Broadcast operation tests
- `concat` - Concat-gather and concat-scatter tests
- `control` - Control flow tests (hash table lookups)
- `element_wise` - Element-wise operation tests (GELU)
- `fa` - Flash attention tests
- `gather` - Gather operation tests
- `matmul` - Matrix multiplication tests
- `reduction` - Reduction operation tests (max, sum)
- `sort` - Sorting tests (topk)
- `transpose` - Transpose operation tests

## Quick Navigation

- To add or update reusable compute code, use the matching domain under
  [`kernels`](kernels).
- To run kernel tests, use the `compile.all` files in the relevant
  [`test/kernel`](test/kernel) subdirectories.
- For accelerator-specific tests, see [`test/accelerator`](test/accelerator).

## Building Tests

Most test directories include [`test/common/Makefile.common`](test/common/Makefile.common).
The common harness is controlled mainly by `TESTCASE`, `PLAT`,
`COMPILER_DIR`, and `QEMU`.

```sh
cd test/kernel/matmul
make clean
make TESTCASE=matmul TYPE=HIF4_HIF4 M=256 N=2048 K=2048 tM=64 tN=64 tK=128 \
    COMPILER_DIR=/path/to/linx/compiler/bin
```

| Variable | Meaning |
| --- | --- |
| `TESTCASE` | Test case name to build. |
| `PLAT=linx` | Builds for the Linx target and defines `__linx`. |
| `COMPILER_DIR` | Directory containing `clang`, `clang++`, `llvm-objdump`, and related tools. |
| `QEMU` | Simulator binary used by `make sim` for Linx-targeted test execution. |

Common targets:

```sh
make TESTCASE=<case> all
make TESTCASE=<case> diss
make TESTCASE=<case> sim
make TESTCASE=<case> debug
make clean
make clean_all
```

## Batch Suites

Many suites provide a local `compile.all` file. Run these from the suite
directory so relative paths and local make variables resolve as intended.

Examples:

```sh
cd test/kernel/matmul && bash compile.all
cd test/kernel/broadcast && bash compile.all
cd test/accelerator/fusion && bash compile.all
```

## Adding Work

Use the existing directory shape when adding code:

1. Add reusable compute kernels under the matching [`kernels`](kernels) domain.
2. Add focused tests under the relevant suite in [`test/kernel`](test/kernel).
3. Add the case to the local `compile.all` file when it should be part of the
   batch suite.

New make-driven test directories should keep the local `Makefile` small and
include [`test/common/Makefile.common`](test/common/Makefile.common) for shared
platform flags, output paths, simulator targets, and cleanup behavior.

## Generated Files

Do not commit generated files from `output/`, object files, executable test
artifacts, local logs, or disassembly files. Keep source changes in `kernels/`
and `test/`.
