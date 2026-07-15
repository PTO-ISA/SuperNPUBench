# Test Navigation

The `test` tree contains kernel test suites. All make-driven suites reuse
[`common/Makefile.common`](common/Makefile.common), so the same `TESTCASE`,
`PLAT`, and `COMPILER_DIR` variables work across directories.

## Directory Map

| Path | Use it for |
| --- | --- |
| [`common`](common) | Shared make rules, platform flags, output layout, simulator targets, `_start.s`, `benchmark.h`. |
| [`tileop_api`](tileop_api) | Active bounded Linx direct-boot TileOP cases consumed by the superproject AI workload flow. |
| [`kernel`](kernel) | Per-operator suites: broadcast, concat, control, element_wise, fa, gather, matmul, reduction, sort, transpose. |

## Kernel Test Suites

| Directory | Operations |
| --- | --- |
| [`kernel/broadcast`](kernel/broadcast) | Broadcast (various shapes). |
| [`kernel/concat`](kernel/concat) | Concat-gather / concat-scatter. |
| [`kernel/control`](kernel/control) | Hash-table lookup (pure tile-op). |
| [`kernel/element_wise`](kernel/element_wise) | GELU. |
| [`kernel/fa`](kernel/fa) | Flash attention. |
| [`kernel/gather`](kernel/gather) | Gather. |
| [`kernel/matmul`](kernel/matmul) | Matmul (HIF4_HIF4, A16W4, MASK). |
| [`kernel/reduction`](kernel/reduction) | Reduction (max, sum). |
| [`kernel/sort`](kernel/sort) | Topk. |
| [`kernel/transpose`](kernel/transpose) | Transpose. |

## Common Build Pattern

```sh
cd test/kernel/matmul
make TESTCASE=matmul TYPE=HIF4_HIF4 M=256 N=2048 K=2048 tM=64 tN=64 tK=128 \
    COMPILER_DIR=/path/to/linx/compiler/bin
```

Platform values:

| Platform | Backend |
| --- | --- |
| `PLAT=linx` | Linx target backend with `__linx` (default). |
| `PLAT=cpu`  | Host CPU simulation (`__cpu_sim__`). |

Common targets: `all`, `diss`, `sim`, `debug`, `clean`, `clean_all`.

Build products are written under the arch-level `output/` directory
(`benchmark/<arch>/output/`), which is gitignored.

## Batch Runs

Each suite ships a local `compile.all`; run it from the suite directory:

```sh
cd test/tileop_api && bash compile.all
cd test/kernel/matmul && bash compile.all
cd test/kernel/broadcast && bash compile.all
```

Whole-backend batch: `./compile_all.sh two-level|one-level|all` from the repo root.

## Adding A Test Case

For an existing make-driven suite:

1. Add the source file under that suite's `src/`.
2. Set `SRC_FILE`, `TARGET`, and suite-specific variables in the local `Makefile`.
3. `include ../../common/Makefile.common` (adjust depth when nested more deeply).
4. Add the case to the local `compile.all` for batch runs.

Minimal local makefile:

```make
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/$(TESTCASE).cpp
TARGET = $(ELF_HEAD)_$(TESTCASE).elf
include ../../common/Makefile.common
```

For a new suite, create a directory with `src/`, a small `Makefile`, and an
optional `compile.all`.

Back to repository overview: [`../README.md`](../README.md).
