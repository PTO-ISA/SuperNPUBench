# Test Navigation

The `test` tree contains kernel test suites. All make-driven suites reuse
[`common/Makefile.common`](common/Makefile.common), so the same `TESTCASE`,
`PLAT`, and `COMPILER_DIR` variables work across directories.

## Directory Map

| Path | Use it for |
| --- | --- |
| [`common`](common) | Shared make rules, platform flags, output layout, simulator targets, `_start.s`, `benchmark.h`. |
| [`kernel`](kernel) | Four-PE per-operator suites under `kernel/`. |

## Kernel Test Suites

Single-thread suites have been retired. The active four-PE suites live under
[`kernel`](kernel); see
[`kernel/README.md`](kernel/README.md) for the
operator list and partition rules.

## Common Build Pattern

```sh
cd test/kernel/matmul
make TESTCASE=matmul COMPILER_DIR=/path/to/linx/compiler/bin \
    B=1 M=256 N=256 K=256 tM=32 tN=32 tK=32
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
