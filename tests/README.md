# Tests

This tree keeps correctness material that is not the primary Linx benchmark
navigation surface. Active Linx benchmark entrypoints live under
[`../benchmarks`](../benchmarks); add new benchmark suites there instead of
recreating the old `test/` tree.

## Directory Map

| Path | Purpose |
| --- | --- |
| [`py_api`](py_api) | Active Python-facing TileOP correctness and golden-comparison flow. |
| [`tileop_layout`](tileop_layout) | TileOP layout and behavior checks that are not cataloged as primary benchmark suites. |

These directories still use the shared benchmark harness through
[`../benchmarks/common/Makefile.common`](../benchmarks/common/Makefile.common),
so the same `TESTCASE`, `PLAT`, `COMPILER_DIR`, and `QEMU` variables work here.

## Common Build Pattern

```sh
cd tests/tileop_layout
make clean
make TESTCASE=TLOAD PLAT=linx COMPILER_DIR=/path/to/linx/compiler/bin
make TESTCASE=TSTORE PLAT=linx COMPILER_DIR=/path/to/linx/compiler/bin
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

Run batch files from their suite directory so relative paths and make variables
resolve as intended:

```sh
cd tests/tileop_layout && bash compile.all
cd tests/py_api && bash compile.all
```

## Python Golden Comparison

```sh
cd tests/py_api
make clean
make TESTCASE=tileop_py
python3 golden_cmp/golden_cmp.py -i tadd
```

For adding golden-comparison cases, see
[`py_api/golden_cmp/README.md`](py_api/golden_cmp/README.md).

Back to the repository overview: [`../README.md`](../README.md).
