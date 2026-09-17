# Test Kernel — Operator Test Suites

Per-operator test code and build scripts. Each operator directory has a
`Makefile`, a `compile.all` (typical configs), and `src/`. All suites include
the shared `common/Makefile.common`.

## Directory Structure

The test tree is flat: every operator sits directly under `test/kernel/` and
is a four-PE (four-thread) SPMD suite.

```
test/kernel/
├── broadcast  concat  conv2d  element_wise  fa  gather
├── matmul  mxquant  reduction  transpose  vec
├── res_check_all.py
└── README.md
```

See [`../../kernels/basic_op/README.md`](../../kernels/basic_op/README.md) for
the operator list, partition rules, and the gfrun / RES_CHECK regression status.

## Build

```bash
# single config
cd test/kernel/matmul
make TESTCASE=matmul COMPILER_DIR="$COMPILER_DIR" B=1 M=256 N=256 K=256 tM=32 tN=32 tK=32

# per-operator batch
cd test/kernel/matmul && bash compile.all

# whole one-level-arch backend (from repo root)
./compile_all.sh one-level
```

Build products are written under the arch-level `output/` directory
(`benchmark/one-level-arch/output/`), which is gitignored.

## Adding a Test

1. Create `test/kernel/<operator>/` with `src/`, `Makefile`, `compile.all`.
2. Add the operator to `compile_all.sh`.
3. `include` the shared `common/Makefile.common`. The relative depth is
   `../../common/Makefile.common` for a top-level operator, one level deeper
   for nested sub-operators (e.g. `element_wise/gelu`).

## See Also
- [Arch-level README](../../README.md)
- [Operator implementations](../../kernels/README.md)
