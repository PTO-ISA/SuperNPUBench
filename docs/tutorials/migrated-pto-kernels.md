# Migrated PTO Tile Kernels

SuperNPUBench includes nine tile-only kernels imported from
[`LinxISA/PTO-Kernel`](https://github.com/LinxISA/PTO-Kernel) at revision
`0eb72cb20b0de99326d984a9a27ddb815e6e4c24`.

## Included Kernels

| Benchmark | Upstream source | PTO 0.57 data path |
| --- | --- | --- |
| `pto_tload_store` | `kernels/memory/tload_store.cpp` | `TLOAD`, `TSTORE` |
| `pto_add` | `kernels/elementwise/add_custom.cpp` | `TLOAD`, `TADD`, `TSTORE` |
| `pto_gemm` | `kernels/matmul/gemm.cpp` | GEMM with destination accumulation |
| `pto_gemm_basic` | `kernels/matmul/gemm_basic.cpp` | Basic FP32 tiled GEMM |
| `pto_gemm_demo` | `kernels/matmul/gemm_demo.cpp` | GEMM followed by tile scaling and addition |
| `pto_gemm_performance` | `kernels/matmul/gemm_performance.cpp` | Repeated tiled GEMM accumulation |
| `pto_mamulb` | `kernels/matmul/mamulb.cpp` | Integer tiled matrix multiply |
| `pto_tmatmul_acc` | `kernels/matmul/tmatmul_acc.cpp` | Explicit `TMATMUL_ACC` coverage |
| `pto_flash_attention` | `kernels/attention/flash_attention.cpp` | Two-stage integer QK and score-V tile matmul |

The [PTO-Kernel benchmark catalog](../benchmarks/catalog/one-level/pto-kernels/index.md)
shows the complete wrapper and migrated source for every case, together with
its exact intrinsic union and build command.

## Tile-Only Policy

The imported kernels use scalar C++ only for tile-grid control. Their data
paths do not index global pointers or provide a scalar execution branch. Every
load, elementwise operation, matrix multiply, conversion, and store is a named
PTO 0.57 intrinsic.

The imported support headers are namespaced under `pto_kernel/`. They do not
shadow the compatibility headers used by existing SuperNPUBench suites.

## Build

```bash
cd benchmark/one-level-arch/test/kernel/pto_kernels
PLAT=linx COMPILER_DIR=/path/to/linx-isa/compiler/llvm/build-linxisa-clang/bin \
  bash compile.all
```

Build one case with:

```bash
make TESTCASE=pto_flash_attention \
  PLAT=linx \
  COMPILER_DIR=/path/to/linx-isa/compiler/llvm/build-linxisa-clang/bin \
  diss
```

## Verify the Intrinsic Boundary

```bash
python3 scripts/verify_pto_kernel_migration.py
python3 scripts/generate_benchmark_manual.py
python3 -m mkdocs build --strict --clean
python3 scripts/verify_golden_manual.py --site site
```

The migration verifier rejects any operation outside the 111-entry PTO 0.57
allowlist, scalar pointer-indexed data paths, missing wrappers, and
manifest/build-list drift.
