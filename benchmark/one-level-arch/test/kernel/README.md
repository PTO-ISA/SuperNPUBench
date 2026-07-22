# Test Kernel — Operator Test Suites

Per-operator test code and build scripts. Each operator directory has a
`Makefile`, a `compile.all` (typical configs), and `src/`. All suites include
`../../common/Makefile.common`.

## Directory Structure

```
test/kernel/
├── broadcast/        element_wise/    matmul/
├── concat/           fa/              reduction/{reducemax_col,row,...}
├── control/          gather/          sort/
├── pto_kernels/      transpose/
└── reduction/
```

## Operator Test Status

| Operator | Configs | Status | Notes |
|----------|---------|--------|-------|
| matmul | 13 | ✓ | FP4/BF16/FP32/FP16/FP8; quantization, mixed precision, reuse |
| fa | 9 | ✓ | 2D unroll, HIF4; `Ydim=1` crashes compiler (Issue #6) |
| transpose | 8 | ✓ | 3D~6D; __half, int32_t |
| reduction | 8 | ✓ | row/col max/sum; int32_t, __half |
| gelu | 8 | ✓ | exact (erf) and tanh; __bf16, __half |
| broadcast | 5 | ✓ | 2D~5D; vectorized |
| gather | 4 | ✓ | large-scale, power-of-2 |
| concat | 4 | ✓ | gather/scatter |
| control | 1 | △ | pure tile-op; run gfsim with `-s core.singleTierMode=true`; `.data` via `gen_data.py` |
| sort | 1 | △ | topk |
| pto_kernels | 9 | ✓ | Tile-only imports pinned to PTO-Kernel; strict 0.57 allowlist |

(Configs reflect `compile.all` typical scenarios; `△` = compiles but needs special run flags / generated data.)

## Build

### Single config

```bash
cd test/kernel/matmul

# FP4 × FP4
make TESTCASE=matmul TYPE=HIF4_HIF4 M=256 N=2048 K=2048 tM=128 tN=128 tK=128

# BF16 × FP4 mixed precision
make TESTCASE=matmul TYPE=A16W4 M=256 N=2048 K=2048 tM=128 tN=128 tK=128

# FP32 mask matmul
make TESTCASE=matmul TYPE=MASK MODE=MASK_FP32 M=256 N=256 K=256 tM=64 tN=64 tK=64
```

### Batch / full

```bash
cd test/kernel/matmul && bash compile.all        # per-operator
./compile_all.sh two-level                        # whole backend (from repo root)
```

### Makefile parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `TESTCASE` | Test case name | `matmul`, `fa_2d_unroll` |
| `TYPE` | Operator type (matmul) | `HIF4_HIF4`, `A16W4`, `MASK` |
| `MODE` | Operator mode | `MASK_FP32`, `BF16x2_NOGATHER` |
| `M/N/K` | Matrix dimensions | `M=256 N=2048 K=2048` |
| `tM/tN/tK` | Tile sizes | `tM=128 tN=128 tK=128` |
| `COMPILER_DIR` | Compiler path | `/path/to/linx/bin` |
| `PLAT` | Platform | `linx` (default), `cpu` |

### Build targets

```bash
make TESTCASE=<case> all      # compile
make TESTCASE=<case> diss     # disassembly
make TESTCASE=<case> sim      # run in simulator
make TESTCASE=<case> debug    # debug mode
make clean                    # clean current operator
make clean_all                # clean all
```

## Operator Details

### Matmul (13 configs)
- `src/HiF4_HiF4.cpp` — FP4×FP4 quantized; `TYPE=HIF4_HIF4`, `VER=MX_NOGATHER|MX_NOGATHER_REUSEA`.
- `src/A16W4.cpp` — BF16×FP4 mixed precision; `TYPE=A16W4`.
- `src/matmul.cpp` — general; `TYPE=MASK`, `MODE=MASK_FP32|MASK_FP16|MASK_FP8`(+ REUSEA/REUSEB/DYNAMIC).

### Flash Attention (9 configs)
- `src/fa_2d_unroll.cpp` (8) — X=1/2, Y=2/4; seq 256/512.
- `src/fa_hif4.cpp` (1) — HIF4 quantized.
- Avoid `Ydim=1` (compiler crash).

### Transpose (8 configs) — see [`transpose/README.md`](transpose/README.md)
2D / 4D(A,B) / 6D(≡5D); __half, int32_t.

### Reduction (8 configs) — see [`reduction/README.md`](reduction/README.md)
`reducemax_{col,row}/`, `reducesum_{col,row}/`; int32_t, __half.

### GELU (8 configs)
`src/gelu.cpp`; __bf16/__half; shapes 24_8_1024 / 128_1024; exact & tanh.

### Broadcast (5 configs) — 2D/3D/4D×2/5D.

### Gather (4 configs) — large (200000,875000), medium (754), power-of-2 (131072).

### Concat (4 configs) — `concat_gather` (2) + `concat_scatter` (2); int32_t, __half.

### Control — `control/hashtable_lookup_simd` (pure tile-op; single-tier gfsim).

### Sort — `sort/topk`.

## Runtime note
- `control` needs `-s core.singleTierMode=true` on gfsim; `.data` comes from `gen_data.py`.

## Adding a Test
1. Create `test/kernel/<operator>/` with `src/`, `Makefile`, `compile.all`.
2. Mirror in the other backend.
3. `include ../../common/Makefile.common` (adjust depth if nested).

## See Also
- [Top-level README](../../README.md)
- [Operator implementations](../kernels/README.md)
