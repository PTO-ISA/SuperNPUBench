# Kernels — Operator Implementations

Header-only operator implementations, organized by function. All operators use
the PTO tile-programming paradigm via `<common/pto_tileop.hpp>` and C++ templates
for type/dimension parameterization.

## Operator List

> **DeepSeek 迁移算子**：`deepseek/` 子目录收录从 TileKernels（TileLang DSL）迁移的 19 个
> tile 版算子（engram/mhc/moe/quant/transpose 五模块），已通过 linx 工具链编译+链接验证。
> 详见 [`deepseek/TileKernels迁移说明.md`](deepseek/TileKernels迁移说明.md) 与各模块 README。

> **PTO-Kernel imports**: `pto_kernels/` contains nine source-backed, tile-only
> memory, elementwise, GEMM, and attention kernels imported at a pinned
> `LinxISA/PTO-Kernel` revision. See [`pto_kernels/README.md`](pto_kernels/README.md).

### 1. Matmul — `matmul/`
- `matmul.hpp` — general matrix multiply; FP32/FP16/FP8; mask, dynamic, vec variants; A/B tile reuse.
- `matmul_mx.hpp` — MX quantized matmul; FP4×FP4, BF16×FP4 mixed precision; microscaling factors.

### 2. Flash Attention — `fa/`
- `fa_2d_unroll.hpp` — 2D unroll (X/Y dims); seq len 256/512.
- `fa_2d_unroll_pto.hpp` — 2D unroll, PTO tile-op variant.
- `fa_unalign_2d_unroll.hpp` — unaligned boundary (seq len not a multiple of tile).
- `fa_hif4.hpp` — HIF4 quantized.
- `fa_dcore.hpp` — DCore-optimized.
- `fa_utils.h` / `fa_fp4_utils.h` — shared helpers.

> Note: in `one-level-arch`, `*_pto.hpp` files are PTO-style variants kept
> alongside the base implementations.

### 3. Broadcast — `broadcast/`
- `broadcast.hpp` — base; `broadcast_07/019/039/Hunyuan.hpp` — 2D~5D shapes;
  `broadcast_vec_*.hpp` — vectorized; `broadcast_mscatter/nocopyout/nomg/simple.hpp` — variants.

### 4. Reduction — `reduction/` (see [`reduction/README.md`](reduction/README.md))
- `reducemax_{colvec,rowvec}.hpp`, `reducesum_{colvec,rowvec}.hpp` — base max/sum.
- `*_single_tree.hpp` — multi-stage tree reduction.
- `*_unalign_120_8.hpp` — 3D unaligned (120×8).
- `cumsum_{colvec,rowvec}.hpp`, `reduceprod_{colvec,rowvec}.hpp`.

### 5. GELU — `element_wise/`
- `gelu.hpp` — polynomial-fitting; exact (erf) and tanh approximation.
- `gelu_origin.hpp` — original erf/tanh implementation.

### 6. Gather — `gather/`
- `gather.hpp` — large-scale, various indexing modes.

### 7. Concat — `concat/`
- `concat_gather.hpp` — gather-based; `concat_scatter.hpp` — scatter-based.

### 8. Transpose — `transpose/`
- `transpose.hpp` — 3D~6D; `transpose_vector_007/050.hpp` — vectorized.

### 9. Control — `control/`
- `hashtable_lookup_simd.hpp` — pure tile-op kernel (no SIMT). Runs on gfsim
  with `-s core.singleTierMode=true`.

### 10. Sort — `sort/`
- `topk.hpp` — Top-K.

### Utils — `utils/`
- `layout_transform.hpp` — ND→ZZ / ND→NN offset calculation.

## Design Principles
1. **Header-only** — easy integration/reuse.
2. **PTO paradigm** — unified tile-operation interface.
3. **Templated** — type and dimension parameterization.
4. **Optimization-oriented** — multiple variants per scenario.

## Usage
```cpp
#include "matmul/matmul.hpp"
matmul_mask<float, M, N, K, tM, tN, tK>(dst, src0, src1);
```

## Optimization Tips
- Pick `tM/tN/tK` per hardware.
- Choose dtype per precision.
- Use `reuseA/reuseB` for repeated computation.
- Prefer vectorized variants (`vec`/`vector`) where available.

## See Also
- [Top-level README](../../README.md)
- [Test suites](../test/kernel/README.md)
