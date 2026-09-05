# Kernels — Operator Implementations

Header-only PTO operator implementations are separated by PE execution model:

- [`single_thread/`](single_thread/README.md): legacy and current single-PE kernels.
- [`multi_thread/`](multi_thread/README.md): explicit four-PE kernels using
  cooperative matrix execution or `get_thread_idx()` data partitioning.

## Operator List

> **DeepSeek 迁移算子**：`single_thread/deepseek/` 子目录收录从 TileKernels（TileLang DSL）迁移的 19 个
> tile 版算子（engram/mhc/moe/quant/transpose 五模块），已通过 linx 工具链编译+链接验证。
> 详见 [`TileKernels迁移说明.md`](single_thread/deepseek/TileKernels迁移说明.md) 与各模块 README。

### 1. Matmul — `single_thread/matmul/`
- `matmul.hpp` — general matrix multiply; FP32/FP16/FP8; mask, dynamic, vec variants; A/B tile reuse.
- `matmul_mx.hpp` — MX quantized matmul; FP4×FP4, BF16×FP4 mixed precision; microscaling factors.

### 2. Flash Attention — `single_thread/fa/`
- `fa_2d_unroll.hpp` / `fa_2d_unroll.hpp` — 2D unroll (X/Y dims); seq len 256/512.
- `fa_unalign_2d_unroll.hpp` / `fa_unalign_2d_unroll.hpp` — unaligned boundary.
- `fa_hif4.hpp` / `fa_hif4.hpp` — HIF4 quantized.
- `fa_dcore.hpp` / `fa_dcore.hpp` — DCore-optimized.
- `sfa.hpp` — Sparse Flash Attention (block-sparse / CSR pattern), two-pass.
- `fa_utils.h` / `fa_fp4_utils.h` — shared helpers.

> Note: in `one-level-arch`, `*.hpp` files are PTO-style variants kept
> alongside the base implementations.

### 3. Broadcast — `single_thread/broadcast/`
- `broadcast.hpp` — base; `broadcast_07/019/039/Hunyuan.hpp` — 2D~5D shapes;
  `broadcast_vec_*.hpp` — vectorized; `broadcast_mscatter/nocopyout/nomg/simple.hpp` — variants.

### 4. Reduction — `single_thread/reduction/`
- `reducemax_{colvec,rowvec}.hpp`, `reducesum_{colvec,rowvec}.hpp` — base max/sum.
- `*_single_tree.hpp` — multi-stage tree reduction.
- `*_unalign_120_8.hpp` — 3D unaligned (120×8).
- `cumsum_{colvec,rowvec}.hpp`, `reduceprod_{colvec,rowvec}.hpp`.

### 5. GELU — `single_thread/element_wise/`
- `gelu.hpp` — polynomial-fitting; exact (erf) and tanh approximation.
- `gelu_origin.hpp` — original erf/tanh implementation.

### 6. Gather — `single_thread/gather/`
- `gather.hpp` — large-scale, various indexing modes.

### 7. Concat — `single_thread/concat/`
- `concat_gather.hpp` — gather-based; `concat_scatter.hpp` — scatter-based.

### 8. Transpose — `single_thread/transpose/`
- `transpose.hpp` — 3D~6D; `transpose_vector_007/050.hpp` — vectorized.

### 9. Control — `single_thread/control/`
- `hashtable_lookup_simd.hpp` — pure tile-op kernel (no SIMT). Runs on gfsim
  with `-s core.singleTierMode=true`.

### 10. Sort — `single_thread/sort/`
- `topk.hpp` / `topk.hpp` — Top-K via radix-bucket histogram.

### 11. DeepSeek 迁移算子 — `single_thread/deepseek/`
- 19 个从 TileKernels (TileLang DSL) 迁移的 tile 版算子:
  engram (2), mhc (5+1), moe (8+1), quant (3+2), transpose (1)
- `_compile_test.cpp` 实例化全部 kernel 用于编译验证
- 23 个独立测试用例 (每个 kernel 一个 ELF)

### Utils — `single_thread/utils/`
- `layout_transform.hpp` — ND→ZZ / ND→NN offset calculation.

## Design Principles
1. **Header-only** — easy integration/reuse.
2. **PTO paradigm** — unified tile-operation interface.
3. **Templated** — type and dimension parameterization.
4. **Optimization-oriented** — multiple variants per scenario.

## Usage
```cpp
#include "single_thread/matmul/matmul.hpp"
matmul_mask<float, M, N, K, tM, tN, tK>(dst, src0, src1);
```

## Optimization Tips
- Pick `tM/tN/tK` per hardware.
- Choose dtype per precision.
- Use `reuseA/reuseB` for repeated computation.
- Prefer vectorized variants (`vec`/`vector`) where available.

## Four-PE Kernels

- Matmul: shared, shared-B-reuse and low-precision cooperative variants.
- FlashAttention: cooperative GMMA implementation.
- Element-wise and data movement: TADD, GELU, Broadcast, Gather and Concat.
- Reduction: TROWSUM plus row-wise Cumsum, Max, Product and Sum.
- Normalization: RMSNorm and binary-accumulation RMSNorm.
- Structured operators: 1x1 Conv2D output-block partitioning and 2D
  Transpose row-block partitioning.

See the [four-PE coverage and limitations](multi_thread/README.md) for the
exact headers, partition rules and gfrun status.

## See Also
- [Top-level README](../../README.md)
- [Test suites](../test/kernel/README.md)
