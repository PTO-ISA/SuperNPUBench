# Kernels — Operator Implementations

Header-only operator implementations, organized by function. All operators use
the PTO tile-programming paradigm via `<common/pto_tileop.hpp>` and C++ templates
for type/dimension parameterization.

## Operator List

> **DeepSeek 迁移算子**：`deepseek/` 子目录收录从 TileKernels（TileLang DSL）迁移的 19 个
> tile 版算子（engram/mhc/moe/quant/transpose 五模块），已通过 linx 工具链编译+链接验证。
> 详见 [`deepseek/TileKernels迁移说明.md`](deepseek/TileKernels迁移说明.md) 与各模块 README。

### 1. Matmul — `matmul/`
- `matmul.hpp` — general matrix multiply; FP32/FP16/FP8; mask, dynamic, vec variants; A/B tile reuse.
- `matmul_mx.hpp` — MX quantized matmul; FP4×FP4, BF16×FP4 mixed precision; microscaling factors.

### 2. Flash Attention — `fa/` (see [`fa/README.md`](fa/README.md))
- `fa_2d_unroll.hpp` / `fa_2d_unroll.hpp` — 2D unroll (X/Y dims); seq len 256/512.
- `fa_unalign_2d_unroll.hpp` / `fa_unalign_2d_unroll.hpp` — unaligned boundary.
- `fa_hif4.hpp` / `fa_hif4.hpp` — HIF4 quantized.
- `fa_dcore.hpp` / `fa_dcore.hpp` — DCore-optimized.
- `sfa.hpp` — Sparse Flash Attention (block-sparse / CSR pattern), two-pass.
- `fa_utils.h` / `fa_fp4_utils.h` — shared helpers.

> Note: in `one-level-arch`, `*.hpp` files are PTO-style variants kept
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

### 10. Sort — `sort/` (see [`sort/README.md`](sort/README.md))
- `topk.hpp` / `topk.hpp` — Top-K via radix-bucket histogram.

### 11. DeepSeek 迁移算子 — `deepseek/` (see [`deepseek/README.md`](deepseek/README.md))
- 19 个从 TileKernels (TileLang DSL) 迁移的 tile 版算子:
  engram (2), mhc (5+1), moe (8+1), quant (3+2), transpose (1)
- `_compile_test.cpp` 实例化全部 kernel 用于编译验证
- 23 个独立测试用例 (每个 kernel 一个 ELF)

### 12. QLI — `qli/` (see [`qli/README.md`](qli/README.md))
- `qli_pto.hpp` — Quant Lightning Indexer baseline（Step1-6 + TROWARGMAX TopK）。
- `qli_pto_opt.hpp` — **多轮 MSD radix-select TopK**（THISTOGRAM Byte3→Byte0 +
  tile pop），SFA 前处理选 token，已通过 set-match 精度验证（推荐版本）。

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
