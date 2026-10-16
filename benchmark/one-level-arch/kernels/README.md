# Kernels — Operator Implementations

Header-only PTO operator implementations, organized by execution model:

- [`basic_op/`](basic_op/README.md): four-PE kernels. Each operator
  directory holds the four-PE wrapper plus the single-PE base implementation it
  partitions, the latter under `<op>/detail/`. Shared helpers live in
  `basic_op/utils/`.
- [`solution/`](../test/solution): fused / end-to-end operator solutions built
  on top of the kernel primitives.

## basic_op/ operators

| Operator | Directory |
|---|---|
| Broadcast | `basic_op/broadcast/` |
| Concat (gather / scatter) | `basic_op/concat/` |
| 1x1 Conv2D | `basic_op/conv2d/` |
| GELU / TADD | `basic_op/element_wise/` |
| FlashAttention | `basic_op/fa/` |
| Gather | `basic_op/gather/` |
| Shared / low-precision Matmul | `basic_op/matmul/` |
| MX-quantized Matmul | `basic_op/mxquant/` |
| Row Cumsum / Max / Prod / Sum | `basic_op/reduction/` |
| 2D Transpose | `basic_op/transpose/` |
| SPMD partition / layout helpers | `basic_op/utils/` |

See [`basic_op/README.md`](basic_op/README.md) for the partition rules,
the base/wrapper split, and the gfrun + RES_CHECK regression status.

## solution/ operators

`gather_v2`, `group_token_old`, `group_token_vec`, `matmul_test`, `mega_moe`,
`moe_combine`, `moe_dispatch`, `normalization` (RMSNorm / GroupNorm grad),
`qli`, `quant`, `quant_batch_matmul`, `quant_sparse_flash_mla`, `view_copy`.

## Design Principles
1. **Header-only** — easy integration/reuse.
2. **PTO paradigm** — unified tile-operation interface.
3. **Templated** — type and dimension parameterization.
4. **Optimization-oriented** — multiple variants per scenario.

## Usage

```cpp
#include "basic_op/matmul/matmul_shared.hpp"
matmul_shared<float, gM, gN, gK, tM, tN, tK>(c_ptr, a_ptr, b_ptr);
```

## See Also
- [Top-level README](../../README.md)
- [Test suites](../test/kernel/README.md)

## PTO element-for examples

The core TopK example is [basic_op/topk/element_atomic_topk.hpp](basic_op/topk/element_atomic_topk.hpp). It alternates public Tile loads/views, ordinary scalar C++ under `#pragma pto element for`, and Tile stores. Its runnable driver and coherence checks are in `test/kernel/sort/element_atomic_topk/`.

The element-for migration retains the original workload configurations in the gfrun suite under `test/kernel/element_wise/tile_element_suite/`. Former single-thread sources now live under `basic_op/`; the verification inventory records their original identities and current paths. QLI demonstrates repeated Tile reduction → element-loop mutation → Tile reduction across helper calls.
