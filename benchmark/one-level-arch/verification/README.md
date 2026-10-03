# Verification — CPU PyTorch Reference for One-Level-Arch Kernels

This directory contains PyTorch-based CPU reference implementations for all
one-level-arch operators. Each function generates deterministic (fixed-seed)
input and expected output tensors that can be compared against ELF simulation
output from `gfrun`/`gfsim`.

## Usage

```bash
# Run all kernel verifications (print shapes)
python3 verify_all.py

# Run one kernel
python3 verify_all.py --kernel matmul

# Print small tensors (< 32 elements)
python3 verify_all.py --kernel rms_norm --print

# Export binary files for offline comparison
python3 verify_all.py --kernel matmul --export ./golden

# Compile and run the TLEA element-wise atomic Top-K on both models
COMPILER_DIR=/path/to/fresh-llvm-build/bin \
LINX_RUNTIME_ROOT=/path/to/linx_blockisa_llvm_musl \
API_INCLUDE=/path/to/Linx-TileOP-API/include \
SSM=/path/to/SuperScalarModel ./run_element_atomic_topk.sh
```

Each run writes a unique evidence directory with a content ID. The directory
contains exact command identities, per-stage exit status, disassembly checks,
model logs, memory dumps, independent goldens, and repository/tool hashes.

## Layout

```
verification/
├── README.md           ← this file
├── verify_all.py       ← all kernel reference functions
└── run_element_atomic_topk.sh ← compile/disassembly/gfrun/gfsim gate
```

## Covered Kernels (37 functions)

### Benchmark operators

| Kernel | Function | Operation |
|--------|----------|-----------|
| matmul | `verify_matmul` | `C = A @ B` (fp32 accumulation) |
| fa_2d_unroll | `verify_fa_2d_unroll` | `O = softmax(Q@Kᵀ/√d) @ V` |
| sfa | `verify_sfa` | Block-sparse Flash Attention (CSR) |
| fa_softmax | `verify_fa_softmax` | `out = softmax(score, dim=-1)` |
| transpose_2d | `verify_transpose_2d` | `out = in.T` |
| transpose_nd | `verify_transpose_nd` | N-D axis swap |
| reducesum_col/row | `verify_reducesum_*` | Column/row sum |
| reducemax_col/row | `verify_reducemax_*` | Column/row max |
| gelu | `verify_gelu` | Custom polynomial GELU |
| broadcast | `verify_broadcast` | N-D broadcast |
| concat_gather | `verify_concat_gather` | N-table concat via gather |
| gather | `verify_gather` | Row-index gather |

### DeepSeek kernels

| Kernel | Function | Operation |
|--------|----------|-----------|
| fused_weight | `verify_fused_weight` | `out = a * b` (element-wise) |
| rms_norm | `verify_rms_norm` | `out = x * rsqrt(mean(x²) + eps)` |
| batched_transpose | `verify_batched_transpose` | Per-batch 2D transpose |
| expand_mhc_fwd | `verify_expand_to_mhc_fwd` | Replicate tokens along MHC axis |
| expand_mhc_bwd | `verify_expand_to_mhc_bwd` | Sum-reduce along MHC axis |
| topk_gate | `verify_topk_gate` | Greedy top-k (tie-break: smallest index) |
| normalize_weight | `verify_normalize_weight` | `out = w / (sum(w) + eps)` |
| group_count | `verify_group_count` | Histogram of group selections |
| aux_fi | `verify_aux_fi` | Load-balancing term |
| cast_back_token | `verify_cast_back_per_token` | Dequantize with per-row scale |
| cast_back_channel | `verify_cast_back_per_channel` | Dequantize with per-col scale |
| per_token_cast | `verify_per_token_cast` | Quantize with per-row scale |
| per_channel_cast | `verify_per_channel_cast` | Quantize with per-col scale |
| swiglu | `verify_swiglu` | SiLU(gate) * up + per-token quant |
| reduce_fused | `verify_reduce_fused` | Weighted sum over top-k experts |
| inplace_unique | `verify_inplace_unique` | In-place dedup (set -1) |
| sinkhorn | `verify_sinkhorn` | Doubly-stochastic normalization |
| fn_normw_merge | `verify_fn_normw_merge` | Column-broadcast multiply |
| mask_indices_tp | `verify_mask_indices_by_tp` | TP-rank masking |
| engram_hash | `verify_engram_hash` | N-gram hash embedding index |
| get_fused_mapping | `verify_get_fused_mapping` | Expert-major token mapping |
| expand_to_fused | `verify_expand_to_fused` | Scatter-replicate to expanded buffer |

## Tolerances

| dtype | rtol | atol |
|-------|------|------|
| fp32 | 1e-4 | 1e-5 |
| fp16/bf16 | 1e-2 | 1e-3 |
| int32 | exact match | — |

## Notes

- GELU uses custom polynomial coefficients (not PyTorch's tanh-gelu)
- Sinkhorn/rsqrt use Newton-Raphson (4 iterations)
- topk_gate ties are broken by **smallest index** (not PyTorch default)
- sfa uses CSR-style block-sparse pattern (local window attention)
