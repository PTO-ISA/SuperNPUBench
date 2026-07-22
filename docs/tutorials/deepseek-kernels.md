# Programming patterns from DeepSeek tile kernels

The DeepSeek kernel collection exercises the same C++ tile model across gating,
mixture-of-experts routing, quantization, and layout conversion. The ports are
source translations pinned to one upstream revision so that every documented
algorithm can be traced to a stable input module.

## What the migration preserves

Each migrated module records its upstream path and revision, translated C++
implementation, intrinsic set, thread partition, and verification level. Tensor
loads, arithmetic, reductions, transformations, and stores remain tile
operations. Scalar C++ is limited to dimensions, loop bounds, indices, and
compile-time specialization.

The migration follows three rules:

1. Each active thread owns a non-overlapping fragment of at least 128 bytes.
2. Missing compound operations are composed from public tile intrinsics.
3. Shared work is represented by block-shared tile values and tile-version
   dependencies, not scalar pointer exchange.

## Family map

| Family | Representative programming problem |
| --- | --- |
| Engram | Hash lookup, gating, and weight-gradient reduction |
| mHC | Expansion, mixing, normalization, recompute, and post transforms |
| MoE | Top-k routing, expert counts, mapping, normalization, and fused reduction |
| Quantization | Per-token, per-channel, and per-block scaling with fused activation |
| Transpose | Batched tile layout conversion |

## Read a kernel page

Start with the core C++ kernel near the top. Then use the algorithm walkthrough
to connect source stages to the intrinsic table, inspect the block/thread
partition, and follow the memory/tile flow. The verification field distinguishes
compile-checked implementations from source-validated interfaces whose compiler
lowering is still in progress.

Open the [DeepSeek kernel reference](../benchmarks/deepseek/index.md) for all 37
standalone source modules and their translated C++ implementations.
