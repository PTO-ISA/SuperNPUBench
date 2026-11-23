# DeepSeek-V4.1-Flash single-core decode

The [WSE example](../wse/README.md) uses this layer with PGAS and SoC L1.SRAM transports.

`decode_layer(layer_idx, token, weights, cache, published, shards, output, memory)` evaluates one text-backbone layer for one decode position. Supply the weights of that layer, the four incoming HC streams and their staggered pre-mix. Keep the layer's window and compressor pair state across tokens. Give the following layer `output.next` and the current query's `published` indices. Layer 20 takes the final encoder HC state. The caller provides RoPE rows, physical window slots and, at layers 1 and 14, host-looked-up Engram rows.

| Zero-based layer | Attention | Side effects |
| --- | --- | --- |
| 0, 1 | SWA | Write the window and attend to visible rows and the sink. |
| 2, 8, 14 | C2A Full | Store both compressor projections in the pair ring. On odd positions publish the pooled KV and index key. Select sparse rows on every query. |
| Other layers 2–19 | C2A Reuse | Read KV and selected physical rows from the most recent Full layer. |
| 20 | C1A Full | Publish KV and index key on every position. Select sparse rows and candidate blocks, including the newest block. |
| 24, 28, 32, 36 | C1A Reindex | Score the layer-20 candidate blocks with this layer's query and publish new row IDs. |
| Other layers 21–39 | C1A Reuse | Read layer-20 KV and the most recent index selection. |

Each layer runs mHC collapse, RMSNorm, Q/KV low-rank projections, RoPE, sink attention, inverse RoPE, grouped output projection and mHC residual mixing. The FFN applies its own mHC transform, sqrt-softplus routing, six routed SwiGLUs and one shared SwiGLU. The bias affects expert selection. The selected unbiased scores are normalized to sum to 1.5. Each routed expert multiplies its activation by its routing weight before MXFP8 requantization and down projection. The combiner sums returned BF16 contributions in original slot order.

## Functional geometry

The example uses batch 1, hidden and head widths 32, two attention heads, two output groups, rank 32, FFN width 64, 16 routed experts and four HC streams with 20 Sinkhorn iterations. The window holds four rows. Compressed history has 32 rows in four permuted pages, sparse Top-K is four, and the C1A pool holds two eight-row blocks. The Linx hosted ABI launches four PEs on one Core. PE0 evaluates the layer and can schedule one or two logical expert shards.

Weights are synthetic. The public types in `types.hpp` describe the weight, activation, cache and publication layouts. Production checkpoint tiling, vocabulary lookup and tokenization are outside the layer boundary. See the [pypto-lib layer](https://github.com/hw-native-sys/pypto-lib/blob/182990263ee87ef29c835ead20a716e3f4aa3d19/models/deepseek_v4_1_flash/decode_layer.py) and [configuration](https://github.com/hw-native-sys/pypto-lib/blob/182990263ee87ef29c835ead20a716e3f4aa3d19/models/deepseek_v4_1_flash/config.py).

## Linx-TileOP computation

| Layer operation | Implementation | Related one-level kernel |
| --- | --- | --- |
| Q, KV, O, index, Engram and all expert MX projections | TLSU gathers read output-major weights in K×N order. For FP4, shifts and OR pair adjacent output channels. A native dual-scale `TGEMV_MX` computes 32 outputs across the full K dimension, including both scale groups in the down projection. The index query keeps its FP32 result. | [MegaMoE](../solution/mega_moe/mega_moe_sim.hpp), [MX matmul](../basic_op/matmul/) |
| FP32 projections, RMSNorm, both mHC paths and Engram | Broadcast products and reductions compute FP32 dots and norms. HF32 rounding uses a U32 bit view. Sinkhorn uses 20 row/column passes and HC post accumulates four fused products in stream order. Engram computes the signed square root and applies its gate across the hidden channels. | [RMSNorm](../solution/normalization/rms_norm/), [reductions](../basic_op/reduction/) |
| Compressor, RoPE, indexer, window/sparse attention and sink | Strided loads and stores address even/odd RoPE lanes. Page scoring and online softmax set the valid row count on each tile. The softmax denominator uses FP32 probabilities, the numerator rounds probabilities to BF16, and the sink adds mass only to the denominator. | [FlashAttention](../basic_op/fa/), [sparse Flash MLA](../solution/quant_sparse_flash_mla/) |
| Activations and cache formats | Group abs-max, scaling and `TCVT` produce FP8 activations/window rows, FP4 compressed KV with group-16 BF16 scales, and FP4 index rows with group-32 UE8M0 scales. Integer tile operations restore signed zero. The C2A compressor corrects exact middle values that round up under nearest-even. | [MX quantization](../basic_op/mxquant/) |
| Sparse Top-K, routing, SwiGLU and combine | `TROWARGMAX`, comparisons and `TSEL` mask each winner. The router gathers the unbiased scores for normalization. SwiGLU clips gate/up independently and applies each route weight before requantization. `TADD` combines contributions in slot order. | [ArgMaxV2](../basic_op/argmaxv2/), [elementwise](../basic_op/element_wise/), [MegaMoE](../solution/mega_moe/) |

Scalar code schedules layers, pages, experts and tiles, validates IDs, maps selected logical rows to physical slots and processes the transport protocol. The arithmetic above uses the [0.59 TileOP API](https://github.com/LinxISA/Linx-TileOP-API/tree/7954b9207c748e298c90b7a86964351ad0e4f565).

## Build and verify

From `benchmark/one-level-arch`, use a 0.59 Linx clang/musl toolchain, matching Linx-TileOP-API headers, and a SuperScalarModel build with hosted common entry and memory dump support. The fixture generator requires Python 3, NumPy and ml_dtypes. Set `COMPILER_DIR` to the compiler bin directory. Set `TILEOP_INCLUDE` when the headers are installed separately.

```sh
make -C test/solution/deepseekv4.1 diss
elf=output/solution/deepseekv4.1/elf/solution_deepseekv4.1_decode.elf
"$GFRUN" -s softcore.multiThreadNum=4 -f "$elf"
python3 test/solution/deepseekv4.1/run.py --elf "$elf" --gfrun "$GFRUN" --all-layers --jobs 4
```

The ELF embeds a valid layer-0 case. The runner replaces its input section with the requested layer's fixture. It requires formal End, zero status and a matching guest-memory dump. The independent NumPy oracle checks each token's attention HC state, routing IDs and weights, individual expert contributions, final HC state, published indices and candidate blocks, and every final cache bank. Reuse fixtures run the source attention in the oracle to derive the inputs. Use `--layers 2 20 24 --start 0` for a cold window and partially filled selections, `--shards 2` for expert dispatch across two logical shards, and `--empty-shard` for a deterministic idle shard.

Replace `--all-layers` with `--regressions` to run the layer-2 HF32 rounding case (seed 7, positions 22–23) and layer 20 across a page boundary using a valid shuffled page table. Duplicate physical pages are rejected before the layer or an upstream publication can modify the cache. Cold C2A and C1A Reuse accept repeated `-1` padding and validate that each selected slot is distinct and belongs to the published logical prefix under the current page table. The suite also checks duplicate window slots and Reindex candidate blocks.

The directed edge test checks all three FP4 midpoint policies, both sides of a midpoint, signed underflow and group scales at the byte level. It also checks sparse padding, tied expert IDs and normalization of very negative logits. A passing hosted run reports `formal_End=1 formal_Abort=0 status=0`.

```sh
make -C test/solution/deepseekv4.1 TESTCASE=tile_edges diss
"$GFRUN" -s softcore.multiThreadNum=4 -f output/solution/deepseekv4.1/elf/solution_deepseekv4.1_tile_edges.elf
```
