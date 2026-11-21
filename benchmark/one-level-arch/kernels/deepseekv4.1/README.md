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

Weights are synthetic. The matrix-vector path unpacks output-major E4M3FN/E2M1 weights and UE8M0 scales and uses PTO tile loads, column broadcast multiplication, row reductions and stores. Intermediate BF16 boundaries and FP32 residuals are explicit. The cache stores packed FP8 window rows, FP4 compressed KV with group-16 BF16 scales, and FP4 index keys with group-32 UE8M0 scales. The public types in `types.hpp` describe the logical layouts used by this benchmark. Production checkpoint tiling, vocabulary lookup and tokenization are outside the layer boundary. See the [pypto-lib layer](https://github.com/hw-native-sys/pypto-lib/blob/182990263ee87ef29c835ead20a716e3f4aa3d19/models/deepseek_v4_1_flash/decode_layer.py) and [configuration](https://github.com/hw-native-sys/pypto-lib/blob/182990263ee87ef29c835ead20a716e3f4aa3d19/models/deepseek_v4_1_flash/config.py).

## Build and verify

From `benchmark/one-level-arch`, use a 0.59 Linx clang/musl toolchain, matching Linx-TileOP-API headers, and a SuperScalarModel build with hosted common entry and memory dump support. The fixture generator requires Python 3, NumPy and ml_dtypes. Set `COMPILER_DIR` to the compiler bin directory. Set `TILEOP_INCLUDE` when the headers are installed separately.

```sh
make -C test/solution/deepseekv4.1 diss
elf=output/solution/deepseekv4.1/elf/solution_deepseekv4.1_decode.elf
"$GFRUN" -s softcore.multiThreadNum=4 -f "$elf"
python3 test/solution/deepseekv4.1/run.py --elf "$elf" --gfrun "$GFRUN" --all-layers --jobs 4
```

The ELF embeds a valid layer-0 case. The runner replaces its input section with the requested layer's fixture. It requires formal End, zero status and a matching guest-memory dump. The independent NumPy oracle checks each token's attention HC state, routing IDs and weights, individual expert contributions, final HC state, published indices and candidate blocks, and every final cache bank. Reuse fixtures run the source attention in the oracle to derive the inputs. Use `--layers 2 20 24 --start 0` for a cold window and partially filled selections, `--shards 2` for expert dispatch across two logical shards, and `--empty-shard` for a deterministic idle shard.
