# Engram — Engram 门控算子（PTO 一层 tile 版）

迁移自 `TileKernels/tile_kernels/engram/`。全部 tile 版，经 linx 工具链编译通过。

## 已迁移

| 文件 | 对应 TileKernels | 功能 | 主指令（实际编译版） |
|------|------------------|------|---------------------|
| `fused_weight_pto.hpp` | `engram/engram_fused_weight_kernel.py` | bf16×bf16→fp32 权重融合 | `TCVT` + `TMUL` |
| `engram_hash_pto.hpp` | `engram/engram_hash_kernel.py` | n-gram 哈希嵌入索引 | `TMULS`+`TXOR`+`TREMS`+`TADDS`（int32 hash） |

## 注

- bf16 tile 列宽须 16 的倍数（32B 对齐）；1 行 tile 须 ≥128B（TLOAD 最小）。
- `engram_hash` 索引/哈希累加跨 ngram 步串行，但每步运算跨 token tile 并行。
