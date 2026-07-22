# Engram — Engram 门控算子（PTO 一层 tile 版）

迁移自 `TileKernels/tile_kernels/engram/`。全部 tile 版，经 linx 工具链编译通过。

## 已迁移

| 文件 | 对应 TileKernels | 功能 | 主指令（实际编译版） |
|------|------------------|------|---------------------|
| `fused_weight_pto.hpp` | `engram/engram_fused_weight_kernel.py` | bf16×bf16→fp32 权重融合 | `TCVT` + `TMUL` |
| `engram_hash_pto.hpp` | `engram/engram_hash_kernel.py` | n-gram 哈希嵌入索引 | `TMULS`+`TXOR`+`TREMS`+`TADDS`（int32 hash） |
| `engram_gate_pto.hpp` | `engram/engram_gate_kernel.py` | Engram gate 前向/反向 source-level tile port | `TLOAD`/`TSTORE`+`TCVT`+`TMUL`+`TROWSUM`+`TRSQRT`+`TABS`+`TSQRT`+`TEXP`+`TRECIP` |
| `engram_grad_w_reduce_pto.hpp` | `engram/engram_grad_w_reduce_kernel.py` | partial `grad_w` 归约并累加 hidden/embed 权重梯度 | `TLOAD`/`TSTORE`+`TCVT`+`TADD`+`TMUL` |

## 注

- bf16 tile 列宽须 16 的倍数（32B 对齐）；1 行 tile 须 ≥128B（TLOAD 最小）。
- `engram_hash` 索引/哈希累加跨 ngram 步串行，但每步运算跨 token tile 并行。
- 新增 source-level tile ports 使用物理 ≥128B thread-local tile fragments；逻辑标量采用有效形状收窄的物理 tile。
