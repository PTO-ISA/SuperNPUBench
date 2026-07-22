# Quant — FP8/FP4 量化算子（PTO 一层 tile 版）

迁移自 `TileKernels/tile_kernels/quant/`。现有端口保留，并补齐
`36d9e45d38e204ebb87e6f6e833821eee0482fe5` 中尚未单独表示的 standalone
量化模块。工具链未暴露的指令用可用指令模拟（详见各 `.hpp` 头部注释）。

## 已迁移

| 文件 | 对应 TileKernels | 功能 | 主指令（实际编译版） |
|------|------------------|------|---------------------|
| `cast_back_pto.hpp` | `quant/cast_back_kernel.py` | FP8/FP4 反量化为 fp32（per-token/per-channel） | `TCVT`+`TROWEXPANDMUL`/`TCOLEXPANDMUL`（TDEQUANT 未提供→显式 TCVT+广播乘） |
| `cast_back_e5m6_pto.hpp` | `quant/cast_back_e5m6_kernel.py` | E5M6 packed 反量化 | `TLOAD`+`TCVT`+`TROWEXPAND`+`TMUL`+`TSTORE`（packed word 粒度；不引入标量 pointer 数据路径） |
| `per_block_cast_pto.hpp` | `quant/per_block_cast_kernel.py` | block 级 FP8/FP4 量化，含 sf-only / precomputed-sf 入口 | `TROWMAX`+`TCOLMAX`+`TMAX`+`TMAXS`+`TRECIP`+`TCOLEXPAND`+`TROWEXPAND`+`TMUL`+`TCVT` |
| `per_block_cast_lossless_pto.hpp` | `quant/per_block_cast_lossless_kernel.py` | FP4→FP8 block lossless recast 源级 tile 端口 | `TLOAD`+`TCVT`+`TMULS`+`TRECIP`+`TCOLEXPAND`+`TROWEXPAND`+`TMUL`+`TSTORE` |
| `per_channel_cast_and_transpose_pto.hpp` | `quant/per_channel_cast_and_transpose_kernel.py` | per-channel 量化并转置 | `TLOAD`+`TTRANS`+`TROWMAX`+`TRECIP`+`TROWEXPAND`+`TMUL`+`TCVT`+`TSTORE` |
| `per_channel_cast_fused_pto.hpp` | `quant/per_channel_cast_fused_kernel.py` | fused per-channel 量化（contiguous no-expand 路径） | `TCVT`+`TCOLMAX`+`TMAX`+`TRECIP`+`TCOLEXPAND`+`TMUL`+`TSTORE` |
| `per_channel_cast_kernel_pto.hpp` | `quant/per_channel_cast_kernel.py` | standalone per_channel_cast 模块入口 | 复用 `per_channel_cast_pto.hpp` 中已存在的列归约量化链 |
| `per_token_cast_pto.hpp` | `quant/per_token_cast_kernel.py`、`per_channel_cast_kernel.py` | 行级/列级 FP8 量化（amax→sf→缩放） | `TMAX`(`TROWMAX`(x),`TROWMAX`(`TMULS`(x,-1)))(absmax 模拟) + `TMAXS`+`TMULS`+`TRECIP`+`TROW/TCOLEXPANDMUL`+`TCVT`（TABS 未提供→max(x,-x)） |
| `per_token_cast_to_e5m6_pto.hpp` | `quant/per_token_cast_to_e5m6_kernel.py` | per-token E5M6 packed cast 源级 tile 端口 | `TROWMAX`+`TMAX`+`TRECIP`+`TROWEXPAND`+`TMUL`+`TCVT`+`TSTORE`（packed word 粒度） |
| `swiglu_backward_and_per_token_cast_pto.hpp` | `quant/swiglu_backward_and_per_token_cast_kernel.py` | SwiGLU backward + x/y grad per-token 量化 | `TCVT`+`TEXP`+`TRECIP`+`TMUL`+`TROWMAX`+`TROWEXPAND`+`TSTORE`（map scatter/gather 不降为标量 pointer 访问） |
| `swiglu_fused_cast_pto.hpp` | `quant/swiglu_forward_and_per_token_cast_kernel.py` | 融合 SwiGLU 前向 + per-token 量化 | `TMULS`(x,-1)(TNEG 未提供)+`TEXP`+`TADDS`+`TRECIP`+`TMUL`(silu) → 量化链 |
| `swiglu_forward_and_per_channel_cast_and_transpose_pto.hpp` | `quant/swiglu_forward_and_per_channel_cast_and_transpose_kernel.py` | SwiGLU forward + per-channel 量化，含 transpose/no-transpose 入口 | `TEXP`+`TRECIP`+`TMUL`+`TTRANS`+`TROWMAX`/`TCOLMAX`+`TROWEXPAND`/`TCOLEXPAND`+`TCVT` |

## 注

- bf16 tile 列宽须 16 的倍数（32B 对齐）；1 行 tile 须 ≥128B（TLOAD 最小）。
- amax 输出 tile 用物理 8 宽 valid1（每行/列一个标量）。
- 新增端口均显式 `static_assert` thread-local tile fragment ≥128B。
- E5M6 和 map-driven fused backward/expand 的源端含 lane bit-pack 或索引
  gather/scatter；本轮只补 source-level tile 端口，不新增标量 pointer-indexed tensor
  数据路径。
