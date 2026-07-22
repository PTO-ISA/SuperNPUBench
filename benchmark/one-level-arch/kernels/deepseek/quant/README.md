# Quant — FP8/FP4 量化算子（PTO 一层 tile 版）

迁移自 `TileKernels/tile_kernels/quant/`。全部 tile 版，经 linx 工具链编译通过。
工具链未暴露的指令用可用指令模拟（详见各 `.hpp` 头部注释）。

## 已迁移

| 文件 | 对应 TileKernels | 功能 | 主指令（实际编译版） |
|------|------------------|------|---------------------|
| `cast_back_pto.hpp` | `quant/cast_back_kernel.py` | FP8/FP4 反量化为 fp32（per-token/per-channel） | `TCVT`+`TROWEXPANDMUL`/`TCOLEXPANDMUL`（TDEQUANT 未提供→显式 TCVT+广播乘） |
| `per_token_cast_pto.hpp` | `quant/per_token_cast_kernel.py`、`per_channel_cast_kernel.py` | 行级/列级 FP8 量化（amax→sf→缩放） | `TMAX`(`TROWMAX`(x),`TROWMAX`(`TMULS`(x,-1)))(absmax 模拟) + `TMAXS`+`TMULS`+`TRECIP`+`TROW/TCOLEXPANDMUL`+`TCVT`（TABS 未提供→max(x,-x)） |
| `swiglu_fused_cast_pto.hpp` | `quant/swiglu_forward_and_per_token_cast_kernel.py` | 融合 SwiGLU 前向 + per-token 量化 | `TMULS`(x,-1)(TNEG 未提供)+`TEXP`+`TADDS`+`TRECIP`+`TMUL`(silu) → 量化链 |

## 注

- bf16 tile 列宽须 16 的倍数（32B 对齐）；1 行 tile 须 ≥128B（TLOAD 最小）。
- amax 输出 tile 用物理 8 宽 valid1（每行/列一个标量）。
