# Mhc — Manifold HyperConnection 算子（PTO 一层 tile 版）

迁移自 `TileKernels/tile_kernels/mhc/`。全部 tile 版，经 linx 工具链编译通过。
工具链未暴露的指令用可用指令模拟（详见各 `.hpp` 头部注释）。

## 已迁移

| 文件 | 对应 TileKernels | 功能 | 主指令（实际编译版） |
|------|------------------|------|---------------------|
| `expand_to_mhc_pto.hpp` | `mhc/expand_kernel.py`（fwd） | token 沿 mhc 维扩展复制 | `TLOAD` + 循环 `TSTORE`（广播写） |
| `expand_to_mhc_bwd_pto.hpp` | `mhc/expand_kernel.py`（bwd） | 沿 mhc 维求和归约（反向） | `TCVT` + `TCOLSUM` |
| `sinkhorn_pto.hpp` | `mhc/sinkhorn_kernel.py`（fwd） | 行列交替归一化 | `TROWMAX`+`TROWEXPANDSUB`+`TEXP`+`TROWSUM`+`TADDS`+`TRECIP`+`TROWEXPANDMUL` + `TCOLSUM`+`TCOLEXPANDMUL`（TROWEXPANDDIV 未提供→recip+mul） |
| `norm_fn_pto.hpp` | `mhc/norm_fn_kernel.py` | RMSNorm + normw 列广播合并 | `TMUL`+`TROWSUM`+牛顿 rsqrt(`TRECIP`+`TMUL`+`TMULS`+`TADDS`)+`TROWEXPANDMUL` / `TCOLEXPANDMUL`（TRSQRT 未提供→牛顿迭代） |
| `multilayer_recompute_pto.hpp` | `mhc/multilayer_recompute_kernel.py` | 多层重算 GEMM 累加链 | `TMATMUL_ACC`(链式累加)+`ACCCVT`（TileLeft/TileRight/TileAcc） |
| `head_compute_mix_pto.hpp` | `mhc/head_compute_mix_kernel.py` | head mix sigmoid 前向/反向 source-level tile port | `TLOAD`/`TSTORE`+`TMULS`+`TADDS`+`TEXP`+`TRECIP`+`TCOLSUM`+`TROWSUM` |
| `pre_split_mixes_pto.hpp` | `mhc/pre_split_mixes_kernel.py` | pre/post/comb mix 拆分前向/反向 source-level tile port | `TLOAD`/`TSTORE`+`TMULS`+`TADDS`+`TEXP`+`TRECIP` |
| `pre_apply_mix_pto.hpp` | `mhc/pre_apply_mix_kernel.py` | pre mix 加权 residual 前向/反向 source-level tile port | `TLOAD`/`TSTORE`+`TCVT`+`TROWEXPAND`+`TCOLEXPAND`+`TMUL`+`TADD`+`TCOLSUM`+`TROWSUM` |
| `post_pto.hpp` | `mhc/post_kernel.py` | post mix 前向/反向 source-level tile port | `TLOAD`/`TSTORE`+`TCVT`+`TROWEXPAND`+`TMUL`+`TADD`+`TROWSUM` |
| `pre_big_fuse_pto.hpp` | `mhc/pre_big_fuse_kernel.py` | fused pre path：RMS、split、Sinkhorn、pre-apply | `TLOAD`/`TSTORE`+`TRSQRT`+`TEXP`+`TRECIP`+`TROWMAX`+`TROWSUM`+`TCOLSUM`+`TROWEXPAND`+`TCOLEXPAND` |

## 注

- `sinkhorn` eps 用 constexpr 局部量（float 非类型模板参数工具链不支持）。
- 3D 张量展平为 2D（global_iterator 仅 2 参）。
- 新增 source-level tile ports 使用物理 ≥128B thread-local tile fragments；逻辑标量/短向量采用有效形状收窄的物理 tile。
