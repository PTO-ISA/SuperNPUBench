# Moe — Mixture of Experts 路由算子（PTO 一层 tile 版）

迁移自 `TileKernels/tile_kernels/moe/`。全部 tile 版，经 linx 工具链编译通过。
工具链未暴露的指令用可用指令模拟（详见各 `.hpp` 头部注释）。

## 已迁移

| 文件 | 对应 TileKernels | 功能 | 主指令（实际编译版） |
|------|------------------|------|---------------------|
| `normalize_weight_pto.hpp` | `moe/normalize_weight_kernel.py` | top-k 路由权重的行归一化 | `TROWSUM`+`TADDS`+`TRECIP`+`TROWEXPANDMUL`（TROWEXPANDDIV 未提供→recip+mul） |
| `topk_gate_pto.hpp` | `moe/topk_gate_kernel.py` | 每 token 选 top-k 专家索引 | `TROWMAX`+`TROWEXPANDSUB`+`TCMP`+逆索引`TSEL`+`TROWEXPAND`（TROWARGMAX 未提供→模拟） |
| `group_count_aux_fi_pto.hpp` | `moe/group_count_kernel.py`、`aux_fi_kernel.py` | group/expert 计数直方图+aux 归一化 | `TEXPANDS`+`TCMP`+`TSEL`+`TROWSUM`+`TINSERT`（THISTOGRAM 未提供→直方图 tile）+`TCVT`/`TMULS` |
| `topk_sum_and_topk_group_idx_pto.hpp` | `moe/topk_sum_and_topk_group_idx_kernel.py` | group 内 top-1/top-2 sum 后稳定选 top-k groups | `TROWMAX`+`TROWEXPANDSUB`+`TCMP`+逆索引`TSEL`+`TINSERT`（warp shuffle rank→tile 归约+稳定 tie-break） |
| `top2_sum_gate_pto.hpp` | `moe/top2_sum_gate_kernel.py` | top-2-sum group 过滤后的 MoE top-k gate 核心路径 | `TEXP`/`TLOG`/`TSQRT`+`TROWMAX`+`TROWSUM`+`TROWEXPANDMUL`+逆索引`TSEL`（mask/physical-map data-dependent 分支不走 pointer-indexed tensor path） |
| `expand_to_fused_pto.hpp` | `moe/expand_to_fused_kernel.py` | token 扩展到 fused 布局（scatter 写） | `TLOAD`+循环`TSTORE`(pos 偏移)+`TEXPANDS` 清零 |
| `reduce_fused_pto.hpp` | `moe/reduce_fused_kernel.py` | 加权归约回 token 级 | `TMULS`+`TADD`（TFMA 未提供→拆乘加）+`TCVT` |
| `inplace_unique_group_indices_pto.hpp` | `moe/inplace_unique_group_indices_kernel.py` | 组索引原地去重 | `TEXTRACT`+`TCMP`+`TSEL`+`TINSERT` pairwise（位图需 cumsum→改 O(k²) pairwise） |
| `mask_indices_by_tp_pto.hpp` | `moe/mask_indices_by_tp_kernel.py` | 按 TP rank 屏蔽并转局部专家索引 | `TDIVS`/`TREMS`/`TSUBS`+`TEXPANDS`+`TCMP`+`TAND`(符号位)+`TSEL`（TCMPS 未提供→广播+TCMP） |
| `get_fused_mapping_pto.hpp` | `moe/get_fused_mapping_kernel.py` | token-topk↔expert-major 映射 | 直方图 `TCMP`+`TSEL`+`TROWSUM` + `TEXPANDS` 区间填充；位置映射需 scan（工具链无，标量 cursor 推进） |

## 注

- `TCMP`/`TSEL` 三参须同 dtype；`topk_gate` 索引运算全用 float、存储时 `TCVT` 回 int。
- `get_fused_mapping` 位置映射需 intra-tile 前缀和(scan)，工具链暂无 cumsum/sort，主体 tile + 标量 cursor。
- `aux_fi_kernel.py` 仍由 `group_count_aux_fi_pto.hpp` 的 combined port 显式代表；未拆成独立文件。
- `top2_sum_gate_pto.hpp` 代表 source-level tile-native 核心路径；源端 mask/fixed-routing/physical-map 分支需要 data-dependent scalar tensor lookup，按本目录约束未引入 pointer-indexed tensor data path。
- 新增 group/gate 端口均用 `static_assert` 约束 thread-local/vector tile 物理片段不小于 128B。
