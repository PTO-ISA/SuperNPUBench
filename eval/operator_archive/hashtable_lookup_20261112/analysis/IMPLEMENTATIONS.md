# basic_op/hashtable_lookup 的实现清单

（2026-11-12 状态；源码在 `benchmark/one-level-arch/kernels/basic_op/hashtable_lookup/`）

| 文件 | 数据契约 | 探测风格 | guard | 定位 |
|---|---|---|---|---|
| `hashtable_lookup.hpp` | — | — | — | 入口，默认 → adaptive |
| `hashtable_lookup_tileop_adaptive.hpp` | AoS 16 B | 32-lane，自适应 2+6（`HASH_ADAPTIVE_TAIL=2` 回 2+2） | 112 B（7 项） | **当前默认** |
| `hashtable_lookup_tileop_masked.hpp` | AoS 16 B | 32-lane，1 槽/轮 masked gather | 无（表尾回绕） | 无 guard 场景 |
| `hashtable_lookup_tileop_stride4_guard.hpp` | AoS 16 B | 32-lane，4 槽/轮 | 48 B（3 项） | 上一版默认/基线 |
| `hashtable_lookup_element.hpp` | AoS 16 B | 32-lane，element-if 逐 lane | 无 | element-region 参考 |
| `hashtable_lookup_soa_wide.hpp` | **SoA** | 64/128/256/512-lane，2+6 | ≥7 项（推荐 128） | 宽 lane 性能参考（需 PR146 工具链） |

## 必要性

- 功能必需：`adaptive`（默认）、`masked`（无 guard）。
- 保留作参考/基线：`stride4_guard`（140,166 历史记录与 A/B）、`element`（element-region 示范）。
- 新增参考：`soa_wide`（唯一 256-lane 2+6 实现；非默认，不能替换 AoS API）。
- 已删除：`hashtable_lookup_tileop.hpp`（仅本地分支存在的 1 槽/轮 full gather 版本）。

## 与 perf kernel 的关系

`soa_wide` 就是 perf kernel（`kernel_adaptive_2_6.hpp`）的算子化版本：同一 2+6 结构、
同一 SoA 数据、同一 batch map 调度；2026-11-12 实跑与 perf kernel 差异 ≤1.2%。
见 `PERFORMANCE_MATRIX.md` 第 2 节。
