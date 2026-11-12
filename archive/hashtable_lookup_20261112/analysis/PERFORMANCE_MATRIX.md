# 性能矩阵与扫描（observation / controlled_comparison）

Metric：gfsim 结束行 `Total Cycles`。配置：`config/random300.json`
（`core.soc_random=true`、`mean_read/write_latency=300`、读写带宽 128 B/cycle、
`core.sharetreg_size_bytes=262144`、`ssb=64`、`TAG_WAY_COUNT=4` 除非注明）。
模型：pr1177 @ 18b24f6ca。所有数字为单次确定性运行。

## 1. lane × 自适应变体 × RWDB（SoA perf kernel，36 组）

| lane | 变体 | RWDB=8 | 16 | 24 | 32 |
|---:|---|---:|---:|---:|---:|
| 64 | 2+2 | 135,819 | 122,285 | 121,951 | 124,107 |
| 64 | 2+4 | 141,153 | 122,524 | 117,339 | 118,607 |
| 64 | 2+6 | 143,096 | 118,207 | 117,524 | 116,635 |
| 128 | 2+2 | 158,089 | 113,091 | 98,149 | 98,129 |
| 128 | 2+4 | 156,016 | 101,328 | 94,104 | 92,755 |
| 128 | 2+6 | 165,628 | 104,769 | 93,469 | 87,674 |
| 256 | 2+2 | 168,059 | 112,852 | 96,767 | 91,150 |
| 256 | 2+4 | 171,499 | 113,404 | 93,145 | 88,902 |
| 256 | 2+6 | 170,132 | 136,217 | 95,444 | 86,331 |

补充点（256-lane 2+6）：RWDB=40 → 82,720；48 → 81,317；64 → 81,437；
48 + `TAG_WAY=7` → **77,312**。

每档最优：8 → 64/2+2（135,819）；16 → 128/2+4（101,328）；24 → 256/2+4（93,145）；
32 → 256/2+6（86,331）。

## 2. 新算子 header `hashtable_lookup_soa_wide.hpp` 实跑（256-lane 2+6）

2026-11-12 用归档 header 重建 fixture（累到的 6144-query SoA 数据 + batch map）：

| RWDB | soa_wide header | 同日 perf kernel | 差异 |
|---:|---:|---:|---:|
| 8 | 169,868 | 170,132 | -0.2% |
| 16 | 136,545 | 136,217 | +0.2% |
| 24 | 96,372 | 95,444 | +1.0% |
| 32 | 87,237 | 86,331 | +1.0% |
| 48 | 82,256 | 81,317 | +1.2% |
| 48 + WAY7 | 76,959 | 77,312 | -0.5% |

结论：header 重构（`lookupBatch` + `run`）与 perf kernel 等价（差异为 codegen）。

## 3. 解释（区分观察与结论）

- observation：RWDB 从 8 加深到 32/48 时，所有 lane/变体都显著变快；256/2+6 从 170,132
  到 81,317（-52%）。
- observation：默认 RWDB=8 时加宽 lane 反而更慢（64→128→256：135,819→158,089→168,059）。
- controlled_comparison：同一 ELF、同一配置，只改 `sl2.RWDB_ENTRY_COUNT`；只改
  `sl2.TAG_WAY_COUNT` 的组已单独标注（r2 way7）。
- derived_bound：在飞 gather 并发 = `floor(RWDB_ENTRY_COUNT / beats_per_gather)`；
  64 lane × 4 B 与 32 lane × 8 B 都是 256 B = 2 beat。
- hypothesis（有支持，未单独消融）：256-lane 在 8 项时受“wave 全有或全无”限制，
  详见 `RWDB_BOTTLENECK.md`。

## 4. 复现

- 矩阵：`outputs/hash64_batchmap_20261105/slots_sweep/repro_matrix/`（9 个 ELF 的
  重建脚本与 36 份日志；md5 与历史一致）。
- soa_wide：`results/r2_256l_2p6_rwdb48/README.md` 与 `build/`。
