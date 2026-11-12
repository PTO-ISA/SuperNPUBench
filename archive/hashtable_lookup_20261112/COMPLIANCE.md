# NDF 合规性审计（draft.8，2026-11-12，revision 2）

**结论：仍为 `incomplete`。** 本轮补齐后可核对的范围扩大到：全部 P 运行都有完整
SwimLane/nocounters 与周期起止口径（r1/r2 重跑周期数与原始 `run.log` 一致：
135,819 / 82,256），外部资产有字段表，111 项要求有覆盖映射。仍**不能**宣称
`material_checked`、`run_reproduced`、`build_reproduced` 或 NDF 整体合规。

## 条款状态

| 条款 | 要求摘要 | 状态 | 缺口/说明 |
|---|---|---|---|
| WORK-ARCHIVE-001 | 三级身份、目录、不可变依赖 ID | 部分 | task/result/run 有；目录用相对路径映射（允许）；四个依赖的 `manifest.json` 已建（含 `dependency_id`/来源/哈希），但 API 快照的上游 commit 仍未知（`GAP-API-1`） |
| WORK-ARCHIVE-002 | 内容要求矩阵（28 项） | 部分 | 新增：`ASSETS.md` 资产字段表、r1/r2 周期起止与分母声明、`requirements_coverage.md` 覆盖表、r2 `algorithm.md` 完整假设/正确性/性能条件。仍缺：r2 PipeView（`GAP-PV-1`）、负控、独立复核、修改详情完整字段 |
| WORK-ARCHIVE-003 | 源码/构建/ELF 对应 | 部分 | 仓库/commit/patch/构建选项/ELF 哈希 ✅；构建日志缺（`GAP-LOG-1`）；编译器二进制逐字节重建未证（`GAP-BIND-2`） |
| WORK-ARCHIVE-004 | 依赖与本地修改 | 部分 | compiler/model/api/dataset 的 `manifest.json` 已建；compiler/model 二进制哈希 ✅；API 快照来源缺（`GAP-API-1`）；修改详情表的上游生命周期字段不全 |
| WORK-ARCHIVE-005 | 输入与正确性 | 部分 | 输入哈希 ✅；checker 身份/脚本/日志未归档；边界用例覆盖表缺；负控 `not_run` |
| WORK-ARCHIVE-006 | 性能口径与 trace | 部分 | r1/r2 的完整 SwimLane + nocounters + swim 日志已归档（`A-R1/R2-TRACE`，含字节数/SHA256）；首 TLOAD→末 TSTORE 的 kernel span 已记录（134,326 / 80,669）；初始化/同步/读回计入范围与“延迟指标无峰值分母”已声明。仍缺：r2 PipeView（`GAP-PV-1`） |
| WORK-ARCHIVE-007 | 优化结论与保留 | ✅ 基本满足 | observation / hypothesis / controlled_comparison / derived_bound 已区分；保留策略见 `analysis/` |
| WORK-ARCHIVE-009 | 协作与不可变 | 部分 | owner/archiver ✅；revision 记录已补（README 末尾）；reviewer 仍为空 |
| WORK-ARCHIVE-010 | 分级验收 | ❌ 未达到 | 111 项覆盖表已建（`requirements_coverage.md`，自评），但未逐条 material check；无独立复核 → 不能 `run_reproduced`；未做独立目录重建 → 不能 `build_reproduced` |
| WORK-ARCHIVE-011 | 算法定义与假设清单 | 部分 | r2 `algorithm.md` 已含数据布局、哈希、探测伪代码、正确性条件、性能成立条件；假设 A-001/2/3 与 `H-RWDB`/`H-WAY` 有 id 与违反行为；ISA 版本仅在目标 triple 层面记录（`linx64v5`），ASL 不适用（gfsim 为计时模型） |

## 本轮新增/更新的证据

| 证据 | 位置 | 覆盖条款 |
|---|---|---|
| 资产字段表（asset_id/required_for/字节/SHA256/位置/状态） | `ASSETS.md` | 002、006、009 |
| 逐文件字节数 + SHA256 | `FILE_MANIFEST.json`、`FILE_MANIFEST.sha256` | 002 |
| 111 项覆盖映射（自评） | `requirements_coverage.md` | 002、010 |
| r1 完整 trace（full + nocounters + log，gzip） | 本机 `outputs/hashtable_lookup_archive_assets/r1_rand300_01/` | 006 |
| r2 完整 trace（full + nocounters + log，gzip） | 本机 `outputs/hashtable_lookup_archive_assets/r2_rand300_rw48/` | 006 |
| kernel span 与口径 | `results/r1.../README.md`、`results/r2.../README.md` | 006 |
| 算法/正确性/性能条件 | `results/r2.../algorithm.md` | 002、011 |
| 依赖 manifest | `dependencies/*/manifest.json` | 001、004 |

## 距合规最近的补齐项（按性价比排序）

1. **独立复核**：由非归档人按 `build/run_perf.sh` 在独立目录重建并运行，记录
   `run_reproduced`；这是 WORK-ARCHIVE-010 的唯一硬门槛。
2. **r2 PipeView**：`GAP-PV-1`，用于支撑“RWDB 并发/停顿”结论；当前结论由
   `analysis/RWDB_BOTTLENECK.md` 的 PMU + 周期对比支撑，未用流水事件。
3. **资产发布**：把 `A-REF-ELF`、`A-R1/R2/R3-ELF`、`A-R1/R2-TRACE` 上传 Release，
   补 `download_verified` 与校验和（`GAP-EXT-1`）。
4. **负控**：对 checker 输入篡改输出，证明错误结果会被拒绝（`not_run`）。
5. **逐条 material check**：按 `requirements_coverage.md` 的 `check_id` 实际执行并
   把 `partial/not_checked` 升级或保持。
6. **构建日志**：补 LLVM/gfsim 的完整构建日志（`GAP-LOG-1`）。

## 现在可以宣称的能力

- `metadata_integrity`：`FILE_MANIFEST.json` + `FILE_MANIFEST.sha256` 可逐文件核对；
  manifest JSON 可解析。
- 作者本地 `build_reproduced` 的**子项**：`build/build_kernel.sh` 重建 r1/r2 ELF
  逐字节一致（md5 `f5d5a0cb…` / sha256 `35218ed0…`），`build/run_perf.sh` 复现 135,819。
- **trace 一致性**：r1/r2 的 SwimLane 重跑周期数与原始 `run.log` 完全一致
  （135,819 / 82,256），日志以 `linx_test_finisher ... pass` 结束。
- **不能**宣称：独立运行复现、数值运行 PASS（r3）、NDF 整体合规。
