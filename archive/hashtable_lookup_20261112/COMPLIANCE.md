# NDF 合规性审计（draft.8，2026-11-12）

**结论：不满足。** 按 NDF §6.1.4，本归档保持 `incomplete`；可以宣称的只有
“目录/身份/结果索引/优化证据链已成型 + 作者本地 build 自检通过”，
**不能**宣称 `material_checked`、`run_reproduced`、`build_reproduced` 或 NDF 合规。

## 条款状态

| 条款 | 要求摘要 | 状态 | 缺口/说明 |
|---|---|---|---|
| WORK-ARCHIVE-001 | 三级身份、目录、不可变依赖 ID | 部分 | task/result/run 有；目录用相对路径映射（允许）；`dependencies/*/manifest.json` 与 `dependency_id` 未建 |
| WORK-ARCHIVE-002 | 内容要求矩阵（28 项） | 部分 | 已覆盖：目标/负责人/结果索引、shape/dtype/layout/tiling、假设与限制、算法简述、本地修改摘要、构建脚本与参数、ELF+`.diss`（本次补）、冻结配置、原始日志、优化证据链、文件哈希清单。缺失：每个 P 运行的完整 SwimLane/nocounters、r2 的 PipeView、周期起止点、MFU/带宽公式与分母、资产交付字段表、负控、独立复核、覆盖表、修改详情完整字段 |
| WORK-ARCHIVE-003 | 源码/构建/ELF 对应 | 部分 | 仓库/commit/patch/构建选项/ELF 哈希 ✅；构建日志缺（`GAP-LOG-1`）；编译器二进制逐字节重建未证（`GAP-BIND-2`） |
| WORK-ARCHIVE-004 | 依赖与本地修改 | 部分 | compiler/model 身份与二进制哈希 ✅；API 快照来源缺（`GAP-API-1`）；修改详情表字段（因果对照、成本归属、上游生命周期）不全；issue 对应仅 #1210 |
| WORK-ARCHIVE-005 | 输入与正确性 | 部分 | 输入哈希 ✅；checker 身份/脚本/日志未归档；边界用例覆盖表缺；负控 `not_run` |
| WORK-ARCHIVE-006 | 性能口径与 trace | ❌ 缺 | 只有 `Total Cycles`；缺 SwimLane/nocounters（r2 没有，r1 的 bundle 未纳入归档）、r2 PipeView、kernel/pure 起止事件、初始化/同步/读回计入说明、MFU/带宽公式与分母 |
| WORK-ARCHIVE-007 | 优化结论与保留 | ✅ 基本满足 | observation / hypothesis / controlled_comparison / derived_bound 已区分；保留策略声明可再明确一句 |
| WORK-ARCHIVE-009 | 协作与不可变 | 部分 | owner/archiver ✅；reviewer 为空；无 revision/supersedes 记录 |
| WORK-ARCHIVE-010 | 分级验收 | ❌ 未达到 | 未做逐条 material check；无独立复核 → 不能 `run_reproduced`；未做独立目录重建运行 → 不能 `build_reproduced`；现有仅为作者自检 |
| WORK-ARCHIVE-011 | 算法定义与假设清单 | 部分 | 假设 A-001/2/3 有 id、类别、违反行为；缺独立 `algorithm.md`（伪代码/数据流图）、ISA/ASL 版本、§3.12 维度清单的完整逐项说明 |

## 距合规最近的补齐项（按性价比排序）

1. **覆盖映射**：引附录 K 的 111 个 `requirement_id`，生成附录 J.4 覆盖表（逐项
   `pass/fail/not_checked/not_applicable` + 理由 + check_id + 证据路径）。这是
   §6.1.4 的硬要求。
2. **trace**：为 r2 补一次 `--swimlane` 运行（rw48）并归档 `*_swim.json.gz` 与
   nocounters（工作区 `run_swimlane.py` 口径）；r1 的 bundle 改引归档相对路径；
   补一次 r2 PipeView。
3. **性能口径**：为 r1/r2 记录首 TLOAD/末 TSTORE 的 cycle 起止、初始化/同步/读回是否计入、
   MFU/带宽公式与分母（或注明“本条不适用”及原因）。
4. **数值验证材料**：checker 版本与 hash、`result.json`、负控（篡改输出应被拒绝）。
5. **依赖治理**：确认 API 快照的上游 commit 或发布为依赖包；补 LLVM/CMake 构建日志；
   每个依赖补 `manifest.json`（类型/来源/版本/构建/哈希/恢复）。
6. **独立复核**：由非归档人执行 `run_reproduced`（并在独立目录重建 → `build_reproduced`）。
7. **资产交付**：按 §3.7.4 字段表把参考 ELF、结果 ELF、trace 发布到 Release 并记录
   `asset_id`/获取方式/状态。

## 现在可以宣称的能力

- `metadata_integrity`：归档内文件哈希清单可核对；manifest JSON 可解析。
- 作者本地 `build_reproduced` 的**子项**：`build/build_kernel.sh` 重建 r1/r2 ELF
  逐字节一致（md5 `f5d5a0cb…` / sha256 `35218ed0…`），`build/run_perf.sh` 复现 135,819。
- **不能**宣称：独立运行复现、数值运行 PASS（r3）、NDF 整体合规。
