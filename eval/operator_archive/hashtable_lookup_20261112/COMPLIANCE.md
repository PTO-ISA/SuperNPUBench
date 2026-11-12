# NDF 合规性审计（draft.8，2026-11-12，revision 4）

**结论：仍为 `incomplete`，但已达成 `material_checked`（下载侧）。** revision 3 补齐
材料与逐条记录；revision 4 发布 Release、重新下载并完成哈希/清单/路径扫描，因此可以
宣称“材料检查通过（记录范围内）”。仍**不能**宣称 `run_reproduced`、
`build_reproduced` 或 NDF 整体合规——这三项需要非归档人独立执行。

## 条款状态

| 条款 | 状态 | 说明 |
|---|---|---|
| WORK-ARCHIVE-001 | 部分 | 四个依赖 manifest 已建；API 上游 commit 仍未知（`GAP-API-1`） |
| WORK-ARCHIVE-002 | 部分 | 材料/口径/覆盖/检查记录齐全；r3 失败运行仅诊断（`GAP-R3-1`），256-lane 无独立 check（`GAP-R2-CHECK`） |
| WORK-ARCHIVE-003 | 部分 | kernel 重建日志已补（r1/r2 字节一致）；LLVM/gfsim 完整构建日志仍缺（`GAP-LOG-1`） |
| WORK-ARCHIVE-004 | 部分 | 资产已发布并下载验证（R017/R018 通过）；API 来源仍未知 |
| WORK-ARCHIVE-005 | 部分 | 负控与 checker 身份已归档；输出读回与边界覆盖仍缺 |
| WORK-ARCHIVE-006 | ✅ 基本满足（自评） | 5 个 P 运行均有日志+full+nocounters；r1/r2 有 PipeView；span/口径/宿主资源齐全；语义标注不适用 |
| WORK-ARCHIVE-007 | ✅ | 优化证据链与保留策略 |
| WORK-ARCHIVE-008 | 部分 | 状态维度显式记录；无独立复核 |
| WORK-ARCHIVE-009 | 部分 | revision 3/4 记录已建；reviewer 仍为空 |
| WORK-ARCHIVE-010 | ❌ 未达到 | `material_checked` 已达成；缺独立 `run_reproduced` / `build_reproduced` |
| WORK-ARCHIVE-011 | ✅（自评） | `algorithm.md` 定义/伪代码/正确性/性能条件/假设表齐全 |

## 验收层级

| 层级 | 状态 | 证据 |
|---|---|---|
| `metadata_integrity` | 通过 | `FILE_MANIFEST.json`（55 文件）+ `sha256sum -c` 56/56 |
| `material_checked` | **通过（下载侧，2026-11-12）** | `validation/material_check.md`：12 附件、11/11 hash、归档清单自校验、下载/解压 payload 个人路径 0 命中 |
| `run_reproduced` | 未做 | 需非归档人在独立目录下载并重跑 r1/r2，产生新周期日志与 SwimLane |
| `build_reproduced` | 未做 | 作者本地重建一致（`validation/build_repro_r3.log`），缺独立环境复核 |

## revision 4 新增证据

| 证据 | 位置 |
|---|---|
| Release（固定 commit `4359f297`）与 11 个资产 | https://github.com/PTO-ISA/SuperNPUBench/releases/tag/hashlk-20261112-r3 |
| 下载/哈希/清单/路径检查记录 | `validation/material_check.md` |
| 资产 URL、字节、SHA256、状态 | `ASSETS.md` |
| 111 项覆盖映射（pass 55 / partial 38 / n/a 2 / not_checked 16） | `requirements_coverage.md` |
| 111 项逐条检查记录 | `validation/check_records.md` |

## 距合规最近的动作

1. **独立复核 `run_reproduced`**：非归档人按 `validation/material_check.md` 的下载步骤，
   在独立目录运行 r1/r2（期望 135,819 / 82,256），生成新周期日志与 SwimLane，并跑
   check 的数值验证；记录复核人/时间/revision/命令/偏差。
2. **独立 `build_reproduced`**：固定工具链下重建 r1/r2 ELF 并比较（作者本地已一致）。
3. 补 `GAP-R2-CHECK` 与 `GAP-LOG-1`；确认 API 快照上游 commit。

## 现在可以宣称的能力

- `metadata_integrity`：通过。
- `material_checked`：通过（下载侧，记录范围内；见 `validation/material_check.md`）。
- 作者本地重建一致（非独立）。
- 比较器负控：错误输出被拒绝（rc=70）。
- **不能**宣称：`run_reproduced`、`build_reproduced`、NDF 整体合规。
