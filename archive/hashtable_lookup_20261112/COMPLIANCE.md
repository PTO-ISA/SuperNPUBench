# NDF 合规性审计（draft.8，2026-11-12，revision 3）

**结论：仍为 `incomplete`，但已从“只可宣称 metadata_integrity”推进到“材料与逐条
检查已自评，等待发布与独立复核”。** revision 3 补齐了：路径整改与扫描、r2 的
rw8/way7 完整 trace、r1/r2 PipeView、宿主资源实测、比较器负控、kernel 重建日志、
修改/假设详情、111 项逐条检查记录。仍**不能**宣称 `material_checked`（下载侧未做）、
`run_reproduced`、`build_reproduced` 或 NDF 整体合规。

## 条款状态

| 条款 | 状态 | 说明 |
|---|---|---|
| WORK-ARCHIVE-001 | 部分 | 四个依赖 manifest 已建；API 上游 commit 仍未知（`GAP-API-1`） |
| WORK-ARCHIVE-002 | 部分 | 资产字段表、周期口径、覆盖表、检查记录、算法文档齐全；r3 失败运行仅诊断、无 256-lane 独立 check（`GAP-R2-CHECK`） |
| WORK-ARCHIVE-003 | 部分 | kernel 重建日志已补（r1/r2 字节一致）；LLVM/gfsim 完整构建日志仍缺（`GAP-LOG-1`） |
| WORK-ARCHIVE-004 | 部分 | 修改/假设详情表已补；API 来源与 Release URL 待补 |
| WORK-ARCHIVE-005 | 部分 | 负控与 checker 身份已归档；输出读回与边界覆盖仍缺 |
| WORK-ARCHIVE-006 | ✅ 基本满足（自评） | 5 个 P 运行均有日志+full+nocounters；r1/r2 有 PipeView；span/口径/宿主资源齐全；语义标注不适用；r2 PipeView 未解析（仅原始数据） |
| WORK-ARCHIVE-007 | ✅ | 优化证据链与保留策略不变 |
| WORK-ARCHIVE-008 | 部分 | 各状态维度已有显式记录，但无独立复核 |
| WORK-ARCHIVE-009 | 部分 | revision 3 记录已建；reviewer 仍为空 |
| WORK-ARCHIVE-010 | ❌ 未达到 | 覆盖表与逐条记录已建（自评）；缺下载后 material、独立 `run_reproduced`、独立 `build_reproduced` |
| WORK-ARCHIVE-011 | ✅（自评） | `algorithm.md` 含定义/伪代码/正确性/性能条件/假设表；ISA/ASL 已陈述 |

## revision 3 新增证据

| 证据 | 位置 |
|---|---|
| 路径整改与扫描（工具 sha256、52 文件、0 个人路径） | `validation/PATH_SCAN.md`、`path_scan_revision3.jsonl` |
| 111 项覆盖映射（pass 51 / partial 42 / n/a 2 / not_checked 16） | `requirements_coverage.md` |
| 111 项逐条检查记录（§6.1.1 字段） | `validation/check_records.md` |
| r2 rw8 / way7 完整 trace（周期与日志一致） | `ASSETS.md` 的 `A-R2-RW8-TRACE` / `A-R2-WAY7-TRACE` |
| r1/r2 PipeView（周期不变） | `A-PIPEV` |
| 宿主资源：15.1 s、峰值 RSS 2.34 GB | `results/r1.../README.md`、`A-PIPEV` |
| 比较器负控：0x5555 pass / 0x3333 fail、rc 0/70 | `validation/negative_control/` |
| kernel 重建：r1 md5、r2 sha256 与归档一致 | `validation/build_repro_r3.log` |
| diss 头规范化（仅第 2 行，正文一致） | `validation/diss_normalization.md` |
| 修改/绕过/硬件假设逐项字段 | `dependencies/LOCAL_MODIFICATIONS.md` |

## 距合规最近的动作

1. **发布 Release**（资产 + 归档 tarball），补 URL 后做下载侧 `material_checked`。
2. **独立复核**：非归档人在独立目录下载并重跑 r1/r2，产生新周期日志与 SwimLane，
   同时跑 check；记录复核人/时间/revision/命令/偏差。
3. **独立重建**：固定工具链下重建 r1/r2 ELF 并比较（作者本地已一致，缺独立环境）。
4. 补 `GAP-R2-CHECK`（256-lane 独立 check）与 `GAP-LOG-1`（LLVM/gfsim 构建日志）。

## 现在可以宣称的能力

- `metadata_integrity`：清单/JSON/hash 全量核对通过。
- 作者本地 `build_reproduced` 子项：r1/r2 ELF 字节一致。
- 路径检查（归档 + release payload）：个人路径 0 命中，命中逐项有判定。
- 比较器负控：错误输出被拒绝（rc=70）。
- **不能**宣称：`material_checked`（下载侧）、`run_reproduced`、`build_reproduced`、
  NDF 整体合规。
