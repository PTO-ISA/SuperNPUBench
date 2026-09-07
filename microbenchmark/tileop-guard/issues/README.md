# tileop-guard 看护 → 线上 issue 索引

每个已提交的线上 issue 对应本目录一个文件。命名：`ISSUE_<tag名>_<生成日期>_issue<编号>.md`，
标题行标注 `【线上 issue #<号>】`，一文件对应一个线上 issue。

## 已提交 issue 清单（7 份）

| 线上 # | 状态 | 目标仓 | tag | 生成日期 | 本地文件 | 覆盖 |
|---|---|---|---|---|---|---|
| **#49** | open | Linx-TileOP-API | ops-20260828 | 2026-09-02 | `ISSUE_ops-20260828_20260902_issue49.md` | 文档缺陷汇总（15 项） |
| **#50** | open | Linx-TileOP-API | ops-20260828 | 2026-09-02 | `ISSUE_ops-20260828_20260902_issue50.md` | 编译器/后端无法处理文档接口（4 项） |
| **#478** | open | SuperScalarModel | ops-20260828 | 2026-09-02 | `ISSUE_ops-20260828_20260902_issue478.md` | 模型未实现/契约拒绝（TIMG2COL/TMRGSORT/TileArray region/range::Subview，4 项） |
| **#560** | open | SuperScalarModel | ops-20260904 | 2026-09-05 | `ISSUE_ops-20260904_20260905_issue560.md` | ops-20260904 缺口 7 项（owner 已重分类：API #62/#63、模型保留、退休） |
| **#87** | open | llvm-project | ops-20260904 | 2026-09-05 | `ISSUE_ops-20260904_20260905_issue87.md` | TGPR2T 后端 Match Instruction Error |
| **#569** | open | SuperScalarModel | ops-20260904 | 2026-09-07 | `ISSUE_ops-20260904_20260907_issue569.md` | 4 类模型执行/校验缺口（range::subview cube/TCMP-reinterpret/TGATHER-TSCATTER/GMOV） |
| **#89** | open | llvm-project | ops-20260904 | 2026-09-07 | `ISSUE_ops-20260904_20260907_issue89.md` | clang-15 内联 bf16 CUBE matmul codegen SIGABRT |

## 关联的责任仓拆分 issue（owner/复核结论，非本目录提交）
- **Linx-TileOP-API #62**：B.FPATR None 路径固定发 RNE（#560 gfrun-1 的 API 侧根因）
- **Linx-TileOP-API #63**：reduction B.DIM 错用 destination 几何（#560 gfrun-2 的 API 侧根因）

## 命名约定
`ISSUE_<tag名>_<生成日期(YYYYMMDD)>_issue<线上号>.md`；标题行末标 `【线上 issue #<号>】`；一文件对应一线上 issue。
切换 tag 前（2026-09-03）的旧增量草稿已删除，以最新验证结果为准。
