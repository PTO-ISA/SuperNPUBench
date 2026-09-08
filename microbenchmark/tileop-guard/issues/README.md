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

## 草稿（待提交）

| 目标仓 | tag | 生成日期 | 本地文件 | 覆盖 |
|---|---|---|---|---|
| Linx-TileOP-API (docs) | ops-20260904 | 2026-09-08 | `ISSUE_ops-20260904_20260908_DRAFT-linx-range-subview.md` | **文档缺陷汇总(4)**：A) `range::subview` 推荐写法(RowMajor)与 B.SUBVIEW CUBE-only 契约矛盾；B) 三处文档对 range::subview 定位矛盾 + cube CELL-ordering 未验证；C) GMOV.md dtype 表缺 FP32(与自身示例/PR#573 矛盾)；D) TROWMAX.md 示例物理 M×N 照抄编译失败(API 要求单列)。（TCOLMAX 结果错=#63 模型 bug，未并入；"补 cube 重载"作开放建议非断言） |
| 待官方裁决(分析指向 model) | ops-20260904 | 2026-09-08 | `ISSUE_ops-20260904_20260908_DRAFT-gfrun-region-tilearray.md` | region TileArray/TASSEMBLY 多 writer(parent>writer) B.ASSEMBLE 被 `PrepareLocalAssemble` 拒；发射侧已自证合规(carrier=parent尺寸+offset/coverage全对+INIT=parent规范)；🆕未提交，owner 待官方裁决；确切行号待 `-g` 重编 |

## 关联的责任仓拆分 issue（owner/复核结论，非本目录提交）
- **Linx-TileOP-API #62**：B.FPATR None 路径固定发 RNE（#560 gfrun-1 的 API 侧根因）
- **Linx-TileOP-API #63**：reduction B.DIM 错用 destination 几何（#560 gfrun-2 的 API 侧根因）

## 命名约定
`ISSUE_<tag名>_<生成日期(YYYYMMDD)>_issue<线上号>.md`；标题行末标 `【线上 issue #<号>】`；一文件对应一线上 issue。
切换 tag 前（2026-09-03）的旧增量草稿已删除，以最新验证结果为准。
