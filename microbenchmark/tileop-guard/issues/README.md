# tileop-guard 看护 → 线上 issue 索引

每个已提交的线上 issue 对应本目录一个 `ISSUE-<号>-<slug>.md`，文件标题行标注线上编号。
过时/已被取代的草稿在 `superseded/`。

## 已提交 issue 清单（7 份）

| 线上 # | 状态 | 目标仓 | 本地文件 | 覆盖 |
|---|---|---|---|---|
| **#49** | open | Linx-TileOP-API | `ISSUE-49-docs.md` | 文档缺陷汇总（15 项：签名缺失/示例不可编译/dtype·形状契约未写/语义未说明） |
| **#50** | open | Linx-TileOP-API | `ISSUE-50-linx-compiler-backend.md` | 编译器/后端无法处理文档接口（*.MASK·GMOV Match Error、bf16 clang abort、reinterpret_tile 模板不兼容，4 项） |
| **#478** | open | SuperScalarModel | `ISSUE-478-gfrun-model-unimpl.md` | 模型未实现/契约拒绝（TIMG2COL / TMRGSORT / TileArray region / range::Subview，4 项） |
| **#560** | open | SuperScalarModel | `ISSUE-560-gfrun-ops20260904.md` | ops-20260904 缺口 7 项（已按 owner 回复重分类责任仓：API #62/#63、模型保留、退休） |
| **#87** | open | llvm-project | `ISSUE-87-linx-tgpr2t.md` | TGPR2T 后端 Match Instruction Error |
| **#569** | open | SuperScalarModel | `ISSUE-569-gfrun-model4.md` | 4 类模型执行/校验缺口（range::subview cube 限制、TCMP 拒 reinterpret 视图、TGATHER/TSCATTER 越界、GMOV 描述符匹配） |
| **#89** | open | llvm-project | `ISSUE-89-linx-bf16-codegen.md` | clang-15 对内联 bf16 CUBE matmul codegen SIGABRT |

## 关联的责任仓拆分 issue（owner/复核结论，非本目录提交）

- **Linx-TileOP-API #62**：B.FPATR None 路径固定发 RNE（#560 gfrun-1 的 API 侧根因）
- **Linx-TileOP-API #63**：reduction B.DIM 错用 destination 几何（#560 gfrun-2 的 API 侧根因）

## superseded/（过时草稿，勿提交）

`ISSUE_*_20260903_0929.md` 三份第二轮增量：线上无独立编号，内容已被本轮取代——
gfrun-N1(TCMP reinterpret)→现 #569 gfrun-9；gfrun-N4(FP19 scale)→现 #560 gfrun-4；
docs-N1(Subview cube)→现 #569 gfrun-8。保留作历史，不再提交。

## 命名约定
`ISSUE-<线上号>-<slug>.md`；每文件标题行末标 `【线上 issue #<号>】`；一文件对应一个线上 issue。
