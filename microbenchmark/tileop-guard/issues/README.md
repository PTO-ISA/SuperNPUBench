# tileop-guard 看护结果 → issue 提交清单

基于 TileOP-API v0.58 文档看护(125 case)整理,按「文档 / 仿真器 / 编译器」拆成 3 份可直接提交的 issue。
每份含统一的组件版本清单 + 通用复现步骤,内部以问题为单位分节。

| issue 文件 | 提交目标仓 | 前缀 | 覆盖问题 |
|---|---|---|---|
| `ISSUE_tileop_docs.md` | Linx-TileOP-API | (docs) | Doc-1..Doc-15:签名缺失 / 示例不可编译 / dtype·形状契约未写 / 语义未说明 |
| `ISSUE_gfrun_NA_model_gaps.md` | SuperScalarModel | `[gfrun][NA]` | gfrun-1..gfrun-4:TIMG2COL / TMRGSORT 未实现桩、TileArray region 未实现、range::Subview descriptor 拒绝 |
| `ISSUE_linx_toolchain_gaps.md` | llvm-project / Linx-TileOP-API | `[linx]` | linx-1..linx-4:`*.MASK`·GMOV 后端 Match Error、bf16 clang abort、reinterpret_tile 模板不兼容 |

**分野原则**:能靠补文档解决的(即便现象是 gfrun assert,也是 gfrun 在执行文档漏写的契约)→ 文档;
补文档也解决不了的 → 按报错组件分到 gfrun 或 linx。部分接口跨两类(TMRGSORT / range / TileArray region),
在两侧分别以对应侧根因立条并互相交叉引用。

**版本清单**(三份 issue 一致):

| 组件 | 分支 | commit |
|---|---|---|
| SuperNPUBench(demo,PR #96) | `ops-20260828` | `d1604cf` |
| SuperScalarModel | `codex/pr-0.58.4-shared-model` | `762a72c` |
| Linx-TileOP-API | `linx` | `d6a52b8` |
| llvm-project | `dev-llvm15_56` | `0f878a8` |
| musl | `linx` | `af0dfc2` |
| jemalloc | `linx` | `4495309` |
| linux-linxisa | `main` | `1055a74` |

看护 demo 代码:https://github.com/PTO-ISA/SuperNPUBench/pull/96
