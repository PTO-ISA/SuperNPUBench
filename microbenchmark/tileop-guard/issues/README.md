# tileop-guard 看护结果 → issue 提交清单

基于 TileOP-API v0.58 文档看护整理,按「文档 / 仿真器 / 编译器」拆成 3 份可直接提交的 issue。
每份含统一的组件版本清单 + 通用复现步骤,内部以问题为单位分节。

## 轮次差异(简述)

看护结果按时间戳分两份报告,本目录 issue 随之分两批;两个时点的差异如下:

- **2026-09-01**(见 `../REPORT_20260901.md`,三态,125 case):首轮看护,暴露文档/模型/工具链三类
  缺口,据此提交下方**首轮 3 份 issue**(Doc-1..15 / gfrun-1..4 / linx-1..4)。
- **2026-09-03**(见 `../REPORT_20260903.md`,四态,126 case):demo 持续完善(读头订正 TSEL/TCMP/
  argmax/MGATHER 等签名、为可钉语义的 run-only 补 host 独立 golden)+ 工具链升级(Linx-TileOP-API
  `d6a52b8→6f230c5`、llvm `0f878a8→25677bb`)后重跑。相对首轮的三点变化:
  1. 一批「无签名→编译失败」「`*.MASK`/GMOV 后端 Match Error」case 随读头订正与工具链升级**转正**;
  2. golden 改钉 pto-spec **预期语义**(不再采信实现),TLOG(ln)/TQUANT(mult/zp)/copy-expand(全广播)/
     TINSERT(保留 base)5 个原「run-only 假绿」落到**精度失败**,作模型缺口 witness → 新增 gfrun-5..8;
  3. 09-03 09:29 读头订正另发现 docs-N / gfrun-N / linx-N **增量 issue**(带时间戳后缀,见文末)。

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

---

## 第二轮新增 issue（2026-09-03 09:29，读头文件订正后新发现）

首轮 3 份 issue 保持不变；第二轮新发现单独成文（带日期时间后缀），版本清单以各文件内为准
（本轮工具链已更新：Linx-TileOP-API `d6a52b8→6f230c5`、llvm `0f878a8→25677bb`；demo 仍经
**SuperNPUBench PR #96** 提供 https://github.com/PTO-ISA/SuperNPUBench/pull/96 ）。

| issue 文件 | 提交目标仓 | 前缀 | 覆盖问题 |
|---|---|---|---|
| `ISSUE_tileop_docs_20260903_0929.md` | Linx-TileOP-API | (docs) | docs-N1 Subview parent 须 cube 布局（未文档化）；docs-N2 reinterpret_tile 使用规则（双视图 + 落盘重置标签），并修正首轮 linx-4「TABS 拒视图」判断 |
| `ISSUE_gfrun_NA_model_gaps_20260903_0929.md` | SuperScalarModel | `[gfrun][NA]` | gfrun-N1 TCMP 处理器拒 reinterpret 视图源（官方修复漏 TCMP）；gfrun-N2 TEXTRACT 任何真实子抽取被判非法（block 维取自源 valid 维）；gfrun-N3 region TASSEMBLY 只落 slot0、其余 slot 归零；gfrun-N4 B.FPATR PreQuant 消费 FP19 scale 参数时丢弃缩放后 payload（S8 输出≡offset、F16≡0；与 gfrun-5 同缺陷族） |
| `ISSUE_linx_toolchain_gaps_20260903_0929.md` | Linx-TileOP-API / llvm | `[linx]` | linx-N1 TSTORE_CUBE 无法接收 Subview 载体（const& vs 非 const `data()` + dtype 强绑），阻断 cube-subview 落盘 |
