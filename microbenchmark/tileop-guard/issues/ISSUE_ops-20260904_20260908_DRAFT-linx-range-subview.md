# [Linx-TileOP-API][docs] 文档缺陷汇总（ops-20260904 tileop-guard 看护发现）：range::subview 推荐写法与 B.SUBVIEW CUBE-only 契约矛盾 + 定位矛盾；GMOV dtype 表缺 FP32；TROWMAX 示例（物理 M×N）照抄编译失败  【草稿·待提交】

> **状态**：本地草稿（尚未提交线上）。目标仓 **LinxISA/Linx-TileOP-API**（文档缺陷亦归此仓）。
> 提交后按命名规范改为 `ISSUE_ops-20260904_20260908_issue<号>.md` 并回填编号。
>
> **口径**：本 issue 记录的是**可确证的文档缺陷**（推荐写法与 ISA 契约矛盾 + 文档定位自相矛盾）。文中"是否给
> `range::subview`+TSTORE 补 CUBE 重载 / 加编译期拒绝"属**接口设计取舍**，作**开放建议**供 API 团队判断，
> 不作既成的代码缺陷断言。
>
> **与 #569 gfrun-8 的关系**：#569 owner 把 `range_subview` run-fail 判为 `demo`（parent 须 Matrix+CUBE，模型拒 RowMajor 是 ASL-correct）——**该裁决对**。本 issue 是**独立的文档层问题**：API 文档 §"Recommended view API" 示例就用 Vec/RowMajor parent，即"照推荐写法写"必然触发那个被模型拒的构造——是文档在推荐一个 ISA 不支持的用法，非 demo 个案。

## 组件版本清单

| 组件 | 仓库 | 分支 / tag | commit |
|---|---|---|---|
| **Linx-TileOP-API（本 issue 目标）** | LinxISA/Linx-TileOP-API | `linx` | `0566283` |
| SuperScalarModel（模型契约来源） | LinxISA/SuperScalarModel | `codex/consolidate-post-main-fixes-20260903` | `49547742` |
| pto-spec（ISA 规范） | PTO-ISA/pto-spec | `main` | `dea0b75e`（0.58.6.0） |
| SuperNPUBench(看护 demo) | [SuperNPUBench PR #96](https://github.com/PTO-ISA/SuperNPUBench/pull/96)（分支 `tileop-guard-batch1`） | — | `<PR#96 tip·push 后回填>` |
| llvm-project | LinxISA/llvm-project | `dev-llvm15_56` | `67d3ac9` |

> 工具链指纹：clang++ md5 `e427d1429c0e`、gfrun md5 `0c433cd11c00`。

---

## 问题 A：推荐的 `range::subview`（RowMajor）+ TSTORE 写法被模型拒

**API 推荐写法**（`docs/tileop-usage/range-modifiers.md` §"Recommended view API"，逐字）：
```cpp
using TileT = Tile<Location::Vec, float, 4, 8, BLayout::RowMajor>;   // 推荐示例的 parent = Vec/RowMajor
using GM   = global_tensor<float, RowMajor<4, 8>>;
TileT tile;  GM gm;
auto zero_based_source = range::subview(tile);
TSTORE(gm, zero_based_source);
```

**模型契约**（pto-spec `asl/block/model/operands/subview-descriptor.asl::BundleCubeSubviewDescriptorOf`）：
subview 的 Local descriptor **必须** `parent.location==TileLocation_Matrix && TileLayoutIsCube(parent.layout)`
且 valid geometry / cell_count 非零，否则在无副作用前以 `Fault_TileLegality` 拒绝。即 **B.SUBVIEW 只接受
Matrix-location + CUBE-layout parent**。

**冲突**：照 API 推荐写法（Vec/RowMajor parent）编译通过、发出 B.SUBVIEW，但 gfrun 执行即拒：
```
gfrun: illegal instruction at 0x0: illegal TSTORE operand or descriptor contract
```
模型此拒**是正确的**（与 subview-descriptor.asl 一致，有 `RejectsLocalNormSubviewParent` 回归）。
**文档缺陷**：文档把"针对 RowMajor parent 的 range::subview+TSTORE"列为推荐写法，而该 layout 的 B.SUBVIEW
按 ISA 契约**永远不可执行**——即文档在推荐一个 ISA 不支持的用法。

**代码层可观测事实（佐证，非缺陷定性）**：`common/pto_tile.hpp` 的 `range::subview` 可包装任意 parent；
唯一消费者 `TSTORE`（`jcore/template_asm.hpp:2459`）的 subview 分支带 `requires(!tile_shape::IsCubeLayout)`
——只对非 CUBE parent 发 B.SUBVIEW；换 CUBE parent 则路由到 cube 重载 `TSTORE_CUBE`（`:3022`），它不处理
subview 且 `Subview::data()` 非 const → 编译失败。故当前 `range::subview`+TSTORE 事实上只能针对 RowMajor
（=不可执行的那种）。**是否让该路径支持 CUBE parent，属接口设计取舍**（见文末开放建议），此处仅陈述现状。

**对照（cube subview 可行，非全盘不可用）**：高层 `TPARTVIEW`+region op（TABS 等）在 **CUBE parent** 上
能发出被模型接受的 B.SUBVIEW（实测编译+跑通，无 Fault）——但其 cube CELL-ordering 被 API 文档自认
"未实现/验证"（见问题 B 第 3 条），实测输出为转置。故 cube subview 通路存在但未完成。

**复现**（在 SuperNPUBench 仓）：
```bash
# 1. 从 PR #96 拉取最新看护 demo 代码
gh pr checkout 96                    # https://github.com/PTO-ISA/SuperNPUBench/pull/96
# 2. 指向工具链并逐 case 复现
source microbenchmark/tileop-guard/env.sh
bash microbenchmark/tileop-guard/run_guard.sh tlsu range_subview      # 推荐写法 → run-fail
bash microbenchmark/tileop-guard/run_guard.sh tlsu tpartview_subview  # TPARTVIEW+CUBE → run-only（转置）
```

**建议**：
- **文档订正（可确证缺陷，主）**：修订 `range-modifiers.md` "Recommended view API"——示例 parent 改为 Matrix/CUBE，
  并明确 B.SUBVIEW 仅适用 CUBE parent（对齐 subview-descriptor.asl）；RowMajor 示例移除或标注"不可执行"。
- **开放建议（接口设计取舍，供 API 团队判断）**：是否为 `range::subview`+`TSTORE` 补 CUBE parent 重载（使一条
  非 TPARTVIEW 的 cube subview 路径可用），或对 RowMajor subview 加编译期拒绝（避免发出注定被模型拒的 bundle）。

---

## 问题 B：三处文档对 `range::subview` 的定位互相矛盾

同一 `range::subview` 在 API 文档里定位不一致，误导选型：

| 文档 | 定位 |
|---|---|
| `docs/tileop-usage/range-modifiers.md`（§Recommended view API / "the ordinary surface"） | 把小写工厂 `range::subview(parent[, base])` 当**常规/推荐接口**；大写 `Subview<>` carrier 才是 lower-level |
| `docs/tileop-usage/migration/pto-0583-migration.md` | 普通 kernel 应迁移到**工厂** `range::subview<bytes,off>()`；手填 SizeCode 的 carrier/专家接口仅兼容/编码测试 |
| `docs/tileop-usage/range-modifiers-developer-guide.md` | **"range::subview/assemble 仅作底层兼容与测试接口，不是业务主入口"**，推荐 **TPARTVIEW/TileArray** |

- 前两者把 `range::subview` 工厂当推荐入口；developer-guide 却把整个 `range::subview` 贬为底层 compat 并改推 TPARTVIEW。
- 且"推荐入口"的推荐示例（RowMajor）恰是问题 A 里模型拒绝的写法。

**另一处文档自认未实现**（`range-modifiers.md` line 118-123）：
> region producer inline-asm path is **intentionally limited to `RowMajor + NoneBox`**; do not use `TCVT` or
> row-wise region producers with **Cube fragments until the Cube binder/CELL ordering path is implemented and validated**.

即 TPARTVIEW+region 的 CUBE 路径 API 自认未验证——与问题 A 实测的 cube 输出转置吻合。

**建议**：统一 range modifier 文档的接口定位（`range::subview` 工厂 vs TPARTVIEW 谁是主入口），并使"推荐写法/示例"
与模型 subview-descriptor.asl 的 CUBE-only 契约一致；补齐或明确标注 cube CELL-ordering 的实现/验证状态。

---

## 问题 C：GMOV.md dtype 表遗漏 FP32（与自身示例矛盾、与实现不符）

**现象**：`tlsu/peer-move/GMOV.md` §"支持的数据类型"（行 16）只列 packed 4-bit 类型
（E2M1X2 / E1M2X2 / HiF4X2 / S4X2 / U4X2），**FP32 不在其中**；但**同页示例（行 90）却用
`TileF32 = Tile<Location::Vec, float, 8, 32, RowMajor>`（float）**——文档自相矛盾。

**佐证 FP32 合法**：SuperScalarModel 对 GMOV 的官方修复 **PR #573**（#569 gfrun-11）新增的回归就是
**"16×16 FP32 descriptor + byte-copy"** → FP32 是合法 GMOV 数据类型。故 dtype 表过窄/漏列。

**复现**：`bash run_guard.sh tlsu gmov`（demo 用 float，跟了文档示例；本身是 gfrun-11 的 witness）。
**建议**：GMOV.md dtype 表补齐 FP32（及其他实际支持的非 packed 类型），与示例/实现一致。

---

## 问题 D：TROWMAX.md（行归约族）示例输出 tile 为物理 M×N，照抄在 ops-20260904 下编译失败

**现象**：ops-20260904 的 TileOP-API 要求**所有**行归约（TROWMAX/MIN/SUM/PROD/ARGMAX…）目的端为
**单列 tile（Cols==1）**——头 `template_asm.hpp` 每个都有 static_assert：
```
template_asm.hpp:13497: static assertion failed ... "TROWMAX destination must be a single-column tile (N x 1)"
  static_assert(tile_shape_out::ValidCol == DYNAMIC || (ValidCol==1 && Cols==1), ...)
```
（TROWSUM=13420、TROWMIN=13570、TROWPROD=13643、TROWARGMAX=13786 同款断言。）
但**同族 per-op 文档示例的 OutputTile 自相矛盾**——有的已改真 M×1、有的仍是物理 M×N（照抄即编译失败）：

| 文档 | OutputTile 示例（行 98） | 照抄结果 |
|---|---|---|
| `reduce-and-expand/row/TROWMAX.md` | `Tile<Vec,float,32,32,RowMajor,32,1>`（物理 M×N） | **编译失败** |
| `.../TROWMIN.md` | `Tile<Vec,float,32,32,RowMajor,32,1>`（物理 M×N） | **编译失败** |
| `.../TROWPROD.md` | `Tile<Vec,float,32,32,RowMajor,32,1>`（物理 M×N） | **编译失败** |
| `.../TROWSUM.md` | `Tile<Vec,float,32,1,RowMajor>`（**真 M×1**） | 编译通过（已修，作对照） |

（实测 TROWMAX：物理 32×32 编译失败；真 M×1 `Tile<Vec,float,32,1,RowMajor,32,1>` 通过。）

**归属**：**文档示例滞后且族内不一致**（TROWSUM 已同步单列约束，TROWMAX/MIN/PROD 未同步）。**独立于 #63**
（后者是运行期 header 把 dst 维度发进 B.DIM 的模型/工具链问题，见 #560 gfrun-2）。
**建议**：TROWMAX/TROWMIN/TROWPROD（及其他仍用物理 M×N 的行归约）示例输出对齐 TROWSUM.md 的真 `M×1`（Cols=1）。

> **附注（TCOLMAX 不列为文档缺陷）**：TCOLMAX.md 示例的物理 32×32（valid 1×32）**能编能跑**（列归约未加单行
> 约束），但结果错（模型 col-reduce 只读 row0）——真 1×N 与物理 M×N 输出**同为 row0**。此为**模型 bug #63
> （#560 gfrun-2）**，非文档缺陷，不并入本 issue。

## 看护落点

- `tlsu/range_subview`（`range::subview`+TSTORE，RowMajor，照推荐写法）→ **run-fail witness**（问题 A）。
- `tlsu/tpartview_subview`（TPARTVIEW+TABS，CUBE）→ **run-only**（cube 通路可跑，CELL-ordering 未验证=转置，问题 B 第 3 条）。
