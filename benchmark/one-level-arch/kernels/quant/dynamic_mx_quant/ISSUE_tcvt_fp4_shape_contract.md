# [Issue] emulator `ValidateOperandContract`（TCVT 分支）：按 physical row/col 严格相等校验，「宽类型 → 打包 fp4」的 TCVT 结构性必崩

## 摘要

`dynamic_mx_quant` 的 OCP fp4 输出 kernel（`TYPE=TAIL_OCP_FP4`）在 data 路径末尾用
`TCVT` 把 fp32 tile 转成打包 fp4（`__fp4_e2m1x2`）。编译与反汇编均正确，但 `gfrun` 在
**块解码组装阶段**（任何 Execute 之前）命中 emulator 断言：

```
PTO 0.58 TCVT requires matching source/destination logical shapes
isa/Block.cpp:1039  ValidateOperandContract
```

根因是 emulator 的 TCVT 形状契约**逐 conjunct 比对 physical `row`/`col`**，而 fp4 是**打包
类型**（1 字节 = 2 个 fp4），其 tile 描述符的 physical 维度天然与源不同 —— **无论编译器把
dst.col 发成 64（继承）还是 32（打包半宽），physical `row` 或 `col` 必有一个 ≠ src** →
断言必失败。缺陷所在仓 **`SuperScalarModel`（emulator 建模层）**，非 ISA、非工具链、非 kernel
逻辑。同一 gfrun 上，**非打包**的 OCP fp8 探针（e4m3，1 字节/元素）全链路跑通（`R2=0`），仅
打包 fp4 被此断言挡住，佐证缺陷专属打包类型。

## 环境

| 项 | 值 |
|---|---|
| 编译仓 | SuperNPUBench `benchmark/one-level-arch` |
| 执行仓 | SuperScalarModel（`bin/gfrun` 执行 elf） |
| 工具链 | env_test `linx_blockisa_llvm_musl`，clang-15，target `linx64v5-linux-musl` |
| 复现根目录 | `benchmark/one-level-arch/test/kernel/quant/dynamic_mx_quant` |
| 触发 kernel | `kernels/quant/dynamic_mx_quant/dynamic_mx_quant_tail_ocp_fp4.hpp`（`TYPE=TAIL_OCP_FP4`） |

### 各仓代码分支（复现基线）

| 仓库 | 分支 / HEAD | commit |
|---|---|---|
| SuperScalarModel（gfrun） | `feat/pto-v058-adaptation` | `5689b3e7` |
| └ 组成 | `origin/feat @ b3227fe5` + cherry-pick #254 `af060f31` + #253 `5689b3e7` | |
| linx-toolchain-build（env_test） | `main` | `e6a31ef` |
| └ Linx-TileOP-API | `8b2ee78`（Restore logical Tile register size） | |

> **基线说明（重要）**：本断言**只有在 scale 路径先被打通后才暴露**。gfrun 必须含
> #254（`af060f31`，`ValidateScalarLogicalTepl` 放宽 dtype 相等为位宽相等，见
> `ISSUE_reinterpret_dtype_tag.md`）与 #253（`5689b3e7`，bf16→e8m0 TCVT，见
> `ISSUE_e8m0_cvt.md`），执行流才会走完 OCP scale、抵达 data 路径的 fp32→fp4 TCVT 并崩在
> 本 issue。裸 `origin/feat @ b3227fe5`（不含 #253/#254）会更早崩在 scale 路径。

## 复现命令

```bash
export COMPILER_DIR=.../env_test/linx-toolchain-build/output/linx_blockisa_llvm_musl/bin
export LINX_SYSROOT=.../env_test/linx-toolchain-build/output/linx_blockisa_llvm_musl/sysroot/usr
cd benchmark/one-level-arch/test/kernel/quant/dynamic_mx_quant
make TESTCASE=dynamic_mx_quant TYPE=TAIL_OCP_FP4 diss   # 编译+反汇编，clean

SuperScalarModel/bin/gfrun -f "$PWD/output/.../dynamic_mx_quant_tail_ocp_fp4.elf"
```

## 报错

```
gfrun: illegal instruction: ASSERTION FAILED:
  srcTile.size() == 1 && dstTile.size() == 1 && srcTile[0] && dstTile[0] &&
  srcTile[0]->tileInfo && dstTile[0]->tileInfo &&
  srcTile[0]->tileInfo->row == dstTile[0]->tileInfo->row &&        // <-- 崩在此 conjunct（8 != 4）
  srcTile[0]->tileInfo->col == dstTile[0]->tileInfo->col &&
  srcTile[0]->tileInfo->validRow == dstTile[0]->tileInfo->validRow &&
  srcTile[0]->tileInfo->validCol == dstTile[0]->tileInfo->validCol &&
  srcTile[0]->tileInfo->layout == dstTile[0]->tileInfo->layout &&
  "PTO 0.58 TCVT requires matching source/destination logical shapes"
  func ValidateOperandContract, file isa/Block.cpp:1039   (EXIT=1)
```

调用链：`SetBlockIsComplete()`（Block.cpp:1136）在块解码组装完成时先调 `UpdateDstTileInfo()`
再调 `ValidateOperandContract()`，**均发生在任何 Execute 之前**。

## 根因分析

### 实证：崩在 `row` conjunct（instrumented gfrun 打印实际 tileInfo）

fp4 输出 TCVT **不发 lb2**（反汇编：块内只有 `B.DIM a6->lb0` / `B.DIM a7->lb1`）。在
`ValidateOperandContract` 入口插桩打印每条 TCVT 的 src/dst tileInfo，得到（`dt`：1=bf16、5=fp32、
13/18=e8m0 系、**11=fp4(e2m1x2)**）：

```
[TCVT-DIAG] srcN=1 dstN=1 src row=8 col=64 vRow=8 vCol=1  dt=1  | dst row=8 col=64 vRow=8 vCol=1  dt=5   ✓ 过
[TCVT-DIAG] srcN=1 dstN=1 src row=8 col=64 vRow=8 vCol=1  dt=18 | dst row=8 col=64 vRow=8 vCol=1  dt=1   ✓ 过
[TCVT-DIAG] srcN=1 dstN=1 src row=8 col=64 vRow=8 vCol=1  dt=5  | dst row=8 col=64 vRow=8 vCol=1  dt=13  ✓ 过
[TCVT-DIAG] srcN=1 dstN=1 src row=8 col=64 vRow=8 vCol=32 dt=5  | dst row=8 col=64 vRow=8 vCol=32 dt=1   ✓ 过（data 路径前置 fp32→bf16）
[TCVT-DIAG] srcN=1 dstN=1 src row=8 col=64 vRow=8 vCol=32 dt=1  | dst row=4 col=64 vRow=8 vCol=32 dt=11  ✗ 崩（fp4 输出）
```

scale 路径的 4 条 TCVT（dt→5/1/13、fp32→bf16）src/dst 形状全等、通过；**唯独最后一条 fp4 输出
TCVT（dst dt=11）** 的 `dst.row=4 ≠ src.row=8`，命中断言。逐 conjunct：

| conjunct | src（上游 fp32 tile） | dst（fp4） | 结果 |
|---|---|---|---|
| physical col | 64 | 64（inherit） | ✓ 相等 |
| validCol | 32 | 32 | ✓ 相等 |
| validRow | 8 | 8 | ✓ 相等 |
| **physical row** | **8** | **4** | **✗ 崩（:1042）** |

### col 单位 = fp4x2 打包字节数（源码实证）

emulator 的 physical `col` 对打包 fp4 计的是 **fp4x2 打包单元（字节）数**，不是单个 fp4 元素数：
- `__fp4_e2m1x2` 的 `BytesOf(FP4)=HF4_DATA_WIDTH=1`（`isa/ISACommon/DataType.h`，1 字节 = 2 个 fp4）。
- `col = physicalCol` 直接取自 lb2 或继承（`UpdateDstTileInfo`）。
- physical `row` 由 `row = size / (physicalCol × BytesOf(elem))` 反推
  （`Block.cpp:1384-1386`），要求 col 按打包单元计才自洽。

### row 数值来源

`Block.cpp:1384-1386`：`dst->tileInfo->row = dst->size / (dstPhysicalCol × elemBytes)`：
- src（fp32，elemBytes=4）：`2048 / (64 × 4) = 8`。
- dst（fp4，elemBytes=`BytesOf(FP4)=1`）：`256 / (64 × 1) = 4`。
- dst 继承了 src 的 col=64（见下），但 fp4 每元素只占 1 字节 → 同 size 下 row 减半 → **8 ≠ 4**。

### inherit 分支确实生效（no-lb2 情形）

`Block.cpp:1250-1263`：`inheritTcvtShape = (tileOp==TCVT && shapeSource)`，
`explicitPhysicalCol = (bdimMask & (1<<2)) && lb2 != 0`。fp4 TCVT 不发 lb2 →
`explicitPhysicalCol=false` → `physicalCol = shapeSource->col = 64`（**dst.col 继承 src.col**）。
即 inherit 生效了，但继承来的 col=64 用 fp4 的 1 字节/元素反推 row 得 4，与 src 的 8 冲突。

### 结构性：打包 fp4 的 physical row 天然减半

「宽类型 → 打包 fp4」的 TCVT 中，dst 继承 src 的 `col=64`，但 fp4 每元素只占 1 字节（`BytesOf=1`），
故同 `size` 下 `row = size/(col×elemBytes)` 必然减半（8→4），physical `row` 天然 ≠ src。这是打包
类型的**结构性**特征，与 BlockSize、具体数值、哪个 fp4 kernel 无关。**而 validRow/validCol 两边恒
相等**（8/8、32/32），只比 valid 维度即可放过。

### 对照：非打包 fp8 在同一 gfrun 上全过

同一 gfrun（`5689b3e7`）跑 OCP fp8 探针 `dynamic_mx_quant_probe_ocp_fp8`（e4m3 输出，1 字节/元素、
非打包）：`Total Block=23, Inst=120, R2=0, EXIT=0`；`_newcalc` 变体 `Block=29, Inst=148, R2=0`。
fp8 输出 TCVT 的 physical row/col 不减半 → 过 `ValidateOperandContract`。说明缺陷**专属打包类型**。

## 定性

- **属 emulator 建模缺陷（过严校验）**，非 ISA/工具链/kernel 缺陷。
- TCVT 到打包类型时，dst 的 physical row/col 与 src 天然不同（打包改变 col↔row↔size），这是**正确**
  的描述符；真实硬件按 datatype 字段解释 bit、按 valid 维度处理数据即可运行，不存在「src/dst physical
  维度必须逐一相等」的约束。
- 断言把 physical `row`/`col` 强行要求相等，等于**禁掉一切「宽类型 → 打包窄类型」的 TCVT**（fp4、
  以及任何 `bits<8` 的打包 dtype）。

## 修复建议（emulator 侧）

放宽 `ValidateOperandContract()` 的 TCVT 分支（`Block.cpp:1038-1050`）：**删掉 physical
`row==row` 与 `col==col` 两条 conjunct，保留 `validRow==validRow` + `validCol==validCol` +
`layout==layout`**，另加 physical ≥ valid 的健全性检查。

```cpp
// 原（1042-1044）：
srcTile[0]->tileInfo->row == dstTile[0]->tileInfo->row &&
srcTile[0]->tileInfo->col == dstTile[0]->tileInfo->col &&
// 改：只比 valid 维度 + layout（physical 维度对打包类型天然不同，不应强等）
// 保留 validRow/validCol/layout 三条；physical row/col 仅做 >= valid 的下界健全性检查
```

因 fp4 两边 validRow/validCol 都相等（8/8、32/32），放宽后 fp4 TCVT 过校验且**不影响正确性**
（数据搬运/转换按 valid 维度进行，physical 只是打包后的字节布局）。

> **注**：历史上曾有提交 `eaa3dfe7 "fix(tile): relax TCVT legality to valid shape only"` 做过
> 此放宽，但该提交已被 force-push 从 `origin/feat/pto-v058-adaptation` 重写丢弃、**不可作为依据**。
> 建议在当前 feat 上**重新独立落地**上述放宽，勿 cherry-pick 已丢弃的提交。

**保留不动**：`Block.cpp:1107` 的 Local TMOV 分支断言（另一处 row/col 相等校验，非 TCVT，不在本
issue 范围）。

### 反证：kernel 侧「加宽 dst tile 骗过断言」会产出错误数据（修复必在 emulator）

为确认「能否不改 emulator、只在 kernel 侧规避」，做了一次对照实验：把 fp4 输出 tile 的 physical
列从打包 `PW/2` 改成 `PW`、valid 列从 `BlockSize/2` 改成 `BlockSize`（即声明成「未打包」宽度）。
此时 dst tile size = `TileM × PW × BytesOf(FP4)` = `8×64×1 = 512B`，继承 col=64 反推
`row = 512/(64×1) = 8 == src.row` → **断言通过、gfrun 跑到底 R2=0**。

但**输出数据是错的**（`M=8, N=64, BlockSize=32` 实测，env_test 工具链 + gfrun `5689b3e7`）：

| 对照项 | 结果 |
|---|---|
| gfrun 断言 | ✅ 过（不再崩 `Block.cpp:1039`） |
| `scale_output` vs golden | ✅ 逐字节相同 |
| **`output`（data）vs golden** | ❌ **不同** |

字节实证：golden 每字节打包 2 个 fp4 半字节（`byte0=0x26` = nibble 2,6）；加宽后的输出把**每个
fp4 摊进一整字节、高半字节恒 0**（`byte0=0x06`、`byte1=0x02`）。即 valid col 声明成 `BlockSize=32`
后，32 个 fp4 被**解包**成 32 字节（而非打包成 16 字节），输出宽度翻倍、越界覆盖相邻 block。

**结论**：加宽 dst tile 只是让 physical `row=512/64=8` 恰好等于 src 而绕过断言，**代价是 fp4
打包布局被破坏**。kernel 必须保持 `tile_o` 打包（physical `PW/2`、valid `BlockSize/2`）；本 issue
的断言是 fp4 data 路径的**真实阻塞**，无法在 kernel 侧无损规避，**修复只能在 emulator 侧**（放宽
physical row/col 校验，如上）。

## 影响面

- 阻断 `dynamic_mx_quant_tail_ocp_fp4` 的 gfrun 执行（OCP fp4 data 路径 fp32→fp4 输出转换）。
- 泛化影响：**任何「宽类型（fp32/bf16/half）→ 打包 fp4（或其他 `bits<8` 打包 dtype）」的 TCVT**，
  在 emulator 上都会被本断言结构性挡住。这是 MX-quant fp4 输出量化的必经手法。
- 不影响非打包输出（fp8 e4m3/e5m2 等 1 字节/元素），已实测同 gfrun 全过。
