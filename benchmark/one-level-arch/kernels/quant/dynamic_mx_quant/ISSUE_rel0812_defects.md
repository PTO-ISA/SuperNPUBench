# [Issue] dynamic_mx_quant（release_ver0812）：5 处 model/toolchain 缺陷

在 `dynamic_mx_quant` kernel 的构建 + gfrun 过程中，命中 5 处不在 release_ver0812 已知问题
清单内的报错。逐一定位后，**根因均在 SuperScalarModel（emulator）或 linx-toolchain-build
（LinxV5 后端），非 SuperNPUBench 侧 kernel 逻辑**。

本文以**独立缺陷为单位**记录，每条给出触发条件、报错信息、原因分析、修复建议，互不依赖。
（规避某问题时链式触发的其它已知问题不在此文范围——本文只列可独立复现、可独立修复的缺陷。）

## 环境

| 项 | 值 |
|---|---|
| 复现仓 | SuperNPUBench `benchmark/one-level-arch`，commit `2b41a9c` |
| 分支 | `feat/dmxq-rel0812` |
| 工具链 | `linx_blockisa_llvm_musl`，clang-15，target `linx64v5-unknown-linux-musl`；llvm-project commit `eb64de8` |
| 执行器 | SuperScalarModel `gfrun`，commit `319294f` |
| 复现根目录 | `benchmark/one-level-arch/test/kernel/quant/dynamic_mx_quant` |
| 环境变量 | `export COMPILER_DIR=.../linx_blockisa_llvm_musl/bin`；`export LINX_SYSROOT=.../sysroot/usr` |

## 缺陷一览

| # | 缺陷所在仓 | 触发条件 | 报错 | 修复位置 |
|---|---|---|---|---|
| 1 | SuperScalarModel (emulator) | gfrun `TAIL_CUBLAS_FP8`（bf16 输入） | `AccumulateBlockInfo.cpp:393` TABS/BF16 | `AccumulateBlockInfo.cpp:229` 加 BF16 |
| 2 | SuperScalarModel (emulator) | gfrun `TAIL_OCP_FP4` | `AccumulateBlockInfo.cpp:607` TROWMAX/U16 | `AccumulateBlockInfo.cpp:525` 对齐 TCOLMAX(:539) |
| 3 | linx-toolchain-build (LinxV5) | 编译 `NONTAIL_OCP_FP4`，`-O1/-O2` | `B.IOT ->u<>` unknown operand | `LinxV5AsmPrinter.cpp:176` + 优化 pass 保留 `%Z` 立即数 |
| 4 | linx-toolchain-build (LinxV5) | 任一含 `layout_type_to_str` 的 TU，`-O0` | `LinxV5InstrInfo.cpp:670` Can't load register from stack slot | `loadRegFromStackSlot:665` 白名单加 `mixedgprnora` |
| 5 | emulator ↔ LinxV5（交互） | gfrun 到达非内联、跨函数收发 tile 的调用链 | `AccumulateBlockInfo.cpp:60` S64 块 dtype 不匹配 | `ValidateLocalTlsu` 按字节宽度匹配，或后端按源 tile dtype 发块 |

---

## 缺陷 1：TABS 作用于 BF16 被 emulator 拒绝

- **缺陷所在仓**：`SuperScalarModel`（emulator）
- **触发条件**：kernel 对 **BF16** tile 执行 `TABS`。入口 `dynamic_mx_quant_common.hpp:702`
  `compute_cublas_scale_tail` 中 `TABS(abs_x, x_in)`，`x_in` 为 BF16。
- **复现命令**：
  ```bash
  make TESTCASE=dynamic_mx_quant TYPE=TAIL_CUBLAS_FP8 diss
  ```

**报错信息**（gfrun）：
```
SuperScalarModel/emulator/engine/AccumulateBlockInfo.cpp:393
ASSERT(IsBasicUnaryTeplDataType(...) &&
       "TEPL opcode/data-type tuple is not defined by PTO ISA v0.2")
```

**原因分析**：emulator 白名单 `IsBasicUnaryTeplDataType`
（`AccumulateBlockInfo.cpp:229-230`）把 TABS 限为 `FP16 || FP32`，不含 BF16：
```cpp
case TileOp::TABS:
    return dataType == DataType::FP16 || dataType == DataType::FP32;
```
而规范 ASL 中 TABS 的 legality handler `TileOperandsLegal_ExecuteTileUnary`
（`pto-spec: asl/tile/model/legality/operand-schema.asl:20`）**不含任何 dtype 白名单**，仅要求
`TileShapeAndTypeMatch(dst,src)`；BF16 是合法 tile dtype
（`pto-spec: asl/tile/model/legality/dtype-layout.asl`）。故 BF16-TABS 为 spec-legal，
emulator 白名单过窄。同函数 `TNEG`（:233-236）已允许 BF16，可见白名单本身支持 BF16 表达。

**修复建议**：`AccumulateBlockInfo.cpp:229` TABS 分支加入 `DataType::BF16`（对齐 `TNEG:233-236`）。

---

## 缺陷 2：TROWMAX 作用于 UINT16 被 emulator 拒绝（与 TCOLMAX 不对称）

- **缺陷所在仓**：`SuperScalarModel`（emulator）
- **触发条件**：kernel 对 **UINT16** tile 执行 `TROWMAX`。入口 `dynamic_mx_quant_common.hpp:639`
  `compute_ocp_scale_tail_boxed_pw` 中 `TROWMAX(max_exp, exp_bits)`，`exp_bits` 为 UINT16 指数位。
- **复现命令**：
  ```bash
  make TESTCASE=dynamic_mx_quant TYPE=TAIL_OCP_FP4 diss
  ```

**报错信息**（gfrun）：
```
SuperScalarModel/emulator/engine/AccumulateBlockInfo.cpp:607
ASSERT(IsReduceAndExpandTeplDataType(...) &&
       "reduce/expand TEPL tuple is not defined by PTO ISA v0.58")
```

**原因分析**：白名单 `IsReduceAndExpandTeplDataType`（`AccumulateBlockInfo.cpp:525-528`）给
**TROWMAX** 的 dtype 集为 `FP16||FP32||INT32`，不含 U16/BF16；而同函数 **TCOLMAX**
（:539-547）含 `INT8/UINT8/INT16/UINT16/INT32/UINT32/BF16`：
```cpp
case TileOp::TROWMAX:  return FP16 || FP32 || INT32;                        // 无 U16/BF16
case TileOp::TCOLMAX:  return FP16||FP32||INT8||UINT8||INT16||UINT16||INT32||UINT32||BF16;
```
但规范 ASL 中 TROWMAX 与 TCOLMAX **共用同一** legality handler
`TileOperandsLegal_ExecuteTileReduction`
（`pto-spec: asl/tile/reduce-and-expand/row-reduction/TROWMAX.asl` 与
`.../column-reduction/TCOLMAX.asl` 的 `legality_handler` 字段一致），该 handler
（`operand-schema.asl:70`）不含 dtype 限制（axis 只影响 destination shape 检查），U16 是合法
tile dtype。故 U16-TROWMAX 为 spec-legal；emulator 给 TROWMAX 配了比 TCOLMAX 窄、且与共用
handler 不符的 dtype 集。

**旁证**：not-tail 变体用 TCOLMAX（`dynamic_mx_quant_common.hpp:800/832`），落在已放行集合内，
故 nontail scale pass 不触发此断言——差异纯在 emulator 的 TROWMAX/TCOLMAX 不对称。

**修复建议**：`AccumulateBlockInfo.cpp:525` TROWMAX 分支 dtype 集扩到与 TCOLMAX（:539-547）一致
（或按共用 handler 语义去掉 dtype 白名单）。

---

## 缺陷 3：`B.IOT ... ->u<>` unknown operand（-O1/-O2 编译失败）

- **缺陷所在仓**：`linx-toolchain-build`（`llvm-project` LinxV5 后端）
- **触发条件**：编译 `nontail_ocp_fp4.cpp`（Axis=32/Post=64/BS=32，fp4 输出 tile `[32,32]`
  RowMajor NoneBox）于 **`-O1` 或 `-O2`**。
- **复现命令**：
  ```bash
  make TESTCASE=dynamic_mx_quant TYPE=NONTAIL_OCP_FP4 diss     # 默认 -O2
  ```

**报错信息**（编译期，报在 vendor 头内联汇编）：
```
tileop-api/jcore/template_asm.hpp:115   (TCVT_T)        "B.IOT %3, mask=15, last, ->%0<%Z4>\n"
tileop-api/jcore/template_asm.hpp:5106  (TCOLEXPANDMUL) "B.IOT %5, %6, mask=15, last, ->%0<%Z7>\n"
→ instantiated:  B.IOT u#1, mask=15, last, ->u<>          // box 为空 <>
error: unknown operand
```

**原因分析**：`%Z` 是 LinxV5 后端自定义操作数修饰符，打印 B.IOT 的 TileSize 文本。打印器
`LinxV5AsmPrinter.cpp:176-183`：
```cpp
if (ExtraCode[0]=='Z' ...) {
    if (!MO.isImm()) return true;              // 返回 true = "unknown operand"
    static const char* TileSizes[] = {"0B","128B",...,"8KB"};
    if ((unsigned)MO.getImm() < 8) OS << TileSizes[MO.getImm()];
    return false;
}
```
当 `%Z` 对应操作数在 MI 层不是立即数（`!MO.isImm()`）时 `return true` → clang 报
"unknown operand"，且因提前返回、box 未写入 → 空 `<>`。

证据链，指向**优化 pass** 而非 kernel：
1. C++ 层该 fp4 `[32,32]` 输出 tile 的 `TilesizeCode = 4`（=1KB，合法枚举），与 fp8 输出
   tile 取值相同（static_assert 实测 fp4→4、fp8→4）；同套 TCVT_T/TCOLEXPANDMUL 模板对 fp8
   输出、tail-fp4 输出均编译干净。
2. 最小复现（单独对 `Tile<Vec,__fp4_e2m1x2,32,32,RowMajor>` 做 TCVT）operand 保持立即数 →
   打印 `<1KB>`，编译干净；仅在整 kernel 上下文失败。
3. 失败随优化等级出现：同一 `nontail_ocp_fp4.cpp` — `-O0`→**0** 处、`-O1`→**6** 处、
   `-O2`→**10** 处 "unknown operand"。

→ `-O1/-O2` 的某个 LinxV5 优化 pass 把经 INLINEASM `"i"` 约束传入的 `%Z` 立即数降级为
非立即数（vreg），触发打印器 `!MO.isImm()` 分支。源码合法、仅 -O 变化即触发，是后端优化
pass miscompile 的签名。

**修复建议**：保证经 INLINEASM 传入、`"i"` 约束的 `%Z` 操作数在优化后仍以立即数抵达
AsmPrinter（或相关 pass 对 INLINEASM imm 操作数做保守处理）。

---

## 缺陷 4：`-O0` 溢出/重载寄存器类不对称，`layout_type_to_str` 崩溃

- **缺陷所在仓**：`linx-toolchain-build`（`llvm-project` LinxV5 后端）
- **触发条件**：以 **`-O0`** 编译任一含 `pto::layout_type_to_str`
  （`two-level-arch/include/common/layout.hpp:59`，返回字符串字面量的 helper，各 kernel 都链入）
  的翻译单元。与具体 kernel 无关。
- **复现命令**（任一 kernel 降 -O0 即可，例）：
  ```bash
  make TESTCASE=dynamic_mx_quant TYPE=TAIL_OCP_FP4 EXTRA_CXXFLAGS=-O0 diss
  ```

**报错信息**（编译期）：
```
llvm_unreachable("Can't load this register from stack slot")
llvm-project/llvm/lib/Target/LinxV5/LinxV5InstrInfo.cpp:670
```

**原因分析**：物化字符串字面量地址的 `PseudoADDTPC_HI` 产出寄存器类 **`mixedgprnora`**
（MIR，`-print-before=regallocfast`，`layout_type_to_str` 的 `sw.bb`）：
```
%8:mixedgprnora = PseudoADDTPC_HI <mcsymbol>, target-flags(linx-tpcrel-hi) @.str
%9:mixedgpr    = ADDI killed %8, target-flags(linx-tpcrel-lo) ...
SDI killed %9, %stack.0.retval, 0
```
store 与 load 处理不对称：
- `storeRegToStackSlot`（`LinxV5InstrInfo.cpp:607-648`）无寄存器类白名单——非 `Tile_ABS` 一律
  `SDI` 无条件溢出，**接受** `mixedgprnora`。
- `loadRegFromStackSlot`（`:650-670`）有白名单 `{GR, LTR, LUR, Tile_ABS, SIMTCGV}`，
  `hasSubClassEq(mixedgprnora)==false` → 落到 `:670` `llvm_unreachable`。

`-O0` 用 Fast RegAlloc，激进溢出/重载短活跃期虚寄存器，故命中 load 路径；`-O2` 的 Greedy 把
`mixedgprnora` 保留在寄存器、不经栈往返，故不触发（但 `-O2` 会命中缺陷 3）。

**修复建议**：`loadRegFromStackSlot:665` 白名单加入 `mixedgprnora`（或 `PseudoADDTPC_HI` 结果
对应的正确寄存器类），与 `storeRegToStackSlot` 对齐。

---

## 缺陷 5：非内联 helper 的 tile 参数经 `TSTORE/TLOAD, S64` 栈传参，被 emulator `ValidateLocalTlsu` 拒绝

- **缺陷所在仓**：`SuperScalarModel`（emulator）↔ `linx-toolchain-build`（LinxV5 后端）**交互缺陷**，
  任一侧修复即可。
- **触发条件**：gfrun 执行到「未内联、跨函数收/发 tile」的调用链——本 kernel 为 cuBLAS scale 路径
  `compute_cublas_scale_{tail,not_tail}` → `compute_cublas_core`。默认 bf16 构建下被**缺陷 1**
  （`TABS(bf16)`）前置掩盖、走不到；经 fp16/fp32 输入越过缺陷 1 后**首个命中**。
- **复现命令**：
  ```bash
  make TESTCASE=dynamic_mx_quant TYPE=TAIL_CUBLAS_FP8_FP16 diss   # fp16 越过缺陷1 后命中
  ```

**报错信息**（gfrun）：
```
SuperScalarModel/emulator/engine/AccumulateBlockInfo.cpp:60  ValidateLocalTlsu
ASSERT(... IsCompatibleDataTile(inst->srcs[1], block->dataType, ...)
       && "Local TSTORE requires one compatible source Tile")
```

**原因分析**：LinxV5 后端传 tile 实参时，把 tile 当 **64-bit 通用字节块**（`TSTORE/TLOAD, S64`）在栈上
memcpy（caller-save spill）——源码从未写过 S64 store，反汇编实测源码仅 U8 scale + e4m3 output 两种
TSTORE，diss 却出现 **5 条 `TSTORE, S64`**。而 emulator `ValidateLocalTlsu`（`AccumulateBlockInfo.cpp:60`）
要求 Local TSTORE 的**源 tile dtype 精确等于 store block 的 dtype**（`IsCompatibleDataTile:270`
`source->tileInfo->dataType == dataType`）。emulator 诊断打印：`block.dtype=INT64`（S64，`elemB=8`）、
维度解成 `1/1/1`，而源 `src.dtype=FP32`、`[validRow=8, col=32]`、`size=1024` → S64 块 ≠ 源 tile FP32，
断言必失败。

tile 参数的栈往返本质是**等宽字节 memcpy**，按字节宽度匹配即可；后端发 S64 通用块、emulator 又按 dtype
精确匹配，两边口径不一致——**交互缺陷，非 kernel 逻辑**。

**修复建议**（任一侧即可）：
- **emulator**：`ValidateLocalTlsu` 对编译器合成的通用块搬运放宽为按字节宽度匹配，不要求 block dtype
  逐一等于源 tile dtype。
- **LinxV5 后端**：传 tile 实参时按源 tile 真实 dtype 发 store/load block，而非统一 S64。

---

## 前提说明（针对缺陷 1/2）

结论以规范 ASL 为权威：`TileOperandsLegal_ExecuteTileUnary` /
`TileOperandsLegal_ExecuteTileReduction` 均无 dtype 白名单，BF16/U16 为合法 tile dtype。
emulator 白名单注释引用的是生成的 `.md` 文档；若最终以硬件实际支持为准、且硬件确不支持
BF16-TABS / U16-TROWMAX，则结论回到「合法但不受支持」，仍需在 emulator 或规范侧明确对齐。
