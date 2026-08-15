# [Issue] dynamic_mx_quant（release_ver0812）：2 处需 ISA 支持确认的 dtype×op 组合

在 `dynamic_mx_quant` kernel 的 gfrun 过程中，命中 2 处 emulator 白名单拒绝：**BF16-TABS**
与 **UINT16-TROWMAX**。二者按规范 ASL 均为 spec-legal（对应 legality handler 无 dtype 白名单），
但 emulator 白名单过窄将其拒绝。

**这两条与另一文档（`ISSUE_rel0812_toolchain_defects.md` 的 toolchain/emulator 缺陷）区别在于**：
它们不是单纯的实现 bug，而是**触及「该 dtype×op 组合硬件/ISA 是否支持」**——若最终以硬件实际支持
为准、且硬件确不支持 BF16-TABS / U16-TROWMAX，则结论回到「合法但不受支持」，需在 **ISA 规范侧**
明确对齐（而非仅改 emulator）。故单列本文，需 ISA owner 确认支持面。

## 环境

复现**跨两个仓库**：kernel 在 **SuperNPUBench** 编译，产出的 elf 由 **SuperScalarModel** 的
`bin/gfrun` 执行。两仓缺一不可。

| 项 | 值 |
|---|---|
| 编译仓 | SuperNPUBench `benchmark/one-level-arch`，分支 `feat/dmxq-rel0812`，commit `fb96b28` |
| 执行仓 | SuperScalarModel（`bin/gfrun` 执行 elf），分支 `feat/pto-v058-adaptation`，commit `319294f` |
| 工具链 | `linx_blockisa_llvm_musl`，clang-15，target `linx64v5-unknown-linux-musl`；llvm-project commit `eb64de8` + Linx-TileOP-API commit `72f8255` |
| 复现根目录（编译） | `benchmark/one-level-arch/test/kernel/quant/dynamic_mx_quant` |
| 环境变量 | `export COMPILER_DIR=.../linx_blockisa_llvm_musl/bin`；`export LINX_SYSROOT=.../sysroot/usr` |

> 上述三仓 commit 与 `SuperNPUBench/README.md` release_ver0812「验证仓库版本」一致。

## 缺陷一览

| # | 缺陷所在仓 | 触发条件 | 报错 | 修复位置 |
|---|---|---|---|---|
| 1 | SuperScalarModel (emulator) / ISA | gfrun `TAIL_CUBLAS_FP8`（bf16 输入） | `AccumulateBlockInfo.cpp:393` TABS/BF16 | `AccumulateBlockInfo.cpp:229` 加 BF16（或 ISA 侧确认支持面） |
| 2 | SuperScalarModel (emulator) / ISA | gfrun `TAIL_OCP_FP4` | `AccumulateBlockInfo.cpp:607` TROWMAX/U16 | `AccumulateBlockInfo.cpp:525` 对齐 TCOLMAX(:539)（或 ISA 侧确认支持面） |

## 复现方式

两条均为**运行时缺陷**：`make ... diss` 会**编译成功**，须再在**仓库根**用 gfrun 执行 elf 才触发断言。

- **编译**：在复现根目录执行 `make TESTCASE=dynamic_mx_quant TYPE=<TYPE> diss`。
  `diss` 只做「编译 + 反汇编（`llvm-objdump -dl`）」，**不执行 gfrun**；产物 elf 位于
  `output/kernel/quant/dynamic_mx_quant/elf/kernel_quant_dynamic_mx_quant/dynamic_mx_quant_<variant>.elf`。
- **gfrun**：
  ```bash
  SuperScalarModel/bin/gfrun -t 1 -f "$PWD/SuperNPUBench/benchmark/one-level-arch/output/kernel/quant/dynamic_mx_quant/elf/kernel_quant_dynamic_mx_quant/dynamic_mx_quant_<variant>.elf"
  ```
  每个缺陷的 `<TYPE>` 与 `<variant>` 见其小节的「复现命令」。

---

## 缺陷 1：TABS 作用于 BF16 被 emulator 拒绝

- **缺陷所在仓**：`SuperScalarModel`（emulator）；**需 ISA 侧确认硬件是否支持 BF16-TABS**
- **触发条件**：kernel 对 **BF16** tile 执行 `TABS`。入口 `dynamic_mx_quant_common.hpp:702`
  `compute_cublas_scale_tail` 中 `TABS(abs_x, x_in)`，`x_in` 为 BF16。
- **复现命令**（运行时：编译 + gfrun 两步）：
  ```bash
  # ① 编译（复现根目录；diss 只编译+反汇编，不跑 gfrun）
  make TESTCASE=dynamic_mx_quant TYPE=TAIL_CUBLAS_FP8 diss
  # ② gfrun 执行（仓库根）
  SuperScalarModel/bin/gfrun -t 1 -f "$PWD/SuperNPUBench/benchmark/one-level-arch/output/kernel/quant/dynamic_mx_quant/elf/kernel_quant_dynamic_mx_quant/dynamic_mx_quant_tail_cublas_fp8.elf"
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
**若最终以硬件实际支持为准、且硬件确不支持 BF16-TABS**，则需在 ISA 规范侧明确该组合不受支持、并由
kernel 侧改走替代路径（如先 `TCVT bf16→fp32` 再 `TABS`）。

---

## 缺陷 2：TROWMAX 作用于 UINT16 被 emulator 拒绝（与 TCOLMAX 不对称）

- **缺陷所在仓**：`SuperScalarModel`（emulator）；**需 ISA 侧确认硬件是否支持 U16-TROWMAX**
- **触发条件**：kernel 对 **UINT16** tile 执行 `TROWMAX`。入口 `dynamic_mx_quant_common.hpp:639`
  `compute_ocp_scale_tail_boxed_pw` 中 `TROWMAX(max_exp, exp_bits)`，`exp_bits` 为 UINT16 指数位。
- **复现命令**（运行时：编译 + gfrun 两步）：
  ```bash
  # ① 编译（复现根目录）
  make TESTCASE=dynamic_mx_quant TYPE=TAIL_OCP_FP4 diss
  # ② gfrun 执行（仓库根）
  SuperScalarModel/bin/gfrun -t 1 -f "$PWD/SuperNPUBench/benchmark/one-level-arch/output/kernel/quant/dynamic_mx_quant/elf/kernel_quant_dynamic_mx_quant/dynamic_mx_quant_tail_ocp_fp4.elf"
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
（或按共用 handler 语义去掉 dtype 白名单）。**若最终以硬件实际支持为准、且硬件确不支持
U16-TROWMAX**，则需在 ISA 规范侧明确 TROWMAX/TCOLMAX 的支持面差异，并由 kernel 侧改走 TCOLMAX
或其它替代归约路径。

---

## 前提说明（针对缺陷 1/2）

结论以规范 ASL 为权威：`TileOperandsLegal_ExecuteTileUnary` /
`TileOperandsLegal_ExecuteTileReduction` 均无 dtype 白名单，BF16/U16 为合法 tile dtype。
emulator 白名单注释引用的是生成的 `.md` 文档；若最终以硬件实际支持为准、且硬件确不支持
BF16-TABS / U16-TROWMAX，则结论回到「合法但不受支持」，仍需在 emulator 或规范侧明确对齐。
**这正是本文与 toolchain/emulator 纯实现缺陷分开单列的原因**——修复动作可能落在 ISA 规范侧，
而非仅 emulator 白名单。
