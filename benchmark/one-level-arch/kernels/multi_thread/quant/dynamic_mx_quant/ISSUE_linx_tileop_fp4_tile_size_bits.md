# [Linx-TileOP-API] fp4(e2m1x2) tile 的 size 位宽用 8-bit（打包容器）算，与 pto-spec / 模型的 4-bit(每元素) 口径不一致，导致宽→打包fp4 的 TCVT 在运行期崩形状契约

> **2026-09-05 更新：`804eb03`（比本 issue 的 `a795b97` 新）仍存在，已本地修复待上游。** 采用下文「建议修复」option (b)：`pto_tile.hpp` 的 `kBytes`/`StorageBytes`（非 Cube 分支）改用已有的 `CubeElementBits`（e2m1x2/e1m2x2/hif4x2/int4x2/uint4x2 → 4，其余 = `type_traits::bits`），`type_traits::bits` 保持 8（storage/reinterpret/stride 不变）。本地补丁分支 `local/fp4-tile-size-fix` `78045d6`。配合模型侧 4 处修复后 `tail_ocp_fp4` 512×256 4-PE 逐字节 pass。详见 RECORD 问题26。

## 环境

- 仓库：`github.com/LinxISA/Linx-TileOP-API` @ `a795b97`
- 配套模型：SuperScalarModel `a5dca25a` + `31f7a8f`（`ElementBits(FP4)=4` 已并入）
- 基线：ops-20260823
- 复现算子：`SuperNPUBench` `dynamic_mx_quant` 的 `TAIL_OCP_FP4` / `NONTAIL_OCP_FP4`
  （data pass 末尾 `TCVT(oq, xf)`，fp32 → 打包 fp4）

## TL;DR

`type_traits<__fp4_e2m1x2>::bits = 8`（把 e2m1x2 当 **8-bit 打包容器**）使 `Cols=64` 的
fp4 tile 算出 **512B**，是「64 个逻辑 fp4 元素 = 256B」的 **2 倍**。这个翻倍的 size 喂进模型
侧按 **4-bit/元素** 反推 row 的公式，行数翻倍（8→16），在 TCVT 运行期形状契约上崩。

**修复须在工具链头把 fp4 tile 的 size 位宽从 8 改成按每逻辑元素 4-bit 计**（256B）。模型侧
`ElementBits(FP4)=4` 已就位（31f7a8f），无需再动。

## 现象（编译期已通过、运行期崩）

kernel（`dynamic_mx_quant_tail_ocp_fp4.hpp:106`）声明输出 tile：

```cpp
using tile_o = Tile<Location::Vec, OutT /*=__fp4_e2m1x2*/, TileM, PW,
                    BLayout::RowMajor, TileM, BlockSize>;   // Cols = PW = 64（逻辑元素列，与源 tile_f 同）
```

1. **编译期 TileLogicalShapeMatch 已通过**（`template_asm.hpp:115` static_assert 不再触发）。
   `tile_o` 与源 `tile_f`（fp32, Cols=64）Rows/Cols 全等。实测 `TAIL_OCP_FP4` 编译 **EXIT=0**。

2. **运行期崩 `SuperScalarModel/isa/Block.cpp:1155` ValidateOperandContract**（TCVT 要求
   src/dst 的 row/col/validRow/validCol/layout 全等），崩在 **physical ROW**：
   `src fp32 row=8`，`dst fp4 row=16`。

## 根因（bit 口径错位）

模型侧 row 推导（`Block.cpp:1558-1560`）：

```
dst.row = dst.size * 8 / (dstPhysicalCol * elementBits)
```

代入 `tile_o`（fp4, Rows=8, Cols=64）：

| 量 | 值 | 来源 |
|---|---|---|
| `dst.size` | **512B** | 工具链头 `pto_tile.hpp:643` `kBytes = Rows*Cols*type_traits<DType>::bits/8 = 8*64*8/8` |
| `type_traits<__fp4_e2m1x2>::bits` | **8** | 工具链头 `type.hpp:62`（把 e2m1x2 当 8-bit 打包容器，storage=uint8_t） |
| `dstPhysicalCol` | 64 | tile 声明 Cols |
| `elementBits` | **4** | 模型 `DataType.h` `ElementBits(FP4)=4`（每 fp4 元素 4-bit） |

→ `row = 512*8 / (64*4) = 16 ≠ src 8`，崩。

**矛盾点**：`type.hpp` 里 e2m1x2 用 8-bit，而 fp6 等窄类型用真实位宽（`__fp6_e3m2` = 6-bit，
`type.hpp:60`）。fp4x2 的 8-bit 是「打包两个 4-bit 的容器位宽」，用于 storage 层没错；但当它被
`kBytes = Rows*Cols*bits/8` 直接乘进 tile size 时，**把 `Cols=64` 当成 64 个 8-bit 打包单元
（=128 个 fp4，512B），而不是 64 个 fp4 逻辑元素（256B）**——size 翻倍。

模型侧 `31f7a8f` 已改用 pto-spec 的 `TileElementBits(FP4)=4` 反推 row，期望的自洽输入是
真实打包的 **256B**（`256*8/(64*4)=8=src`）；但当前工具链头无法由任一 kernel 声明产出 256B
（`size` 恒 = `Rows*Cols*8/8`）。**未对齐的那半在工具链头。**

## 复现

```bash
# 工具链
export COMPILER_DIR=.../linx-toolchain-build/output/linx_blockisa_llvm_musl/bin
export LINX_SYSROOT=.../linx-toolchain-build/output/linx_blockisa_llvm_musl/sysroot/usr

cd SuperNPUBench/benchmark/one-level-arch/test/kernel/multi_thread/quant/dynamic_mx_quant
make TESTCASE=dynamic_mx_quant TYPE=TAIL_OCP_FP4          # 编译 EXIT=0
# gfrun 该 ELF → Block.cpp:1155 ValidateOperandContract 崩，src row=8 vs dst row=16
```

## 建议修复（双侧统一到 pto-spec：TileElementBits(FP4)=4）

- **工具链头（本 issue 主体，待修）**：让 fp4(e2m1x2 / e1m2x2 / hif4x2) tile 的 **size 位宽按
  每逻辑元素 4-bit** 计，使 `Tile<fp4, Rows=8, Cols=64>` → `kBytes = 8*64*4/8 = 256B`。两种落法：
  - 在 `pto_tile.hpp` 的 `kBytes` 公式对 4-bit 打包类型特判（除 2）；或
  - 为 fp4x2 引入「tile-size 位宽」概念（4）与「storage 容器位宽」（8）分离，`kBytes` 用前者。
  - 注意别破坏 storage 层（`sizeof(__fp4_e2m1x2)=1B`）与 32B 列对齐检查（`pto_tile.hpp:722`）。
- **模型侧**：`ElementBits(FP4)=4` 已就位（`31f7a8f`），无需再改。
- 对齐后：`size=256B → row = 256*8/(64*4) = 8 = src` → 契约通过。

## 不该采用的方案

- **kernel 侧把 Cols 写成 `PW/2=32`（打包字节数）**：会违反 pto-spec「columns = 逻辑元素」语义，
  并触发编译期 `template_asm.hpp:115` static_assert（32 ≠ 源 64）。要「除 2」的是**字节 size**，
  属工具链头职责，kernel 声明层表达不了。
- **删 `Block.cpp` 的 physical row/col 校验只留 valid**：与 pto-spec `TileOperandsLegal_TCVT` /
  `TCVT()` 契约冲突，是掩盖而非对齐口径。

## 影响面

所有用 TCVT 把宽类型（fp32/bf16/half）转成打包 fp4（`__fp4_e2m1x2` 等）的 kernel data pass，
即 `dynamic_mx_quant` 的 tail/nontail OCP-FP4 全部路径。scale pass（e8m0）不受影响。
