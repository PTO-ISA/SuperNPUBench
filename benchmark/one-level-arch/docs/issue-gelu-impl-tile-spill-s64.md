# [gelu] 跨函数调用传递 tile 时以 S64/NORM 溢出，丢失 BF16 元素类型，导致 gelu_impl 入口 TCVT(BF16→FP32) 出错（编译器缺陷）

> 修订说明：本文档此前将该问题定性为“`gelu_impl` 未 inline 导致 tile 溢出到栈的性能开销”，并断言 `S64` 是“只搬原始比特、不做格式转换、类型安全”的存储格式。**该结论是错误的。** 经对 `-latest` 工具链（`origin/dev-llvm15_56`）所产 ELF 反汇编核实，这是一个**编译器正确性缺陷**：tile 作为函数参数跨调用边界传递时，编译器以 `S64` + `B.DATR NORM.normal` 溢出，**没有保留 tile 的 BF16 元素类型**；callee 入口重新加载得到的 tile 处于 `NORM`（原始/64 位标量）表示而非 BF16，紧随其后的 `BSTART.TEPL TCVT, BF16`（BF16→FP32）拿到的是类型不符的数据，转换出错。下面为修正后的分析。

## 摘要

gelu 算子数据类型为 `__bf16`。`gelu()` 通过函数调用把 tile（`inTile`/`outTile`）按引用传给未被 inline 的 `gelu_impl()`。编译器在调用边界为 tile 寄存器生成溢出代码：

- 调用前 `BSTART.TLSU TSTORE, S64` + `B.DATR NORM.normal` —— 把 **BF16** tile 以 `S64`（64-bit/PE 标量）/`NORM` 表示存入栈；
- callee 入口 `BSTART.TLSU TLOAD, S64` + `B.DATR NORM.normal` —— 从栈恢复，但恢复出的 tile 仍是 `NORM` 表示，**不是 BF16**；
- 紧接着 `BSTART.TEPL TCVT, BF16`（BF16→FP32）—— 源类型声明为 BF16，实际输入 tile 却是 `NORM`，类型不符 → 转换报错/结果错误。

**这不是性能问题，而是编译器在“tile 跨调用参数溢出”lowering 路径上的正确性缺陷**：溢出/恢复未保留 tile 的原生元素表示（BF16）。给 `gelu_impl` 加 `inline` 只是消除了调用边界、绕开了这条有缺陷的溢出路径（workaround），并不修复编译器。

---

## 环境

| 项 | 值 |
|---|---|
| 工具链 worktree | `linx-toolchain-build-latest`（已构建并通过 gelu 端到端验证） |
| llvm-project | `origin/dev-llvm15_56` @ `86959776b`（`src/llvm-project`，detached） |
| Linx-TileOP-API | `origin/linx` @ `6a43784`（`src/Linx-TileOP-API`，detached） |
| clang | `15.0.4 (linx64v5-musl-local 86959776bd1fb22dcc8e73b57ec2276c65d44f38)`，Target `linx64v5-unknown-linux-musl` |
| 算子 | `element_wise/gelu` |
| 配置 | `DTYPE=__bf16 tMs=2048 gMs=24*8*1024 SHAPE_NAME=24_8_1024 Approximate=false` |
| 编译命令 | `make TESTCASE=gelu DTYPE=__bf16 tMs=2048 gMs=24*8*1024 SHAPE_NAME=24_8_1024 Approximate=false diss` |
| COMPILER_DIR | `linx-toolchain-build-latest/output/linx_blockisa_llvm_musl/bin` |
| ELF | `output/kernel/element_wise/gelu/elf/kernel_element_wise_gelu_gelu_Approximate_false_DType__bf16_tM2048_SHAPE24_8_1024.elf` |
| 反汇编 | 同上 `.elf.diss` |

> 该缺陷同样复现于 `temp/shared-tload-integration-20260811`（`eb64de8af` / `72f8255`）分支的构建产物，pattern 完全一致，说明在稳定分支与临时集成分支上均存在。

---

## 涉及的源文件

### 1. 测试入口 `test/kernel/element_wise/gelu/src/gelu.cpp`

```cpp
#include <common/pto_tileop.hpp>
#include <cstdint>
#include <cstdio>
#include "fileop.h"
#include "element_wise/gelu_pto.hpp"

#ifndef DTYPE
#define DTYPE int32_t
#endif
#ifndef tMs
#define tMs 512
#endif
#ifndef gMs
#define gMs (24 * 512 * 1024)
#endif
#ifndef Approximate
#define Approximate false
#endif

int main() {
    using dtype = DTYPE;
    dtype  input_buf[gMs  + 11];
    dtype  output_buf[gMs + 17];
    dtype* input  = input_buf  + 11;
    dtype* output = output_buf + 17;
    gelu<dtype, gMs, tMs>(input, output, Approximate);
}
```

### 2. Kernel 实现 `kernels/element_wise/gelu_pto.hpp`

源码注释（第 9 行）即把数据流描述为：
`TLOAD (half) -> TCVT(fp16→fp32) -> tile 指令链计算 GELU -> TCVT(fp32→fp16) -> TSTORE (half)`
（源码以“fp16/half”泛指半精度；本配置 `DTYPE=__bf16`，即 BF16。）

#### gelu 主函数（line 229-277）

```cpp
template<typename dtype, int gM, int tM>
void gelu(
    dtype *in_ptr,
    dtype *out_ptr,
    bool approximate = false
    ) {
    using gm_shape       = global_tensor<dtype, RowMajor<1, gM>>;
    using tile_shapeData = Tile<Location::Vec, dtype, 1, tM, BLayout::RowMajor>;
    using tile_shapeFP32 = Tile<Location::Vec, float, 1, tM, BLayout::RowMajor>;
    // ...
    itIn  gIIter(in_ptr);
    itOut gOIter(out_ptr);

    tile_shapeData inTile, outTile;
    tile_shapeFP32 tmpCvt;

    for (int i = 0; i < Mb; ++i) {
        auto gI = gIIter(0, i);
        auto gO = gOIter(0, i);
        TLOAD(inTile, gI);                                    // GM -> tile (BF16)
        gelu_impl<tile_shapeData, tile_shapeFP32>(inTile, outTile, tmpCvt);  // ← 非 inline 调用，tile 跨边界传递
        TSTORE(gO, outTile);                                  // tile -> GM
    }
}
```

#### gelu_impl 函数（line 150-223）— **没有 inline 标注**

```cpp
template<typename tile_shapeData, typename tile_shapeFP32>
void gelu_impl(                                              // ← 无 inline / always_inline
    tile_shapeData  &inTile,
    tile_shapeData  &outTile,
    tile_shapeFP32  &tmpCvt
) {
    using fp_t = typename tile_shapeFP32::DType;   // float

    tile_shapeFP32 xTile;        // x = (float)input
    // ...

    // Step 1: bf16 -> fp32
    TCVT(xTile, inTile);         // ← 入口处第一件事就是把 inTile(BF16) 转成 fp32

    // Step 2..6: clamp / 多项式 Horner / exp / recip ...
    // Step 7: fp32 -> bf16
    TCVT(outTile, scratchTile);
}
```

**注意**：`gelu_impl` 是 template 但未加 inline，函数体较大（6 个 tile 变量 + 多步计算），编译器选择不 inline，编译为独立函数（反汇编 `0x114ba` 有独立符号）。

### 3. Makefile `test/kernel/element_wise/gelu/Makefile`

```makefile
ifeq ($(TESTCASE), gelu)
DEFINES += -DDTYPE=$(DTYPE) -DtMs=$(tMs) -DgMs=$(gMs) -DApproximate=$(Approximate)
TARGET = $(ELF_HEAD)_$(TESTCASE)_Approximate_$(Approximate)_DType$(DTYPE)_tM$(tMs)_SHAPE$(SHAPE_NAME).elf
endif
SRC_FILE +=  $(TEST_ROOT)/$(CASE_SRC_DIR)/$(TESTCASE).cpp
include ../../../common/Makefile.common
```

---

## 复现方法

### 前置条件

工具链已构建（`llvm-project origin/dev-llvm15_56 @ 86959776b` + `Linx-TileOP-API origin/linx @ 6a43784`），安装到 `linx-toolchain-build-latest/output/linx_blockisa_llvm_musl`。

### 编译命令

```bash
export PATH=/opt/homebrew/opt/make/libexec/gnubin:$PATH
export COMPILER_DIR=/Users/blacktraker/Programming/gitproj/DV4/linx-toolchain-build-latest/output/linx_blockisa_llvm_musl/bin

cd /Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/test/kernel/element_wise/gelu

make TESTCASE=gelu DTYPE=__bf16 tMs=2048 gMs=24*8*1024 \
    SHAPE_NAME=24_8_1024 Approximate=false diss
```

> `gMs=24*8*1024` 中的 `*` 会被 zsh 当通配符报错，须用 `bash` 执行 `compile.all` 或加引号。

### 验证命令

```bash
DISS=output/kernel/element_wise/gelu/elf/kernel_element_wise_gelu_gelu_Approximate_false_DType__bf16_tM2048_SHAPE24_8_1024.elf.diss

# gelu_impl 是独立函数
grep "gelu_impl" $DISS

# 关键：调用边界的 tile 溢出用 S64 + NORM.normal（而非 BF16）
grep -nE "BSTART\.TLSU.*(TLOAD|TSTORE), S64" $DISS
grep -nE "B\.DATR\s*NORM\.normal" $DISS

# 紧跟在 S64 reload 之后的 TCVT, BF16 —— 输入已是 NORM 而非 BF16
grep -nE "BSTART\.TEPL\s*TCVT, BF16" $DISS
```

---

## 反汇编分析（基于 `-latest` 实际产物 `.diss`）

### gelu() 主函数：TLOAD(BF16) → 溢出 → 调用 → 恢复 → TSTORE(BF16)

```asm
; ---- 从 GM 加载 BF16 输入（类型正确，BF16）----
1137a: 28011181  BSTART.TLSU  TLOAD, BF16          ; GM -> inTile，元素表示 = BF16  ← 类型起点
1137e: B.DIM s3, 0, ->lb0
11382: B.DIM s4, 0, ->lb1
11386: B.DIM zero, 2048, ->lb2
1138a: B.IOT mask=1111, last, ->t<4KB>
1138e: B.IOR [a0,s5],[]

; ---- 编译器为调用 gelu_impl 溢出 inTile：用 S64 + NORM.normal（不是 BF16！）----
11392: C.BSTART.STD
1139e: addi t#1, 32, ->a0
113a2: 80111181  BSTART.TLSU  TSTORE, S64           ; ← 把 BF16 tile 以 S64 格式存入栈
113a6: 19f01023  B.DATR        NORM.normal, Null    ; ← 元素表示退化为 NORM（原始/64位标量），BF16 类型丢失
113aa: B.IOT t#1, mask=1111, last
113ae: B.IOR [a0],[]
113b2: B.DIM zero, 512, ->lb0
113b8: B.DIM zero, 512, ->lb2

; ---- 调用 gelu_impl（独立函数）----
113bc: C.BSTART.STD  DIRECT, 0x114ba <gelu_impl>
113be: c.setret 0x113f0, ->ra

; ---- 返回后从栈恢复 outTile：仍是 S64 + NORM.normal（不是 BF16）----
113f0: C.BSTART.STD
113f2: addi s6, 32, ->a0
113f6: 80011181  BSTART.TLSU  TLOAD, S64           ; ← 从栈恢复，S64
113fa: 01f01023  B.DATR        NORM.normal, Zero    ; ← 恢复出的是 NORM，不是 BF16
113fe: B.IOT mask=1111, last, ->t<4KB>
11402: B.IOR [a0],[]
11406: B.DIM zero, 512, ->lb0
1140c: B.DIM zero, 512, ->lb2
11410: 28111181  BSTART.TLSU  TSTORE, BF16          ; ← tile -> GM，却又声明 BF16（与刚 TLOAD 出的 NORM 表示不符）
```

### gelu_impl()：入口 S64/NORM reload → 紧接 TCVT, BF16（出错点）

```asm
00000000000114ba <gelu_impl>:
114ba: C.BSTART.STD
114bc: subi sp, 256, ->sp
114c0: addi a0, 32, ->a0
114c4: 80011181  BSTART.TLSU  TLOAD, S64            ; ← 从栈加载 tile 参数 inTile
114c8: 01f01023  B.DATR        NORM.normal, Zero     ; ← 仍是 NORM，不是 BF16  ★类型丢失★
114cc: B.IOT mask=1111, last, ->t<4KB>
114d0: B.IOR [a0],[]
114d4: B.DIM zero, 512, ->lb0
114da: B.DIM zero, 512, ->lb2

; ---- 真正计算的第一步：bf16 -> fp32 ----
114e6: 29b19181  BSTART.TEPL  TCVT, BF16           ; ← 源类型声明 BF16（BF16→FP32）
114ea: 18101023  B.DATR         FP32, byte0, Null    ;    目的表示 FP32
114ee: B.IOT t#1, mask=1111, last, ->t<4KB>
...   ; 后续 Horner / exp / recip 等计算
```

`TCVT, BF16` 声明源类型为 BF16，但它的输入 tile（`inTile`）刚由 `TLOAD, S64` + `B.DATR NORM.normal` 恢复，**元素表示是 NORM（64-bit/PE 原始标量）而非 BF16**。源类型与实际 tile 表示不一致 → 转换拿到的是类型不符的数据 → **报错/结果错误**。

### gelu_impl() 出口：TCVT 产出的 BF16 又被以 S64/NORM 溢出回栈（对称缺陷）

```asm
; ---- fp32 -> bf16 转换（结果此时确为 BF16 表示）----
1176c: 09b19181  BSTART.TEPL  TCVT, FP32           ; FP32 -> BF16
11770: 18501023  B.DATR         BF16, byte0, Null   ; ← 结果 outTile 现在是 BF16（正确）
11774: B.IOT t#1, mask=1111, last, ->t<4KB>
11778: B.DIM a0, 0, ->lb0
1177c: B.DIM a2, 0, ->lb1

; ---- 出口把 outTile 存回栈：又是 S64 + NORM.normal（BF16 再次丢失）----
11780: 80111181  BSTART.TLSU  TSTORE, S64           ; ← BF16 tile 以 S64 存回栈
11784: 19f01023  B.DATR        NORM.normal, Null     ; ← 退化为 NORM，BF16 类型再次丢失
11788: B.IOT t#1, mask=1111, last
1178c: B.IOR [a1],[]
1179a: C.BSTART.STD  RET
1179c: addi sp, 256, ->sp
```

### S64/NORM 溢出汇总（实测地址）

| # | 位置 | 指令 | B.DATR | 用途 | 是否保留 BF16 |
|---|---|---|---|---|---|
| 1 | gelu() `113a2` | `TSTORE, S64` | `NORM.normal, Null` | 调用前溢出 inTile(BF16) 到栈 | ✗ BF16→NORM |
| 2 | gelu() `113f6` | `TLOAD, S64`  | `NORM.normal, Zero` | 返回后恢复 outTile | ✗ 仍是 NORM |
| 3 | gelu_impl() `114c4` | `TLOAD, S64`  | `NORM.normal, Zero` | 入口加载 inTile 参数 → 喂给 `TCVT, BF16` | ✗ NORM 而非 BF16 → **TCVT 出错** |
| 4 | gelu_impl() `11780` | `TSTORE, S64` | `NORM.normal, Null` | 出口把 outTile(BF16) 存回栈 | ✗ BF16→NORM |

---

## 根因分析（修正）

### 1. 这是编译器正确性缺陷，不是性能问题

此前文档把现象（调用边界出现 `TLOAD/TSTORE S64`）归因为“未 inline 带来的额外内存访问开销”，并把 `inline` 当作修复。**这是误判**。问题的本质是：跨调用传递 tile 时，编译器生成的溢出/恢复**没有保留 tile 的原生元素类型（BF16）**，导致 callee 入口拿到一个类型不符（NORM）的 tile，下游 `TCVT, BF16` 因此出错。这是一个 correctness bug，与是否 inline 无关——只要存在“tile 跨调用边界”的内核，缺陷就会被触发。

### 2. `S64` + `NORM.normal` 不是类型透明的比特拷贝（纠正旧结论）

旧文档断言：“`S64` 只是把 tile 寄存器的原始比特搬到栈上，不做格式转换，类型安全”。**该断言与反汇编不符**：

- GM 加载使用 `BSTART.TLSU TLOAD, BF16`，tile 的元素表示是 **BF16**；
- 而调用边界的溢出/恢复一律是 `BSTART.TLSU TSTORE/TLOAD, S64` 配 `B.DATR NORM.normal`——元素表示被改写为 **NORM（64-bit/PE 原始标量）**，BF16 的元素粒度与表示随之丢失；
- callee 入口的 `TCVT, BF16` 声明源类型为 BF16，但其输入 tile 实际为 NORM 表示，二者不一致。

也就是说，`S64/NORM` 的溢出对半精度浮点（BF16，同理 FP16/__half）**不是无损往返**：它把 tile 以 64-bit 标量粒度打包存取，元素类型信息未随溢出保留，恢复出来“不再是 BF16”。这正是“**fp16 打包成 S64 存了，但加载出来不是 fp16，导致后续 TCVT(fp16→fp32) 报错**”的指令级成因。

### 3. 缺陷位于“tile 跨调用参数溢出”的 lowering 路径

`gelu_impl` 未被 inline，tile 以引用参数跨函数传递。LinxV5 后端在调用边界为 caller-saved 的 tile 寄存器生成 spill/reload 时，**选择了 `S64` + `NORM.normal`，而非按 tile 的原生元素表示（`BF16`）来存取**。于是在 callee 入口，被恢复的 tile 的元素表示与 `TCVT, BF16` 的源类型声明不一致，转换拿到类型不符的数据。

> 对比：GM↔tile 的存取（`TLOAD/TSTORE, BF16`）正确保留了 BF16 表示；只有**调用边界**的 tile 溢出退化为 S64/NORM。这进一步说明问题出在跨调用 spill 的 lowering，而非 TLSU/TCVT 指令本身。

---

## 影响

- **正确性**：gelu 主循环对每个 tile block 调用一次 `gelu_impl`，循环次数 = `gM/tMs = 24*8*1024/2048 = 96` 次。每次入口的 `TCVT, BF16` 都在 NORM 表示的 tile 上执行 → 每次 GELU 计算都拿到错误输入，结果不可信。
- **对称性**：输出路径同样被破坏——`gelu_impl` 出口把 BF16 结果以 S64/NORM 存回栈，`gelu()` 恢复后再以 `TSTORE, BF16` 写回 GM，存在同样的类型不一致。
- **性能（次要）**：96 次调用产生 96×4 = 384 次额外 TLSU（每个 tile 4KB），但这只是表象；在类型被破坏的前提下，性能讨论已无意义。

---

## 修复建议（修正）

### 正确的修复点：编译器后端

缺陷属于 **LLVM LinxV5 后端在“tile 函数参数 / 跨调用 tile spill”lowering 路径**。溢出/恢复必须保留 tile 的原生元素表示：

- 调用边界为 tile 寄存器生成的 `BSTART.TLSU TSTORE/TLOAD`，其 `B.DATR` 应携带 tile 的元素类型（本例 `BF16`），而非 `NORM.normal`；存储格式也应与元素类型匹配，而非一律 `S64`。
- 这样 callee 入口恢复出的 tile 仍是 BF16 表示，`TCVT, BF16` 的源类型与实际 tile 表示一致，转换正常。

简言之：**spill/reload 的 `B.DATR` 与存储格式要随 tile 的元素类型走（BF16/FP16/FP32…），不能固定降级为 `S64`+`NORM.normal`。**

### `inline` 只是 workaround，不是修复

给 `gelu_impl` 加 `__attribute__((always_inline)) inline`：

```cpp
template<typename tile_shapeData, typename tile_shapeFP32>
__attribute__((always_inline)) inline
void gelu_impl(
    tile_shapeData  &inTile,
    tile_shapeData  &outTile,
    tile_shapeFP32  &tmpCvt
) {
```

其作用是**消除函数调用边界**，使有缺陷的跨调用 tile 溢出路径根本不被触发：`TCVT, BF16` 直接在原 BF16 tile 上执行，于是看似“修好了”。但：

1. 它没有修复编译器——任何**必须跨调用传递 tile** 的内核仍会命中该缺陷；
2. 它只是把缺陷掩盖在“恰好不需要跨调用 spill”的情形下。

因此 `inline` 是**规避手段（workaround）**，编译器后端修复才是**真正的修复**。

### 验证方法

1. 后端修复后，**保持 `gelu_impl` 不加 inline**，重新编译：
   ```bash
   make TESTCASE=gelu DTYPE=__bf16 tMs=2048 gMs=24*8*1024 \
       SHAPE_NAME=24_8_1024 Approximate=false diss
   ```
2. 检查反汇编：调用边界的 tile spill 应保留元素类型——
   ```bash
   DISS=output/kernel/element_wise/gelu/elf/*.elf.diss
   # 应为 TSTORE/TLOAD, BF16（或随 tile 类型），且 B.DATR 为 BF16，而非 S64/NORM.normal
   grep -nE "BSTART\.TLSU.*(TLOAD|TSTORE)" $DISS
   grep -nE "B\.DATR" $DISS
   ```
   修复前：`..., S64` + `NORM.normal`；修复后：与 tile 元素类型一致的表示。
3. 确认 `gelu_impl` 仍为独立函数（未 inline）时，入口 `TCVT, BF16` 的输入 tile 表示亦为 BF16，转换不再报错；在 QEMU 模型上 `make ... sim` 跑通且结果正确。

---

## 附：错因对照

| 项 | 旧（错误）结论 | 修正后结论 |
|---|---|---|
| 性质 | 性能问题（溢出开销） | 编译器正确性缺陷（类型丢失） |
| `S64` 语义 | “只搬原始比特、不做格式转换、类型安全” | `S64`+`NORM.normal` 把元素表示退化为 NORM，**不保留 BF16**，对半精度浮点非无损 |
| `TCVT, BF16` 为何出错 | 未提及（旧文档未发现此问题） | 入口 reload 得到 NORM tile，源类型声明 BF16 与实际不符 → 转换报错 |
| `inline` 的作用 | 当作“修复” | 只是消除调用边界、规避缺陷的 workaround；后端按元素类型 spill/reload 才是真修复 |
