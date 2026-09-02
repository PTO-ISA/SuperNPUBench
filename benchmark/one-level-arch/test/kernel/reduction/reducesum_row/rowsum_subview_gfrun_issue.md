# [gfrun] Vector/NORM B.SUBVIEW 与 B.ASSEMBLE 尚不受支持

## 问题概述

新版 PTO tile region API 已能为 Vector/NORM RowMajor Tile 生成
`B.SUBVIEW` 和 `B.ASSEMBLE`。`rowsum_subview` 算子可以正常完成编译、链接和
反汇编，但使用 gfrun 执行时，会在第一条 `TROWSUM` 的 TEPL shape 校验处
触发断言。

目前观察到 SuperScalarModel 的 region 指令实现与编译器语义存在以下差异：

1. `OP_B_IOT` 在后续 `B.SUBVIEW` 生效前执行 TEPL shape 校验，校验看到的
   是父 Tile，而不是 subview Tile。
2. `B.SUBVIEW` 当前仅支持 Cube layout，尚不支持 Vector/NORM layout。
3. local `B.ASSEMBLE` 当前同样仅支持 Cube layout，无法组装 Vector/NORM
   fragments。

## 复现环境

- 模型仓库：`LinxISA/SuperScalarModel`
- gfrun：`/Users/blacktraker/Programming/gitproj/DV4/SuperScalarModel/bin/gfrun`
- 编译器：`/Users/blacktraker/Programming/gitproj/DV4/linx-toolchain-build/output/linx_blockisa_llvm_musl/bin`
- 算子源码：`benchmark/one-level-arch/test/kernel/reduction/reducesum_row/src/rowsum_subview.cpp`
- 数据类型：FP32
- 父 Tile：`32x32`，共 4096B
- Subview：沿 M/row 轴切成 4 个 `8x32` Tile，每个 1024B
- 输出 fragment：每个 fragment 的有效 shape 为 `8x1`，物理 shape 为
  `8x4`，共 128B

## 复现步骤

在 SuperNPUBench 根目录执行：

```bash
export COMPILER_DIR=/Users/blacktraker/Programming/gitproj/DV4/linx-toolchain-build/output/linx_blockisa_llvm_musl/bin

make -C benchmark/one-level-arch/test/kernel/reduction/reducesum_row \
  TESTCASE=rowsum_subview \
  ROWSUM_ROWS=32 ROWSUM_COLS=32 ROWSUM_PARTS=4 \
  COMPILER_DIR="$COMPILER_DIR" diss

/Users/blacktraker/Programming/gitproj/DV4/SuperScalarModel/bin/gfrun \
  -f benchmark/one-level-arch/output/kernel/reduction/reducesum_row/elf/\
kernel_reduction_reducesum_row_rowsum_subview_Rows32_Cols32_Parts4.elf
```

## 编译结果

算子可以正常编译和链接。生成文件为：

```text
benchmark/one-level-arch/output/kernel/reduction/reducesum_row/elf/
kernel_reduction_reducesum_row_rowsum_subview_Rows32_Cols32_Parts4.elf

benchmark/one-level-arch/output/kernel/reduction/reducesum_row/elf/
kernel_reduction_reducesum_row_rowsum_subview_Rows32_Cols32_Parts4.elf.diss
```

反汇编中包含预期的 4 条 `B.SUBVIEW` 和 4 条 `B.ASSEMBLE`：

```text
B.SUBVIEW ... range_base=0
B.SUBVIEW ... range_base=8
B.SUBVIEW ... range_base=16
B.SUBVIEW ... range_base=24

B.ASSEMBLE INIT
B.ASSEMBLE MIDDLE
B.ASSEMBLE MIDDLE
B.ASSEMBLE LAST
```

Subview 偏移使用新版 API 规定的 128B range address unit。每个 `8x32` FP32
Subview 为 1024B，即 8 个 address units，所以四段偏移依次为
`0/8/16/24`。

## 实际结果

gfrun 在第一条 `TROWSUM` 处失败：

```text
ValidateReduceAndExpandTepl assertion failed
```

断言位置：

```text
emulator/engine/AccumulateBlockInfo.cpp:853
```

触发校验时，模型仍使用父 Tile `32x32` 作为 source Tile，而当前 `TROWSUM`
指令的 `B.DIM` 描述的是经过 `B.SUBVIEW` 切分后的 `8x32` source Tile，导致
source shape 校验失败。

## 预期结果

- `B.SUBVIEW` 在 TEPL shape 校验前更新当前输入操作数的有效 Tile 描述。
- Vector/NORM RowMajor Tile 可以作为 `B.SUBVIEW` 的 parent 和 subview。
- local `B.ASSEMBLE` 可以将 Vector/NORM Tile fragments 按
  INIT/MIDDLE/LAST 阶段组装为目标 Tile。
- 算子完成 4 次 `TROWSUM`、fragment assemble 和 `TSTORE`，并与 CPU
  rowsum golden 保持一致。

## 根因分析

### 1. SUBVIEW 生效时序晚于 TEPL 校验

`emulator/engine/AccumulateBlockInfo.cpp` 的 `OP_B_IOT` 处理流程先调用：

```cpp
ValidateReduceAndExpandTepl(currentBlock, inst);
```

直到后续读取 `B.SUBVIEW` 时才调用：

```cpp
currentBlock->HandleBSubview(*inst);
```

因此 `ValidateReduceAndExpandTepl` 使用的是未经 subview 修正的父 Tile
描述。对本例而言，validator 看到的是 `32x32`，而不是当前 fragment 的
`8x32`。

### 2. HandleBSubview 仅支持 Cube layout

当前 `Block::HandleBSubview` 要求 parent Tile 使用 Cube layout，并通过
`CubeCellDescribeSubview` 计算子区域。这条路径不能表达编译器已经支持的
Vector/NORM RowMajor region view。

### 3. Local assemble 仅支持 Cube layout

`emulator/SoftCore.cpp` 中的 `SoftCore::PrepareLocalAssemble` 要求目标 block
和 layout 满足类似以下条件：

```cpp
blockType == CUBE && IsCubeLayout(dstLayout)
```

因此，即使调整了 subview 与 shape validation 的执行顺序，Vector/NORM
`B.ASSEMBLE` 仍会在后续执行阶段失败。

## 建议修复

1. 调整 region modifier 的解析和校验顺序：先收集并应用属于当前操作数的
   `B.SUBVIEW`，再执行 TEPL shape validation。
2. 为 `Block::HandleBSubview` 增加 NORM layout 路径，根据 range base、
   range size 和父 Tile 元数据构造 fragment 描述，不使用 Cube cell layout
   的计算方法。
3. 扩展 `SoftCore::PrepareLocalAssemble`，支持 TEPL/Vector NORM fragments
   按 INIT/MIDDLE/LAST 阶段写入同一个目标 Tile。
4. 保证 range base 按 ISA/API 规定的 128B address unit 解释。

## 建议回归测试

- NORM RowMajor 父 Tile 的多段 `B.SUBVIEW + TROWSUM`。
- 非零 range base 的 NORM subview，覆盖 128B address unit 换算。
- NORM fragments 的 `B.ASSEMBLE INIT/MIDDLE/LAST`。
- Assemble 后执行 `TSTORE`，并与 CPU rowsum golden 做数值比较。
- 保留现有 Cube-layout SUBVIEW/ASSEMBLE 用例，确认新增 NORM 路径不会引入
  Cube 行为回归。

## 影响范围

该问题会影响所有通过 PTO region API 对 Vector Tile 使用 `TPARTVIEW` 或
`TASSEMBLY` 的算子，包括但不限于：

- 分块 reduction；
- 分块 element-wise；
- 需要从父 Vector Tile 借用多个局部 view 的算子；
- 需要将多个 Vector fragments 汇总到一个父 Tile 的算子。

现阶段这些算子可以由编译器生成合法的 region 指令，但无法通过 gfrun 完成
功能模拟和数值验证。
