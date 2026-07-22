# LinxISA 张量编程指南

## 1. 概述

LinxISA（灵犀指令集架构）采用**块结构化指令**设计，将张量计算操作封装在不同的块类型中。本指南面向程序员，介绍如何使用 LinxISA 进行张量编程。

### 1.1 核心概念

- **块指令（Block Instruction）**：LinxISA 的基本执行单元，包含块头和块体
- **张量块类型**：
  - `VPAR` / `VSEQ`：向量数据块（并行/串行）
  - `CUBE`：矩阵数据块
  - `TMA`：数据搬运块
  - `TEPL`：模板数据块

### 1.2 数据类型支持

| 类型 | 位宽 | 说明 |
|------|------|------|
| FP64 | 64-bit | 双精度浮点 |
| FP32 | 32-bit | 单精度浮点 |
| TF32 | 19-bit | TensorFloat-32 |
| FP16 | 16-bit | 半精度浮点 |
| BF16 | 16-bit | Brain Float 16 |
| FP8 | 8-bit | FP8 (e4m3/e5m2) |
| INT64/32/16/8 | 64/32/16/8-bit | 有符号整数 |
| UINT64/32/16/8 | 64/32/16/8-bit | 无符号整数 |

## 2. 编程模型

### 2.1 块指令结构

```
┌─────────────────────────────────┐
│         Block Header            │
│  - Block Type (块类型)          │
│  - Branch Info (分支信息)       │
│  - Loop Bounds (循环边界)       │
│  - Scheduling Hints (调度提示)  │
└─────────────────────────────────┘
┌─────────────────────────────────┐
│         Block Body              │
│  - Micro-instructions (微指令)  │
│  - 实际计算/访存操作            │
└─────────────────────────────────┘
```

### 2.2 执行模型

LinxISA 采用**主核 + 专用核**的异构架构：

- **BCC（Block Control Core）**：主核，负责调度和标量计算
- **Cube Core**：矩阵计算专用核
- **Vector Core**：向量计算专用核（64 lanes）
- **MTC/TMA**：数据搬运专用核

## 3. 向量编程

### 3.1 向量块类型

- **VPAR**：向量并行块，支持多 lane 并行执行
- **VSEQ**：向量串行块，按顺序执行

### 3.2 向量算术运算

```cpp
// 向量加法
V.ADD vd, vs1, vs2      // vd = vs1 + vs2
V.ADDI vd, vs1, imm     // vd = vs1 + imm

// 向量减法
V.SUB vd, vs1, vs2      // vd = vs1 - vs2

// 向量乘法
V.MUL vd, vs1, vs2      // vd = vs1 * vs2
V.MADD vd, vs1, vs2, vs3 // vd = vs1 * vs2 + vs3

// 向量除法
V.DIV vd, vs1, vs2      // vd = vs1 / vs2
```

### 3.3 向量比较与选择

```cpp
// 比较操作
V.CMP.EQ vd, vs1, vs2   // vd = (vs1 == vs2)
V.CMP.LT vd, vs1, vs2   // vd = (vs1 < vs2)

// 条件选择
V.CSEL vd, vs1, vs2, vm // vd = vm ? vs1 : vs2
```

### 3.4 向量访存

```cpp
// 普通访存
V.LW vd, offset(base)   // 向量加载字
V.SW vd, offset(base)   // 向量存储字

// 桥接访存（跨 lane）
V.LW.BRG vd, offset(base)
V.SW.BRG vd, offset(base)
```

## 4. 矩阵编程

### 4.1 CUBE 块

CUBE 块用于矩阵乘法（GEMM）运算：

```
C = A × B + C (可选累加)
```

### 4.2 矩阵乘法示例

```cpp
// 矩阵乘法：C[M,N] = A[M,K] × B[K,N]
// 使用 CUBE 块
bstart.cube
  // 加载 A 和 B 到 L0 buffer
  // 执行矩阵乘法
  // 存储结果 C
bstop
```

### 4.3 矩阵数据类型

CUBE 块支持的数据类型组合：

| 累加器类型 | A 类型 | B 类型 |
|-----------|--------|--------|
| INT32 | INT8 | INT8 |
| FP32 | FP16 | FP16 |
| FP32 | BF16 | BF16 |
| FP32 | FP32 | FP32 |

## 5. 数据搬运

### 5.1 TMA 块

TMA（Tile Memory Access）块用于高效的数据搬运：

```asm
BSTART.TLOAD INT32
  // B.ARG / B.DIM / B.IOR / B.IOT 描述符
```

### 5.2 搬运模式

- **连续搬运**：连续内存区域
- **分块搬运**：支持 stride 和 tiling
- **Gather/Scatter**：非连续内存访问

## 6. 编程最佳实践

### 6.1 数据布局

- 矩阵数据推荐使用 Row-Major 布局
- 向量数据注意对齐要求（32 字节对齐）
- 使用 TEPL 块进行数据重排

### 6.2 性能优化

1. **数据复用**：充分利用 Tile Register 的空间
2. **流水线重叠**：计算与数据搬运重叠
3. **向量化**：尽量使用 VPAR 块而非 VSEQ
4. **分块计算**：大矩阵使用分块策略

### 6.3 同步机制

- Block 内部指令按程序序执行
- Block 之间通过调度器保证依赖关系
- 使用 `BSTOP` 显式同步点

## 7. 常见模式

### 7.1 矩阵乘法 + 激活函数

```cpp
// 1. 矩阵乘法 (CUBE)
C = A × B

// 2. 激活函数 (VPAR)
C = ReLU(C)  // 或 GELU, Sigmoid 等
```

### 7.2 归约操作

```cpp
// 行归约：每行求和
result[i] = sum(C[i, :])

// 列归约：每列求和
result[j] = sum(C[:, j])
```

### 7.3 Flash Attention

```cpp
// Q, K, V 矩阵
// 1. 计算注意力分数：S = Q × K^T / sqrt(d)
// 2. Softmax：P = softmax(S)
// 3. 输出：O = P × V
```

## 8. 调试与验证

### 8.1 仿真工具

- **gfrun**：功能模型，快速验证正确性
- **gfsim**：周期精确模型，性能评估

### 8.2 常见问题

1. **数据类型不匹配**：检查 CUBE 块的类型约束
2. **对齐错误**：确保数据满足对齐要求
3. **Tile Register 溢出**：合理分配 Tile 资源

## 9. 参考资源

- [LinxISA 官方文档](https://linxisa.github.io/linx-isa/)
- [LinxISA GitHub](https://github.com/LinxISA/linx-isa)
- SuperNPUBench 算子示例

---

**文档版本**: 1.0
**最后更新**: 2026-01-08
