# PTO ISA Tile 编程指南

## 1. 概述

PTO ISA（Pervasive Tile Operations）是一套面向 Tile（张量）操作的指令集架构。本指南面向程序员，介绍如何使用 PTO ISA 的 Tile 指令进行张量编程。

### 1.1 核心概念

- **Tile（张量）**：二维数据块，是 PTO ISA 的基本操作单元
- **TileType**：Tile 的类型，决定其存储位置和执行管线
  - `Vec`：通用向量操作（UB，256 KB）
  - `Mat`：矩阵操作（L1，512 KB）
  - `Left`：矩阵乘左操作数（L0A，64 KB）
  - `Right`：矩阵乘右操作数（L0B，64 KB）
  - `Acc`：矩阵乘累加器（L0C，256 KB）
- **Valid Region**：Tile 的有效区域，定义操作的迭代域
- **Auto/Manual Mode**：自动模式 vs 手动模式

### 1.2 数据类型支持

| 类型 | 位宽 | 说明 |
|------|------|------|
| FP64 | 64-bit | 双精度浮点 |
| FP32 | 32-bit | 单精度浮点 |
| TF32 | 19-bit | TensorFloat-32 |
| HF32 | 19-bit | High-precision Float 32 |
| FP16 | 16-bit | 半精度浮点 |
| BF16 | 16-bit | Brain Float 16 |
| FP8 | 8-bit | FP8 (e4m3/e5m2) |
| INT64/32/16/8 | 64/32/16/8-bit | 有符号整数 |
| UINT64/32/16/8 | 64/32/16/8-bit | 无符号整数 |

## 2. 编程模型

### 2.1 Tile 与 Valid Region

Tile 是 PTO ISA 的核心数据结构：

```cpp
// Tile 定义
Tile<Vec, float, 16, 16> myTile;  // 16x16 浮点 Tile

// Valid Region
myTile.GetValidRow()  // 有效行数
myTile.GetValidCol()  // 有效列数
```

**Valid Region** 定义了 Tile 操作的实际迭代域。只有 Valid Region 内的元素会被处理。

### 2.2 TileType 与内存层次

| TileType | 内存位置 | 容量 | 对齐要求 | 用途 |
|----------|---------|------|---------|------|
| Vec | UB | 256 KB | 32 B | 通用逐元素操作 |
| Mat | L1 | 512 KB | 32 B | 矩阵操作 |
| Left | L0A | 64 KB | 32 B | 矩阵乘 A 操作数 |
| Right | L0B | 64 KB | 32 B | 矩阵乘 B 操作数 |
| Acc | L0C | 256 KB | 32 B | 矩阵乘累加器 |

### 2.3 Auto vs Manual 模式

**Auto 模式**：编译器自动管理 Tile 分配和同步

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void auto_mode() {
    Tile<Vec, float, 16, 16> a, b, c;
    // 编译器自动插入 TASSIGN 和 TSYNC
    TADD(c, a, b);
}
```

**Manual 模式**：程序员显式管理 Tile 分配和同步

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void manual_mode() {
    Tile<Vec, float, 16, 16> a, b, c;
    
    // 显式分配 Tile 到硬件资源
    TASSIGN(a, 0x1000);
    TASSIGN(b, 0x2000);
    TASSIGN(c, 0x3000);
    
    // 显式同步
    RecordEvent e0 = TLOAD(a, ga);
    RecordEvent e1 = TLOAD(b, gb);
    TSYNC(e0, e1);
    
    TADD(c, a, b);
    TSYNC();
    
    TSTORE(gc, c);
}
```

## 3. Tile 指令分类

### 3.1 View 与 Tile Buffer

| 指令 | 功能 |
|------|------|
| `pto.make_tensor_view` | 创建 Tensor View |
| `pto.get_tensor_view_dim` | 获取 View 维度 |
| `pto.get_tensor_view_stride` | 获取 View 步长 |
| `pto.tensor_view_addr` | 获取 View 地址 |
| `pto.partition_view` | 分区 View |
| `pto.alloc_tile` | 分配 Tile |
| `pto.subset` | 创建子集 |
| `pto.set_validshape` | 设置 Valid Shape |

### 3.2 同步与配置

| 指令 | 功能 |
|------|------|
| `pto.tsync` | Tile 同步 |
| `pto.syncall` | 全局同步 |
| `pto.tassign` | 分配 Tile 到硬件资源 |
| `pto.talias` | 创建 Tile 别名 |

### 3.3 逐元素操作（Elementwise）

| 指令 | 功能 |
|------|------|
| `pto.tadd` | Tile 加法 |
| `pto.tsub` | Tile 减法 |
| `pto.tmul` | Tile 乘法 |
| `pto.tdiv` | Tile 除法 |
| `pto.tmax` | Tile 最大值 |
| `pto.tmin` | Tile 最小值 |
| `pto.tcmp` | Tile 比较 |
| `pto.tabs` | Tile 绝对值 |
| `pto.texp` | Tile 指数 |
| `pto.tsqrt` | Tile 平方根 |
| `pto.trelu` | Tile ReLU 激活 |
| `pto.tcvt` | Tile 类型转换 |

### 3.4 Tile-标量操作

| 指令 | 功能 |
|------|------|
| `pto.tadds` | Tile + 标量 |
| `pto.tsubs` | Tile - 标量 |
| `pto.tmuls` | Tile * 标量 |
| `pto.tdivs` | Tile / 标量 |
| `pto.tmaxs` | Tile 与标量取最大值 |
| `pto.tmins` | Tile 与标量取最小值 |
| `pto.taxpy` | AXPY 操作 (a*x + y) |

### 3.5 归约与扩展（Reduce & Expand）

| 指令 | 功能 |
|------|------|
| `pto.trowsum` | 行求和 |
| `pto.tcolsum` | 列求和 |
| `pto.trowmax` | 行最大值 |
| `pto.tcolmax` | 列最大值 |
| `pto.trowmin` | 行最小值 |
| `pto.tcolmin` | 列最小值 |
| `pto.trowexpand` | 行扩展 |
| `pto.tcolexpand` | 列扩展 |

### 3.6 内存与数据搬运

| 指令 | 功能 |
|------|------|
| `pto.tload` | 从全局内存加载到 Tile |
| `pto.tstore` | 从 Tile 存储到全局内存 |
| `pto.tprefetch` | Tile 预取 |
| `pto.mgather` | 掩码 Gather |
| `pto.mscatter` | 掩码 Scatter |

### 3.7 矩阵运算

| 指令 | 功能 |
|------|------|
| `pto.tmatmul` | 矩阵乘法 (GEMM) |
| `pto.tmatmul_acc` | 矩阵乘加 |
| `pto.tmatmul_bias` | 矩阵乘加偏置 |
| `pto.tgemv` | 矩阵向量乘 |
| `pto.tgemv_acc` | 矩阵向量乘加 |

### 3.8 布局与重排

| 指令 | 功能 |
|------|------|
| `pto.tmov` | Tile 移动 |
| `pto.textract` | 提取子 Tile |
| `pto.tinsert` | 插入子 Tile |
| `pto.tconcat` | Tile 拼接 |
| `pto.treshape` | Tile 重塑 |
| `pto.ttrans` | Tile 转置 |
| `pto.tfillpad` | 填充 Padding |
| `pto.tpack` | Tile 打包 |

## 4. 编程示例

### 4.1 矩阵加法

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void matrix_add() {
    // 定义 16x16 浮点 Tile
    Tile<Vec, float, 16, 16> a, b, c;
    
    // Auto 模式：编译器自动管理
    TADD(c, a, b);
}
```

### 4.2 矩阵乘法

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void matrix_multiply() {
    // 矩阵乘操作数需要特定 TileType
    Tile<Left, half, 16, 16> a;    // L0A
    Tile<Right, half, 16, 16> b;   // L0B
    Tile<Acc, float, 16, 16> c;    // L0C
    
    // 执行 C = A × B
    TMATMUL(c, a, b);
}
```

### 4.3 数据加载与计算

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void load_compute_store() {
    Tile<Vec, float, 16, 16> a, b, c;
    
    // Manual 模式
    TASSIGN(a, 0x1000);
    TASSIGN(b, 0x2000);
    TASSIGN(c, 0x3000);
    
    // 从全局内存加载
    RecordEvent e0 = TLOAD(a, ptr_a);
    RecordEvent e1 = TLOAD(b, ptr_b);
    TSYNC(e0, e1);
    
    // 计算 c = a + b
    TADD(c, a, b);
    TSYNC();
    
    // 存储到全局内存
    TSTORE(ptr_c, c);
}
```

### 4.4 归约操作

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void row_sum() {
    Tile<Vec, float, 16, 64> input;
    Tile<Vec, float, 16, 1> output;
    
    // 对每行求和
    TROWSUM(output, input);
}
```

### 4.5 Flash Attention 模式

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void flash_attention() {
    // Q, K, V 矩阵
    Tile<Left, half, 16, 16> Q, K;
    Tile<Right, half, 16, 16> V;
    Tile<Acc, float, 16, 16> S, O;
    
    // 1. 计算注意力分数: S = Q × K^T / sqrt(d)
    TMATMUL(S, Q, K);
    
    // 2. Softmax (需要额外的标量操作)
    // ...
    
    // 3. 输出: O = P × V
    TMATMUL(O, P, V);
}
```

## 5. 性能优化

### 5.1 Tile 复用

```cpp
// 好的做法：复用 Tile
Tile<Vec, float, 16, 16> temp;
for (int i = 0; i < N; i++) {
    TLOAD(temp, ptr[i]);
    TADD(result, result, temp);
}

// 不好的做法：频繁分配新 Tile
for (int i = 0; i < N; i++) {
    Tile<Vec, float, 16, 16> temp;  // 每次循环都分配
    TLOAD(temp, ptr[i]);
    TADD(result, result, temp);
}
```

### 5.2 流水线重叠

```cpp
// 计算与数据搬运重叠
RecordEvent load_event = TLOAD(tile_next, ptr_next);
TCOMPUTE(tile_current);
TSYNC(load_event);
// 交换 tile_current 和 tile_next
```

### 5.3 选择合适的 TileType

```cpp
// 矩阵乘法：使用专用 TileType
Tile<Left, half, 16, 16> a;    // 用于矩阵乘 A
Tile<Right, half, 16, 16> b;   // 用于矩阵乘 B
Tile<Acc, float, 16, 16> c;    // 用于累加器

// 逐元素操作：使用 Vec
Tile<Vec, float, 16, 16> x;    // 用于逐元素操作
```

### 5.4 布局优化

```cpp
// RowMajor 布局通常性能更好
Tile<Vec, float, 16, 64, RowMajor> tile;

// 避免不必要的转置
// 如果可能，直接在原始布局上操作
```

## 6. 同步机制

### 6.1 RecordEvent

`RecordEvent` 是 PTO ISA 的同步令牌：

```cpp
RecordEvent e0 = TLOAD(a, ptr_a);
RecordEvent e1 = TLOAD(b, ptr_b);

// 等待两个加载完成
TSYNC(e0, e1);

// 现在可以安全使用 a 和 b
TADD(c, a, b);
```

### 6.2 隐式同步

在 Auto 模式下，编译器会自动插入必要的同步：

```cpp
// Auto 模式：编译器自动同步
TADD(c, a, b);  // 编译器确保 a, b 已加载完成
```

### 6.3 显式同步

在 Manual 模式下，需要显式管理同步：

```cpp
// Manual 模式：显式同步
TLOAD(a, ptr_a);
TSYNC();  // 等待加载完成
TADD(c, a, b);
TSYNC();  // 等待计算完成
TSTORE(ptr_c, c);
```

## 7. 约束与限制

### 7.1 类型匹配

所有操作数必须具有相同的元素类型：

```cpp
// 正确：类型匹配
Tile<Vec, float, 16, 16> a, b, c;
TADD(c, a, b);

// 错误：类型不匹配
Tile<Vec, float, 16, 16> a;
Tile<Vec, half, 16, 16> b;
TADD(c, a, b);  // 编译错误
```

### 7.2 形状约束

矩阵乘法有严格的形状约束：

```cpp
// C[M,N] = A[M,K] × B[K,N]
// 约束：
// - A.GetValidRow() = C.GetValidRow() = M
// - A.GetValidCol() = B.GetValidRow() = K
// - B.GetValidCol() = C.GetValidCol() = N
```

### 7.3 TileType 约束

不同指令对 TileType 有特定要求：

```cpp
// 矩阵乘法
Tile<Left, half, 16, 16> a;    // 必须是 Left
Tile<Right, half, 16, 16> b;   // 必须是 Right
Tile<Acc, float, 16, 16> c;    // 必须是 Acc
TMATMUL(c, a, b);

// 逐元素操作
Tile<Vec, float, 16, 16> x, y, z;  // 必须是 Vec
TADD(z, x, y);
```

## 8. 调试与验证

### 8.1 仿真工具

- **gfrun**：功能模型，快速验证正确性
- **gfsim**：周期精确模型，性能评估

### 8.2 常见问题

1. **类型不匹配**：确保所有操作数类型一致
2. **形状不匹配**：检查矩阵维度约束
3. **TileType 错误**：使用正确的 TileType
4. **同步缺失**：确保在 Manual 模式下正确同步
5. **Valid Region 越界**：不要访问 Valid Region 外的元素

### 8.3 性能分析

使用 gfsim 分析性能：

```bash
gfsim -f program.elf --stop_cycle 1000000
```

查看输出中的性能统计：
- 周期数
- 指令吞吐量
- 资源利用率

## 9. 参考资源

- [PTO ISA 官方文档](https://pto-isa.github.io/docs/isa/tile/)
- [PTO ISA GitHub](https://github.com/PTO-ISA/pto-isa)
- SuperNPUBench 算子示例

---

**文档版本**: 1.0  
**最后更新**: 2026-01-08
