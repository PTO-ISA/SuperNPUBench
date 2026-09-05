# Transpose (PTO one-level-arch)

## 功能

张量转置，交换指定轴。提供通用 N-D 转置和 2D 硬件转置两种实现。

| 变体 | 场景 | 实现方式 |
|------|------|----------|
| `tile_transpose_nd` | 任意 N-D 轴交换 | offset 计算 + MGATHER |
| `tile_transpose_2d` | 2D 矩阵转置 | 硬件 TTRANS |
| `transpose_007` | 4096×3 → 3×4096 | TLOAD + TTRANS + TSTORE |
| `transpose_050` | 8×64×64 → 64×8×64 | TLOAD + TSTORE (逐行复制) |

## 输入输出

| 参数 | 类型 | 说明 |
|------|------|------|
| `input` | `DType*` | 输入张量 |
| `output` | `DType*` | 输出张量 (轴交换后) |
| `input_shape` / `output_shape` | `uint32_t*` | (ND) 形状描述 |

## Tile 类型

### ND 变体
| Tile | 用途 |
|------|------|
| `Tile<Vec, DType, 1, TileElements, RowMajor>` | 数据 tile |
| `Tile<Vec, uint32_t, 1, TileElements, RowMajor>` | offset tile (字节偏移) |

### 2D 变体
| Tile | 用途 |
|------|------|
| `Tile<Vec, DType, TileRows, TileCols, RowMajor>` | 源 tile |
| `Tile<Vec, DType, TileCols, TileRows, RowMajor>` | 转置后目标 tile |

## 调用的 TileOp

### ND 变体
| 操作 | 说明 |
|------|------|
| `TCI` | 生成线性索引 |
| `TEXPANDS` | 初始化 offset 为 0 |
| `TDIVS` | 整除 (提取商) |
| `TMULS` + `TSUB` | 取模 (提取坐标 = idx - (idx/shape)*shape) |
| `TMULS` + `TADD` | 累加 stride × coord × sizeof 到 offset |
| `MGATHER` | 按 offset 从输入 gather |
| `TSTORE` | 存储输出 |

### 2D 变体
| 操作 | 说明 |
|------|------|
| `TLOAD` | 加载源 tile |
| `TTRANS` | 硬件 2D 转置 (TileRows×TileCols → TileCols×TileRows) |
| `TSTORE` | 存储转置结果 |

## 实现方式

### ND 转置 (`tile_transpose_nd`)
对每个输出 tile: `TCI` 生成线性索引 → 逐维 (Rank 编译期展开) divmod 提取坐标 →
轴交换后映射回输入坐标 → `TMULS`×stride×sizeof 累加到 offset tile →
`MGATHER` 按偏移 gather → `TSTORE`。

设计要点: 维度循环在标量核上编译期展开，坐标提取用 tile op 并行处理整个 tile，
避免逐元素 `__vec__` 循环。

### 2D 转置 (`tile_transpose_2d`)
双重 (row_tile, col_tile) 循环: `TLOAD` (TileRows×TileCols) →
`TTRANS` (转置为 TileCols×TileRows) → `TSTORE` 到交换后的输出位置。
四象限 tail 处理。

## 源文件

| 文件 | 说明 |
|------|------|
| `transpose.hpp` | 通用 ND + 2D 转置 |
| `transpose_vector_007.hpp` | 4096×3 → 3×4096 |
| `transpose_vector_050.hpp` | 8×64×64 → 64×8×64 (逐行复制) |
