# Concat (PTO one-level-arch)

## 功能

N-D 张量拼接，将多个输入张量沿指定维度连接成一个输出张量。
提供两种实现方式：

| 变体 | 方向 | 实现方式 |
|------|------|----------|
| `concat_gather` | 输出→输入 | 对每个输出元素计算源 offset，MGATHER 读取 |
| `concat_scatter` | 输入→输出 | 对每个输入元素计算目标 offset，MSCATTER 写入 |

## 输入输出

| 参数 | 类型 | 说明 |
|------|------|------|
| `in_ptr` | `DType*` | 输入张量 (多个拼接) |
| `out_ptr` | `DType*` | 输出张量 (拼接后) |
| `in_shape` / `out_shape` | `size_t*` | 形状描述 |

## Tile 类型

| Tile | 用途 |
|------|------|
| `Tile<Vec, DType, 1, tM, RowMajor>` | 数据 tile |
| `Tile<Vec, uint32_t, 1, tM, RowMajor>` | (gather) offset tile |
| `Tile<Vec, uint16_t, 1, tM, RowMajor>` | (scatter) offset tile (16-bit) |

Tail 变体使用 `ValidCol` 处理非整除列。

## 调用的 TileOp

### Gather 变体
| 操作 | 说明 |
|------|------|
| `TCI` | 生成输出线性索引 |
| `TEXPANDS` | 初始化 offset tile 为 0 |
| `TDIVS` | 整除提取商 |
| `TREMS` | 取模提取坐标 |
| `TMULS` | 坐标 × stride |
| `TADD` | 累加到 offset |
| `MGATHER` | 按 offset 从输入 gather |
| `TSTORE` | 存储输出 |

### Scatter 变体
| 操作 | 说明 |
|------|------|
| `TCI` | 生成输入线性索引 |
| `TEXPANDS` | 初始化 offset tile 为 0 |
| `TDIVS` / `TREMS` | 提取坐标 |
| `TMULS` + `TADD` | 累加到输出 offset |
| `TLOAD` | 加载输入数据 tile |
| `MSCATTER` | 按 offset scatter 到输出 |

## 实现方式

### Gather (`concat_gather`)
对每个输出 tile: `TCI` 生成索引 → 逐维 `TDIVS`/`TREMS` 提取坐标 →
在 CONCAT_DIM 处分裂坐标为 (n_tile, offset_in) → `TMULS`×stride + `TADD` 累加
跨张量跳转 offset → `MGATHER` 从输入读取 → `TSTORE`。

### Scatter (`concat_scatter`)
对每个输入 tile: `TCI` 生成索引 → 逐维提取坐标 → 用输出 stride 映射到输出位置 →
`TLOAD` 输入数据 → `MSCATTER` 写入到输出。

## 源文件

| 文件 | 说明 |
|------|------|
| `concat_gather.hpp` | Gather 模式 (输出驱动) |
| `concat_scatter.hpp` | Scatter 模式 (输入驱动) |
