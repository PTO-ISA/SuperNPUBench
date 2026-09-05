# Broadcast (PTO one-level-arch)

## 功能

N-D 广播操作，将低维张量扩展到高维。支持多种广播模式：

| 变体 | 广播 | 实现方式 |
|------|------|----------|
| `broadcast` (generic) | N-D 任意维度 | MGATHER (offset 计算 + gather) |
| `broadcast_vec_019` | (B,1,K)→(B,N,K) | TLOAD + TINSERT×N |
| `broadcast_vec_039` | (B,1,K)→(B,N,K), K=2^n | TLOAD + TINSERT×N |
| `broadcast_vec_07` | (N,1)→(N,C) | TLOAD + TROWEXPAND |

## 输入输出

| 参数 | 类型 | 说明 |
|------|------|------|
| `in_ptr` | `dtype*` | 输入张量 |
| `out_ptr` | `dtype*` | 输出张量 |
| `in_shape` / `out_shape` | `size_t*` | (generic) 形状描述 |

## Tile 类型

| Tile | 用途 |
|------|------|
| `Tile<Vec, dtype, 1, tM, RowMajor>` | 数据 tile |
| `Tile<Vec, uint32_t, 1, tM, RowMajor>` | (generic) offset tile |
| `Tile<Vec, dtype, kTileBatch, tileCols, RowMajor, VR, VC>` | (vec 变体) 输入/输出 tile |

Tail 变体使用 `ValidCol` 处理非整除列。

## 调用的 TileOp

### Generic N-D 变体
| 操作 | 说明 |
|------|------|
| `TCI` | 生成线性索引 [base, base+1, ...] |
| `TEXPANDS` | 初始化 offset tile 为 0 |
| `TDIVS` | 整除: idx / shape (商) |
| `TREMS` | 取模: idx % shape (余数 = 坐标) |
| `TMULS` | 坐标 × stride |
| `TADD` | 累加到 offset |
| `MGATHER` | 按 offset 从输入 gather |
| `TSTORE` | 存储输出 |

### Vec 变体
| 操作 | 说明 |
|------|------|
| `TLOAD` | 加载输入 tile |
| `TINSERT` | (019/039) 将输入插入到 N 个列位置 |
| `TROWEXPAND` | (07) 列 0 广播到所有列 |
| `TSTORE` | 存储输出 |

## 实现方式

### Generic N-D (`broadcast`)
对每个输出 tile: `TCI` 生成线性索引 → 从内到外逐维 `TREMS`/`TDIVS` 提取坐标 →
`TMULS`×stride + `TADD` 累加到 offset → `MGATHER` 从输入 gather → `TSTORE`。

### Vec 019/039 (`broadcast`)
对每个 batch tile: `TLOAD` (kTileBatch×kInner) → `TINSERT`×kBCast 插入到 N 个
不重叠的列位置 → `TSTORE`。039 要求 K 为 2 的幂。

### Vec 07 (`broadcast`)
对每个 N tile: `TLOAD` (kTileRows×1) → `TROWEXPAND` 广播到 (kTileRows×kC) → `TSTORE`。

## 源文件

| 文件 | 说明 |
|------|------|
| `broadcast.hpp` | Generic N-D 广播 (offset+gather) |
| `broadcast_vec_019.hpp` | (B,1,K)→(B,N,K) via TINSERT |
| `broadcast_vec_039.hpp` | (B,1,K)→(B,N,K), K=2^n |
| `broadcast_vec_07.hpp` | (N,1)→(N,C) via TROWEXPAND |
