# Gather (PTO one-level-arch)

## 功能

行索引 gather: `out[j,:] = table[idx[j],:]`。按索引从表中提取数据行。
支持大尺度 (gK=131072) 和 2 的幂维度。

## 输入输出

| 参数 | 类型 | 说明 |
|------|------|------|
| `in_data_ptr` | `dtype*` | 查找表 (gK×gN) |
| `in_offset_ptr` | `otype*` | 行索引 (gM) |
| `out_ptr` | `dtype*` | 输出 (gM×gN) |

## Tile 类型

| Tile | 类型 | 用途 |
|------|------|------|
| 索引 tile | `Tile<Vec, otype, 1, tM, RowMajor>` | 行索引 (uint32/int32) |
| 数据 tile | `Tile<Vec, dtype, tM, tN, RowMajor>` | gather 结果 |

Tail 变体: `rmd_m` (行尾)、`rmd_n` (列尾)、`rmd_mn` (角落)。

## 调用的 TileOp

| 操作 | 说明 |
|------|------|
| `TLOAD` | 加载行索引 tile (1×tM) |
| `MGATHER` | 按行索引从表中 gather (tM×tN) |
| `TSTORE` | 存储结果 tile |

## 实现方式

双重 Mb×Nb 循环遍历输出 tiles。对每个 (j, i):
1. `TLOAD` 加载 tM 个行索引 (1×tM tile)
2. 调整表全局指针 `table + i*tN` (按列分块偏移)
3. `MGATHER` 按行索引从调整后的表地址 gather tM×tN 数据
4. `TSTORE` 存储结果

Tail 处理: rmd_M (行尾)、rmd_N (列尾)、角落 (rmd_M×rmd_N)。

## 源文件

| 文件 | 说明 |
|------|------|
| `gather.hpp` | 行索引 gather 实现 |
