# Flash Attention (PTO one-level-arch)

## 功能

Flash Attention 算子集合，包含多种变体:

| 变体 | 说明 |
|------|------|
| `sfa` | 块稀疏 Flash Attention (CSR 格式，两遍: online softmax + P·V) |
| `fa_2d_unroll` | 稠密 Flash Attention (2D 展开，Xdim×Ydim 分组并行) |
| `fa_hif4` | HiF4 (MXFP4) 量化 Flash Attention |
| `fa_softmax_pto` | PTO softmax (Flash Attention 的 softmax 组件) |
| `fa_dcore` | fa_2d_unroll 的薄封装 |
| `fa_unalign_2d_unroll` | fa_2d_unroll 的薄封装 (非对齐场景) |

## 输入输出

| 参数 | 类型 | 说明 |
|------|------|------|
| `out_ptr` | `dtype*` | 输出 O (Sq×vD) |
| `q_ptr` | `dtype*` | Query (Sq×qD) |
| `k_ptr` | `dtype*` | Key (Skv×qD) |
| `v_ptr` | `dtype*` | Value (Skv×vD) |
| `kv_idx_ptr` / `kv_off_ptr` | `const int*` | (sfa) CSR 块稀疏索引 |
| `scale_q/k/v` | `uint32_t*` | (hif4) 每 64 个逻辑 K 元素一个 U32 scale carrier |

## Tile 类型

| Tile | 类型 | 用途 |
|------|------|------|
| Q | `CubeTileM16/M32<dtype, kTm, qD>` | Query 左矩阵操作数 |
| K | `CubeTileN8<dtype, qD, kTk>` | K^T 右矩阵操作数 |
| V | `CubeTileN8<dtype, kTk, vD>` | Value 右矩阵操作数 |
| W_acc | `CubeAccumulatorM16/M32<float, kTm, kTk>` | QK^T 累加器 |
| W | `Tile<Vec, float, kTm, kTk, ColMajor>` | FP32 score tile |
| W_cast | `Tile<Vec, dtype, kTm, kTk, ColMajor>` | 转回 dtype |
| W_left | `CubeTileM16/M32<dtype, kTm, kTk>` | P·V 的左操作数 |
| O_acc | `CubeAccumulatorM16/M32<float, kTm, vD>` | P·V 累加器 |
| O | `Tile<Vec, float, kTm, vD, ColMajor>` | FP32 输出 |
| Max/Sum | `Tile/VecTile<..., kTm, 1>` | 行 max/sum 状态 (1 标量/行) |
| Scale | `Tile<Scaling/Vec, uint32_t, ..., RowMajor>` | (hif4) group-64 U32 缩放因子 tile |

## 调用的 TileOp

### 核心矩阵操作
| 操作 | 说明 |
|------|------|
| `TLOAD_CUBE` | GM→CUBE CELL 转换并加载 Q/K/V tile |
| `TMATMUL` | Q × K^T (首次) |
| `TMATMUL_ACC` | Q × K^T (累加) |
| `TMATMUL_MX` | (hif4) 带缩放的 Q × K^T |
| `TSTORE_CUBE` | CUBE CELL→GM 转换（必要时作为 Cube/Vec 边界） |

### Softmax 计算
| 操作 | 说明 |
|------|------|
| `TMULS` | score × scale (1/√d) |
| `TROWMAX` | 对每个 query 行求最大值 → `[kTm,1]` |
| `TMAX` | 更新全局 max |
| `TSUB` | score - new_max |
| `TEXP` | exp(score - max) |
| `TROWEXPANDSUB` / `TROWEXPANDEXPDIF` | 按 query 行广播 softmax 状态 |
| `TROWSUM` | 对每个 query 行求和 → `[kTm,1]` |
| `TADD` | 更新 Sum |
| `TMUL` | rescale old sum |

### P·V 和输出
| 操作 | 说明 |
|------|------|
| `TCVT` | dtype 转换 / Vec→Left 装箱 |
| `TCOLEXPANDMUL_TEPL` / `TROWEXPANDMUL` | 广播乘 (1/l 归一化) |
| `TRECIP` | 1 / sum |
| `TADD` | 累加到 O |
| `TSTORE` | 存储输出 |

## 实现方式

### SFA (块稀疏, `sfa.hpp`)
两遍设计:
- **Pass 1** (reduce): 对每个 Q 块，遍历 CSR 活跃 K/V 块: Q·K^T → scale →
  online softmax 更新 (rowmax → rescale old → exp → rowsum → add)
- **Pass 2** (attend): 重新加载 Q (缩短 live range) → 计算 p=exp(score-m)/l →
  cast to Left → `TMATMUL`(p, V) → `TADD` 到 O

避免 `TMATMUL_ACC` 的 tile 寄存器溢出问题，使用 fresh `TMATMUL`。

### 2D Unroll (`fa_2d_unroll.hpp`)
单遍 online softmax，Xdim 个 Q 块 × Ydim 个 K/V 块并行展开。
每个 K/V 组: QK → per-Q softmax (Ydim 路归约树) → P·V → 在线更新 O。
Ydim∈{1,2,4} 有硬编码的归约树。不处理 tail (要求 Qb%Xdim==0, Kb%Ydim==0)。

### HiF4 (`fa_hif4.hpp`)
Q/K/V 使用 `__fp4_hif4x2` packed-x2 存储，并通过 `TMATMUL_MX` 携带
group-64 U32 scale。softmax 采用 FP32 Cube/Vec 状态；P 通过 packed-x2
转换进入第二次 Matrix-MX，替代已退役的 `TQUANT<MXFP4>` 接口。

## 源文件

| 文件 | 说明 |
|------|------|
| `sfa.hpp` | 块稀疏 Flash Attention (两遍) |
| `fa_2d_unroll.hpp` | 稠密 2D 展开 Flash Attention |
| `fa_hif4.hpp` | HiF4 量化 Flash Attention |
| `fa_softmax.hpp` | PTO softmax 组件 |
| `fa_dcore.hpp` | fa_2d_unroll 薄封装 |
| `fa_unalign_2d_unroll.hpp` | fa_2d_unroll 薄封装 |
| `fa_utils.h` | 辅助工具 |
| `fa_fp4_utils.h` | FP4 辅助工具 |
