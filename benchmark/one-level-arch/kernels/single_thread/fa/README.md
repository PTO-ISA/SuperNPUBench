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
| `scale_q/k/v` | `uint8_t*` | (hif4) per-tile 缩放因子 |

## Tile 类型

| Tile | 类型 | 用途 |
|------|------|------|
| Q | `TileLeft<dtype, kTm, qD>` | Query (boxed 512B) |
| K | `TileRight<dtype, kTk, qD>` | Key (boxed 512B) |
| V | `TileRight<dtype, kTk, vD>` | Value (boxed 512B) |
| W_acc | `TileAcc<float, kTm, kTk>` | QK^T 累加器 |
| W | `Tile<Vec, float, kTm, kTk, ColMajor>` | FP32 score tile |
| W_cast | `Tile<Vec, dtype, kTm, kTk, ColMajor>` | 转回 dtype |
| W_left | `TileLeft<dtype, kTm, kTk>` | 重新装箱为 Left (P·V 的左操作数) |
| O_acc | `TileAcc<float, kTm, vD>` | P·V 累加器 |
| O | `Tile<Vec, float, kTm, vD, ColMajor>` | FP32 输出 |
| Max/Sum | `Tile<Vec, float, kTm, 8, ColMajor, kTm, 1>` | 行 max/sum 状态 (1 标量/行) |
| Scale | `Tile<Scaling, uint8_t, ...>` | (hif4) 缩放因子 tile |

## 调用的 TileOp

### 核心矩阵操作
| 操作 | 说明 |
|------|------|
| `TLOAD` | 加载 Q/K/V tile |
| `TMATMUL` | Q × K^T (首次) |
| `TMATMUL_ACC` | Q × K^T (累加) |
| `TMATMUL_MX` | (hif4) 带缩放的 Q × K^T |
| `ACCCVT` | Acc → Vec 转换 |

### Softmax 计算
| 操作 | 说明 |
|------|------|
| `TMULS` | score × scale (1/√d) |
| `TCOLMAX` / `TROWMAX` | 列/行最大值 → Max tile |
| `TMAX` | 更新全局 max |
| `TSUB` | score - new_max |
| `TEXP` | exp(score - max) |
| `TCOLEXPANDSUB_TEPL` | 列广播减 (用 _TEPL 变体规避工具链 bug) |
| `TCOLSUM` / `TROWSUM` | 列/行求和 → Sum tile |
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
  online softmax 更新 (colmax → rescale old → exp → colsum → add)
- **Pass 2** (attend): 重新加载 Q (缩短 live range) → 计算 p=exp(score-m)/l →
  cast to Left → `TMATMUL`(p, V) → `TADD` 到 O

避免 `TMATMUL_ACC` 的 tile 寄存器溢出问题，使用 fresh `TMATMUL`。

### 2D Unroll (`fa_2d_unroll.hpp`)
单遍 online softmax，Xdim 个 Q 块 × Ydim 个 K/V 块并行展开。
每个 K/V 组: QK → per-Q softmax (Ydim 路归约树) → P·V → 在线更新 O。
Ydim∈{1,2,4} 有硬编码的归约树。不处理 tail (要求 Qb%Xdim==0, Kb%Ydim==0)。

### HiF4 (`fa_hif4.hpp`)
使用 `TMATMUL_MX` 带缩放因子。softmax 概率通过 `TQUANT<MXFP4>` 量化为 FP4。
P·V 也使用 `TMATMUL_MX`。qD 必须等于 vD。

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
