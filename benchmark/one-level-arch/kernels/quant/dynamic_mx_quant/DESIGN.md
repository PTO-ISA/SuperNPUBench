# DynamicMxQuant PTO-ISA 扩展设计文档

## 1. 概述

### 目标

将 DynamicMxQuant 实现为完整的多文件 PTO-ISA kernel，覆盖 Ascend C 参考实现中的三种 scale 算法、两种 axis 模式和 M 维度尾块处理。

### 设计原则

- **算法驱动划分**：文件按核心算法差异划分，而非实现分支
- **公共抽象复用**：类型定义、常量、scale 算法集中在 common 模块
- **模板参数控制变体**：ScaleAlg、TileM、BlockSize 通过模板参数分发，不分文件
- **公式一致性**：量化缩放使用 `y = x × 2^(-E)`（精确 2 的幂次），与 Ascend C 参考实现数学等价

### 覆盖范围

| 维度 | 当前实现 |
|------|---------|
| 输入类型 | BF16 |
| 输出类型 | FP8_E4M3FN |
| scaleAlg | 0=OCP, 1=cuBLAS, 2=DynamicRange |
| axis | 尾轴 + 非尾轴 |
| blockSize | 32（TileM=8 时上限 512，TileM=4 时上限 1024） |
| round_mode | rint |
| scale 输出 | uint16_t（E8M0 格式，stride=BlockSize padded） |
| M 维度尾块 | 支持（递归模板实例化） |
| K 维度 | 必须为 BlockSize 的倍数 |

---

## 2. 文件结构

```
kernels/quant/dynamic_mx_quant/
├── dynamic_mx_quant_common.hpp    # 公共模块：类型、常量、scale 算法
├── dynamic_mx_quant_tail.hpp      # 尾轴入口：2D 循环 [M, K/blockSize]
├── dynamic_mx_quant_nontail.hpp   # 非尾轴入口：2D 循环 [Pre*Post, Axis/blockSize]
├── DESIGN.md                      # 本文档
└── RECORD.md                      # 问题记录（TileSize 约束等）
```

### 文件职责

| 文件 | 职责 | 核心差异 |
|------|------|---------|
| `common` | 常量定义、三种 scale 算法实现（输出 scale_byte + shared_exp） | 算法核心 |
| `tail` | 尾轴量化入口，2D 循环 + M 尾块递归，inv_scale 构造 + 量化 | 循环结构 |
| `nontail` | 非尾轴量化入口，将 [Pre, Axis, Post] 视为 [Pre*Post, Axis] | 循环结构 |

---

## 3. Common 模块设计

### 3.1 模板参数体系

```cpp
// 尾轴入口函数模板参数
template <
    int M,                        // 行数
    int K,                        // 量化轴大小
    ScaleAlg Alg = ScaleAlg::OCP, // scale 算法枚举
    int TileM = 8,                // 每次处理的行数
    int BlockSize = 32            // 量化块大小
>
void dynamic_mx_quant_tail(__bf16 *x, __fp8_e4m3 *y, uint16_t *scale);

// 非尾轴入口函数模板参数
template <
    int Pre,                      // 量化轴之前的维度乘积
    int Axis,                     // 量化轴大小
    int Post,                     // 量化轴之后的维度乘积
    ScaleAlg Alg = ScaleAlg::OCP, // scale 算法枚举
    int TileM = 8,                // 每次处理的行数
    int BlockSize = 32            // 量化块大小
>
void dynamic_mx_quant_nontail(__bf16 *x, __fp8_e4m3 *y, uint16_t *scale);
```

### 3.2 常量表

#### 3.2.1 BF16 相关常量

| 常量名 | 值 | 说明 |
|--------|---|------|
| `BF16_EXP_MASK` | `0x7F80` | BF16 指数位掩码 |
| `BF16_ABS_MASK` | `0x7FFF` | BF16 绝对值掩码（去符号位） |
| `BF16_EXP_BIAS` | `0x7F00` | BF16 指数偏置（2^127 的位模式） |
| `BF16_SHR_NUM` | `7` | 提取原始指数的右移位数 |
| `BF16_NAN_PATTERN` | `0x7F81` | 自定义 NaN 哨兵（BF16 位模式） |
| `BF16_SPECIAL_EXP` | `0x0040` | sharedExp=127 时的 inv_scale（最小 BF16 次正规数） |
| `BF16_SCALE_BIAS` | `0x7F00` | 构造 inv_scale 的偏置值（= BF16_EXP_BIAS，即 `0x7F00 - sharedExp` 得到 `2^(-shared_exp)` 的 BF16 位模式） |

#### 3.2.2 FP8 E4M3 相关常量

| 常量名 | 值 | 说明 |
|--------|---|------|
| `FP8_E4M3_EMAX` | `0x0400` | FP8 E4M3 最大正则数指数（BF16 位模式） |
| `FP8_E4M3_DST_MAX` | `448.0f` | FP8 E4M3 最大可表示值 |
| `FP8_E4M3_INV_DST_MAX` | `1/448.0f` | DST_MAX 的倒数 |
| `FP8_NAN_BYTE` | `0x00FF` | E8M0 NaN 哨兵值 |

#### 3.2.3 FP32 相关常量

| 常量名 | 值 | 说明 |
|--------|---|------|
| `FP32_EXP_MASK` | `0x7F800000` | FP32 指数位掩码 |
| `FP32_MANTISSA_MASK` | `0x007FFFFF` | FP32 尾数掩码（23 位） |
| `FP32_EXP_BIAS` | `0x3F800000` | FP32 指数偏置 |
| `FP32_SHR_NUM` | `23` | 提取 FP32 指数的右移位数 |

### 3.3 Tile 类型定义

```cpp
// 输入 tile：从 GM 加载的 BF16 原始数据
using tile_x     = Tile<Location::Vec, __bf16,     TileM, BlockSize, BLayout::RowMajor>;

// FP32 计算 tile：精度转换后的中间数据
using tile_f     = Tile<Location::Vec, float,       TileM, BlockSize, BLayout::RowMajor>;

// 输出 tile：量化结果
using tile_o     = Tile<Location::Vec, __fp8_e4m3,  TileM, BlockSize, BLayout::RowMajor>;

// scale/shared_exp tile：每行 BlockSize 个元素（实际仅首列有效）
using tile_scale = Tile<Location::Vec, uint16_t,    TileM, BlockSize, BLayout::RowMajor>;
```

### 3.4 Scale 算法

三种算法共享相同的接口签名，通过 `ScaleAlg` 枚举分发。

#### 3.4.1 接口定义

```cpp
// 输入：x_bf16（原始 BF16 数据 tile）
// 输出：scale_byte（E8M0 格式）
//       shared_exp_out（BF16 位模式差值 = E << 7，用于构造 inv_scale）
template <ScaleAlg Alg, int TileM, int BlockSize>
void compute_scale(
    Tile<Vec, __bf16, TileM, BlockSize> &x_bf16,
    Tile<Vec, uint16_t, TileM, BlockSize> &scale_byte,
    Tile<Vec, uint16_t, TileM, BlockSize> &shared_exp_out);
```

`shared_exp_out` 的含义：BF16 位模式差值，等于 `E × 0x0080`（即 `E << 7`）。
在主流程中通过 `TXORS + TADDS` 转换为 `inv_scale = 2^(-E)` 的 BF16 位模式。

#### 3.4.2 scaleAlg=0（OCP MxFP8/MxFP4）

**算法概述**：提取每个元素的指数位，取块内最大指数，计算 `shared_exp = max_exp - emax`，scale = 2^shared_exp。

**PTO-ISA 实现步骤**：

```
TANDS(exp_bits, x_u16, BF16_EXP_MASK)         // 提取 BF16 指数位
TROWMAX(max_exp, exp_bits)                     // 块内最大指数
TCMPS(eq_nan, max_exp, BF16_EXP_MASK)          // NaN/Inf 检测
TMAXS(max_exp, max_exp, FP8_E4M3_EMAX)         // 下限钳位 emax
TEXPANDS(emax_tile, FP8_E4M3_EMAX)
TSUB(shared_exp_out, max_exp, emax_tile)       // shared_exp = max_exp - emax
TSHRS(scale_byte, shared_exp_out, 7)           // scale_byte = shared_exp >> 7
TEXPANDS(nan_byte, FP8_NAN_BYTE)
TSEL(scale_byte, eq_nan, nan_byte)             // NaN/Inf → 0xFF
```

**指令数**：~10 条

#### 3.4.3 scaleAlg=1（NVIDIA cuBLAS MxFP8）

**算法概述**：计算 `S = amax * inv_dst_max`，提取 FP32 指数，尾数非零时指数 +1（向上取整），构造 shared_exp = E << 7。

**PTO-ISA 实现步骤**：

```
TABS(abs_x, x_f32)                             // 取绝对值
TROWMAX(max_abs, abs_x)                        // 块内最大绝对值（FP32）
TMULS(max_abs, max_abs, FP8_E4M3_INV_DST_MAX)  // S = amax / dst_max
TCAST(s_bits, max_abs)                         // 重新解释为 uint32
TSHRS(exp_bits, s_bits, 23)                    // 提取 FP32 指数位
TANDS(man_bits, s_bits, FP32_MANTISSA_MASK)    // 提取尾数位
TEXPANDS(s_bits, 0)
TCMP(man_nz, man_bits, s_bits)                 // 尾数非零检测
TADDS(s_bits, exp_bits, 1)                     // exp + 1
TSEL(exp_bits, man_nz, s_bits)                 // 尾数非零 → exp += 1
TCAST(scale_byte, exp_bits)                    // 转换为 uint8（E8M0）
TSHLS(shared_exp_out, scale_byte, 7)           // shared_exp = E << 7
```

**指令数**：~12 条

#### 3.4.4 scaleAlg=2（Dynamic Dtype Range MxFP4）

**算法概述**：对最大绝对值加尾数修正值（0x003F），加法进位到指数位实现向上取整，计算 `shared_exp = corrected_exp - emax`。

**PTO-ISA 实现步骤**：

```
TANDS(abs_x, x_u16, BF16_ABS_MASK)             // 取绝对值（位操作）
TROWMAX(max_abs, abs_x)                        // 块内最大绝对值
TADDS(max_abs, max_abs, 0x003F)                // 加修正值（尾数进位到指数）
TANDS(exp_bits, max_abs, BF16_EXP_MASK)        // 提取进位后的指数
TCMPS(eq_nan, exp_bits, BF16_EXP_MASK)         // NaN/Inf 检测
TMAXS(exp_bits, exp_bits, FP8_E4M3_EMAX)       // 下限钳位 emax
TEXPANDS(emax_tile, FP8_E4M3_EMAX)
TSUB(shared_exp_out, exp_bits, emax_tile)      // shared_exp = exp - emax
TSHRS(scale_byte, shared_exp_out, 7)           // scale_byte = shared_exp >> 7
TEXPANDS(nan_byte, FP8_NAN_BYTE)
TSEL(scale_byte, eq_nan, nan_byte)             // NaN/Inf → 0xFF
```

**指令数**：~11 条

#### 3.4.5 三种算法对比

| 维度 | OCP (0) | cuBLAS (1) | DynamicRange (2) |
|------|---------|------------|-------------------|
| 归约输入 | 指数位 (`x & EXP_MASK`) | 绝对值（FP32 TABS） | 绝对值 (`x & ABS_MASK`) |
| 归约操作 | TROWMAX（U16） | TROWMAX（FP32） | TROWMAX（U16） |
| 核心计算 | `max_exp - emax` | `ceil(log2(amax * inv_max))` | `(amax + 0x003F) & EXP_MASK - emax` |
| 尾数感知 | 否 | 是（显式提取+条件+1） | 是（加法进位隐式） |
| shared_exp 构造 | `TSUB(max_exp, emax)` | `TSHLS(E, 7)` | `TSUB(corrected_exp, emax)` |

### 3.5 量化主计算（tail/nontail 入口中）

三种 scale 算法共享相同的量化主流程：

```
步骤 1: 加载 + compute_scale
  TLOAD(xq, gx)                                // 从 GM 加载输入 tile
  compute_scale<Alg>(xq, scale_byte, shared_exp) // 计算 scale_byte + shared_exp

步骤 2: 精度转换
  TCVT(xf, xq)                                // BF16 → FP32

步骤 3: 构造 inv_scale = 2^(-E)
   // inv_scale = BF16_SCALE_BIAS - shared_exp = 0x7F00 - shared_exp
  // 等价变换：0x3F80 - x = ~x + 0x3F81 = TXORS(x, 0xFFFF) + TADDS(0x3F81)
  TXORS(neg_exp, shared_exp, 0xFFFF)           // neg_exp = ~shared_exp
  TADDS(inv_scale, neg_exp, 0x3F81)            // inv_scale = 2^(-E) 的 BF16 位模式

步骤 4: 缩放输入数据
  TCAST(inv_bf16, inv_scale)                   // uint16 → BF16（位模式不变）
  TCVT(inv_scale_f, inv_bf16)                  // BF16 → FP32（浮点值转换）
  TMUL(xf, xf, inv_scale_f)                    // 逐元素乘：xf[i,j] *= inv_scale_f[i,j]

步骤 5: 类型转换 + 存储
  TCAST(oq, xf)                                // FP32 → FP8 E4M3
  TSTORE(gs, scale_byte)                       // 存储 scale（E8M0）
  TSTORE(gy, oq)                               // 存储量化结果
```

**关键设计**：`shared_exp` 是 BF16 位模式差值（= E << 7），通过位操作直接构造 `2^(-E)` 的 BF16 位模式，保证量化缩放是精确的 2 的幂次。

---

## 4. 尾轴 Kernel 设计

### 4.1 入口签名

```cpp
template <int M, int K, ScaleAlg Alg = ScaleAlg::OCP, int TileM = 8, int BlockSize = 32>
void dynamic_mx_quant_tail(__bf16 *x, __fp8_e4m3 *y, uint16_t *scale);
```

### 4.2 循环结构

```
full_m = M / TileM                             // 完整 tile 数
M_tail = M % TileM                             // 尾块行数
numKb = K / BlockSize                          // K 方向量化块数

for m in [0, full_m):                          // 外层：行方向（完整 tile）
  for kb in [0, numKb):                        // 内层：量化块方向
    quantize_tile(...)

if constexpr (M_tail > 0):                     // M 维度尾块（递归）
  dynamic_mx_quant_tail<M_tail, K, Alg, TileM, BlockSize>(...)
```

### 4.3 GM 张量布局

```cpp
// 输入：[M, K]，行主序，连续
using gm_x = global_tensor<__bf16,    RowMajor<M, K>>;

// 输出：[M, K]，行主序，连续（fp8 存储为 uint8）
using gm_y = global_tensor<uint8_t,   RowMajor<M, K>>;

// scale：[M, numKb * BlockSize]，stride=BlockSize padded
using gm_s = global_tensor<uint16_t,  RowMajor<M, numKb * BlockSize>>;
```

---

## 5. 非尾轴 Kernel 设计

### 5.1 入口签名

```cpp
template <int Pre, int Axis, int Post, ScaleAlg Alg = ScaleAlg::OCP, int TileM = 8, int BlockSize = 32>
void dynamic_mx_quant_nontail(__bf16 *x, __fp8_e4m3 *y, uint16_t *scale);
```

### 5.2 处理策略

将 `[Pre, Axis, Post]` 视为 `[Pre * Post, Axis]` 的 2D 矩阵：
- 每行是一个 `(pre, post)` 组合，沿 axis 方向连续
- 复用与 tail 相同的 2D 循环结构
- M 维度尾块同样通过递归处理

### 5.3 GM 张量布局

```cpp
using gm_x = global_tensor<__bf16,    RowMajor<Pre * Post, Axis>>;
using gm_y = global_tensor<uint8_t,   RowMajor<Pre * Post, Axis>>;
using gm_s = global_tensor<uint16_t,  RowMajor<Pre * Post, numKb * BlockSize>>;
```

---

## 6. PTO-ISA Intrinsic 依赖

### 6.1 必需 Intrinsic 清单

| 类别 | Intrinsic | 用途 |
|------|-----------|------|
| 加载/存储 | TLOAD, TSTORE | GM ↔ UB 数据搬运 |
| 类型转换 | TCVT, TCAST | 精度转换、位模式重解释 |
| 算术 | TMUL, TADDS, TSUB, TMULS | 逐元素乘、标量加、减法、标量乘 |
| 位操作 | TANDS, TXORS | 掩码提取、位翻转 |
| 移位 | TSHRS, TSHLS | 指数提取、位模式构造 |
| 归约 | TROWMAX | 行最大值（U16 和 FP32） |
| 比较 | TCMPS | 条件掩码生成 |
| 选择 | TSEL | 条件赋值 |
| 广播 | TEXPANDS | 标量广播到 tile |
| 绝对值 | TABS | 浮点绝对值（仅 cuBLAS） |
| 最大值 | TMAXS | 标量最大值钳位 |

### 6.2 关键约束

- **位操作仅支持整数类型**：TANDS、TXORS、TSHRS、TSHLS 的 tile 元素必须是 `uint16_t` 或 `uint32_t`
- **TROWMAX 支持整数和浮点 tile**：OCP/DynamicRange 用 U16，cuBLAS 用 FP32
- **TSEL 需要匹配类型**：条件 mask 和 src0/src1 的元素类型必须一致
- **TileSize 约束**：TLOAD/TSTORE 的 tile 必须满足 `logicalTileBytes ∈ [512B, 32KB]`（详见 RECORD.md）

---

## 7. 测试计划

### 7.1 测试矩阵

| 测试维度 | 取值 | 组合数 |
|---------|------|--------|
| ScaleAlg | OCP, CUBLAS, DYNAMIC_RANGE | 3 |
| Axis | 尾轴, 非尾轴 | 2 |
| **总计** | | **6** |

### 7.2 验证方法

- **编译验证**：所有组合通过 `make TESTCASE=... diss` 编译成功（含 res_check=on）
- **精度验证**：`run_precision_check.py` 编排 gen → compile → QEMU → compare 流程

### 7.3 测试文件结构

```
test/kernel/quant/dynamic_mx_quant/
├── src/
│   ├── tail_ocp.cpp                    # 尾轴 OCP
│   ├── tail_cublas.cpp                 # 尾轴 cuBLAS
│   ├── tail_dynamic_range.cpp          # 尾轴 DynamicRange
│   ├── nontail_ocp.cpp                 # 非尾轴 OCP
│   ├── nontail_cublas.cpp              # 非尾轴 cuBLAS
│   ├── nontail_dynamic_range.cpp       # 非尾轴 DynamicRange
│   ├── gen_dynamic_mx_quant_data.py    # Golden 数据生成
│   ├── dynamic_mx_quant_data_compare.py # 精度比对
│   └── run_precision_check.py          # 全流程编排
├── Makefile                             # TYPE 分支构建规则
└── compile.all                          # 全量编译脚本（含 res_check）
```
