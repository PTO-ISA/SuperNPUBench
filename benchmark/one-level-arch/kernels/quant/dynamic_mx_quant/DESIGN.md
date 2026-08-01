# DynamicMxQuant PTO-ISA 扩展设计文档

## 1. 概述

### 目标

将当前简化的 `dynamic_mx_quant_pto.hpp`（单文件、BF16→FP8、min-max 量化、尾轴、blockSize=32）扩展为完整的多文件实现，覆盖 Ascend C 参考实现中的三种 scale 算法、两种 axis 模式和多种输出类型。

### 设计原则

- **算法驱动划分**：文件按核心算法差异划分，而非实现分支
- **公共抽象复用**：类型定义、常量、scale 算法集中在 common 模块
- **模板参数控制变体**：dst_type、blockSize、round_mode 通过模板参数分发，不分文件

### 覆盖范围

| 维度 | 当前实现 | 目标实现 |
|------|---------|---------|
| 输入类型 | BF16 | BF16, FP16, FP32 |
| 输出类型 | FP8_E4M3FN | FP4_E2M1, FP4_E1M2, FP8_E4M3FN, FP8_E5M2 |
| scaleAlg | 简化 min-max | 0=OCP, 1=cuBLAS, 2=DynamicRange |
| axis | 尾轴 | 尾轴 + 非尾轴 |
| blockSize | 32 | 32, 64, ..., 1024 |
| round_mode | rint | rint, round, floor |
| scale 输出 | FP32 | E8M0（uint8 字节） |
| scale shape | [M, K/32] stride=8 | [M, K/blockSize] stride=8 |

---

## 2. 文件结构

```
kernels/quant/dynamic_mx_quant/
├── dynamic_mx_quant_common.hpp    # 公共模块：类型、常量、scale 算法
├── dynamic_mx_quant_tail.hpp      # 尾轴入口：2D 循环 [M, K/blockSize]
├── dynamic_mx_quant_nontail.hpp   # 非尾轴入口：3D 循环 [pre, axis/blockSize, post]
└── DESIGN.md                      # 本文档
```

### 文件职责

| 文件 | 职责 | 核心差异 |
|------|------|---------|
| `common` | tile 类型定义、常量表、三种 scale 算法实现 | 算法核心 |
| `tail` | 尾轴量化入口，2D 循环，连续内存访问 | 循环结构 |
| `nontail` | 非尾轴量化入口，3D 循环，stride 内存访问 | 循环结构 + 内存模式 |

### 与旧文件的关系

旧的 `kernels/quant/dynamic_mx_quant_pto.hpp` 保留作为对照参考，不参与新实现的编译。

---

## 3. Common 模块设计

### 3.1 模板参数体系

```cpp
// 入口函数模板参数
template <
    int M,                    // 行数（尾轴）或 pre_axis 大小（非尾轴）
    int K,                    // 量化轴大小
    int ScaleAlg,             // scale 算法：0=OCP, 1=cuBLAS, 2=DynamicRange
    typename InputDType,      // 输入类型：__bf16, half, float
    typename OutputDType,     // 输出类型：__fp8_e4m3, __fp8_e5m2, __fp4_e2m1x2, __fp4_e1m2x2
    int TileM = 8,            // 每次处理的行数
    int BlockSize = 32,       // 量化块大小
    int RoundMode = 0         // 舍入模式：0=rint, 1=round, 2=floor
>
```

### 3.2 常量表

常量按 `InputDType` 和 `OutputDType` 模板特化，编译期确定。

#### 3.2.1 输入类型相关常量

| 常量名 | BF16 | FP16 | FP32 | 说明 |
|--------|------|------|------|------|
| `EXP_MASK` | `0x7F80` | `0x7F80`（转 BF16 后） | `0x7F800000` | 指数位掩码 |
| `ABS_MASK` | `0x7FFF` | `0x7FFF` | `0x7FFFFFFF` | 绝对值掩码（去符号位） |
| `EXP_BIAS` | `0x7F00` | `0x7F00`（BF16 位模式） | `0x7F000000` | 2^127 的位模式 |
| `SHR_NUM` | `7` | `7` | `23` | 提取原始指数的右移位数 |
| `INVALID_EXP` | `0x7C00` | `0x7C00` | `0x7F800000` | NaN/Inf 指数位 |

#### 3.2.2 输出类型相关常量

| 常量名 | FP4_E2M1 | FP4_E1M2 | FP8_E4M3FN | FP8_E5M2 | 说明 |
|--------|----------|----------|------------|----------|------|
| `EMAX` | `0x0100` | `0x0000` | `0x0400` | `0x0780` | 目标类型最大正则数指数（BF16 位模式） |
| `DST_MAX` | `6.0f` | `3.5f` | `448.0f` | `57344.0f` | 目标类型最大可表示值 |
| `INV_DST_MAX` | `1/6.0f` | `1/3.5f` | `1/448.0f` | `1/57344.0f` | DST_MAX 的倒数 |
| `NAN_BYTE` | `0xFF` | `0xFF` | `0xFF` | `0xFF` | E8M0 NaN 哨兵值 |

#### 3.2.3 通用常量

| 常量名 | 值 | 说明 |
|--------|---|------|
| `SCALE_BIAS` | `0x7F00` | BF16 位模式中 2^127 的表示，用于构造 inv_scale |
| `NAN_PATTERN` | `0x7F81` | 自定义 NaN 哨兵（BF16 位模式） |
| `SPECIAL_EXP` | `0x0040` | sharedExp=127 时的 inv_scale（最小 BF16 次正规数） |
| `FP32_MANTISSA_MASK` | `0x007FFFFF` | FP32 尾数掩码（23 位） |
| `FP32_HALF_MANTISSA` | `0x00400000` | FP32 尾数中点（2^22），次正规数舍入阈值 |
| `ADD_VALUE_MAN1` | `0x003F` | 尾数修正值（dstTypeMax=0 或 6） |
| `ADD_VALUE_MAN2` | `0x001F` | 尾数修正值（dstTypeMax=7） |

### 3.3 Tile 类型定义

```cpp
// 输入 tile：从 GM 加载的原始数据
using tile_x    = Tile<Location::Vec, InputDType, TileM, BlockSize, BLayout::RowMajor>;

// FP32 计算 tile：精度转换后的中间数据
using tile_f    = Tile<Location::Vec, float, TileM, BlockSize, BLayout::RowMajor>;

// 整数 tile：位操作中间数据（指数提取、掩码等）
// 元素宽度与 InputDType 相同（BF16/FP16 → uint16_t，FP32 → uint32_t）
using tile_int  = Tile<Location::Vec, uint_type<sizeof(InputDType)>, TileM, BlockSize, BLayout::RowMajor>;

// 归约 tile：每行一个标量（amax、scale 等）
// 物理形状 [TileM, 8]，逻辑形状 [TileM, 1]
using tile_amax = Tile<Location::Vec, float, TileM, 8, BLayout::RowMajor, TileM, 1>;

// 整数归约 tile：位操作后的每行标量（max_exp 等）
using tile_amax_int = Tile<Location::Vec, uint_type<sizeof(InputDType)>, TileM, 8, BLayout::RowMajor, TileM, 1>;

// 输出 tile：量化结果
using tile_o    = Tile<Location::Vec, OutputDType, TileM, BlockSize, BLayout::RowMajor>;
```

### 3.4 Scale 算法

三种算法共享相同的接口签名，通过 `ScaleAlg` 模板参数分发。

#### 3.4.1 接口定义

```cpp
// 输入：amax_int（每行的整数位模式，含义因算法而异）
// 输出：scale_byte（E8M0 格式，每行 1 字节）
//       inv_scale（BF16 位模式，每行 1 个 uint16_t，用于缩放输入数据）
template <int ScaleAlg, typename TileAmaxInt, typename TileScaleByte, typename TileInvScale>
void compute_scale(TileAmaxInt &amax_int, TileScaleByte &scale_byte, TileInvScale &inv_scale);
```

#### 3.4.2 scaleAlg=0（OCP MxFP8/MxFP4）

**算法概述**：提取每个元素的指数位，取块内最大指数，计算 `shared_exp = max_exp - emax`，scale = 2^shared_exp。

**输入含义**：`amax_int` = 块内最大指数位（`x & EXP_MASK` 后 TROWMAX 归约）

**PTO-ISA 实现步骤**：

```
步骤 1: NaN/Inf 检测
  TCMP(ne_mask, amax_int, EXP_MASK)           // ne_mask = (amax_int != EXP_MASK) → 正常数据

步骤 2: emax 下限钳位
  TCMP(le_mask, amax_int, EMAX)              // le_mask = (amax_int <= EMAX)
  TSELECT(amax_int, le_mask, EMAX, amax_int) // if amax_int <= EMAX: amax_int = EMAX

步骤 3: 计算 shared_exp
  TSUB(shared_exp, amax_int, EMAX)           // shared_exp = amax_int - emax（BF16 位模式差）

步骤 4: 构造 forward scale（E8M0 字节）
  TSHRS(scale_byte, shared_exp, SHR_NUM)     // scale_byte = shared_exp >> 7 = 原始整数 E
  TSELECT(scale_byte, ne_mask, scale_byte, NAN_BYTE)  // NaN/Inf → 0xFF

步骤 5: 构造 inv_scale（BF16 位模式 = 2^(-E)）
  // inv_scale = SCALE_BIAS - shared_exp = 0x7F00 - shared_exp
  // PTO-ISA 无 TSUBSC（scalar - tile），需等价变换：
  //   0x7F00 - x = ~x + 0x7F01 = TXORS(x, 0xFFFF) + TADDS(0x7F01)
  TXORS(neg_exp, shared_exp, 0xFFFF)         // neg_exp = ~shared_exp
  TADDS(inv_scale, neg_exp, 0x7F01)          // inv_scale = ~shared_exp + 0x7F01 = 0x7F00 - shared_exp

步骤 6: inv_scale 特殊值处理
  TCMP(zero_mask, shared_exp, 0)             // zero_mask = (shared_exp == 0) → scale=1
  TCMP(eq_mask, shared_exp, SCALE_BIAS)      // eq_mask = (shared_exp == 0x7F00) → E=127
  TSELECT(inv_scale, ne_mask, inv_scale, NAN_PATTERN)   // NaN/Inf → NaN 哨兵
  TSELECT(inv_scale, zero_mask, 0, inv_scale)           // scale=1 → inv_scale=0
  TSELECT(inv_scale, eq_mask, SPECIAL_EXP, inv_scale)   // E=127 → 最小次正规数
```

**数据流**：
```
x(uint16) → TANDS(&EXP_MASK) → exp_bits(uint16)
  → TROWMAX → amax_int(uint16, [TileM,1])
  → compute_scale<0> → scale_byte(uint8), inv_scale(uint16)
```

#### 3.4.3 scaleAlg=1（NVIDIA cuBLAS MxFP8）

**算法概述**：计算 `S = amax * inv_dst_max`，提取 FP32 指数并向上取整（尾数非零时 +1），构造 scale = 2^E。

**输入含义**：`amax_int` = 块内最大绝对值（`x & ABS_MASK` 后 TROWMAX 归约）

**PTO-ISA 实现步骤**：

```
步骤 1: 转换为 FP32
  TCVT(amax_f, amax_int)                     // amax_f = (float)amax_int

步骤 2: NaN/Inf 和零检测
  TCAST(amax_u, amax_f)                      // amax_u = 重新解释为 uint
  TCMP(normal_mask, amax_u, EXP_MASK)        // normal = (amax_u < EXP_MASK) → 非 NaN/Inf
  TCMP(nz_mask, amax_f, 0.0f)                // nz_mask = (amax_f != 0)

步骤 3: 归一化
  TMULS(s_fp32, amax_f, INV_DST_MAX)         // S = amax / dst_max

步骤 4: 提取指数和尾数
  TCAST(s_bits, s_fp32)                      // s_bits = S 的 IEEE 位模式
  TSHRS(exp_bits, s_bits, FP32_SHR_NUM)      // exp = S 的偏置指数（>>23）
  TANDS(man_bits, s_bits, FP32_MANTISSA_MASK) // man = S 的尾数位

步骤 5: 指数向上取整
  // 条件 p0: 正规数 (0 < exp < 254) 且尾数非零 → exp += 1
  TCMP(gt0, exp_bits, 0)                     // exp > 0
  TCMP(lt254, exp_bits, 254)                 // exp < 254
  TCMP(man_nz, man_bits, 0)                  // mantissa != 0
  TAND(p0, gt0, lt254)                       // 0 < exp < 254
  TAND(p0, p0, man_nz)                       // AND mantissa != 0

  // 条件 p1: 次正规数 (exp == 0) 且尾数 > 0.5 → exp += 1
  TCMP(eq0, exp_bits, 0)                     // exp == 0
  TCMP(gt_half, man_bits, FP32_HALF_MANTISSA) // mantissa > 2^22
  TAND(p1, eq0, gt_half)

  // 合并舍入条件
  TOR(round_up, p0, p1)                      // round_up = p0 | p1

  // 应用舍入
  TADDS(exp_plus1, exp_bits, 1)              // exp + 1
  TSELECT(extract_exp, round_up, exp_plus1, exp_bits)  // if round_up: exp+1, else exp

步骤 6: NaN/Inf 和零覆盖
  TSELECT(extract_exp, normal_mask, extract_exp, 0xFF)  // NaN/Inf → 0xFF
  TSELECT(extract_exp, nz_mask, extract_exp, 0)         // max==0 → 0

步骤 7: 构造 forward scale（E8M0 字节）
  TCAST(scale_byte, extract_exp)             // 打包为 uint8

步骤 8: 构造 inv_scale（BF16 位模式）
  TSHLS(exp_bf16, extract_exp, 7)            // 移入 BF16 指数位
  // inv_scale = 0x7F00 - exp_bf16（同 OCP 步骤 5 的等价变换）
  TXORS(neg_exp, exp_bf16, 0xFFFF)
  TADDS(inv_scale, neg_exp, 0x7F01)

步骤 9: inv_scale 特殊值处理
  TSELECT(inv_scale, normal_mask, inv_scale, NAN_PATTERN)  // NaN/Inf → NaN
  TSELECT(inv_scale, nz_mask, inv_scale, 0)                // max==0 → 0
```

**数据流**：
```
x(uint16) → TANDS(&ABS_MASK) → abs_x(uint16)
  → TROWMAX → amax_int(uint16, [TileM,1])
  → TCVT → amax_f(float)
  → compute_scale<1> → scale_byte(uint8), inv_scale(uint16)
```

#### 3.4.4 scaleAlg=2（Dynamic Dtype Range MxFP4）

**算法概述**：对最大绝对值加尾数修正值后取指数位（加法进位实现指数向上取整），计算 `shared_exp = corrected_exp - emax`。

**输入含义**：`amax_int` = 块内最大绝对值（同 cuBLAS）

**约束**：仅支持 FP4_E2M1 + blockSize=32

**PTO-ISA 实现步骤**：

```
步骤 1: NaN/Inf 检测
  TANDS(exp_only, amax_int, EXP_MASK)        // 提取指数位
  TCMP(normal_mask, exp_only, EXP_MASK)      // normal = (exp != 0xFF)

步骤 2: emax 下限检测
  TCMP(lt_emax, exp_only, EMAX)             // lt_emax = (exp < emax)

步骤 3: 尾数修正 + 指数进位
  // addValue = ADD_VALUE_MAN1 (dstTypeMax=0/6) 或 ADD_VALUE_MAN2 (dstTypeMax=7)
  TADDS(amax_add, amax_int, ADD_VALUE)       // 加修正值到尾数
  TANDS(corrected, amax_add, EXP_MASK)       // 提取进位后的指数

步骤 4: emax 下限钳位
  TSELECT(corrected, lt_emax, EMAX, corrected)  // if exp < emax: clamp to emax

步骤 5: 计算 shared_exp
  TSUB(shared_exp, corrected, EMAX)          // shared_exp = corrected - emax

步骤 6-9: 构造 scale 和 inv_scale（与 OCP 步骤 4-6 完全相同）
  TSHRS(scale_byte, shared_exp, SHR_NUM)
  TSELECT(scale_byte, normal_mask, scale_byte, NAN_BYTE)
  TXORS(neg_exp, shared_exp, 0xFFFF)
  TADDS(inv_scale, neg_exp, 0x7F01)
  TCMP(zero_mask, shared_exp, 0)
  TCMP(eq_mask, shared_exp, SCALE_BIAS)
  TSELECT(inv_scale, normal_mask, inv_scale, NAN_PATTERN)
  TSELECT(inv_scale, zero_mask, 0, inv_scale)
  TSELECT(inv_scale, eq_mask, SPECIAL_EXP, inv_scale)
```

**数据流**：
```
x(uint16) → TANDS(&ABS_MASK) → abs_x(uint16)
  → TROWMAX → amax_int(uint16, [TileM,1])
  → compute_scale<2> → scale_byte(uint8), inv_scale(uint16)
```

#### 3.4.5 三种算法对比

| 维度 | OCP (0) | cuBLAS (1) | DynamicRange (2) |
|------|---------|------------|-------------------|
| 归约输入 | 指数位 (`x & EXP_MASK`) | 绝对值 (`x & ABS_MASK`) | 绝对值 (`x & ABS_MASK`) |
| 归约操作 | TROWMAX（整数） | TROWMAX（整数） | TROWMAX（整数） |
| 核心计算 | `max_exp - emax` | `ceil(log2(amax * inv_max))` | `(amax + addValue) & EXP_MASK - emax` |
| 尾数感知 | 否 | 是（显式提取+条件+1） | 是（加法进位隐式） |
| 次正规数处理 | 否 | 是（exp=0 且 man>0.5） | 否 |
| 适用输出 | FP4 + FP8 | FP4 + FP8 | 仅 FP4_E2M1 |
| 指令数（约） | ~10 条 | ~20 条 | ~12 条 |

### 3.5 量化主计算

三种 scale 算法共享相同的量化主流程（在 tail/nontail 入口中调用）：

```
步骤 1: 加载 + 精度转换
  TLOAD(xq, gx)                              // 从 GM 加载输入 tile
  TCVT(xf, xq)                               // 转换为 FP32

步骤 2: 计算 amax（整数位模式）
  TCAST(xi, xq)                              // 重新解释为整数
  TANDS(amax_input, xi, MASK)                // OCP: &EXP_MASK, cuBLAS/DR: &ABS_MASK
  TROWMAX(amax_int, amax_input)              // 行归约取最大值

步骤 3: 计算 scale
  compute_scale<ScaleAlg>(amax_int, scale_byte, inv_scale)

步骤 4: 缩放输入数据
  // inv_scale 是 BF16 位模式（uint16），需转换为 FP32 浮点值
  TCVT(inv_scale_f, inv_scale)               // BF16 位模式 → FP32 浮点值
  TROWEXPANDMUL(outf, xf, inv_scale_f)       // 行广播乘：outf[i,j] = xf[i,j] * inv_scale[i]

步骤 5: 类型转换 + 存储
  TCAST(oq, outf)                            // FP32 → 目标类型
  TSTORE(gs, scale_byte)                     // 存储 scale（E8M0 字节）
  TSTORE(gy, oq)                             // 存储量化结果
```

**注意**：步骤 4 中 `inv_scale` 是 BF16 位模式（uint16），直接 TCVT 会将其作为整数值转换而非 BF16 浮点值。需要先将 uint16 位模式重新解释为 BF16，再转换为 FP32：

```
TCAST(inv_scale_bf16, inv_scale)             // uint16 → BF16（位模式不变）
TCVT(inv_scale_f, inv_scale_bf16)            // BF16 → FP32（浮点值转换）
```

---

## 4. 尾轴 Kernel 设计

### 4.1 入口签名

```cpp
template <int M, int K, int ScaleAlg, typename InputDType, typename OutputDType,
          int TileM = 8, int BlockSize = 32, int RoundMode = 0>
void dynamic_mx_quant_tail(InputDType *x, OutputDType *y, uint8_t *scale);
```

### 4.2 循环结构

```
kTM = M / TileM                              // M 方向 tile 数
numKb = K / BlockSize                         // K 方向量化块数

for m in [0, kTM):                           // 外层：行方向
  for kb in [0, numKb):                      // 内层：量化块方向
    gx = x_iter(m, kb)                       // 输入 tile 地址
    gy = y_iter(m, kb)                       // 输出 tile 地址
    gs = s_iter(m, kb)                       // scale 地址
    quantize_tile<ScaleAlg, ...>(gx, gy, gs) // 调用 common 中的量化主计算
```

### 4.3 GM 张量布局

```cpp
// 输入：[M, K]，行主序，连续
using gm_x = global_tensor<InputDType, RowMajor<M, K>>;

// 输出：[M, K]，行主序，连续
using gm_y = global_tensor<uint8_t, RowMajor<M, K * sizeof(OutputDType)>>;

// scale：[M, numKb * kScaleStride]，stride=8 padded
// 每行 numKb 个量化块，每个块的 scale 占 8 字节（仅首字节有效）
using gm_s = global_tensor<uint8_t, RowMajor<M, numKb * kScaleStride>>;
```

### 4.4 内存访问模式

尾轴量化的内存访问是**完全连续**的：
- 输入 tile `[TileM, BlockSize]` 在 GM 中连续存储（行主序，BlockSize 是最后一维的子段）
- 输出 tile 同样连续
- scale 按 stride=8 存储，每个量化块写 1 字节

---

## 5. 非尾轴 Kernel 设计

### 5.1 入口签名

```cpp
template <int Pre, int Axis, int Post, int ScaleAlg, typename InputDType, typename OutputDType,
          int TilePre = 8, int BlockSize = 32, int RoundMode = 0>
void dynamic_mx_quant_nontail(InputDType *x, OutputDType *y, uint8_t *scale);
```

### 5.2 循环结构

```
kTP = Pre / TilePre                          // pre 方向 tile 数
numKb = Axis / BlockSize                      // axis 方向量化块数

for p in [0, kTP):                           // 外层：pre_axis 方向
  for kb in [0, numKb):                      // 中层：量化块方向
    for post in [0, Post):                   // 内层：post_axis 方向
      // 计算 GM 地址（stride 访问）
      gx = x + (p * TilePre) * (Axis * Post) + (kb * BlockSize) * Post + post
      gy = y + ...
      gs = scale + ...
      quantize_tile<ScaleAlg, ...>(gx, gy, gs)
```

### 5.3 内存访问模式

非尾轴量化的内存访问是**stride 访问**：
- 同一量化块内的 BlockSize 个元素在内存中**不连续**，间隔 `Post` 个元素
- 无法使用单次 TLOAD 加载完整的 `[TilePre, BlockSize]` tile

### 5.4 处理策略

**方案 A：逐 post 元素处理（简单，初始实现）**

对每个 post 元素单独加载 `[TilePre, BlockSize]` 的连续数据：
- 将 tensor 视为 `[Pre * Post, Axis]` 的 2D 矩阵
- 每行是 `(pre, post)` 组合，沿 axis 方向连续
- 量化块变为 `[TilePre * 1, BlockSize]`（每次只处理 1 个 post 元素）

```
for p in [0, Pre):
  for post in [0, Post):
    for kb in [0, numKb):
      // 加载 [1, BlockSize] 连续数据
      row = p * Post + post
      gx = x_iter(row, kb)
      quantize_tile(...)
```

**方案 B：批量 post 处理（优化，后续实现）**

将多个 post 元素打包到一个 tile 中：
- tile 形状 `[TilePre, BlockSize * PostBatch]`
- 加载后按 post 分组处理
- 减少 TLOAD 次数，提高内存带宽利用率

**初始实现采用方案 A**，后续可根据性能需求升级到方案 B。

### 5.5 GM 张量布局

```cpp
// 输入：[Pre, Axis, Post]，行主序
using gm_x = global_tensor<InputDType, RowMajor<Pre * Post, Axis>>;

// 输出：[Pre, Axis, Post]，行主序
using gm_y = global_tensor<uint8_t, RowMajor<Pre * Post, Axis * sizeof(OutputDType)>>;

// scale：[Pre * Post, numKb * kScaleStride]
using gm_s = global_tensor<uint8_t, RowMajor<Pre * Post, numKb * kScaleStride>>;
```

---

## 6. PTO-ISA Intrinsic 依赖

### 6.1 必需 Intrinsic 清单

| 类别 | Intrinsic | 用途 |
|------|-----------|------|
| 加载/存储 | TLOAD, TSTORE | GM ↔ UB 数据搬运 |
| 类型转换 | TCVT, TCAST | 精度转换、位模式重解释 |
| 算术 | TMULS, TADDS, TSUB, TRECIP | 标量乘、加法、减法、倒数 |
| 位操作 | TANDS, TORS, TXORS | 掩码提取、位翻转 |
| 移位 | TSHRS, TSHLS | 指数提取、位模式构造 |
| 归约 | TROWMAX | 行最大值（支持整数 tile） |
| 比较 | TCMP | 条件掩码生成 |
| 选择 | TSELECT | 条件赋值 |
| 广播 | TROWEXPANDMUL | 行广播乘法 |
| 绝对值 | TABS | 浮点绝对值（min-max 量化备选） |

### 6.2 关键约束

- **位操作仅支持整数类型**：TANDS、TORS、TXORS、TSHRS、TSHLS 的 tile 元素必须是 `int8/16/32/64` 或 `uint8/16/32/64`
- **TROWMAX 支持整数 tile**：通过 `type_traits<DType>::TypeCode` 分发，无整数限制
- **TSELECT 需要匹配类型**：条件 mask 和 src0/src1 的元素类型必须一致
- **TCMP 输出类型**：产生 packed predicate/mask tile，需确认与 TSELECT 的 cond 参数兼容

---

## 7. 测试计划

### 7.1 测试矩阵

| 测试维度 | 取值 | 组合数 |
|---------|------|--------|
| ScaleAlg | 0, 1, 2 | 3 |
| InputDType | BF16, FP16 | 2 |
| OutputDType | FP8_E4M3FN, FP4_E2M1 | 2 |
| Axis | 尾轴, 非尾轴 | 2 |
| BlockSize | 32 | 1 |
| **总计** | | **24**（排除无效组合后约 16） |

### 7.2 验证方法

- **编译验证**：所有组合通过 `make TESTCASE=... diss` 编译成功
- **功能验证**：gfrun 通过（编译测试模式，取函数地址）
- **精度验证**（后续）：与 Ascend C 参考实现的输出对比

### 7.3 测试文件结构

```
test/kernel/quant/dynamic_mx_quant/
├── src/
│   └── dynamic_mx_quant.cpp     # 测试入口（编译测试模式）
├── Makefile                      # 构建规则
└── compile.all                   # 全量编译脚本
```

---

## 8. 实现计划

### Phase 1：Common 模块 + 尾轴 OCP（当前阶段）

1. 实现 `dynamic_mx_quant_common.hpp`
   - 常量表（模板特化）
   - tile 类型定义
   - `compute_scale<0>`（OCP 算法）
2. 实现 `dynamic_mx_quant_tail.hpp`
   - 2D 循环结构
   - 调用 common 中的量化主计算
3. 编译验证 BF16 → FP8_E4M3FN + scaleAlg=0 + 尾轴

### Phase 2：cuBLAS + DynamicRange

4. 在 common 中添加 `compute_scale<1>`（cuBLAS）
5. 在 common 中添加 `compute_scale<2>`（DynamicRange）
6. 编译验证三种 scaleAlg

### Phase 3：多类型支持

7. 添加 FP16 输入支持
8. 添加 FP4 输出支持
9. 编译验证所有类型组合

### Phase 4：非尾轴

10. 实现 `dynamic_mx_quant_nontail.hpp`
11. 编译验证非尾轴场景

---

## 9. 待确认事项

| 编号 | 事项 | 影响 |
|------|------|------|
| 1 | TCMP 输出 mask 与 TSELECT cond 参数的类型兼容性 | 所有 scale 算法的条件分支 |
| 2 | TCAST uint16 ↔ BF16 位模式重解释是否正确 | inv_scale 构造 |
| 3 | TROWMAX 对 uint16_t tile 的实际行为 | 整数归约 |
| 4 | TSHLS 对 uint16_t tile 的实际行为 | cuBLAS inv_scale 构造 |
| 5 | scale 输出的 E8M0 字节在 GM 中的对齐要求 | TSTORE 地址计算 |
| 6 | 非尾轴方案 A 的 TLOAD 是否支持 stride 地址 | 非尾轴 kernel 可行性 |
