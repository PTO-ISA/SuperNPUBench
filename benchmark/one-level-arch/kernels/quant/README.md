# DynamicMxQuant — 动态 MX 量化

## 原 Ascend C 算子规格

**源路径**: `ops-nn/quant/dynamic_mx_quant`  
**Proto 定义**: `op_graph/dynamic_mx_quant_proto.h`  
**支持平台**: Ascend 950PR / Ascend 950DT（arch35）

### 功能说明

在给定的轴 `axis` 上，按每 `blocksize` 个元素分组，计算每组对应的量化尺度 `mxscale`，然后对组内每个元素除以 `mxscale`，按 `round_mode` 转换到目标类型 `dst_type`，得到量化结果 `y`。

### 算子 Proto 定义

**源文件**: `op_graph/dynamic_mx_quant_proto.h`

```cpp
REG_OP(DynamicMxQuant)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT4_E2M1, DT_FLOAT4_E1M2, DT_FLOAT6_E3M2, DT_FLOAT6_E2M3,
                           DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2}))
    .OUTPUT(mxscale, TensorType({DT_FLOAT8_E8M0}))
    .ATTR(axis, Int, -1)
    .ATTR(round_mode, String, "rint")
    .ATTR(dst_type, Int, DT_FLOAT4_E2M1)
    .ATTR(blocksize, Int, 32)
    .ATTR(scale_alg, Int, 0)
    .ATTR(dst_type_max, Float, 0.0)
    .ATTR(max_low_bound, Float, 0.0)
    .OP_END_FACTORY_REG(DynamicMxQuant)
```

### 算子属性（Attributes）

| 属性名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `axis` | Int | -1 | 量化发生的轴，取值范围 [-rank(x), rank(x)) |
| `round_mode` | String | "rint" | 舍入模式："rint"（四舍六入五成双）、"round"（四舍五入）、"floor"（向下取整） |
| `dst_type` | Int | DT_FLOAT4_E2M1 | 输出 y 的数据类型，支持 FLOAT4_E2M1 / FLOAT4_E1M2 / FLOAT8_E4M3FN / FLOAT8_E5M2 |
| `blocksize` | Int | 32 | 量化块大小，32 的倍数，≤1024，不为 0；scale_alg=2 时必须为 32 |
| `scale_alg` | Int | 0 | scale 算法：0=OCP MxFP8/MxFP4，1=NVIDIA cuBLAS MxFP8，2=Dynamic Dtype Range MxFP4 |
| `dst_type_max` | Float | 0.0 | 用户指定的目标类型最大值；仅 scale_alg=2 + FP4_E2M1 + blockSize=32 时生效；取值 0.0（使用类型默认最大值）或 6.0~12.0；其中 0.0/6.0/7.0 走 OCP 风格（指数位+尾数修正），其他值走 cuBLAS 风格（`S=Amax/dstTypeMax`） |
| `max_low_bound` | Float | 0.0 | 每个 block 的 Amax 下限钳位值；仅 scale_alg=1 时生效，scale_alg≠1 时必须为 0.0；必须非负 |

### 输入/输出张量

| 张量 | 输入/输出 | 数据类型 | Shape 约束 |
|------|----------|---------|-----------|
| `x` | 输入 | FLOAT16, BF16, FLOAT | ND 格式，rank 1-7 |
| `y` | 输出 | FLOAT4_E2M1, FLOAT4_E1M2, FLOAT8_E4M3FN, FLOAT8_E5M2 | 与 x 相同 |
| `mxscale` | 输出 | FLOAT8_E8M0 | rank = rank(x)+1，axis 维度偶数对齐，末维=2 |

### 约束条件

- 当 dst_type 为 FLOAT8_E5M2 或 FLOAT8_E4M3FN 时，round_mode 仅支持 "rint"
- 当 dst_type 为 FLOAT4_E2M1 或 FLOAT4_E1M2 时，round_mode 支持 "rint"、"floor"、"round"
- 当 dst_type 为 FLOAT4_E2M1 或 FLOAT4_E1M2 时，输入 x 的最后一维必须能被 2 整除
- blocksize 必须是 32 的倍数（非零），且 ≤ 1024
- 当 scale_alg=2 时，blocksize 必须为 32
- dst_type_max 仅支持 0.0 或 6.0~12.0，仅在 scale_alg=2 时生效
- 当 x 的数据类型为 FLOAT 时，blocksize 必须为 32，且量化轴维度不能小于 32

**mxscale shape 计算公式**：
```
rank(mxscale) = rank(x) + 1
axis_change = axis if axis >= 0 else axis + rank(x)
mxscale.shape[axis_change] = (ceil(x.shape[axis] / blocksize) + 1) / 2   // 偶数对齐
mxscale.shape[-1] = 2
其他维度与 x 一致
```

### 计算公式

所有 scaleAlg 的核心流程相同：在 axis 维度上按 blocksize 分组，计算每组的 mxscale，然后对组内元素缩放并转换到目标类型。差异在于 mxscale 的计算方式。

#### scaleAlg = 0（OCP MxFP8/MxFP4）

```
max_exp    = max_i(exponent_bits(V_i))                     // 提取每个元素的指数位，取最大值
shared_exp = max(max_exp, emax) - emax                     // 下限钳位 emax，保证 shared_exp ≥ 0
mxscale    = 2^shared_exp                                  // E8M0 格式
P_i        = cast_to_dst_type(V_i / mxscale, round_mode)
```

通过位操作提取 BF16/FP32 的指数位（`x & 0x7f80`），取块内最大指数，减去 emax 得到 shared_exp。

#### scaleAlg = 1（NVIDIA cuBLAS MxFP8，仅 FP8）

```
Amax    = max(|block_values|)
Amax    = max(Amax, maxLowBound)                           // V3 新增，maxLowBound > 0 时生效
S_fp32  = Amax * (1 / Amax(DType))                         // 浮点乘法
E_int   = floor(log2(S_fp32))                              // 提取 FP32 指数位
若 S_fp32 为正规数（0 < E_int < 254）且 mantissa(S_fp32) > 0，则 E_int += 1
若 S_fp32 为非正规数（E_int == 0）且 mantissa(S_fp32) > 0.5，则 E_int += 1
S_ue8m0 = 2^E_int
R_fp32  = 1 / fp32(S_ue8m0)
d_i     = DType(d_fp32_i * R_fp32)
```

对 `S_fp32` 的指数向上取整（尾数非零时），确保量化不溢出。非正规数处理仅在 FP8 输出路径中实现。

#### scaleAlg = 2（Dynamic Dtype Range MxFP4，仅 FP4_E2M1，blockSize=32）

根据 `dstTypeMax` 的取值分两种计算方式：

**当 dstTypeMax = 0.0 / 6.0 / 7.0 时**（OCP 风格 + 尾数修正）：

```
Amax_abs   = max_i(|V_i|)                                  // 绝对值最大值（含尾数位）
addValue   = dstTypeMax∈{0.0, 6.0} ? 0x003f : 0x001f      // 尾数修正值
corrected  = (Amax_abs + addValue) & expMask               // 加法进位到指数位，等效 ceil
shared_exp = max(corrected, FP4_E2M1_max_exp) - FP4_E2M1_max_exp
mxscale    = 2^shared_exp
P_i        = cast_to_dst_type(V_i / mxscale, round_mode)
```

修正值加到尾数上，若尾数足够大则进位到指数位，实现指数向上取整。

**当 dstTypeMax 为其他值（6.0~12.0 范围内，≠0/6/7）时**（cuBLAS 风格）：

```
Amax    = max(|block_values|)
S_fp32  = Amax / dstTypeMax                                // 使用用户指定的最大值
E_int   = floor(log2(S_fp32))
若 S_fp32 为正规数（0 < E_int < 254）且 mantissa(S_fp32) > 0，则 E_int += 1
S_ue8m0 = 2^E_int
R_fp32  = 1 / fp32(S_ue8m0)
d_i     = DType(d_fp32_i * R_fp32)
```

#### emax（目标类型最大正则数的指数位）

| dst_type       | emax |
|----------------|------|
| FLOAT4_E2M1    | 2    |
| FLOAT4_E1M2    | 0    |
| FLOAT8_E4M3FN  | 8    |
| FLOAT8_E5M2    | 15   |

### 场景矩阵

从公式和参数组合角度，有效场景由以下维度决定：

| 维度 | 取值范围 |
|------|---------|
| x 数据类型 | FLOAT16, BF16, FLOAT |
| dst_type | FP4_E2M1, FP4_E1M2, FP8_E4M3FN, FP8_E5M2 |
| scaleAlg | 0, 1, 2 |
| round_mode | rint, round, floor |
| axis | 尾轴 (-1), 非尾轴 |
| blockSize | 32, 64, 96, ..., 1024 |

**参数约束**（排除无效组合）：

| 约束 | 说明 |
|------|------|
| scaleAlg=1 → dst_type ∈ {FP8_E4M3FN, FP8_E5M2} | cuBLAS 风格仅支持 FP8 |
| scaleAlg=2 → dst_type = FP4_E2M1 | Dynamic Dtype Range 仅支持 FP4_E2M1 |
| scaleAlg=2 → blockSize = 32 | 必须为 32 |
| dst_type ∈ {FP8} → round_mode = rint | FP8 仅支持 rint |
| dst_type ∈ {FP4} → round_mode ∈ {rint, round, floor} | FP4 支持三种舍入模式 |
| x = FLOAT → blockSize = 32 | FP32 输入仅支持 blockSize=32 |
| x = FLOAT → axis 维度 ≥ 32 | FP32 输入量化轴不能小于 32 |
| dst_type ∈ {FP4} → x 末维为偶数 | FP4 输出要求输入末维对齐 |

**典型场景举例**：

| 场景 | x dtype | dst_type | scaleAlg | round_mode | axis | blockSize |
|------|---------|----------|----------|------------|------|-----------|
| OCP FP4 量化 | BF16 | FP4_E2M1 | 0 | rint | 尾轴 | 32 |
| OCP FP8 量化 | BF16 | FP8_E4M3FN | 0 | rint | 尾轴 | 32 |
| cuBLAS FP8 量化 | BF16 | FP8_E4M3FN | 1 | rint | 尾轴 | 32 |
| 动态范围 FP4 量化 | BF16 | FP4_E2M1 | 2 | rint | 尾轴 | 32 |
| OCP FP4 非尾轴量化 | BF16 | FP4_E2M1 | 0 | floor | 非尾轴 | 64 |
| FP32 输入 FP8 量化 | FLOAT | FP8_E4M3FN | 0 | rint | 尾轴 | 32 |

---

## 当前 PTO-ISA 实现

### 文件结构

```
kernels/quant/dynamic_mx_quant/
├── DESIGN.md                             # 设计文档
├── RECORD.md                             # 问题记录（TileSize 约束等）
├── dynamic_mx_quant_common.hpp           # 公共模块：类型、常量、scale 算法
├── dynamic_mx_quant_tail.hpp             # 尾轴入口：2D 循环 [M, K/blockSize]
└── dynamic_mx_quant_nontail.hpp          # 非尾轴入口：2D 循环 [Pre*Post, Axis/blockSize]
```

### 覆盖场景

| 维度 | 当前实现 |
|------|---------|
| 输入类型 | BF16 |
| 输出类型 | FP8_E4M3FN |
| axis | 尾轴 + 非尾轴 |
| blockSize | 32（TileM=8 时上限 512，TileM=4 时上限 1024） |
| scaleAlg | 0（OCP）、1（cuBLAS）、2（DynamicRange） |
| round_mode | rint |
| scale 输出类型 | uint16（E8M0 格式，stride=BlockSize padded） |
| M 维度尾块 | 支持（递归模板实例化，M % TileM != 0 时自动处理） |
| K 维度 | 必须为 BlockSize 的倍数 |
| 量化缩放 | `y = x × 2^(-E)`（精确 2 的幂次，与 Ascend C 数学等价） |

### 三种 Scale 算法实现

#### 1. OCP（scaleAlg=0）— `compute_ocp_scale`

**核心思路**：提取 BF16 指数位，取块内最大指数，计算 `shared_exp = max_exp - emax`，右移得到 scale byte。

**PTO-ISA 实现**：
```cpp
TANDS(exp_bits, x_u16, BF16_EXP_MASK);        // 提取指数位
TROWMAX(max_exp, exp_bits);                    // 块内最大指数
TCMPS(eq_nan, max_exp, BF16_EXP_MASK);         // NaN/Inf 检测
TMAXS(max_exp, max_exp, FP8_E4M3_EMAX);        // 下限钳位 emax
TSUB(shared_exp_out, max_exp, emax_tile);      // shared_exp = max_exp - emax
TSHRS(scale_byte, shared_exp_out, BF16_SHR_NUM); // scale_byte = shared_exp >> 7
TSEL(scale_byte, eq_nan, nan_byte);            // NaN/Inf → 0xFF
```

**指令数**：~10 条
**特点**：纯位操作，无浮点运算

#### 2. cuBLAS（scaleAlg=1）— `compute_cublas_scale`

**核心思路**：计算 `S = amax * inv_dst_max`，提取 FP32 指数，尾数非零时指数 +1（向上取整）。

**PTO-ISA 实现**：
```cpp
TABS(abs_x, x_f32);                            // 取绝对值
TROWMAX(max_abs, abs_x);                       // 块内最大绝对值（FP32）
TMULS(max_abs, max_abs, FP8_E4M3_INV_DST_MAX); // S = amax / dst_max
TCAST(s_bits, max_abs);                        // 重新解释为 uint32
TSHRS(exp_bits, s_bits, FP32_SHR_NUM);         // 提取指数位
TANDS(man_bits, s_bits, FP32_MANTISSA_MASK);   // 提取尾数位
TCMP(man_nz, man_bits, zero_u32);              // 尾数非零检测
TADDS(s_bits, exp_bits, 1);                    // exp + 1
TSEL(exp_bits, man_nz, s_bits);                // 尾数非零 → exp += 1
TCAST(scale_byte, exp_bits);                   // 转换为 uint8（E8M0）
TSHLS(shared_exp_out, scale_byte, 7);          // shared_exp = E << 7
```

**指令数**：~12 条
**特点**：浮点运算 + 位操作，尾数感知舍入

#### 3. DynamicRange（scaleAlg=2）— `compute_dynamic_range_scale`

**核心思路**：对最大绝对值加尾数修正值（0x003F），加法进位到指数位实现向上取整。

**PTO-ISA 实现**：
```cpp
TANDS(abs_x, x_u16, BF16_ABS_MASK);            // 取绝对值（位操作）
TROWMAX(max_abs, abs_x);                       // 块内最大绝对值
TADDS(max_abs, max_abs, ADD_VALUE);            // 加修正值（0x003F）
TANDS(exp_bits, max_abs, BF16_EXP_MASK);       // 提取进位后的指数
TCMPS(eq_nan, exp_bits, BF16_EXP_MASK);        // NaN/Inf 检测
TMAXS(exp_bits, exp_bits, FP8_E4M3_EMAX);      // 下限钳位 emax
TSUB(shared_exp_out, exp_bits, emax_tile);     // shared_exp = exp - emax
TSHRS(scale_byte, shared_exp_out, BF16_SHR_NUM); // scale_byte = shared_exp >> 7
TSEL(scale_byte, eq_nan, nan_byte);            // NaN/Inf → 0xFF
```

**指令数**：~11 条
**特点**：纯位操作，加法进位隐式实现指数向上取整

### 两种 Axis 模式

#### 1. 尾轴（axis=-1）— `dynamic_mx_quant_tail`

**输入形状**：`[M, K]`，量化轴为最后一维  
**循环结构**：2D 循环 `[M/TileM, K/BlockSize]`  
**内存访问**：连续访问（行主序，BlockSize 是最后一维的子段）

```cpp
for (int m = 0; m < kTM; ++m) {
    for (int kb = 0; kb < numKb; ++kb) {
        // 加载 [TileM, BlockSize] 连续数据
        // 计算 scale 和量化
        // 存储结果
    }
}
```

#### 2. 非尾轴（axis≠-1）— `dynamic_mx_quant_nontail`

**输入形状**：`[Pre, Axis, Post]`，量化轴为中间维度  
**循环结构**：3D 循环 `[Pre*Post/TileM, Axis/BlockSize]`（将 Pre*Post 视为行）  
**内存访问**：stride 访问（同一量化块内的元素间隔 Post 个元素）

```cpp
// 将 [Pre, Axis, Post] 视为 [Pre*Post, Axis] 的 2D 矩阵
for (int m = 0; m < kTM; ++m) {
    for (int kb = 0; kb < numKb; ++kb) {
        // 加载 [TileM, BlockSize] 数据（stride 访问）
        // 计算 scale 和量化
        // 存储结果
    }
}
```

### 算法流程（通用）

```
对每个 [TileM, BlockSize] 的 tile:
  1. TLOAD xq(bf16)
  2. compute_scale<Alg>(xq, scale_byte, shared_exp)  → 计算 E8M0 scale + shared_exp
  3. TCVT xf(fp32) ← xq(bf16)
  4. TXORS(neg_exp, shared_exp, 0xFFFF)              → 位翻转
  5. TADDS(inv_scale, neg_exp, 0x3F81)               → inv_scale = 2^(-E) 的 BF16 位模式
  6. TCAST(inv_bf16, inv_scale)                       → uint16 → bf16（位模式不变）
  7. TCVT(inv_scale_f, inv_bf16)                      → bf16 → fp32（浮点值转换）
  8. TMUL(xf, xf, inv_scale_f)                        → 逐元素乘：y = x × 2^(-E)
  9. TCAST(oq, xf)                                    → fp32 → fp8_e4m3fn
  10. TSTORE scale_byte 和量化结果
```

**量化缩放公式**：`y = x × 2^(-E)`，其中 E 由 scale 算法确定。这是精确的 2 的幂次缩放，与 Ascend C 参考实现数学等价。

### 模板参数

#### 尾轴 kernel

```cpp
template <int M, int K, ScaleAlg Alg = ScaleAlg::OCP, int TileM = 8, int BlockSize = 32>
void dynamic_mx_quant_tail(__bf16 *x, __fp8_e4m3 *y, uint16_t *scale);
```

- `M`: 行数（支持 M % TileM != 0，尾块自动递归处理）
- `K`: 列数（量化轴），须为 BlockSize 的倍数
- `Alg`: scale 算法枚举（OCP / CUBLAS / DYNAMIC_RANGE）
- `TileM`: 每次处理的行数，默认 8
- `BlockSize`: 量化块大小，默认 32

#### 非尾轴 kernel

```cpp
template <int Pre, int Axis, int Post, ScaleAlg Alg = ScaleAlg::OCP, int TileM = 8, int BlockSize = 32>
void dynamic_mx_quant_nontail(__bf16 *x, __fp8_e4m3 *y, uint16_t *scale);
```

- `Pre`: 量化轴之前的维度乘积
- `Axis`: 量化轴大小，须为 BlockSize 的倍数
- `Post`: 量化轴之后的维度乘积
- `Alg`: scale 算法枚举
- `TileM`: 每次处理的行数（Pre*Post 维度），默认 8
- `BlockSize`: 量化块大小，默认 32

### 与 OCP MX 标准的差异

| 项目 | OCP 标准 | 当前实现 | 状态 |
|------|---------|---------|------|
| scale 计算 | `2^(max_exp - emax)`（基于指数位） | 已实现（OCP/cuBLAS/DynamicRange 三种算法） | ✅ 已对齐 |
| 量化缩放 | `y = x × 2^(-E)`（精确 2 的幂次） | `TMUL(xf, xf, inv_scale_f)` 其中 inv_scale_f = 2^(-E) | ✅ 已对齐 |
| scale 类型 | FLOAT8_E8M0 | uint16（stride=BlockSize padded） | 简化输出布局 |
| scale shape | `[M, ceil(K/32)/2, 2]`（偶数对齐+交织） | `[M, K/BlockSize]`（stride=BlockSize padded） | 简化输出布局 |

### 待扩展

- [ ] FP4 输出（`__fp4_e1m2x2`）
- [ ] FP16 / FP32 输入支持
- [ ] K 维度尾块处理（K 非 BlockSize 倍数时的 padding）
- [ ] 多 round_mode 支持（round / floor）
- [ ] 非尾轴 stride 访问优化（批量 post 元素处理）
- [ ] scale 输出偶数对齐+交织布局（完全对齐 OCP 标准）
