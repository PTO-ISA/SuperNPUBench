# DynamicMxQuant PTO-ISA 扩展设计文档

## 1. 概述

### 目标

将 DynamicMxQuant 实现为完整的多文件 PTO-ISA kernel，覆盖 Ascend C 参考实现中的三种 scale 算法、两种输出 dtype（FP8/FP4）与两种 axis 模式，并按 AscendC 的 **scaleAlg↔dstType 合法性**逐场景专门化。

### 设计原则

- **场景专门化划分**：每个 kernel 文件只实现**一个** `(axis, scaleAlg, dstType)` 场景，文件内**无** `if constexpr(Alg)` 分支。可复用的常量、类型 trait、scale 算法、finalize、内存别名重解释集中在 `common.hpp`。
- **对齐 AscendC 为唯一基准**：三种 scale 算法逐 op 对齐 AscendC `ComputeScale{Ocp,Cublas,DynamicDtypeRange}`，golden 亦为其 Python 移植。
- **scaleAlg↔dstType 合法性对齐 AscendC**：合法组合来自 `op_host/arch35/dynamic_mx_quant_tiling_arch35.cpp`——非法组合（cuBLAS-FP4、DynRange-FP8）不生成 kernel。
- **emax 由输出 dtype 派生**：`emax` 是输出 dtype 最大正则数的（<<7）指数域位模式，通过 `constexpr` trait `emax_bits<OutT>()`（对应 AscendC `GetFp8MaxExp<T>()` / `GetFp4MaxExp<T>()`）从输出类型推导，**不作为调用方自由参数**。FP8_E4M3=`0x0400`，FP4_E2M1=`0x0100`。
- **公式一致性**：量化缩放使用 `y = x × 2^(-E)`（精确 2 的幂次），reciprocal 由 `finalize_recip_u16`（OCP 新算法）/ `finalize_scale_recip_u16`（DynRange） / `compute_cublas_core`（cuBLAS）的 bf16 位模式给出。
- **无寄存器 bitcast**：linx（`-D__linx`）无 `TCAST`（`#ifndef __linx`）。float↔int 的位重解释只能经 scratch-HBM 同宽度别名（`TSTORE` 后以另一 dtype `TLOAD`），见 `reinterpret_u16_to_bf16` / `reinterpret_f32_to_u32`。
- **两遍结构降低寄存器压力**：镜像 AscendC 的 ComputeScale→ComputeData 分离——先算 scale/recip 并立即 `TSTORE` scale_byte，再在数据遍加载 bf16 值。避免同时存活过多 tile 触发 LinxV5 对 <512B tile 的溢出断言（`LinxV5RegisterInfo.cpp:403`）。

### scaleAlg × dstType 合法性（对齐 AscendC）

| 输出 dtype | scaleAlg=0 OCP | scaleAlg=1 cuBLAS | scaleAlg=2 DynRange |
|---|:--:|:--:|:--:|
| FP4_E2M1 | ✓ | ✗ | ✓ |
| FP4_E1M2 | ✓ | ✗ | ✗ |
| FP8_E4M3FN | ✓ | ✓ | ✗ |
| FP8_E5M2 | ✓ | ✓ | ✗ |

本次交付取每算法一个代表性合法 dtype：**OCP→{FP8_E4M3, FP4_E2M1}**、**cuBLAS→FP8_E4M3**、**DynRange→FP4_E2M1**。× {tail, nontail} = **8 个 kernel 配置**。

### 覆盖范围

| 维度 | 当前实现 |
|------|---------|
| 输入类型 | BF16（16bit 域）、FP16 / FP32（32bit 域，FP16 先 upcast 到 FP32）——对齐 AscendC `INPUT_SUPPORT_DTYPE_SET`（tiling `:70`） |
| 输出类型 | FP8_E4M3FN（OCP/cuBLAS × tail/nontail = 4 配置）、FP4_E2M1（OCP/DynRange × tail/nontail = 4 配置） |
| scaleAlg | 0=OCP, 1=cuBLAS, 2=DynamicRange（三种全部保留，按合法性配对 dtype） |
| axis | 尾轴 + 非尾轴 |
| emax | 由 `emax_field<OutT, Domain>()` 从输出 dtype 派生、按输入计算域宽移位（16bit: FP8_E4M3=0x0400 / FP4_E2M1=0x0100；32bit: `<<23`，如 FP4_E2M1=0x01000000） |
| blockSize | 32（TileM=8 时上限 512，TileM=4 时上限 1024） |
| round_mode | rint |
| scale 输出 | FLOAT8_E8M0（1 字节/块）。归约轴块数 `numKb=归约维/BlockSize` 偶数对齐。**PTO-ISA 契约（ADR-0101）= 纯 planar，无交织**：尾轴 `[..., evenAlign(K/BlockSize)]`（A-scale `[group_M,G]`）、非尾轴 `[evenAlign(Axis/BlockSize), Post]`（B-scale `[G,N]`）。AscendC 的 parity 交织非 PTO-ISA 契约（RECORD 问题5 已解除）。详见 §4.3 / §5.3 |
| M 维度尾块 | tail 支持（boxed 部分 tile：物理 TileM×BlockSize + ValidRow=M_tail） |
| K 维度 | 必须为 BlockSize 的倍数 |

---

## 2. 文件结构

```
kernels/multi_thread/quant/dynamic_mx_quant/
├── dynamic_mx_quant_common.hpp             # 公共模块：常量、emax trait、6 个 scale 算法、finalize、reinterpret
├── dynamic_mx_quant_tail_ocp_fp8.hpp       # 尾轴 · OCP · FP8_E4M3
├── dynamic_mx_quant_tail_cublas_fp8.hpp    # 尾轴 · cuBLAS · FP8_E4M3
├── dynamic_mx_quant_tail_ocp_fp4.hpp       # 尾轴 · OCP · FP4_E2M1
├── dynamic_mx_quant_tail_dynrange_fp4.hpp  # 尾轴 · DynRange · FP4_E2M1
├── dynamic_mx_quant_nontail_ocp_fp8.hpp    # 非尾轴 · OCP · FP8_E4M3
├── dynamic_mx_quant_nontail_cublas_fp8.hpp # 非尾轴 · cuBLAS · FP8_E4M3
├── dynamic_mx_quant_nontail_ocp_fp4.hpp    # 非尾轴 · OCP · FP4_E2M1
├── dynamic_mx_quant_nontail_dynrange_fp4.hpp # 非尾轴 · DynRange · FP4_E2M1
├── DESIGN.md                               # 本文档（预期全功能设计）
├── README.md                               # 当前已实现状态
└── RECORD.md                               # 问题记录（TileSize / 32B 对齐 / fp4 tile 切分 / linx 缺口）
```

每个 kernel 文件是**自包含**的：调用 `common.hpp` 中对应的单个 `compute_*_scale_{tail,not_tail}`，完成 ComputeScale→ComputeData 两遍主流程，不再有跨算法分支。

---

## 3. Common 模块设计

### 3.1 emax / inv_dst_max trait（由输出 dtype 派生，按输入域宽移位）

emax 是「输出 dtype 最大正则数的**无偏**指数」，与输出 dtype 绑定；再按**输入计算域宽**移进指数位域。

```cpp
template <typename OutT> constexpr int unbiased_emax();     // 无偏值：E4M3=8 E5M2=15 E2M1=2 E1M2=0

// Domain ∈ {16,32}：16bit(bf16 位域, <<7) / 32bit(fp32 位域, <<23)
template <typename OutT, int Domain> constexpr uint32_t emax_field();  // 对应 GetFp4MaxExp/GetFp8MaxExp
// 16bit: E4M3=0x0400 E5M2=0x0780 E2M1=0x0100 E1M2=0x0000
// 32bit: 无偏值 <<23，如 E2M1=0x01000000（= AscendC FP4_E2M1_FP32_MX_MAX_EXP）

template <typename OutT> constexpr float inv_dst_max();     // cuBLAS 专用（FP8-only）
// __fp8_e4m3 -> 1/448   __fp8_e5m2 -> 1/57344
```

各 kernel 与 scale 算法内部一律用 `emax_field<OutT, Domain>()`，**不接收 emax 调用方参数**——严格镜像 AscendC 由输出 tile 元素类型推导 emax、再按输入域宽（`GetFp4MaxExp<uint16_t/uint32_t>`）选位域形式的做法。

**输入 dtype 与指数域宽**（对齐 AscendC `common.h:124-192` 的二元域选择 `IsSame<T,uint16_t>`——不维护三套布局，只有 16/32bit 两套）：

| 计算域 | 输入 dtype | 指数掩码 | 移位 | recip bias | E2M1 emax_field |
|---|---|:--:|:--:|:--:|:--:|
| 16bit | bf16（= 截断 fp32，指数同为 8 位 bias127，直接在 16bit 位域算） | `0x7f80` | 7 | `0x7f00` | `0x0100` |
| 32bit | fp16 → 先 upcast 到 fp32；fp32 直接 | `0x7f800000` | 23 | fp32 bias | `0x01000000` |

> fp16 自身的 5 位指数 / bias15 布局**从不使用**——upcast 到 fp32（精确，fp16⊂fp32）后复用 32bit 路径，避免第三套常量。bf16 因指数域与 fp32 完全相同，可直接在 16bit 位域算而不必 upcast。

> **上表的域宽按算法区分（关键点）**：**cuBLAS** 的 scale 计算**恒在 32bit 域**（fp32 输入时 maxExp 用 uint32；bf16/fp16 输入时 16bit maxExp 也一律上转到 32bit 再算），最后缩回 uint16 存 scale/recip。**OCP / DynRange** 的 scale finalize **恒在 16bit 域**（emax 用 16bit 值）——32bit 那一行仅作用于 **fp32 输入时的 max-exp 抽取**（读 `0x7f800000`、uint32 `ReduceMax`），随即 `ShiftRights(16)+Pack` 缩回 bf16 域 uint16，finalize 不进 32bit。两算法的 **ComputeData 都提升到 fp32**。

### 3.2 常量表

#### 3.2.1 BF16 相关常量

| 常量名 | 值 | 说明 |
|--------|---|------|
| `BF16_EXP_MASK` | `0x7F80` | BF16 指数位掩码 |
| `BF16_ABS_MASK` | `0x7FFF` | BF16 绝对值掩码（去符号位） |
| `BF16_EXP_BIAS` | `0x7F00` | BF16 指数偏置（recip 基值） |
| `BF16_SHR_NUM` | `7` | 提取原始指数的右移位数 |
| `BF16_NAN_PATTERN` | `0x7F81` | 自定义 NaN 哨兵（BF16 位模式） |
| `BF16_SPECIAL_EXP` | `0x0040` | sharedExp=0x7F00 时的 recip |
| `BF16_ADD_VALUE_MAN1` | `0x003F` | DynRange 尾数向上取整修正值 |

#### 3.2.2 输出 dtype emax / max-normal 常量

| 常量名 | 值 | 说明 |
|--------|---|------|
| `FP8_E4M3_EMAX` | `0x0400` | FP8 E4M3 最大正则数指数（BF16 位模式，448） |
| `FP8_E5M2_EMAX` | `0x0780` | FP8 E5M2（57344） |
| `FP4_E2M1_EMAX` | `0x0100` | FP4 E2M1 最大正则数指数（6） |
| `FP4_E1M2_EMAX` | `0x0000` | FP4 E1M2 |
| `FP8_E4M3_INV_DST_MAX` | `1/448.0f` | cuBLAS 1/dstMax |
| `FP8_NAN_BYTE` | `0x00FF` | E8M0 NaN 哨兵值 |

#### 3.2.3 FP32 相关常量（cuBLAS 路径）

| 常量名 | 值 | 说明 |
|--------|---|------|
| `FP32_EXP_MASK` | `0x7F800000` | FP32 指数位掩码（finite 判据） |
| `FP32_MANTISSA_MASK` | `0x007FFFFF` | FP32 尾数掩码（23 位） |
| `FP32_EXP_BIAS_CUBLAS` | `0x00007F00` | cuBLAS recip 基值（低 16 位 = bf16 bias） |
| `FP32_NUMBER_254` | `0x000000FE` | 指数上界守卫（exp<254） |
| `FP32_NUMBER_HALF` | `0x00400000` | 尾数 0.5 守卫（p1） |
| `FP32_SHR_NUM` | `23` | 提取 FP32 指数的右移位数 |
| `CLAMP_MIN` | `1e-12f` | amax 下限钳位（maxLowBound_） |

### 3.3 Scale 算法（6 个，无分发器）

每种算法有 `_tail`（沿 BlockSize 列归约，用 `TROWMAX`）与 `_not_tail`（沿 BlockSize 行归约，用 `TCOLMAX`）两个变体，均输出 `(scale_byte, recip_out)`，`emax` 经 `emax_bits<OutT>()` 内部派生。**不再有 `compute_scale_{tail,not_tail}` 分发器**——各 kernel 直接调用其唯一算法。

```cpp
template <typename OutT, int TileM, int BlockSize>
void compute_ocp_scale_tail(x_u16, scale_byte, recip_out);            // 位视图
template <typename OutT, int TileM, int BlockSize>
void compute_cublas_scale_tail(x_f32, scale_byte, recip_out);         // 值视图（FP32）
template <typename OutT, int TileM, int BlockSize>
void compute_dynamic_range_scale_tail(x_u16, scale_byte, recip_out); // 位视图
// _not_tail 三个同构（TROWMAX→TCOLMAX；cuBLAS not_tail 取 bf16 值视图 + TCVT→fp32）
```

`recip_out` 与 `scale_byte` 的收尾按算法分三条路径：

- **当前 OCP（scaleAlg=0）**：已改为 **bf16→e8m0 直转**——scale 字节由 `Cast<bf16→e8m0>` 直接产出（**不再走**下面的 `>>7` + `eq_inf` select），recip 由 `finalize_recip_u16` 产出，详见 §3.3.1。
- **DynRange（scaleAlg=2）及已过时的旧版 OCP**：走公共 `finalize_scale_recip_u16(shared_exp, eq_inf, ...)`（下面的收尾）。
- **cuBLAS（scaleAlg=1）**：走 `compute_cublas_core`。

`finalize_scale_recip_u16` 收尾（DynRange / 旧 OCP，镜像 AscendC）：
```
scale_byte = shared_exp >> 7 ; select(eq_inf ? 0x00FF : scale_byte)
recip      = 0x7F00 - shared_exp
             select(eq_inf         ? 0x7F81 : recip)   // NaN
             select(shared_exp==0  ? 0      : recip)   // 下溢归零
             select(shared_exp==0x7F00 ? 0x0040 : recip)
```

#### 3.3.1 scaleAlg=0（OCP MxFP8/MxFP4）— bf16→e8m0 直转（当前算法）

> 本节以「0 代码」为基准、以**预期实现**为目标描述当前 OCP scale 的设计。对应实现见
> `common.hpp` `ocp_scale_mulcast_from_maxexp` / `finalize_recip_u16`，镜像 AscendC
> `dynamic_mx_quant_tail_axis_fp8_ocp_new.h` 的 `ComputeScaleOcp`（`Mul` + `Cast<e8m0>`）。
> 旧的「clamp 到 emax 再减 emax + `>>7`」路径见 §3.3.1-旧（已过时）。

**核心思路**：硬件支持 bf16→e8m0 的 `Cast` 直转（e8m0 存的就是 bf16 值的有偏指数字段），
且 OCP 的 shared scale 恒为 2 的幂（零尾数）——故不必再在整数指数域手工 `clamp`/减法/移位：
一次 bf16 乘法得到 `2^(E_max-emax)`，一次 `Cast<bf16→e8m0>` 直接落成 scale 字节。emax 仍由
`emax_bits<OutT>()` 派生（FP8=0x0400 / FP4=0x0100），是 OCP 同时支持 FP8 与 FP4 的唯一区别。

```
提取指数位并归约得每块 max_exp        // TANDS(0x7F80) + TROWMAX/TCOLMAX（有偏 bf16 指数域，2^E_max 位型）
max_bf     = reinterpret<bf16>(max_exp)   // 位重解释：2^E_max 的 bf16 值
shared_bf  = max_bf × 2^(-emax)           // TMULS，乘数 recip_emax_bits<OutT>() = 0x3F80 - emax_bits<OutT>()
scale_e8m0 = Cast<bf16→e8m0>(shared_bf)   // TCVT 直转；硬件把 inf/nan 映射到 0xFF
```

**为什么乘数 `0x3F80 - emax_bits<OutT>()` 就是 `2^(-emax)` 的 bf16 位型**：`0x3F80` 是 bf16 的
`1.0`（指数域 127<<7、零尾数），`emax_bits<OutT>()` 是 `emax<<7`（同为零尾数 2 的幂），两者
指数域整数相减即得 `(127-emax)<<7`、尾数仍零 → bf16 值 `2^(-emax)`。故
`2^E_max × 2^(-emax) = 2^(E_max-emax)`，`Cast` 取其指数字段即 E8M0 字节 `E_max-emax`
（**与旧算法逐比特同值**）。e4m3→`0x3B80`(2⁻⁸)、e5m2→`0x3800`(2⁻¹⁵)、fp4_e2m1→`0x3E80`(2⁻²)、
fp4_e1m2→`0x3F80`(2⁰)。

**recip 收尾**（`finalize_recip_u16`，仍在 uint16 位域，逻辑同旧算法、仅少 scale 的 `>>7`）：
```
shared_u16 = reinterpret<uint16>(shared_bf)
recip      = 0x7F00 - shared_u16
             select(max_exp==0x7F80   ? 0x7F81 : recip)   // inf/nan（取自乘前 max_exp）
             select(max_exp==0        ? 0      : recip)   // 整块全 0（取自乘前 max_exp）
             select(shared_u16==0x7F00? 0x0040 : recip)   // special（取自 shared 位型）
```
`eq_inf`/`eq_zero` 取自**乘之前**的 `max_exp`（镜像 ocp_new 的 `Compare(xMaxExp,expMask)` /
`Compare(xMaxExp,0)`），`eq_special` 取自 `shared` 的位型。

**两处相对旧算法的边界**（详见 RECORD 问题7）：
- **边界1（inf/nan）**：scale 字节现依赖 `Cast<e8m0>(inf/nan)==0xFF`（无显式 Select），而非旧路径的 `eq_inf→0x00FF`。
- **边界2（极小非零下溢）**：旧路径 `clamp` 后 recip 归 0；新路径给出 bf16 denormal recip。

#### 3.3.1-旧 scaleAlg=0（OCP）— clamp + 减 emax（已过时）

> ⚠️ **已过时**：当前代码已改为 §3.3.1 的 bf16→e8m0 直转。本节仅描述旧的整数指数域
> clamp/减法/移位算法，保留供对照。旧算法目前仅 `bak/` 下**未调试**的 fp8 kernel
> （`compute_ocp_scale_{tail,not_tail}`）仍在使用。

提取每个元素指数位，取块内最大指数，`shared_exp = max(max_exp, emax) - emax`。emax 由 `emax_field<OutT, Domain>()` 给出——**这正是 OCP 同时支持 FP8 与 FP4 的关键**：唯一区别是 emax（16bit 域 FP8=0x0400 / FP4=0x0100）。

> **精度位宽关键点（对齐 AscendC）——OCP 的 scale 计算恒为 16bit**。无论输入 dtype 如何，`ComputeScaleOcp` 与 `finalize_scale_recip_u16` **全程 uint16**（emax 用 16bit 值 0x0100/0x0400），scale/recip 也是 uint16。**只有 fp32 输入的 max-exp 抽取**这一步临时走 32bit（`And(0x7f800000)`、uint32 域 `ReduceMax`），随后立刻 `ShiftRights(FP32_PACK_SHR_NUM=16)` + `Pack<uint16_t>` **缩窄回 bf16 域 uint16 指数**（<<7 形式，掩码 0x7f80）再交给 finalize——**不是** SHR=23、**也不是** emax_field<OutT,32>，scale finalize 从不在 32bit 域进行（见 AscendC `ComputeMaxExpOcpFp32` :516-554，narrow shift=16；`ComputeScaleOcp` 全 uint16）。这与 **cuBLAS 恒在 32bit 域算 scale**（`intCalcType = fp32 输入时 uint32`，把 16bit maxExp 上转到 32bit，最后缩回 uint16）形成对照；两算法的 **ComputeData 都提升到 fp32** 计算。下面代码以 bf16 输入的 16bit 域为代表——OCP 唯一随输入变化的是 max-exp 抽取的读宽（fp32 用 `0x7f800000`/uint32 后缩回 uint16），scale finalize 逻辑与位宽不变。

```
TANDS(exp_bits, x_bits, EXP_MASK<Domain>)       // 16bit:0x7f80 / 32bit:0x7f800000
TROWMAX(max_exp, exp_bits)                      // nontail 用 TCOLMAX
TCMPS(eq_inf, max_exp, EXP_MASK<Domain>)        // 钳位前 NaN/Inf
TMAXS(max_exp, max_exp, emax_field<OutT,Domain>())
TSUB(shared_exp, max_exp, emax_tile)
finalize_scale_recip_u16(shared_exp, eq_inf, scale_byte, recip_out)
```

**为什么位域内一次 `max_exp - emax` 直接得到 E8M0 字节（bias 抵消推导）**

aclnnV3 文档的公式在**无偏指数 / 实数域**给出：

```
e          = floor(log2(max|V|))     // 无偏指数
shared_exp = e - emax                // 无偏域相减
mxscale    = 2^shared_exp            // 实数（恒为 2 的幂）
scaleByte  = 编码 mxscale 到 E8M0
```

而 kernel 全程在**有偏指数位域**（`E << 7`，`E = e + 127` 是 bf16 存储的有偏指数）上做整数运算，直接算出要存的 E8M0 字节，从不构造无偏 `e` 或实数 `mxscale`。两者等价的推导：

1. **E8M0 编码定义**：字节 `B` 代表 `2^(B - 127)`；故编码 `2^p` 所需字节 `B = p + 127`（+127 是 E8M0 偏置）。
2. **公式路线**得到的字节：`B = shared_exp + 127`。
3. **代入** `shared_exp = e - emax` 且 `E_amax = e + 127`：

   ```
   B = shared_exp + 127
     = (e - emax) + 127
     = (e + 127) - emax        // 把 +127 挪到 e 旁
     = E_amax - emax           // E_amax = e + 127
   ```

4. **kernel 路线**（`TROWMAX` 抽有偏指数 → `TSUB emax`）直接得 `E_amax - emax`，与公式的编码字节 `B` **逐比特相同**。

**关键**：bf16 有偏指数的 +127 与 E8M0 的 +127 是**同一个 127**，恰好抵消 —— 所以公式里「无偏相减 → `2^·` → 再 +127 编码」三步，在 kernel 里压成一次整数减法。`emax_bits<OutT>()` 已是 `<<7` 的位域常量（FP4=`0x0100`=`2<<7`），故 `TSUB` 同时完成「`- emax`」与「有偏→E8M0」的偏置换算。深层原因：`mxscale` 恒为 2 的幂，`2^·`（幂）与「存 E8M0 时取指数」互逆直接抵消，kernel 只在指数（整数）上活动，无 log、无幂、无除法。

> 命名提示：上面代码里的变量 `shared_exp` 承载的其实是**有偏 E8M0 字节的 `<<7` 位域形式**（`(E_amax - emax) << 7`），并非文档里的无偏 `shared_exp`；`finalize_scale_recip_u16` 内再 `>> 7` 得到最终 scale 字节。数值例：`max|V|=48, emax=2` → 公式 `e=5, shared_exp=3, mxscale=8, B=130`；kernel `E_amax=132, 132-2=130`；同为字节 `130`（= `2^3`）。

#### 3.3.2 scaleAlg=1（NVIDIA cuBLAS MxFP8）—— FP8-only

`S = max(amax, 1e-12) * inv_dst_max<OutT>()`，提取 FP32 指数/尾数，**带守卫地**向上取整，再处理 NaN/零。镜像 AscendC `ComputeScaleCublas`（line 935），**不是**简单“尾数非零就 +1”。

**守卫条件**：
- `roundup = p0 || p1`，`p0 = (exp>0) && (exp<254) && (man>0)`，`p1 = (exp==0) && (man>0x400000)`
- `finite = raw_amax_bits < 0x7F800000`（NaN/Inf → scale=0xFF, recip=0x7F81）
- `nonzero = raw_amax_bits != 0`（零 → scale=0, recip=0）

GT/LT 用 EQ 模拟（`a<K ⟺ min(a,K-1)==a`；`a>0 ⟺ !(a==0)`），位重解释走 `reinterpret_f32_to_u32`。cuBLAS 仅对 FP8 合法，故 `inv_dst_max<OutT>()` 对非 FP8 静态报错。

#### 3.3.3 scaleAlg=2（Dynamic Dtype Range MxFP4）—— FP4-only

对最大绝对值加尾数修正值（0x003F），进位到指数位实现向上取整，`shared_exp = corrected_exp - emax`。

```
TANDS(abs_x, x_u16, BF16_ABS_MASK)
TROWMAX(max_abs, abs_x)                         // nontail 用 TCOLMAX
TANDS(xexp_only, max_abs, BF16_EXP_MASK)
TCMPS(eq_inf, xexp_only, BF16_EXP_MASK)
invalid = (min(xexp_only, emax-1)==xexp_only)   // xexp_only < emax
TADDS(x_add, max_abs, 0x003F) ; TANDS(x_add, x_add, BF16_EXP_MASK)
TSEL(x_add, invalid, emax_tile)                 // invalid ? emax : x_add
TSUB(shared_exp, x_add, emax_tile)
finalize_scale_recip_u16(shared_exp, eq_inf, scale_byte, recip_out)
```

> DynRange kernel 以 `static_assert(std::is_same_v<OutT, __fp4_e2m1x2>)` 强制 FP4_E2M1，emax=0x0100——与 AscendC（该算法仅对 FP4 合法）一致。

#### 3.3.4 三种算法对比

| 维度 | OCP (0) | cuBLAS (1) | DynamicRange (2) |
|------|---------|------------|-------------------|
| 合法 dstType | FP8 & FP4 | FP8 only | FP4 only |
| 归约输入 | 指数位 (`x & EXP_MASK`) | 绝对值（FP32 TABS） | 绝对值 (`x & ABS_MASK`) |
| 归约操作 | TROWMAX（U16） | TROWMAX（FP32） | TROWMAX（U16） |
| 核心计算 | `max_exp - emax` | `ceil(log2(amax*inv_max))+守卫` | `(amax+0x003F)&EXP_MASK - emax` |
| 尾数感知 | 否 | 是（显式提取+条件+1） | 是（加法进位隐式） |
| emax | `emax_bits<OutT>()` | 折入 `inv_dst_max` | `0x0100`（FP4 固定） |

### 3.4 量化主计算（各 kernel 入口中）

各 kernel 采用 **ComputeScale → ComputeData 两遍结构**：先只让本算法所需视图存活算出 scale/recip 并立即存储 scale，再在数据遍加载输入值。因专门化，**无 `if(Alg)` 分支**：

```
// ---- ComputeScale 遍（按 InT if constexpr 正则化，镜像 AscendC ComputeMaxExp{Ocp,Cublas}{Bf16,Half,Fp32}）----
[OCP/DynRange -> 统一 uint16 bf16-指数域 x_u16]
    InT==__bf16 : TLOAD(x_u16, gxu)                              // 指针 reinterpret，原路径不变（零回归）
    InT==__half : TLOAD(xin, gx) ; half_to_bf16bits(xin→x_u16)   // TCVT half→bf16(TRUNC)+取指数位；inf/nan 靠下游 eq_inf==0x7F80
    InT==float  : TLOAD(xin, gx) ; f32_to_bf16expbits(xin→x_u16) // f32→u32 + TANDS(0x7F800000) + TSHRS(16) + narrow uint16
    compute_{ocp,dynamic_range}_scale_*(x_u16, scale_byte, recip)
[cuBLAS -> 统一 fp32 amax]  TLOAD(xq_s, gx) ; compute_cublas_scale_{tail,not_tail}<OutT, InT, ...>(xq_s, ...)
    //  内部：TABS + TROWMAX/TCOLMAX 在 InT 值域；if constexpr(InT==float) 免 fp32-cast，否则 TCVT InT→fp32；compute_cublas_core 不变
TSTORE(gs, scale_byte)

// 由 recip 位模式构造 fp32 乘子（无 TCAST，走内存别名）
reinterpret_u16_to_bf16(recip → inv_bf16) ; TCVT(inv_scale_f, inv_bf16)

// ---- ComputeData 遍（按 InT if constexpr 三分支，镜像 AscendC ComputeData line 785）----
TLOAD(xq, gx)
    InT==float          : TMUL/TExpandMul(xq, xq, inv_scale_f)   // 直接 fp32 域乘，免前置 cast
    InT∈{__bf16,__half} : TCVT(xf, xq) ; TMUL/TExpandMul(xf, xf, inv_scale_f)  // 先升 fp32 再乘
    // tail 用 TROWEXPANDMUL/TMUL（逐元素/沿行），nontail 用 TCOLEXPANDMUL（沿列广播）
preserve_neg_zero(...)                          // -0 保号，对齐 AscendC common.h:224-254
TCVT(oq, ...)                                   // fp32 → OutT（FP8 或 FP4 打包），rint 单步舍入
TSTORE(gy, oq)
```

> **类型分派集中于「输入正则化」一处**（对齐 AscendC 把类型差异收敛到 `Compute()` line 920-940）：scale 域后半段（`compute_*_scale_*` 的 clamp/finalize）与整条 ComputeData 结构三类型共享。发射能力经 `dtype_probe.cpp` 反汇编确认。bigbs cuBLAS pass1 的 bf16 分支保持 uint16 abs-bit 域（精确匹配 AscendC 且 `TEXPANDS(0)` uint16 种子合法），half/fp32 用值域归约 + peeled-首子块种子 + running-`TMAX`。

**输出落 dtype（fp8 与 fp4）一律用裸 `TCVT` 单步直转即正确，不照搬 AscendC 的 fp32→bf16→窄类型两步 + 显式栅格舍入。** PTO-ISA 的 `TCVT` 原生支持 `fp32 → fp8` 与 `fp32 → fp4` 的单步直转；AscendC 的 `Reg::Cast` 缺这两个组合，被迫 `fp32 → bf16 → 窄类型` 两步。二者是不同的封装机制，不是同一条硬件约束——本次实现 kernel 全部直转，不移植 AscendC 的两步绕行。

---

## 4. 尾轴 Kernel 设计

### 4.1 入口签名（4 个专门化 tail kernel）

> **落地差异（当前实现，见 README「Tile 旋钮编译期推导」）**：`TileM`/`TileN`/`R_sub` **已从公开签名移除**，改由算子输入 + `InT` 预算 constexpr 推导（`max_tilem`/`pick_tilen`/`max_rsub`）；`InT` 既做预算感知，**又是真实数据路径**（`fp16(__half)`/`bf16(__bf16)`/`fp32(float)`，`if constexpr` 分派，`static_assert(InT ∈ {__bf16,__half,float})`，见 §3.4）。非尾轴入口 `if constexpr` 无合法 `TileN` 时**自动路由**到 `_bigbs` 方案 A（独立 bigbs TYPE 已删）。下方签名 `__bf16 *x` 是默认实参，实际形参为 `InT *x`；`TileM`/`TileN` 保留仅作**结构示意**。

```cpp
template <int M, int K, int TileM=8, int BlockSize=32, typename OutT=__fp8_e4m3>
void dynamic_mx_quant_tail_ocp_fp8(__bf16 *x, OutT *y, uint16_t *scale);

template <int M, int K, int TileM=8, int BlockSize=32, typename OutT=__fp8_e4m3>
void dynamic_mx_quant_tail_cublas_fp8(__bf16 *x, OutT *y, uint16_t *scale);

template <int M, int K, int TileM=8, int BlockSize=32, typename OutT=__fp4_e2m1x2>
void dynamic_mx_quant_tail_ocp_fp4(__bf16 *x, OutT *y, uint16_t *scale);      // 阻塞

template <int M, int K, int TileM=8, int BlockSize=32, typename OutT=__fp4_e2m1x2>
void dynamic_mx_quant_tail_dynrange_fp4(__bf16 *x, OutT *y, uint16_t *scale); // 阻塞
```

### 4.2 循环结构

```
full_m = M / TileM ; M_tail = M % TileM ; numKb = K / BlockSize
for m in [0, full_m): for kb in [0, numKb): quantize_tile(...)   // 满行块
// 尾块：不递归。保持物理 tile 形状 = TileM×BlockSize（logicalTileBytes ≥ 512B，
// 满足 IsValidActiveSize），但每个 tile 用 boxed 有效区域 ValidRow = M_tail，
// 仅触及活跃行；行块用物理索引 full_m 寻址（iterator i-stride 用物理 Rows）。
// TileM'=M_tail 递归会造出 sub-512B tile 并触发 LinxV5 溢出断言，故禁用。
if constexpr (M_tail > 0) {
    using tile_x_r = Tile<..., __bf16, TileM, BlockSize, RowMajor, M_tail, BlockSize>; // 物理 TileM，ValidRow=M_tail
    for kb in [0, numKb): quantize_tile_boxed(index=full_m, kb, ValidRow=M_tail)
}
```

### 4.3 GM 张量布局

```cpp
using gm_x = global_tensor<__bf16,  RowMajor<M, K>>;
// FP8：输出 tile [TileM, BlockSize]，gm_y = RowMajor<M, K>（uint8 存储）
// FP4：输出 tile [TileM, BlockSize/2]（2 值/字节打包），gm_y = RowMajor<M, K/2>
// scale：E8M0 1 字节/块，紧凑平铺 [M, scaleCols]，scaleCols = evenAlign(K/BlockSize)
constexpr int scaleCols = ((K / BlockSize + 1) / 2) * 2;
using gm_s = global_tensor<uint8_t, RowMajor<M, scaleCols>>;
```

**尾轴 scale 不需要交织**。AscendC 目标布局 `[M, ceil(K/BlockSize)/2, 2]` 的配对轴（K 块）在同一行内本就内存连续，展平后 `(kb/2)*2 + kb%2 == kb` 顺序不变，故紧凑平铺 `[M, scaleCols]` 与交织布局**逐字节等价**，一次 `TSTORE`/Compact 搬出即可（对齐 AscendC host `tiling_arch35.cpp:415-416`「尾轴且块数为偶数可直接写回 mxScale」、`swiglu_mx_quant_axis_last.h:585-592` 的 `PaddingMode::Compact` 直搬）。尾块 padding 列置零。

---

## 5. 非尾轴 Kernel 设计

### 5.1 入口签名（4 个专门化 nontail kernel）

```cpp
template <int Axis, int Post, int BlockSize=32, int TileN=32, typename OutT=__fp8_e4m3>
void dynamic_mx_quant_nontail_ocp_fp8(__bf16 *x, OutT *y, uint16_t *scale);
// 4 个 nontail kernel 共用两遍 skeleton，各调各自的 compute_*_scale_not_tail；
// _ocp_fp4 / _dynrange_fp4 输出 tile [BlockSize, TileN/2]（打包，TileN=64）
```

### 5.2 处理策略

将量化轴视为“行”（`Axis`），非量化轴视为“列”（`Post`）：沿 `Axis` 按 BlockSize=32 分块归约（`TCOLMAX`，每列独立），2D 循环 `[numKb, numN]=[Axis/BlockSize, Post/TileN]`。数据遍用 `TCOLEXPANDMUL` 沿列广播乘。

### 5.3 GM 张量布局

```cpp
using gm_x = global_tensor<__bf16,  RowMajor<Axis, Post>>;
// FP8：gm_y = RowMajor<Axis, Post>；FP4：gm_y = RowMajor<Axis, Post/2>
// scale：E8M0 1 字节/块，planar [scaleRows, Post]，scaleRows = evenAlign(numKb)
using gm_s = global_tensor<uint8_t, RowMajor<((numKb + 1) / 2) * 2, Post>>; // 纯 planar
```

> **【2026-09-03 更正】PTO-ISA 不要求交织，本节以下描述的是 AscendC 约定（历史留档）。**
> pto-spec `d0ce06ad`（ADR-0101）定义 matmul 消费 MX scale 为 **Shared 普通 Tile，B-scale
> 纯 planar `[G,N]`**（`matmul_shared_lowp.hpp` 坐实），故非尾轴 scale **正解 = planar，无需交织**。
> 下述 AscendC 的 `Reg::Interleave` / `DataCopy<DIST_INTLV_B8>` parity zip 是 Ascend 硬件打包约定，
> **不属 PTO-ISA scale 契约**；golden 已去交织，kernel 现有 planar 逐字节 pass（RECORD 问题5 已解除）。

**（以下为 AscendC 交织约定的历史描述）** AscendC 非尾轴 scale 交织：配对轴是行（量化轴），目标 shape `[ceil(numKb/2), Post, 2]` 的 parity 维 `2` 追加在**列维 Post 之后**当最内层，把相邻两个 block 行（even=rows[0:32)，odd=rows[32:64)）**按列 zip**：`offset = rowPair*2*Post + col*2 + parity`。

AscendC 实现两种等价形态：
- **标准 `dynamic_mx_quant`**：kernel 先把每 32 行 block 的 scale **平铺**写 workspace，再由 `dynamic_mx_quant_post.h`（`ComputeInterleaveVF` / `Reg::Interleave`，`:455-477`）读回**交织**写最终 mxScale。host `tiling_arch35.cpp:415-416` 规定所有非尾轴强制走此 post。
- **融合 `swiglu_mx_quant`**：`axis_not_last.h` 在 UB 内一次处理 64 行（2 个 block），先平铺算 scale（`:349-373`），再 `ComputeInterleave`（`DataCopy<DIST_INTLV_B8>`，`:255-267,374-380`）zip，最后带 `×2` 列步长搬出（`:526-532`）。

两者最终 mxScale 逐字节一致。数值例（`[Axis=128, Post=4]`，4 个 block 行 B0..B3，`S[block][col]`）：
```
平铺（中间态）: S00 S01 S02 S03 S10 S11 S12 S13 ...
交织（mxScale）: S00 S10 S01 S11 S02 S12 S03 S13 ...  // B0/B1 同列相邻，shape [2,4,2]
```

---

## 6. PTO-ISA Intrinsic 依赖

| 类别 | Intrinsic | 用途 |
|------|-----------|------|
| 加载/存储 | TLOAD, TSTORE | GM↔UB 搬运；亦用于内存别名位重解释 |
| 类型转换 | TCVT | 数值转换（bf16↔fp32、fp32→fp8/fp4、u32→u16 窄化）。**无 TCAST** |
| 算术 | TMUL, TCOLEXPANDMUL, TADDS, TSUB, TMULS | 逐元素乘、沿列广播乘、标量加/减/乘 |
| 位操作 | TANDS | 掩码提取 |
| 移位 | TSHRS, TSHLS | 指数提取、位模式构造 |
| 归约 | TROWMAX / TCOLMAX | tail 行最大 / nontail 列最大（U16 和 FP32） |
| 比较 | TCMPS, TCMP | 产 packed predicate；kernel 用原生 `TCMPS<CmpMode>`（6 模式），dead-reference `compute_cublas_core` 保留 min/max EQ 模拟 |
| 选择 | TSEL | 条件赋值 `mask?src:dst`；cuBLAS 守卫复合条件（`&&`/`||`）用**嵌套 TSEL** 组合（PTO ISA 合规，取代旧的数据域 TAND/TOR/TNOT 掩码代数——已全部移除，见 README 合规小节） |
| 广播 | TEXPANDS | 标量广播到 tile |
| 绝对值 | TABS | 浮点绝对值（仅 cuBLAS） |
| 最大/最小 | TMAXS, TMINS | 标量钳位；LT/GT 模拟 |

### 关键约束

- **位操作仅整数类型**：TANDS/TSHRS/TSHLS 的 tile 元素须为 `uint16_t`/`uint32_t`。
- **无 TCAST（`-D__linx`）**：float↔int 位重解释只能经 scratch-HBM 同宽度别名。
- **TCMP/TCMPS 仅 EQ**：GT/LT 用 EQ 模拟。
- **TSEL 语义**：`TSEL(dst, mask, src)` = `mask ? src : dst`。
- **tile 溢出下限 512B**：LinxV5 对 tile 溢出槽断言 `RegSize ∈ [512B, 64KB]`（`LinxV5RegisterInfo.cpp:403`）→ 两遍结构压低峰值活 tile。
- **32B 列对齐 / TileSize 约束**：详见 RECORD.md 问题1、2。
- **FP4 输出 tile 切分**：`pto_tile.hpp:649` 断言 RowMajor NoneBox 需 `Cols * bits % 256 == 0`；fp4（bits=8）单 block Cols=16 不满足，须每 tile ≥ 2 MX block（Cols=32=64 值）。fp4 发射本身可用，详见 RECORD.md 问题2。

---

## 7. 测试计划

### 7.1 测试矩阵（8 配置）

> 本表为**预期全功能**测试矩阵；各配置的**当前实现状态**（编译/链接/待改）见 README.md。

| 配置 | axis | scaleAlg | dstType | fp4 tile 切分策略 |
|------|------|----------|---------|------|
| TAIL_OCP_FP8 | 尾轴 | OCP | FP8_E4M3 | — |
| TAIL_CUBLAS_FP8 | 尾轴 | cuBLAS | FP8_E4M3 | — |
| TAIL_OCP_FP4 | 尾轴 | OCP | FP4_E2M1 | 每 block 子归约 + TCONCAT 到 64 宽（RECORD 问题2） |
| TAIL_DYNRANGE_FP4 | 尾轴 | DynRange | FP4_E2M1 | 同尾轴 fp4 方案 |
| NONTAIL_OCP_FP8 | 非尾轴 | OCP | FP8_E4M3 | — |
| NONTAIL_CUBLAS_FP8 | 非尾轴 | cuBLAS | FP8_E4M3 | — |
| NONTAIL_OCP_FP4 | 非尾轴 | OCP | FP4_E2M1 | TileN=64 plain tile，归约不变 |
| NONTAIL_DYNRANGE_FP4 | 非尾轴 | DynRange | FP4_E2M1 | TileN=64 plain tile（同 nontail OCP） |

### 7.2 验证方法

- **编译验证**：4 个 FP8 配置通过 `make TESTCASE=dynamic_mx_quant TYPE=<...> diss`（含 `res_check=on`）编译+链接成功。
- **精度验证**：`run_precision_check.py` 编排 gen→compile→QEMU→compare；`--dtype FP8|FP4` 选择 golden 与解码路径。尚未落地的配置在编排中跳过（当前落地范围见 README.md）。

> 编译/运行时实际遇到的问题（skew、emulator 断言等）均记录在 RECORD.md，不在本设计文档范围内。

### 7.3 测试文件结构

```
test/kernel/multi_thread/quant/dynamic_mx_quant/
├── src/
│   ├── tail_ocp_fp8.cpp / tail_cublas_fp8.cpp / tail_ocp_fp4.cpp / tail_dynrange_fp4.cpp
│   ├── nontail_ocp_fp8.cpp / nontail_cublas_fp8.cpp / nontail_ocp_fp4.cpp / nontail_dynrange_fp4.cpp
│   ├── gen_dynamic_mx_quant_data.py     # Golden（--dtype FP8|FP4，FP4 打包 2 nibble/字节）
│   ├── dynamic_mx_quant_data_compare.py # 精度比对（FP8 字节 / FP4 nibble 解码）
│   └── run_precision_check.py           # 全流程编排（8 配置）
├── Makefile                             # 8 个 TYPE 分支 + FP4_PROBE + 遗留 BS*/TM4* sweep
└── compile.all                          # 各配置编译入口（当前落地范围见 README.md）
```

### 7.4 FP4 输出 tile 切分约束

**fp4 发射本身可用**：fp32→`__fp4_e2m1x2` 单步 `TCVT` + `TSTORE` 发射真实指令，由探针 `test/kernel/multi_thread/quant/dynamic_mx_quant/src/fp4_probe.cpp`（`make TESTCASE=dynamic_mx_quant TYPE=FP4_PROBE diss`）反汇编验证（`BSTART.TEPL TCVT, FP32` + `B.DATR e2m1x2, byte0`；`BSTART.TLSU TSTORE, e2m1x2`），无 Match-Instruction-Error、无对齐断言。

约束落在 **fp4 输出 tile 的切分**：`type_traits<__fp4_e2m1x2>::bits == 8`，RowMajor+NoneBox 的 `Cols_packed × 8 % 256 == 0` 要求每行 ≥ 32 打包列 = **64 个 fp4 值 = 2 个 MX block**（`pto_tile.hpp:649`）。BlockSize=32 的单 block fp4 输出仅 16 打包列，不满足。两轴按打包轴与 reduce 轴是否重合分别处理（详见 RECORD 问题2）：

- **非尾轴**：打包轴（Post）与 reduce 轴（行/`TCOLMAX`）正交，`TileN=64` 走 plain RowMajor NoneBox（`tile_o=[32,32]`），归约零改动。
- **尾轴**：打包轴与 reduce 轴重合。**不 2-block 打包**，改用**列装箱补齐物理宽**：物理列宽补齐到 `PW = ⌈BlockSize/64⌉×64`（BS=32→64），每 op 列装箱到有效 BlockSize（value/scale tile 物理 PW、有效 BlockSize）。`TROWMAX` 按 **ValidCol** 归约（cpu_sim `TRowMax.hpp:13` 循环 `j<ValidCol`），物理补齐**不合并** block、每块归约独立正确；fp4 输出 tile 物理 `PW/2` 字节（`PW%64==0` 故 32B 对齐）、有效 `BlockSize/2` 字节；boxed `TLOAD`/`TSTORE` 只搬 ValidCol 列（`blk_tload` 计数取 `GetValidCol()`），尾块不越界读过 N；因物理 PW≠每块步长，基址折叠定位每 block（输入 `x+kb*BlockSize`、输出 `y+kb*(BlockSize/2)`）。**无 concat、无配对、无零块**，仅需 `N % BlockSize == 0`（单 block/tile 亦可）。奇数 numKb 时显式往偶数对齐的 padding scale 列补 0x00 E8M0（golden `_pad_to_even` 用 `2^-127`==0x00）。scale 路径复用 `compute_ocp_scale_tail_boxed_pw`（物理宽 PW 参数化的 boxed OCP scale）。

列装箱是 plain RowMajor（有效列 < 物理列），非 fractal。boxed ColMajor fractal（`fa_hif4.hpp` 先例）作为更次备选，其寄存器侧 fractal 落盘字节序需运行期核实。

---

## 附录 A：AscendC 算子规格基线（对齐依据）

> 本附录是本设计所对齐的**唯一正确性基线**——原 Ascend C `DynamicMxQuant` 算子规格。
> kernel 与 golden 的每一步都以此为准；实现层面的位模式细节见 §3。

### A.1 原 Ascend C 算子规格

- **源路径**：`ops-nn/quant/dynamic_mx_quant`
- **Proto 定义**：`op_graph/dynamic_mx_quant_proto.h`
- **支持平台**：Ascend 950PR / Ascend 950DT（arch35）

在给定的轴 `axis` 上，按每 `blocksize` 个元素分组，计算每组对应的量化尺度 `mxscale`，
然后对组内每个元素除以 `mxscale`，按 `round_mode` 转换到目标类型 `dst_type`，得到量化结果 `y`。

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

### A.2 算子属性

| 属性名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `axis` | Int | -1 | 量化发生的轴，取值范围 [-rank(x), rank(x)) |
| `round_mode` | String | "rint" | 舍入模式："rint"（四舍六入五成双）、"round"（四舍五入）、"floor"（向下取整） |
| `dst_type` | Int | DT_FLOAT4_E2M1 | 输出 y 的数据类型，支持 FLOAT4_E2M1 / FLOAT4_E1M2 / FLOAT8_E4M3FN / FLOAT8_E5M2 |
| `blocksize` | Int | 32 | 量化块大小，32 的倍数，≤1024，不为 0；scale_alg=2 时必须为 32 |
| `scale_alg` | Int | 0 | scale 算法：0=OCP MxFP8/MxFP4，1=NVIDIA cuBLAS MxFP8，2=Dynamic Dtype Range MxFP4 |
| `dst_type_max` | Float | 0.0 | 用户指定的目标类型最大值；仅 scale_alg=2 + FP4_E2M1 + blockSize=32 时生效；取值 0.0（使用类型默认最大值）或 6.0~12.0；其中 0.0/6.0/7.0 走 OCP 风格（指数位+尾数修正），其他值走 cuBLAS 风格（`S=Amax/dstTypeMax`） |
| `max_low_bound` | Float | 0.0 | 每个 block 的 Amax 下限钳位值；仅 scale_alg=1 时生效，scale_alg≠1 时必须为 0.0；必须非负 |

### A.3 输入/输出张量

| 张量 | 输入/输出 | 数据类型 | Shape 约束 |
|------|----------|---------|-----------|
| `x` | 输入 | FLOAT16, BF16, FLOAT | ND 格式，rank 1-7 |
| `y` | 输出 | FLOAT4_E2M1, FLOAT4_E1M2, FLOAT8_E4M3FN, FLOAT8_E5M2 | 与 x 相同 |
| `mxscale` | 输出 | FLOAT8_E8M0 | rank = rank(x)+1，axis 维度偶数对齐，末维=2 |

### A.4 约束条件

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

### A.5 计算公式

> 以下公式以 `aclnnDynamicMxQuantV3.md` 官方文档为准，已通过 Python 参考实现（dtypes.py
> `mx_quantize`）和 AscendC kernel 代码交叉验证。

所有 scaleAlg 的核心流程相同：在 axis 维度上按 blocksize 分组，计算每组的 mxscale，然后对
组内元素**除以 mxscale**，按 round_mode 转换到目标类型。差异在于 mxscale 的计算方式。

**emax（目标类型最大正则数的指数位）**

| dst_type       | emax | max_norm（最大正则数） |
|----------------|------|----------------------|
| FLOAT4_E2M1    | 2    | 6.0                  |
| FLOAT4_E1M2    | 0    | 1.75                 |
| FLOAT8_E4M3FN  | 8    | 448.0                |
| FLOAT8_E5M2    | 15   | 57344.0              |

**scaleAlg = 0（OCP MxFP8/MxFP4）**

```
shared_exp = floor(log2(max_i(|V_i|))) - emax
mxscale    = 2^shared_exp
P_i        = cast_to_dst_type(V_i / mxscale, round_mode)
```

边界处理（kernel 实现细节，文档未显式描述）：
- `max_i(|V_i|) == 0` 时：`shared_exp = -inf`，`mxscale = 0`，量化结果全零
- `max_i(|V_i|)` 为 NaN/Inf 时：`scale = 0xFF`（E8M0 NaN），量化结果传播 NaN

**scaleAlg = 1（NVIDIA cuBLAS MxFP8，仅 FP8）**

```
Step 1: Amax   = max({|d_i|})
Step 2: Amax   = max(Amax, maxLowBound)              // maxLowBound>0 时（V3 新增）
Step 3: S_fp32 = Amax / Amax(DType)                  // Amax(DType)=目标精度最大值
Step 4: 从 S_fp32 提取无偏指数 E_int / 尾数 M_fixp，向上取整：
          E_int+1  若 S_fp32 正规且 E_int<254 且 M_fixp>0
          E_int+1  若 S_fp32 非正规且 M_fixp>0.5
          E_int    否则
Step 5: S_ue8m0 = 2^E_int ; R_fp32 = 1 / fp32(S_ue8m0)
Step 6: d_i     = DType(d_fp32_i × R_fp32)
```

**scaleAlg = 2（Dynamic Dtype Range MxFP4，仅 FP4_E2M1，blockSize=32）**

当 dstTypeMax = 0.0 / 6.0 / 7.0（OCP 风格 + 尾数修正）：
```
shared_exp = ceil(log2(max_i(|V_i|))) - emax   若尾数高比特前一/两位为1且尾数非全零
           = floor(log2(max_i(|V_i|))) - emax  其它
mxscale    = 2^shared_exp
P_i        = cast_to_dst_type(V_i / mxscale, round_mode)
```
- 修正值加到尾数上，进位到指数位实现 `ceil`；`dstTypeMax∈{0.0,6.0}` 修正值 `0x003F`，`7.0` 为 `0x001F`

当 dstTypeMax ≠ 0.0/6.0/7.0（cuBLAS 风格）：
```
Amax    = max({|d_i|}) ; S_fp32 = Amax / Amax(DType)
E_int   = E_int+1  若 S_fp32 正规且 E_int<254 且 M_fixp>0 ; 否则 E_int
S_ue8m0 = 2^E_int ; R_fp32 = 1 / fp32(S_ue8m0) ; d_i = DType(d_fp32_i × R_fp32)
```

**Kernel 位模式实现 ↔ 文档公式对应关系**

| 文档公式 | Kernel 位操作实现 | 数学等价性 |
|---------|------------------|-----------|
| `floor(log2(max\|Vi\|))` | `maxExp = max(x_i & 0x7F80)`，提取 BF16 指数位 | BF16 指数位 = `floor(log2(\|V\|)) + 127` |
| `shared_exp = ... - emax` | `sharedExp = maxExp - (emax << 7)` | 偏置域减法等价 |
| `mxscale = 2^shared_exp` | `scale_byte = sharedExp >> 7`（E8M0） | E8M0 值 = `2^(byte - 127)` |
| `V_i / mxscale` | `x × recipScale`，`recipScale = 0x7F00 - sharedExp` | `recipScale` 的 BF16 值 = `2^(-shared_exp)` |
| `cast_to_dst_type(...)` | `Cast<FP8/FP4>(x_fp32, SAT + roundMode)` | 硬件饱和 + 舍入 |
| `S_fp32 = Amax / Amax(DType)` | `S = maxAbs_fp32 × inv_dtype_max`（预计算倒数） | 乘法代替除法 |
| `E_int` 提取 | `exp = S_bits >> 23`（FP32 指数位） | IEEE 754 偏置指数 |
| `M_fixp > 0` | `mantissa = S_bits & 0x007FFFFF`，比较非零 | 尾数位直接检测 |

`recipScale = 0x7F00 - sharedExp`，其 BF16 浮点值 = `2^(-actual_shared_exp)` = `1/mxscale`，
精确 2 的幂次、无浮点除法误差。特殊值：`sharedExp==0`→`recipScale=0`（全零块）；
`sharedExp==0x7F00`→`0x0040`；NaN/Inf→`scale=0xFF, recipScale=0x7F81`。

### A.6 场景矩阵（AscendC 全量有效场景）

| 维度 | 取值范围 |
|------|---------|
| x 数据类型 | FLOAT16, BF16, FLOAT |
| dst_type | FP4_E2M1, FP4_E1M2, FP8_E4M3FN, FP8_E5M2 |
| scaleAlg | 0, 1, 2 |
| round_mode | rint, round, floor |
| axis | 尾轴 (-1), 非尾轴 |
| blockSize | 32, 64, 96, ..., 1024 |

**参数约束（排除无效组合）**

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

> 本设计交付的是该全量矩阵的一个**代表性合法子集**（见 §1「scaleAlg × dstType 合法性」
> 与 §7.1 测试矩阵）：BF16 输入、rint、每算法一个代表 dtype、× {tail, nontail}。
