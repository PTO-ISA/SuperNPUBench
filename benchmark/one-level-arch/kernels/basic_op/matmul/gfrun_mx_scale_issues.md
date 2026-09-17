# Issue: gfrun MX (Microscaling) scale 处理偏离 OCP 标准

## 概述

在 multi_thread/matmul 的 MXFP8 / MXFP4 数值校验中，发现 gfrun 功能模型在 E8M0 scale
处理上存在 **3 处偏离 OCP Microscaling (MX) 标准** 的行为。使用标准 golden
（OCP MX 规范：E8M0 bias=127，A & B 双 scale，FP4 双 nibble）校验时，MXFP8 和 MXFP4
均 **FAIL**，而 FP8（非 MX）PASS。

## 环境

| 项目 | 版本 |
|------|------|
| gfrun | SuperScalarModel `762a72c3` (2026-09-01) |
| 编译器 | linx-toolchain-build main (`adcb87948` / TileOP `f94bc12`) |
| 测试日期 | 2026-09-01 |
| 测试配置 | M=256, N=256, K=512 (MXFP8) / K=1024 (MXFP4), tM=128, tN=256, multiThreadNum=4 |

## 校验结果总表

| 数据类型 | 标准 golden | gfrun 输出 | max_abs | 结果 |
|---------|-----------|-----------|---------|------|
| FP8 E4M3 (非 MX) | FP32 upcast matmul | 匹配 | **0.0** | ✅ PASS |
| MXFP8 (FP8 + E8M0) | OCP MX: bias=127, A&B scale | 偏离（见 Issue 1-2） | **72.83** | ❌ FAIL |
| MXFP4 (FP4 E2M1 + E8M0) | OCP MX: bias=127, A&B scale, 双 nibble | 偏离（见 Issue 1-3） | **509.5** | ❌ FAIL |

---

## Issue 1: E8M0 scale bias = 254（标准为 127）

### 标准行为

OCP MX 规范定义 E8M0 为 8 位纯指数格式，bias = 127：

```
scale_value = 2^(byte - 127)
```

- 0x7F (127) → 2^0 = **1.0**（identity scale）
- 0x00 (0) → 2^(-127) ≈ 0
- 0xFE (254) → 2^127（极大）

### gfrun 实际行为

通过 all-1.0 FP8 数据 + 固定 scale byte 的对照实验，测得 gfrun 实际的 scale 映射为：

```
scale_value = 2^(byte - 254)
```

- 0x7F (127) → 2^(-127) ≈ **5.88e-39**（应为 1.0）
- 0xFE (254) → 2^0 = **1.0**（应为 2^127，极大值）
- 0xFD (253) → 2^(-1) = **0.5**
- 0xFF (255) → **NaN**（特殊值，溢出）
- 0x00 (0) → 2^(-254) → underflow 为 **0.0**

### 实验证据

输入：M=256, N=256, K=512, all-1.0 FP8 E4M3 (0x38), A scale 均匀填充指定 byte, B scale = 0x00

| Scale byte | 标准 (bias=127) 预期 | gfrun 实际输出 | 匹配的 bias |
|-----------|-------------------|--------------|------------|
| 0x00 | K × 2^(-254) ≈ 0 | 0.0 | 254 (2^(-254) underflow) |
| 0x7F | K × 2^0 = 512.0 | 3.009e-36 | 254 (512 × 2^(-127)) |
| 0x80 | K × 2^1 = 1024.0 | 6.019e-36 | 254 (512 × 2^(-126)) |
| 0xFE | K × 2^127 (溢出) | 512.0 | 254 (512 × 2^0) |
| 0xFF | K × 2^128 (溢出) | NaN | 254 (特殊值) |

等效理解：标准 MX 组合公式 `scale = 2^(e_a + e_b - 2×127)` 中，gfrun 似乎将
`2×bias` 写成了 `254` 但 e_b 始终为 0（见 Issue 2），使得有效公式退化为
`2^(e_a - 254)`。

### 影响

所有非 0xFE 的 scale byte 都会产生与标准不同的 scale 因子。使用标准 bias=127
计算 golden 时，MXFP8/MXFP4 的 gfrun 输出与 golden 不匹配。

---

## Issue 2: MX matmul 仅应用 A (src0) 的 scale，忽略 B (src1) 的 scale

### 标准行为

OCP MX 规范要求 TMATMUL 同时应用 A 和 B 的 E8M0 scale：

```
C[i,j] = Σ_k (A[i,k] × 2^(e_a[i,k÷32] - 127)) × (B[k,j] × 2^(e_b[k÷32,j] - 127))
```

组合 scale = `2^(e_a + e_b - 254)`，A 和 B 的 scale 均参与计算。

### gfrun 实际行为

gfrun 功能模型**仅应用 A 的 scale**，B 的 scale 虽然被读取（gfrun 日志可见
`data read to file done: .../src1_scale.bin`）但未参与 TMATMUL 计算。

### 实验证据

输入：M=256, N=256, K=512, all-1.0 FP8 E4M3, A scale 和 B scale 分别设为不同值

| A scale | B scale | 标准 golden (A×B) | gfrun 实际 | 匹配模型 |
|---------|---------|-----------------|-----------|---------|
| 0xFE→1.0 | 0xFE→1.0 | 512 × 1.0 = 512 | **512.0** | 仅 A (512×1.0) |
| 0xFE→1.0 | 0x7F→2^-127 | 512 × 2^(-127) ≈ 0 | **512.0** | 仅 A (512×1.0) |
| 0x7F→2^-127 | 0xFE→1.0 | 512 × 2^(-127) ≈ 0 | **3.009e-36** | 仅 A (512×2^-127) |
| 0x7F→2^-127 | 0x7F→2^-127 | 512 × 2^(-254) ≈ 0 | **3.009e-36** | 仅 A (512×2^-127) |

B scale 变化不影响输出，证明 B scale 被忽略。

### 内核侧确认

内核代码 `matmul_shared_lowp.hpp` 正确地将两个 scale 都传入 TMATMUL_MX 指令：

```cpp
TMATMUL_MX<3>(tC, tA, tAScale, tB, tBScale, mxOptions);
TMATMUL_MX_ACC<3>(tC, tC, tA, tAScale, tB, tBScale, mxOptions);
```

tAScale 和 tBScale 均通过 TLOAD 加载、通过 global_iterator 迭代。问题出在
gfrun 功能模型的 TMATMUL_MX 实现未使用 tBScale 操作数。

### 影响

MXFP8 和 MXFP4 的 golden 计算中 B 的 scale 无法生效，导致标准 golden（同时应用
A 和 B 的 scale）与 gfrun 输出不匹配。

---

## Issue 3: FP4 E2M1x2 仅处理低 nibble，忽略高 nibble

### 标准行为

FP4 E2M1x2 格式将 2 个 FP4 E2M1 值打包在 1 个字节中（低 nibble = 偶数 K 索引，
高 nibble = 奇数 K 索引）。TMATMUL 应同时处理两个 nibble，使 K 维度完整参与
点积。

### gfrun 实际行为

gfrun 功能模型**仅处理每个字节的低 nibble**（4 bit），高 nibble 被完全忽略。
效果上 K 维度的有效元素数从 K 降为 K/PACKED_FACTOR = K/2。

### 实验证据

输入：M=256, N=256, K=1024, A 和 B 的每个字节 = 0x32（低 nibble=0x2=1.0, 高 nibble=0x3=1.5）

| 模型 | 预期 | gfrun 实际 |
|------|------|-----------|
| 双 nibble | (K/2)×(1.0×1.0 + 1.5×1.5) = 512×3.25 = 1664 | — |
| 仅低 nibble | (K/2)×(1.0×1.0) = 512×1.0 = 512 | **512.0** ✅ |
| 仅高 nibble | (K/2)×(1.5×1.5) = 512×2.25 = 1152 | — |

交叉验证：A=0x32 (low=1.0, high=1.5), B=0x23 (low=1.5, high=1.0)
- 双 nibble: (K/2)×(1.0×1.5 + 1.5×1.0) = 512×3.0 = 1536
- 仅低 nibble: (K/2)×(1.0×1.5) = 512×1.5 = **768** ← gfrun 实际输出 = 768.0 ✅

不同 FP4 值的全 1.0 验证（identity scale 0xFE）：

| FP4 nibble | FP4 值 | K×val² (标准) | (K/2)×val² (gfrun) | gfrun 实际 |
|-----------|--------|-------------|-------------------|-----------|
| 0x1 | 0.5 | 256.0 | 128.0 | **128.0** ✅ |
| 0x2 | 1.0 | 1024.0 | 512.0 | **512.0** ✅ |
| 0x3 | 1.5 | 2304.0 | 1152.0 | **1152.0** ✅ |
| 0x4 | 2.0 | 4096.0 | 2048.0 | inf (溢出) |

### 影响

MXFP4 的有效 K 维度减半，标准 golden（使用全部 K 个 FP4 值）与 gfrun 输出不匹配。

---

## 附加发现：Scale group 按 stored 元素计数（非 logical K）

在 MXFP4 (PACKED_FACTOR=2) 中，scale group 为 **32 个 stored 元素**（= 32 字节），
而非 32 个 logical K 元素（= 16 字节）。这意味着每个 scale 覆盖 32×PACKED_FACTOR
= 64 个 logical K 值。

这与内核代码一致：`kScaleK = (kStoredTK + kScaleGroup - 1) / kScaleGroup`
使用 stored K 计算 scale 组数。但 `gmAScale` 的 global_tensor 维度使用
`gK / kScaleGroup`（logical K），导致 scale 文件大小为 [M, K/32]，但仅有
前 K_stored/32 个元素被使用。

此行为本身不是 bug（内核和模型一致），但与标准 MX 的 "32 个 logical 元素 per
scale group" 不同，需确认是否为设计意图。

---

## 复现方法

```bash
export COMPILER_DIR=/Users/blacktraker/Programming/gitproj/DV4/linx-toolchain-build/output/linx_blockisa_llvm_musl/bin
cd /Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench

# 编译 + 验证（标准 OCP MX golden）
python3 /tmp/mt_lowp_standard_verify.py
```

验证脚本使用 PyTorch 生成随机 FP8 E4M3 数据，numpy 生成随机 FP4 E2M1 数据，
按 OCP MX 标准（bias=127, A&B scale, FP4 双 nibble）计算 golden，与 gfrun
输出比较。

## 建议修复方向

1. **Issue 1 (bias)**: 检查 gfrun 中 TMATMUL_MX 的 E8M0 解码逻辑，将 bias 从 254
   改为标准 127，或确认 ISA 规范中 E8M0 的实际 bias 定义。
2. **Issue 2 (B scale 忽略)**: 检查 TMATMUL_MX 的功能模型实现，确认 tBScale
   操作数是否参与 scale 计算。内核侧已正确传入，问题在模型侧。
3. **Issue 3 (FP4 高 nibble 忽略)**: 检查 FP4 E2M1x2 的 unpack 逻辑，确认
   TMATMUL 是否对高 nibble 也执行乘加。内核侧打包格式正确，问题在模型侧。
