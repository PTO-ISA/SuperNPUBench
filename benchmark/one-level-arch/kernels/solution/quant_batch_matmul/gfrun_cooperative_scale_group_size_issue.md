# Issue: cooperative TMATMUL_MX 对 packed FP4x2（HiF4X2/E2M1X2）scale group size 使用 logical 而非 carrier 单位，导致 2x 精度偏差

## 摘要

4-PE cooperative `TMATMUL_MX` 在处理 packed FP4x2 格式（`__fp4_hif4x2` / `__fp4_e2m1x2`）的 MX scale 时，`MatrixScaleGroupSize` 返回 **logical** group size（64 for HiF4, 32 for MXFP4），但 cooperative 执行路径（`ExecuteOrSuspendCollective`）中的矩阵 K 维度 `k` 以 **carrier** 单位传入（`tKv = tK / 2`）。两者单位不匹配导致 scale 索引步幅翻倍，实际每 128 logical elements 共享 1 个 scale word（而非 ISA 规定的 64），产生一致的 **2x 精度偏差**。

## 影响范围

- 所有使用 packed FP4x2 格式（PackedFactor=2）的 4-PE cooperative `TMATMUL_MX` / `TMATMUL_MX_ACC` 内核
- 包括 `__fp4_hif4x2`（HiF4, scale group=64 logical）和 `__fp4_e2m1x2`（MXFP4, scale group=32 logical）
- **不影响** PackedFactor=1 的 FP8 格式（`__fp8_e4m3` 等），因为其 carrier K == logical K，单位一致
- 单 PE 路径（非 cooperative）不受影响，因为该路径使用 `FIXED_MATRIX_ZOOM_HIF4_SIZE = 32`（carrier 单位）而非 `MatrixScaleGroupSize`

## 环境

| 组件 | 版本/Commit |
|------|-------------|
| SuperScalarModel | `e8d2cc23`（merge origin/main 含 `de3ea07f` ScaleB 修复） |
| SuperNPUBench | `quant_batch_matmul_hif4_mt` 4-PE cooperative HiF4X2 kernel |
| 工具链 | linx-toolchain-build `linx_blockisa_llvm_musl` (clang 15.0.4, TileOP-API v0.58.3) |
| 测试形状 | M=256, N=256, K=256, tM=64, tN=32, tK=64, B=1 |

## 复现步骤

```bash
# 编译 hif4 4-PE cooperative kernel (res_check=on)
cd SuperNPUBench/benchmark/one-level-arch/test/solution/quant_batch_matmul_test
make TESTCASE=quant_batch_matmul_test_hif4 \
    M=256 N=256 K=256 tM=64 tN=32 tK=64 B=1 \
    PLAT=linx res_check=on \
    COMPILER_DIR=<toolchain>/bin

# 运行精度验证（ones 输入，最易复现 2x 偏差）
cd src
python3 verify_hif4.py \
    -d <elf_path> \
    --gfrun <SuperScalarModel>/bin/gfrun \
    --gfrun-args "-t 1 -s softcore.multiThreadNum=4 -f" \
    --ones --timeout 300
```

### 预期行为

gfrun 输出与 golden model 一致（atol=5e-2, rtol=5e-2），`verify_hif4.py` 报告 PASS。

### 实际行为

gfrun 输出恰好为 golden 的 **1/2**：

```
actual head: [[38.28125 38.28125 ...]]   （全部一致）
golden head: [[76.5625  76.5625  ...]]   （全部一致）
ratio = golden / actual = 2.0
```

## 根因分析

### 1. `MatrixScaleGroupSize` 返回 logical 单位

**文件：`emulator/engine/CubeEngine.cpp:32-35`**

```cpp
size_t MatrixScaleGroupSize(DataType type)
{
    return type == DataType::HIF4 ? 64u : 32u;
}
```

- HiF4：返回 64（logical K elements per scale group）
- MXFP4：返回 32（logical K elements per scale group）

这些值在 **logical** 维度下是正确的（HiF4 每 64 个逻辑元素共享 1 个 U32 scale word；MXFP4 每 32 个逻辑元素共享 1 个 E8M0 byte）。

### 2. Cooperative 路径的 `k` 以 carrier 单位传入

**文件：`emulator/SoftCore.cpp:2496-2506`（cooperative TMATMUL_MX scale 应用）**

```cpp
if (rightNeedsScale) {
    for (size_t inner = 0; inner < k; ++inner) {       // k = carrier K (tKv)
        for (size_t col = 0; col < n; ++col) {
            const size_t groupSize = MatrixScaleGroupSize(rightType);  // 64 (logical!)
            const uint64_t scaleRaw =
                rightScale[(inner / groupSize) * n + col];             // 步幅错误
            const uint32_t scaleBits = MatrixScaleToFP32(
                rightType, scaleRaw, inner % groupSize);               // lane 错误
            right[inner * n + col] = CubeCalculate::EleMul(
                right[inner * n + col], scaleBits, DataType::FP32,
                this->status);
        }
    }
}
```

**文件：`emulator/SoftCore.cpp:2559-2568`（左操作数 A 的相同问题）**

```cpp
if (leftNeedsScale) {
    const size_t groupSize = MatrixScaleGroupSize(leftType);  // 64 (logical!)
    const size_t scaleBlocks = (k + groupSize - 1u) / groupSize;
    for (size_t row = 0; row < sourceRows; ++row) {
        for (size_t inner = 0; inner < k; ++inner) {          // k = carrier K
            const uint64_t scaleRaw =
                leftScale[row * scaleBlocks + inner / groupSize];
            // ...
        }
    }
}
```

### 3. 单位不匹配导致 scale 索引步幅翻倍

以 HiF4X2 测试用例（tK=64, tKv=32, gK=256, gKv=128）为例：

| 维度 | logical 值 | carrier 值 | `MatrixScaleGroupSize` 返回值 |
|------|-----------|-----------|------------------------------|
| K (tile) | 64 | 32 (tKv) | 64 (logical) |
| K (global) | 256 | 128 (gKv) | 64 (logical) |
| scale blocks | 4 (= 256/64) | — | — |

在 cooperative 路径中 `k = tKv = 32`（carrier），但 `groupSize = 64`（logical）：

- `inner` 范围：0..31（carrier）
- `inner / groupSize` = `inner / 64` → 对所有 inner ∈ [0, 31]，结果恒为 **0**
- 即 32 个 carrier 元素全部使用第 0 个 scale word，而实际应有 `32 / 32 = 1` 个 scale group

对于 gK=256（gKv=128, 4 个 scale blocks）：
- 正确行为：`inner`（carrier 0..127）应按 `inner / 32` 分成 4 组
- 实际行为：`inner / 64` 只分成 2 组（0..63 → group 0, 64..127 → group 1）
- 每 128 个 carrier（= 256 logical）共享 1 个 scale word，而非每 64 个 carrier（= 128 logical）

### 4. 2x 偏差的数值推导

以 `--ones` 测试为例：

- 输入 A = B = 1.0（全 1）
- scale_val = 1.0 / 3.5 = 0.2857
- E6M2 encode → byte 0xb9，decode → 0.3125
- HiF4 quantize 1.0/0.3125 = 3.2 → code 7 (value=1.75)

**Golden model（正确逻辑）：**
- dequant_per_element = 1.75 × 0.3125 = 0.546875
- output = K × dequant² = 256 × 0.546875² = **76.5625**

**gfrun actual（bug 行为）：**
- 由于 scale group 翻倍，等效 scale 变为 0.3125 / √2... 不，实际上 scale 被重用了
- output = **38.28125** = 76.5625 / 2

比值恒为 2.0，因为每 2 个 scale group 共用 1 个 scale word，等效于 scale group size 翻倍。

### 5. 单 PE 路径不受影响

**文件：`emulator/engine/CubeEngine.cpp:1719-1723`（单 PE non-cooperative 路径）**

```cpp
uint32_t kscaleA = (dataTypeA == DataType::HIF4)
    ? (k / FIXED_MATRIX_ZOOM_HIF4_SIZE)    // 32 (carrier!)
    : (k / FIXED_MATRIX_ZOOM_SIZE);         // 32 (carrier!)
```

单 PE 路径使用 `FIXED_MATRIX_ZOOM_HIF4_SIZE = 32`（carrier 单位），与 `k` 的 carrier 单位一致，无此 bug。

## 对比：单 PE vs cooperative 路径的 scale group size

| 路径 | scale group size 来源 | 单位 | 与 k 的单位匹配 |
|------|----------------------|------|----------------|
| 单 PE (CubeEngine TMATMULMX) | `FIXED_MATRIX_ZOOM_HIF4_SIZE = 32` | carrier | ✅ 一致 |
| Cooperative (SoftCore ExecuteOrSuspendCollective) | `MatrixScaleGroupSize(HIF4) = 64` | logical | ❌ 不一致 |

## 涉及文件

| 文件 | 行号 | 角色 | 状态 |
|------|------|------|------|
| `emulator/engine/CubeEngine.cpp` | 32-35 | `MatrixScaleGroupSize` 定义 | **BUG 来源**（返回 logical 而非 carrier） |
| `emulator/SoftCore.cpp` | 2496-2506 | cooperative 右操作数 scale 应用 | **受影响**（使用错误的 groupSize） |
| `emulator/SoftCore.cpp` | 2559-2568 | cooperative 左操作数 scale 应用 | **受影响**（使用错误的 groupSize） |
| `emulator/engine/CubeEngine.cpp` | 24-26 | `FIXED_MATRIX_ZOOM_HIF4_SIZE = 32` | 正确（carrier 单位） |
| `emulator/engine/CubeEngine.cpp` | 1719-1723 | 单 PE 路径 kscale 计算 | 正确（使用 carrier 单位） |

## 修复建议

### 方案 A：`MatrixScaleGroupSize` 返回 carrier 单位（推荐）

将 `MatrixScaleGroupSize` 的返回值改为 carrier 单位，即除以 PackedFactor：

```cpp
size_t MatrixScaleGroupSize(DataType type)
{
    // Carrier group size = logical group size / PackedFactor.
    // HiF4: 64 logical / 2 = 32 carriers per scale word.
    // MXFP4/MXFP8: 32 logical / 1 = 32 carriers per scale byte.
    // Both resolve to 32 carriers, matching FIXED_MATRIX_ZOOM_*_SIZE.
    return 32u;  // unified carrier group size
}
```

或更精确地保留类型区分：

```cpp
size_t MatrixScaleGroupSize(DataType type)
{
    switch (type) {
        case DataType::HIF4:   return 32u;  // 64 logical / PackedFactor(2)
        case DataType::FP4:    return 16u;  // 32 logical / PackedFactor(2)
        default:               return 32u;  // 32 logical / PackedFactor(1)
    }
}
```

**注意**：需确认 `MatrixScaleGroupSize` 的所有调用点是否都期望 carrier 单位。单 PE 路径不使用此函数（使用 `FIXED_MATRIX_ZOOM_*_SIZE`），故不受影响。

### 方案 B：cooperative 路径中将 k 转换为 logical 单位

在 `ExecuteOrSuspendCollective` 中，对 packed FP4x2 类型将 `k` 乘以 PackedFactor 后再用于 scale 索引：

```cpp
const size_t logicalK = (IsPackedFp4x2(rightType)) ? k * 2 : k;
const size_t groupSize = MatrixScaleGroupSize(rightType);  // logical
// ... 使用 logicalK 替代 k 进行 scale 索引
```

此方案改动范围更大，且需同时修改 left/right 两条路径。

### 方案 C：新增 carrier 版本的 group size 函数

```cpp
size_t MatrixScaleCarrierGroupSize(DataType type)
{
    return type == DataType::HIF4 ? 32u : 32u;
}
```

在 cooperative 路径中使用 `MatrixScaleCarrierGroupSize`，单 PE 路径继续使用 `FIXED_MATRIX_ZOOM_*_SIZE`。

## 验证证据

### ones 输入测试（M=256, N=256, K=256, tM=64, tN=32, tK=64）

```
status: FAIL
max_abs: 38.28125
golden_at_max: 76.5625
actual_at_max: 38.28125
ratio = golden / actual = 2.0 (恒定)
```

- gfrun 输出全部一致（38.28125），表明 scale 被均匀应用（无 E1_8/E1_16 干扰）
- golden model 使用 `MatrixScaleToFP32` 的单 U32 word 解码（与 cooperative 路径一致）
- 2x 比值精确对应 scale group size 翻倍（64 carrier → 128 carrier per scale word）

### random 输入测试（seed=42, input_scale=0.5）

```
status: FAIL
max_abs: 17.4377
mse: 1.1614e+01
ratio varies (非恒定 2x，因不同 scale group 被错误合并)
```

随机输入下偏差不恒定，因为不同 scale group 的 scale word 被错误地共用，导致部分元素 scale 偏大/偏小。

## 相关 issue

- `gfrun_scaleB_runtime_check_issue.md`（已由 `de3ea07f` 修复）：cooperative TMATMUL_MX 的 ScaleB runtime check 参数顺序 bug。该修复使 gfrun 不再崩溃，但暴露了本 issue 的精度偏差问题。
- `hif4x2_compile_blocked_issue.md`：HiF4X2 在旧 TileOP API 下无法用于 Cube 路径的编译问题（已通过 PTO v0.58 cooperative SharedTile 路径解决）。

## 补充说明

本 issue 的发现过程：

1. 将 `quant_batch_matmul_hif4.hpp` 从单 PE 改写为 4-PE cooperative 模式（参照 `quant_batch_matmul_mxfp4_mt.hpp`）
2. 编译通过，但 gfrun 崩溃于 ScaleB runtime check（第一个 bug，已由 `de3ea07f` 修复）
3. 合并 origin/main 修复后重新构建 gfrun，gfrun 不再崩溃
4. 运行精度验证，发现 2x 偏差
5. 排除 golden model 错误（确认 `MatrixScaleToFP32` 解码逻辑一致）
6. 排除 `ScaleLExtract` 干扰（确认 cooperative 路径使用 `MatrixScaleToFP32` 而非 `ScaleLExtract`）
7. 定位到 `MatrixScaleGroupSize` 返回 logical 单位与 cooperative 路径 carrier 单位的 `k` 不匹配
