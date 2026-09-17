# HiF4X2 `TMATMUL_MX` 无法端到端执行：TileOP 合法性校验与 gfrun 4PE 协同校验均缺少支持

## 问题概述

按照当前 PTO 规范，`HiF4X2` 只能作为 Matrix-MX 输入，scale 载体为 `U32`，
每 64 个逻辑 K 元素共用一个 scale。将 `multi_thread/matmul` 的 HIF4X2
配置从普通 `TMATMUL` 改为 `TMATMUL_MX` 后，仍然无法端到端执行：

1. **TileOP 编译期阻塞**：公开 TileOP 头文件未将 `__fp4_hif4x2` 纳入
   Matrix-MX 输入集合，并将 MX scale 固定校验为 E8M0/group-32，导致合法的
   HiF4X2 + U32/group-64 调用被 `static_assert` 拒绝。
2. **gfrun 执行期阻塞**：临时绕过上述头文件校验后，原编译器后端可以生成正确的
   `TMATMULMX, HiF4x2` 指令；但 gfrun 的 4PE SharedTile 协同路径在
   `isFloatType()` 中遗漏 `DataType::HIF4`，执行前断言退出。

因此这是两个独立问题：**编译器后端可以编码该指令，但公开 TileOP 接口尚未完整
实现规范；gfrun 已有部分 HiF4 Matrix-MX 实现，但 4PE 协同路径的前置合法性校验
仍会拒绝它。**

## 规范依据

本地 `pto-spec`：`main@961fa81e`。

文件：`asl/tile/model/legality/matrix-functions.asl`

```asl
// Each Matrix-MX primary side MUST independently select group-32 E8M0 scale
// for MX FP8/FP4 carriers or group-64 raw U32 scale for HiF4X2. HiF4X2 MUST
// be accepted only by Matrix-MX input roles; ordinary Matrix MUST not gain it.
```

同一文件还明确规定：

```asl
if data_type == TileDataType_HiF4X2 then return 64; end;
...
if data_type == TileDataType_HiF4X2 then return TileDataType_U32; end;
```

测试用例据此使用：

- A/B dtype：`__fp4_hif4x2`
- 指令接口：`TMATMUL_MX<3>` / `TMATMUL_MX_ACC<3>`
- ScaleA dtype/shape：`uint32_t [M, ceil(K/64)]`
- ScaleB dtype/shape：`uint32_t [ceil(K/64), N]`

## 测试环境

| 组件 | 版本 |
|---|---|
| SuperNPUBench | 当前本地工作树，测试日期 2026-09-02 |
| 编译器 checkout | `linx-toolchain-build/main` |
| TileOP | `linx@ace71b6` |
| clang | 15.0.4，target `linx64v5-unknown-linux-musl` |
| pto-spec | `main@961fa81e` |
| gfrun / SuperScalarModel | `codex/pr-0.58.4-shared-model@762a72c3` |
| PE 配置 | `softcore.multiThreadNum=4` |
| 算子配置 | B=1, M=256, N=256, K=1024, tM=128, tN=256, tK=1024 |

编译器路径：

```text
/Users/blacktraker/Programming/gitproj/DV4/linx-toolchain-build/
  output/linx_blockisa_llvm_musl/bin
```

## 复现 1：公开 TileOP 接口编译失败

```bash
export COMPILER_DIR=/Users/blacktraker/Programming/gitproj/DV4/linx-toolchain-build/output/linx_blockisa_llvm_musl/bin

make -B -C benchmark/one-level-arch/test/kernel/multi_thread/matmul diss \
  TESTCASE=matmul_lowp LP_MODE=HIF4X2 \
  COMPILER_DIR="$COMPILER_DIR" \
  B=1 M=256 N=256 K=1024 tM=128 tN=256 tK=1024 \
  OBJ_ROOT=/tmp/hif4x2_matmul_mx
```

结果：`COMPILE_FAIL`，主要断言如下：

```text
Matrix A/B dtypes must be supported inputs from one numeric class
ScaleA presence must match the PTO MX type contract
ScaleB presence must match the PTO MX type contract
MX ScaleA dtype must be E8M0
MX ScaleB dtype must be E8M0
MX ScaleA valid shape must be M x ceil(K/32)
MX ScaleB valid shape must be ceil(K/32) x N
```

### TileOP 侧根因

`include/common/pto_tile.hpp` 中：

- `matrix_mx_input_supported()` 未包含 `__type_fp4_hif4x2`；
- `matrix_mx_input_needs_scale()` 因此错误地返回 `false`；
- 通用 `matrix_numeric_class()` 也未包含该类型，MX 路径无法推导 FP32 累加器。

`include/jcore/template_asm.hpp::validate_matrix_scale_contract()` 中：

- scale dtype 被固定为 `__type_fp8_e8m0`；
- scale group 被固定为 32；
- shape 被固定为 `ceil(K/32)`。

这些条件与规范要求的 HiF4X2 `U32 + group-64` 不一致。

> 注意：规范明确要求 HiF4X2 **仅用于 Matrix-MX**。修复时不应简单地让普通
> `TMATMUL` 也接受 HiF4X2；类型分类和累加器推导需要能够区分普通 Matrix 与
> Matrix-MX 路径。

## 复现 2：编译后端能够生成正确指令

仅在 `/tmp` 中使用临时 TileOP 头文件覆盖上述合法性校验，不修改编译器源码或安装
目录；同一编译器后端可以成功生成 ELF 和反汇编。

关键反汇编：

```text
BSTART.TLSU  TLOAD, HiF4x2
BSTART.TLSU  TLOAD, U32
...
BSTART.CUBE  TMATMULMX, HiF4x2
B.DATR       HiF4x2, byte0, Null
B.DIM        ..., 1024, ->lb2
B.IOS        S0, mask=1111       # A
B.IOS        S2, mask=1111       # ScaleA (U32)
B.IOS        S1, mask=1111       # B
B.IOS        S3, mask=1111       # ScaleB (U32)
```

这说明：

- clang/LLVM 后端认识 `TMATMULMX, HiF4x2`；
- HiF4X2、U32 scale、四个 Shared source binder 都能被编码；
- 当前首先需要修复的是 TileOP 公共接口/合法性层，而不是指令编码后端。

## 复现 3：gfrun 4PE 协同路径断言失败

对上述 ELF 执行：

```bash
/Users/blacktraker/Programming/gitproj/DV4/SuperScalarModel/bin/gfrun \
  -s softcore.multiThreadNum=4 \
  -f /tmp/hif4x2_matmul_mx.elf
```

结果：`RUN_FAIL`。

```text
gfrun: illegal instruction: ASSERTION FAILED:
((isFloatType(leftType) && isFloatType(rightType)) || ... ) &&
"cooperative TMATMUL input types are not in the current ASL matrix class"

func ExecuteOrSuspendCollective
file emulator/SoftCore.cpp:1726
```

### gfrun 侧根因

`emulator/SoftCore.cpp::ExecuteOrSuspendCollective()` 中存在两个不一致的集合：

```cpp
const auto isMxInputType = [](DataType type) {
    return ... || type == DataType::FP4 || type == DataType::FP4_1 ||
           type == DataType::HIF4;
};
```

该集合已经允许 `HIF4`；同文件的 `MatrixScaleGroupSize()`、
`MatrixScaleCarrierType()` 也已经分别返回 64 和 U32，`CubeEngine.cpp` 中也存在
HiF4 Matrix-MX 的解码、缩放和计算路径。

但紧接着的浮点类别检查遗漏了 `HIF4`：

```cpp
const auto isFloatType = [](DataType type) {
    return type == DataType::FP32 || ... ||
           type == DataType::FP4 || type == DataType::FP4_1;
    // missing DataType::HIF4
};
```

因此它先通过 `isMxInputType(HIF4)`，随后又被 `isFloatType(HIF4)` 拒绝，尚未进入
CubeEngine 的 HiF4 计算实现。

## 结果汇总

| 阶段 | 结果 | 判断 |
|---|---|---|
| 公开 TileOP `TMATMUL_MX` 编译 | FAIL | TileOP 合法性接口问题 |
| 临时绕过 TileOP 校验后生成 ELF | PASS | 编译器指令编码后端支持 |
| 反汇编检查 HiF4X2/U32/四源绑定 | PASS | 指令与 operand schema 正确生成 |
| gfrun 4PE SharedTile 执行 | FAIL | gfrun 协同路径类型校验遗漏 HIF4 |
| 数值对比 | 未执行 | gfrun 在进入 CubeEngine 前断言退出 |

## 建议修复

### TileOP / 编译器接口

1. 在 **Matrix-MX 专用**输入合法性中加入 `__type_fp4_hif4x2`，保持普通
   Matrix 路径继续拒绝该类型。
2. 由 primary dtype 分别推导 A/B 的 scale group 和 carrier：
   - HiF4X2：U32、group-64；
   - 其他需要 scale 的 MX 类型：E8M0、group-32。
3. scale shape 分别校验为 `ceil(K/groupA)` 和 `ceil(K/groupB)`。
4. 增加 HiF4X2 Matrix-MX 的编译测试，覆盖 Local/Shared、MX/MX_ACC。

### gfrun

1. 在协同 Matrix-MX 的浮点类型分类中加入 `DataType::HIF4`，或统一复用一个
   Matrix/Matrix-MX 合法性函数，避免 `isMxInputType()` 与 `isFloatType()` 漂移。
2. 增加 4PE SharedTile HiF4X2 + U32/group-64 scale 的执行与数值回归。
3. 修复后继续验证 CubeEngine 的 U32 scale 解码、64-lane 分组及 packed nibble 顺序。

## 结论

HIF4X2 用普通 `TMATMUL` 确实是错误接口；改为 `TMATMUL_MX` 后，用例语义符合
当前规范。但当前版本仍不能跑通，原因并非用例继续选错接口，而是：

- **TileOP 公共接口尚未实现 HiF4X2 Matrix-MX 的完整合法性规则；**
- **gfrun 4PE 协同执行路径也遗漏了 HIF4 的浮点类别。**

两个组件都需要修复后，才能完成端到端数值验证。
