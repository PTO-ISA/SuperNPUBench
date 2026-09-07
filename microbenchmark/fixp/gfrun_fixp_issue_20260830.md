# [gfrun][PTO 0.58.4] FIXP TMATMUL 94 个用例中 26 个被模型解码或契约检查拒绝

## 问题摘要

使用更新后的主 `linx-toolchain-build` 编译 `SuperNPUBench/microbenchmark/fixp` 全部 94 个模式：

| 阶段 | PASS | FAIL | TIMEOUT |
| --- | ---: | ---: | ---: |
| 编译并生成 `.elf.diss` | 94 | 0 | — |
| gfrun | 68 | 26 | 0 |

26 个失败全部发生在 gfrun 的指令解码或 CUBE operand-contract 检查阶段，尚未进入 benchmark 数值结果比较。逐项对照当前 TileOP API 的编译期约束、生成的反汇编和 gfrun 实现后，未发现算子侧 dtype、shape、layout 或 operand-role 写错；失败集中为 4 类模型问题：

| 类别 | 数量 | 结论 |
| --- | ---: | --- |
| RowMax/GroupMax 多目的 B.IOT | 14 | gfrun 未保留非 `last` 的 D 目的操作数 |
| MX 可选 ScaleA/ScaleB | 6 | gfrun 固定按“两侧都有 scale”计算源操作数数量 |
| B.FPATR TransA/TransB/CScale 位 | 5 | gfrun decode pattern 将合法扩展位固定为 0 |
| 无符号 TMATMUL.ACC | 1 | TileOP 契约规定 U32 accumulator，CubeEngine 仍只接受 S32/F32 |

因此本轮没有通过修改算子来绕过断言。此类绕行会删除被测特性或改变测试语义，例如把 U32 ACC 改为 S32、给 FP16/BF16 MX 人为添加 scale、删除 RowMax/GroupMax 输出。

## 验证环境

| 组件 | 分支 | commit |
| --- | --- | --- |
| SuperNPUBench | `main` | `58d436c01f9a` |
| linx-toolchain-build | `main` | `e6a31efb4cfb` |
| llvm-project | `dev-llvm15_56` | `adcb879481d8` |
| Linx-TileOP-API | `linx` | `955030977c25` |
| SuperScalarModel / gfrun | `codex/pr-0.58.4-shared-model` | `687c37b1563e` |

编译器：

```text
clang version 15.0.4 (linx64v5-musl-local adcb879481d8feb73e17a4134f8ae955bd21ee32)
Target: linx64v5-unknown-linux-musl
```

本轮 TileOP API 相比上一轮从 `c080ec7e` 更新到 `95503097`。相关代码变化主要是统一 CUBE `TLOAD/TSTORE` 入口和 range modifier 接口；代表性失败用例的 `BSTART.CUBE/B.FPATR/B.DIM/B.IOT/B.IOS` 序列与更新前一致。

安装到工具链中的 `template_asm.hpp` 与 `src/Linx-TileOP-API` 源文件 SHA-1 相同：

```text
6ff45e009aa1b8045bb8f3df052772b93a160e72
```

## 复现步骤

### 1. 编译全部 FIXP 模式

```bash
cd /path/to/SuperNPUBench

export COMPILER_DIR=/Users/blacktraker/Programming/gitproj/DV4/linx-toolchain-build/output/linx_blockisa_llvm_musl/bin

bash microbenchmark/fixp/compile.all \
  COMPILER_DIR="$COMPILER_DIR" \
  OBJ_ROOT=/tmp/fixp_regression
```

本轮 94 个模式均成功生成一对文件：

```text
fixp_tmatmul_<mode>_M32_N32_K32_tM32_tN32_tK32.elf
fixp_tmatmul_<mode>_M32_N32_K32_tM32_tN32_tK32.elf.diss
```

### 2. 执行 gfrun

普通模式：

```bash
gfrun -t 1 -f <elf>
```

cooperative Shared 模式 `shared`、`s8_shared`、`trans_a`、`trans_b`、`trans_ab`：

```bash
gfrun -t 1 -s softcore.multiThreadNum=4 -f <elf>
```

PASS 判定条件为返回码 0，且日志同时包含：

```text
Reach the End of Benchmark
R2 = 0
```

## 失败清单与根因

### A. 多目的 RowMax/GroupMax：14 FAIL

影响模式：

```text
rowmax
rowmax_init
s8_rowmax
f16_groupmax
rowgroup_maxabs
groupmax_8
groupmax_16
groupmax_32
groupmax_48
groupmax_64
groupmax_80
groupmax_96
groupmax_112
groupmax_128
```

其中 13 个失败于：

```text
CUBE D must hold the M x N FPATR output region
```

`s8_rowmax` 失败于：

```text
CUBE local source stream does not match the current ASL contract
```

TileOP API 的多输出发射顺序是：先发布 D，再发布 RowMaxOut/GroupMaxOut，只有最后一个目的带 `last`：

```asm
B.IOT mask=1111, ->D<DstSize>
B.IOT mask=1111, last, ->RowMaxOut<RowSize>
```

`rowmax` 的 gfrun `-t 2` trace 显示，第一条非 `last` 的目的 B.IOT 没有产生 destination，只有最后的 RowMaxOut 被识别：

```text
B.IOT 0x6
B.IOT 0x4 -> U<1KB>
ASSERTION FAILED: ... "CUBE D must hold the M x N FPATR output region"
```

因此 RowMaxOut/GroupMaxOut 被误当成 `dstTile[0]`（D）。FP32/F16 模式随即因容量不足失败；`s8_rowmax` 的两个物理 Tile 恰好同为 1 KiB，先通过容量检查，随后因 `dstTile.size()==1` 而不是 2 失败。

算子侧的 RowMax/GroupMax dtype 和 valid shape 已由 `validate_matrix_postprocess_contract()` 编译期检查：

```text
RowMaxOut: accumulator dtype, M x 1
GroupMaxOut: accumulator dtype, M x ceil(N / GroupN)
```

建议修复：在 gfrun B.IOT 解码/operand 构造路径中，无论是否为 `last`，都应保留编码存在的 destination，并按 bundle 顺序追加到 `block.dstTile`。

### B. MX 可选 scale：6 FAIL

影响模式：

```text
mx_scale0
mx_scale_a
mx_scale_b
gemv_mx_scale0
gemv_mx_scale_a
gemv_mx_scale_b
```

统一断言：

```text
CUBE local source stream does not match the current ASL contract
```

当前 TileOP API 的 MX 契约按每侧输入 dtype 独立决定 scale 是否存在：

```text
FP16/BF16: 不需要 E8M0 scale
FP8/FP4: 需要对应侧 E8M0 scale
```

因此三个模式的合法数学源数量分别为：

```text
scale0:  A, B                   -> 2
scale_a: A, ScaleA, B           -> 3
scale_b: A, B, ScaleB           -> 3
```

但 `isa/Block.cpp::ValidateLocalCubeMatrixContract()` 当前只要 `withMx` 为真，就无条件为 A/B 两侧各增加一个 scale：

```cpp
const size_t mathematicalSources = (withAcc ? 1u : 0u) + 1u +
                                   (withMx ? 1u : 0u) + 1u +
                                   (withMx ? 1u : 0u) +
                                   (withBias ? 1u : 0u) +
                                   (block.cScaleEn ? 1u : 0u);
```

建议修复：先从 A/B 的实际 dtype 计算 `leftNeedsScale/rightNeedsScale`，再动态计算 mathematical source count 和 A/B/scale 索引；TMATMUL_MX 与 TGEMV_MX 应共用同一套逻辑。

### C. B.FPATR 扩展位无法解码：5 FAIL

影响模式：

```text
acc_cscale
mxacc_cscale
trans_a
trans_b
trans_ab
```

统一断言：

```text
m_handlers.find(grp) != m_handlers.cend()
```

这些用例在执行 B.FPATR 时即被归入未注册的 instruction group。当前合法编码包括：

```text
base      0x2023
TransA    bit 7
TransB    bit 8
CScaleEn  bit 9
bit 10    reserved, must be 0
```

但 `isa/codec/decodefiles/block32.decode` 当前 pattern 为：

```text
B_FPATR  ...... ... .... . . . . 010 0 0000 010 001 1  @B_FPATR
```

它只匹配扩展位为 0 的基础编码。例如 `acc_cscale` 的 `0x00002223` 和 `trans_a` 的 `0x000020a3` 无法被识别为 B.FPATR。

算子侧已经满足 API 约束：CScale 是 U8、CUBE_M32、valid Mx1；transpose 只用于 cooperative Shared primary。

建议修复：将 TransA/TransB/CScaleEn 对应位改为 decode field/wildcard，只保留 bit 10 为固定 0，然后重新生成 decoder，并增加 0x20a3、0x2123、0x21a3、0x2223 的解码单测。

### D. U32 TMATMUL.ACC 被旧执行约束拒绝：1 FAIL

影响模式：

```text
unsigned_acc
```

断言：

```text
TMATMUL.ACC C must be an MxN AccType (S32/F32) Local Tile
```

TileOP API 的统一数值类型契约明确规定：

```text
floating input -> FP32 accumulator
signed input   -> S32 accumulator
unsigned input -> U32 accumulator
```

该用例使用 U8 A/B、U32 C/D、CUBE_M32 MxN layout，符合 API 静态约束。gfrun 的 Local contract 已能推导 unsigned accumulator，但 `CubeEngine.cpp` 的 TMATMUL_ACC 执行分支仍硬编码：

```cpp
accInfo->dataType == DataType::FP32 ||
accInfo->dataType == DataType::INT32
```

建议修复：执行路径与统一 `CubeAccumulatorType(leftType, rightType)` 对齐，接受 UINT32，并使用推导出的 accumulator type 读取和计算 C。

## 为什么判定不是算子问题

1. 94/94 模式通过当前已安装 TileOP API 的编译期 dtype/layout/shape/operand-role 静态检查。
2. FIXP 算子使用 `CubeTileM16/M32`、`CubeTileN8` 和 `CubeAccumulatorM16/M32`，未使用已淘汰的普通 NORM CUBE carrier。
3. CScale、RowMax、GroupMax、MX scale 和 transpose 均通过 API 专项 contract validator。
4. 所有 26 个失败都在 gfrun decode/preflight 阶段触发，没有数值 mismatch。
5. TileOP 更新前后代表性失败用例的 CUBE compute bundle 未变化，运行结果仍为同一组 68 PASS / 26 FAIL。

## 预期结果

1. gfrun 能完整解析多目的 B.IOT，并正确保留 D、RowMaxOut、GroupMaxOut 的顺序。
2. MX operand arity 按 A/B dtype 独立决定 ScaleA/ScaleB 是否存在。
3. B.FPATR 能解码 TransA、TransB 和 CScaleEn 合法位。
4. 无符号普通矩阵的 TMATMUL.ACC 使用 U32 accumulator。
5. 94 个 ELF 均能越过模型合法性检查并进入实际功能执行；若之后存在数值问题，再按独立问题分析。

## 附件

完整复现包（94 个 ELF + 94 个 `.elf.diss`）：

```text
SuperNPUBench_microbenchmark_fixp_all_20260830_tileop9550309.tar.gz
SHA-256: cd7d9dc93fafa5478ff6b3cf4f5416beb20cec91f66534b332db487b366a94e6
```

当前 gfrun PASS-only 包（68 个 ELF + 68 个 `.elf.diss`）：

```text
SuperNPUBench_microbenchmark_fixp_gfrun_pass_20260830_tileop9550309.tar.gz
SHA-256: 2d8085d55880c8903fd0abcaa4dea0ddd507a6b70b3e31e8dafd797c2193580b
```

完整运行结果由 `summary.tsv` 记录，字段为：

```text
elf, mode, config, status, rc, note
```
