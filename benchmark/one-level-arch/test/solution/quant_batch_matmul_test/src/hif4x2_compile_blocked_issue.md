# HiF4x2 (`__fp4_hif4x2`) 在新 TileOP API 下无法用于任何 MX/Cube 路径

## 问题描述

尝试在 superopbench `ops-20260904` + TileOP API `f94bc12` 环境下创建 hif4x2 的 MX matmul 变体，使用模式与 `HiF4_HiF4.cpp`（`matmul_hif4x2_mx`）一致：

```cpp
using fp4_t = __fp4_hif4x2;
using tileA = CubeTileA<__fp4_hif4x2, tM, tK>;   // 或者 Tile<Left, ..., CubeM32>
using tileB = CubeTileN8<__fp4_hif4x2, tK, tN>;
using tileAScale = Tile<Scaling, uint32_t, tM, kScaleK, RowMajor>;
TMATMUL_MX(c, a, sa, b, sb, fixp::keep_acc());
```

## 验证版本

| 组件 | Commit |
|---|---|
| SuperNPUBench | `a0ddcc3` (`ops-20260904`) |
| Linx-TileOP-API | `f94bc12` |
| llvm-project | `adcb87948` |

## 复现步骤

```bash
# 使用 __fp4_hif4x2 + uint32_t scale
make TESTCASE=quant_batch_matmul_test_hif4 \
     M=64 N=64 K=64 tM=32 tN=32 tK=64 \
     PLAT=linx COMPILER_DIR=<toolchain>/bin
```

编译失败。

## 关键日志

### 错误 1：CubeLayout 直接拒绝 HiF4X2

```
pto_tile.hpp:909:3: error: static assertion failed:
  CUBE CELL layouts support only 4/8/16/32-bit element widths
  and reject HiF4X2
```

根因在 `pto_tile.hpp:909`：

```cpp
static_assert(!IsCubeLayout || ((CubeElementBits == 4 || ... || CubeElementBits == 32)
              && type_traits<__fp4_hif4x2>::TypeCode != __type_fp4_hif4x2),
              "CUBE CELL layouts ... reject HiF4X2");
```

`__fp4_hif4x2` 被**显式排除**在所有 CubeLayout（CubeM32/CubeM16/CubeN8）之外。

### 错误 2：uint32_t scale 不被 MX contract 接受

即使改用 `__fp4_e1m2x2`（受支持的 fp4 类型），`Tile<Scaling, uint32_t, ...>` 也会被拒绝：

```
template_asm.hpp:2703:5: MX ScaleA dtype must be E8M0
```

MX contract (`validate_matrix_scale_contract`) 要求 scale 的 dtype 必须是 `__fp8_e8m0`，不接受 `uint32_t`。

### 错误 3：`matmul_hif4x2_mx` 因此成为死代码

`benchmark/one-level-arch/kernels/single_thread/matmul/matmul_mx.hpp:48` 中的 `matmul_hif4x2_mx` 函数模板使用 `CubeTileA<__fp4_hif4x2, ...>`，在该 TileOP API 版本下完全无法实例化。`HiF4_HiF4.cpp` 也因此无法编译。

## 影响范围

- `__fp4_hif4x2` 不能用于任何 CUBE 操作（TMATMUL、TMATMUL_MX）
- `uint32_t` 作为 MX scale 类型不被 contract 接受
- `matmul_hif4x2_mx` 函数无法在当前 API 下编译
- 唯一可工作的 fp4 MX 组合：`e2m1x2`/`e1m2x2` + `e8m0` + `smatrix_wfactor=32`（group-32）
