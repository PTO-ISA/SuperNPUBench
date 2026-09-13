# Tile 连续轴 32B 对齐 static_assert：打包 fp4（单 MX block=16B）无法构造 RowMajor/NoneBox tile

## 摘要

`pto_tile.hpp:649-656` 在 **Tile 类型实例化时** 有一条 static_assert，要求
`RowMajor + NoneBox` 布局下**连续轴（-1 轴 Cols）的物理字节宽必须是 32 字节（256 位）的整数倍**：

```
Cols * type_traits<DType>::bits % (32 * 8) == 0
```

对打包类型 `__fp4_e2m1x2`（`type_traits<>::bits == 8`，2 个 fp4 值/字节）而言，一个 MX block =
32 个 fp4 值 = **16 个打包字节**。用它构造 `[Rows, 16]` 的 `RowMajor/NoneBox` tile 时
`16 * 8 = 128`，`128 % 256 != 0`，**编译期 static_assert 失败**。必须把连续轴撑到 **2 个 MX
block（Cols=32 打包字节 = 64 个 fp4 值）** 才能通过。

这直接约束了 `dynamic_mx_quant` 的 fp4 输出 kernel（被迫 `TileN % 64 == 0` 或把物理宽 pad 到
64），也让「一个 tile 只装一个 MX block」的自然写法不可用。

想请教维护者：**这条 32B 下界对 sub-32B 的打包/窄类型是否为有意约束？能否放宽（或提供官方的
box/fractal 豁免路径作为推荐写法）？**

---

## 环境

| 项 | 值 |
|---|---|
| 编译器 | `clang version 15.0.4 (linx64v5-musl-local abc023323098a5bf547b6e778aaf63794a3a1876)` |
| Target | `linx64v5-unknown-linux-musl` |
| 头文件 | `.../lib/clang/15.0.4/include/tileop-api/common/pto_tile.hpp` |
| 关键宏 | `-D__linx -DENABLE_TENSOR_INSTR` |
| 编译选项 | `-c -mlxbc -fenable-matrix -O2 -std=c++20`（其余 `-mllvm` 见下方完整命令行） |

---

## 约束位置（`pto_tile.hpp:649-656`）

```cpp
static_assert(
    (BFractal_ == BLayout::RowMajor && SFractal_ == SLayout::NoneBox && Cols * type_traits<DType>::bits % (32 * 8) == 0) ||
    (BFractal_ == BLayout::ColMajor && SFractal_ == SLayout::NoneBox && Rows * type_traits<DType>::bits % (32 * 8) == 0) ||
    (SFractal_ != SLayout::NoneBox) && (Rows % InnerRows == 0 && Cols % InnerCols == 0),
    "BFractal_ is RowMajor and SFractal_ is NoneBox: Rows must be 32 bytes align, \
      BFractal_ is ColMajor and SFractal_ is NoneBox: Cols must be 32 bytes align, \
      SFractal_ in not NoneBox: Rows/Cols must be integer multiple of InnerRows/InnerCols."
);
```

规则（按布局）：

| 布局 | 连续轴 | 约束 |
|---|---|---|
| `RowMajor` | -1 轴（Cols） | `Cols * sizeof(DType) % 32B == 0` |
| `ColMajor` | -2 轴（Rows） | `Rows * sizeof(DType) % 32B == 0` |
| `SLayout != NoneBox`（box/fractal） | — | 仅需 `Rows%InnerRows==0 && Cols%InnerCols==0`（**不受 32B 字节下界约束**） |

换算成连续轴最小元素数：

| DType | 每元素字节 | 连续轴最小元素数 |
|---|---|---|
| `__fp4_e2m1x2`（打包，2 值/字节） | 1 字节 = 2 值 | **64（= 32 打包字节 = 2 个 MX block）** |
| `__fp8_e4m3` / `uint8` | 1 | 32 |
| `bf16` / `uint16` | 2 | 16 |
| `float` | 4 | 8 |

> 注：断言只看 tile 的**物理 Cols（分配宽度）**，不看 valid/box 收窄宽度——因此靠 `ValidCol=1`
> 收窄一个逻辑上很窄的 tile 并不能绕过它，被卡的永远是物理连续轴。

---

## 最小复现

文件：`test/kernel/multi_thread/quant/dynamic_mx_quant/src/align32_probe.cpp`

```cpp
#include <common/pto_tileop.hpp>
#include <cstdint>

using namespace pto;

int main() {
#ifdef ALIGN_OK
    // Cols = 32 打包字节（= 2 个 MX block = 64 个 fp4 值）：32*8 % 256 == 0 -> OK
    using tile_o = Tile<Location::Vec, __fp4_e2m1x2, 8, 32, BLayout::RowMajor>;
#else
    // Cols = 16 打包字节（= 1 个 MX block = 32 个 fp4 值）：16*8 % 256 == 128 != 0 -> FAIL
    using tile_o = Tile<Location::Vec, __fp4_e2m1x2, 8, 16, BLayout::RowMajor>;
#endif
    tile_o oq;        // 实例化 Tile 类型 -> 触发 static_assert
    (void)sizeof(oq);
    return 0;
}
```

### 构建命令

```bash
export COMPILER_DIR=/path/to/linx_blockisa_llvm_musl/bin
export LINX_SYSROOT=/path/to/linx_blockisa_llvm_musl/sysroot/usr
cd SuperNPUBench/benchmark/one-level-arch/test/kernel/multi_thread/quant/dynamic_mx_quant

# 复现失败（默认，Cols=16 单 block）
make TESTCASE=dynamic_mx_quant TYPE=ALIGN32_PROBE

# 对照通过（Cols=32 两 block）
make TESTCASE=dynamic_mx_quant TYPE=ALIGN32_PROBE CC_OPTS=-DALIGN_OK
```

完整底层编译命令（Makefile 展开后）：

```
clang++ -c -mlxbc -fenable-matrix -O2 -mllvm -enable-all-vector-as-tilereg=true \
  -mllvm -linxv5-enable-HL-Inst-Opt=true -mllvm -linxv5-enable-dim-opt=true \
  -mllvm -linxv5-enable-ldst-bridge=false -mllvm -linxv5-enable-continuous-mem-opt=true \
  -mllvm -linxv5-enable-tile-clock-hand=false -mllvm -linxv5-enable-simt-clock-hand=true \
  -mllvm -enable-misched=false -std=c++20 -D__linx -DENABLE_TENSOR_INSTR \
  -I <one-level-arch>/include -I <one-level-arch>/test/common \
  -I <one-level-arch>/test/common/src -I <one-level-arch>/kernels -I <one-level-arch>/models \
  src/align32_probe.cpp -o align32_probe.o
```

---

## 实际报错（FAIL 案例，Cols=16）

```
pto_tile.hpp:649:3: error: static assertion failed due to requirement
'((pto::BLayout)0 == BLayout::RowMajor && (pto::SLayout)0 == SLayout::NoneBox &&
  Cols * type_traits<__fp4_e2m1x2>::bits % (32 * 8) == 0) ||
 ((pto::BLayout)0 == BLayout::ColMajor && (pto::SLayout)0 == SLayout::NoneBox &&
  Rows * type_traits<__fp4_e2m1x2>::bits % (32 * 8) == 0) ||
 ((pto::SLayout)0 != SLayout::NoneBox) && (Rows % InnerRows == 0 && Cols % InnerCols == 0)':
 BFractal_ is RowMajor and SFractal_ is NoneBox: Rows must be 32 bytes align,
 BFractal_ is ColMajor and SFractal_ is NoneBox: Cols must be 32 bytes align,
 SFractal_ in not NoneBox: Rows/Cols must be integer multiple of InnerRows/InnerCols.
  static_assert(
  ^
align32_probe.cpp:36:12: note: in instantiation of template class
 'pto::Tile<pto::Location::Vec, __fp4_e2m1x2, 8, 16, pto::BLayout::RowMajor, 8, 16,
            pto::SLayout::NoneBox, 512, pto::PadValue::Null>' requested here
    tile_o oq;        // 实例化 Tile 类型 -> 触发 static_assert
           ^
1 error generated.
make: *** [.../Makefile.common:138: .../align32_probe.o] Error 1
```

## 对照结果（OK 案例，Cols=32，`-DALIGN_OK`）

编译 + 链接通过，`exit 0`，产出
`output/.../elf/kernel_multi_thread_quant_dynamic_mx_quant/dynamic_mx_quant_align32_probe.elf`。

**唯一变量是 tile 连续轴 Cols（16 → 32）**，其余完全一致，证明失败根因即该 32B 对齐 static_assert。

---

## 对 dynamic_mx_quant 的影响

fp4 输出 tile 的**打包轴（Post）与量化归约轴无关**，但 32B 下界把它顶到 ≥ 2 个 MX block：

- **尾轴** kernel：单个 MX block 的 fp4 输出物理宽 = `BlockSize/2 = 16` 打包字节 < 32B，故 `[TileM,16]`
  这类单 block 中间 tile **根本无法构造**。当前规避：把物理宽 pad 到 `PW = ⌈BlockSize/64⌉×64`
  并对每个 op 列装箱到有效 `BlockSize`（见 `dynamic_mx_quant_tail_ocp_fp4.hpp`）。
- **非尾轴** kernel：连续轴 = 自由的 `TileN`，被迫 `TileN % 64 == 0`（≥ 2 block）。它与 TileSize
  上界（`Rows*Cols*sizeof ≤ 8192`）在同一根 `TileN` 上对撞——大 BlockSize（≥96）下「下界 > 上界」
  **无合法 TileN**，只能改走「切分归约轴」的 `_bigbs` 变体。

（这两条都是可用的规避，但都源于「单 MX block 的打包 fp4 tile 不能是 RowMajor/NoneBox」这一条约束。）

---

## 问题 / 期望

1. 这条 `RowMajor/NoneBox` 连续轴 32B 下界，对 **sub-32B 的打包/窄类型（尤其打包 fp4，单 block=16B）**
   是否为**有意的硬件粒度约束**？还是可以放宽到「≥ 硬件最小突发 / 元素粒度」？
2. 若属有意约束：**box/fractal（`SLayout != NoneBox`）** 是否是官方推荐的、用来承载单 MX block
   打包 fp4 tile 的写法？（该分支不受 32B 字节下界约束，只要求 `Rows%InnerRows==0 && Cols%InnerCols==0`。）
   若是，能否在文档/报错信息里给出打包类型的推荐 InnerRows/InnerCols 组合？
3. 报错文案 `"Rows must be 32 bytes align"` 与 `RowMajor` 分支**实际卡的是 Cols（-1 轴）**不符，
   建议同时修正文案，减少排查成本。

---

## 附：本仓库内相关文件

- 复现探针：`test/kernel/multi_thread/quant/dynamic_mx_quant/src/align32_probe.cpp`
- Makefile TYPE：`ALIGN32_PROBE`
- 背景分析：`kernels/multi_thread/quant/dynamic_mx_quant/RECORD.md` 问题2（32 字节对齐约束分析）
- fp4 发射本身可用（约束只落在 tile 切分、非发射能力）：`src/fp4_probe.cpp`（`TYPE=FP4_PROBE`）
