# v300 1×1 Conv2D：ColMajor `TLOAD/TSTORE` 布局信息丢失问题

## 1. 结论先行

这个问题不是算子没有编译成功，也不是 `TMATMUL` 没有生成。ELF 可以正常编译，`gfrun` 也能正常执行到 benchmark 结束点，但 ColMajor 全局内存访问的数值语义不正确。

目前定位结论如下：

| 组件 | 判断 | 依据 |
|---|---|---|
| LLVM/Clang 后端 | 不是主要问题 | 反汇编与 TileOP inline asm 请求一致，没有发生错误改写 |
| Linx TileOP API 目标实现 | 主要问题 | API 接受 `global_tensor<..., ColMajor<...>>`，但生成的 NORM `TLOAD/TSTORE` 只携带 `GetStride(3)=1`，没有编码目标/源 GM 为 ColMajor 的信息 |
| gfrun 功能模型 | 不是根本原因，但存在次要问题 | 模型按 NORM 语义执行 `address = base + (row × stride + col) × element_size`；指令中没有 ColMajor 信息，模型无法推断另一种布局。但模型没有拒绝 `stride < valid_col` 的冲突组合，导致静默写错 |
| ISA/API 契约 | 需要明确 | 如果 NORM `TLOAD/TSTORE` 只支持 row-strided ND，则 API 应拒绝 ColMajor；如果 ISA 支持 ND↔DN，则 TileOP 必须编码 layout conversion，模型也要实现对应语义 |

因此，建议首先在 **Linx-TileOP-API** 修正或限制 ColMajor `TLOAD/TSTORE`。这不是单纯修改 `gfrun` 的地址公式就能可靠解决的问题，因为当前指令中没有足够信息区分：

```text
RowMajor，row stride = 1
ColMajor，row stride = 1、column stride = Rows
```

两者在当前 `B.IOR [base, 1]` 编码中无法区分。

## 2. 测试环境

| 组件 | 分支/状态 | Commit |
|---|---|---|
| SuperNPUBench | `main` | `f38ee5f8a2e06b9707cfea7326629f44f79c4e05` |
| linx-toolchain-build-latest | `main`（该 worktree 为 detached HEAD，提交与 `main`/`origin/main` 一致） | `e6a31efb4cfb17f1f1c33265cbf6dbb61bbba156` |
| llvm-project | `dev-llvm15_56` | `86959776bd1fb22dcc8e73b57ec2276c65d44f38` |
| Linx-TileOP-API | `origin/linx`（该 worktree 为 detached HEAD） | `8b2ee780ddcc5d7a04c50e337d67eadc3637a17c` |
| SuperScalarModel | `exp/shared-capacity-lb-semantics-20260819` | `4758e93fe44afd065dada533e43c2351c4287fe2` |

TileOP API 工作区还有与 SharedTile/group matmul 相关的本地修改，但普通 Local Tile 的 ColMajor `TLOAD/TSTORE` 路径没有被这些修改改变。

```bash
export SUPERNPUBENCH=/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench
export COMPILER_DIR=/Users/blacktraker/Programming/gitproj/DV4/linx-toolchain-build-latest/output/linx_blockisa_llvm_musl/bin
export GFRUN=/Users/blacktraker/Programming/gitproj/DV4/SuperScalarModel/bin/gfrun
```

### 2.1 `GetStride(3)=1` 的历史追溯

需要区分两件事：ColMajor tensor 的逻辑 row stride，以及最终传入 `B.IOR` 的数值。

1. `2026-06-22`，提交 `91e3d7a8166181882e862aecc05f94e07d5abf1d`（`Create fa-56 TileOP import baseline`）已经定义：

   ```cpp
   using ColMajor = MatrixLayout<Row, Col, 1, kStride>;
   ```

   同一个 baseline 中，静态 ColMajor `global_tensor` 的 stride 也是：

   ```cpp
   Stride<1, 1, Rows * Cols, 1, Rows>
   ```

   因此在当前可追溯的完整仓库历史中，ColMajor 的 `RowStride`/`GetStride(3)` 从一开始就是 1。

2. `2026-07-09`，提交 `970097b2b74464b145abe522c1b3d96ed7fc3778` 首次加入当前 generic `TLOAD/TSTORE`。当时传入的是：

   ```cpp
   gm_shape::RowStride * sizeof(typename gm_shape::DType)
   ```

   所以 ColMajor FP32 最终汇编中看到的 stride 是 `1×4=4`，FP16 是 `1×2=2`，并不是 1。

3. `2026-08-14 09:39:40 +0800`，提交 `f35d3aad30ea53100f0cf5b26254264bb0edca98`：

   ```text
   [tileop-api] Fix TLOAD/TSTORE/THISTOGRAM GM stride: logical elements, not bytes
   ```

   该提交根据 PTO v0.58 契约和 `LinxISA/llvm-project#48` 删除了 `sizeof(DType)`，把 stride 单位从字节改为逻辑元素：

   ```diff
   - gm_shape::RowStride * sizeof(typename gm_shape::DType)
   + gm_shape::RowStride
   ```

   从这个提交开始，静态 ColMajor FP32 的汇编 stride 从 4 变成了 1。

4. `2026-08-14 11:05:30 +0800`，提交 `6a4378428e4f69c23923cec4a99c759a968b724b`：

   ```text
   [tileop-api] Fix dynamic TLSU GM stride binding
   ```

   该提交将静态类型成员改成对象运行时访问：

   ```diff
   - gm_shape::RowStride
   + src.GetStride(3)  // TLOAD
   + dst.GetStride(3)  // TSTORE
   ```

   对静态 ColMajor 来说，两者都等于 1，因此这次提交没有再次改变静态用例的数值；它主要修复动态 tensor 的 stride 绑定。

5. 截至远端 `origin/linx` 的 `c02dae6587aaa4262e4dceb5191bb6f6d1232f65`（`2026-08-19 17:15:37 +0800`），generic `TLOAD/TSTORE` 仍使用 `GetStride(3)`，之后没有 ColMajor 修复。

所以，“以前不是 1”的记忆对应的是 `f35d3aad` 之前汇编中按字节计算的 4。不能简单恢复乘 `sizeof(DType)`：当前 ISA 和模型约定的 stride 单位是逻辑元素，恢复成 4 会重新引入 issue #48，且仍不能完整表达 ColMajor 的 column stride/layout。

## 3. 测试代码

新增文件：

```text
benchmark/one-level-arch/kernels/conv2d/v300_conv2d.hpp
benchmark/one-level-arch/test/kernel/conv2d/src/v300_conv2d.cpp
benchmark/one-level-arch/test/kernel/conv2d/Makefile
```

### 3.1 算子代码

算子把 NCHW 1×1 Conv2D 展开为：

```text
[H×W, C_in] × [C_in, C_out] -> [H×W, C_out]
```

输入 NCHW 和权重 OIHW 均使用 ColMajor 矩阵视图。输出默认也是 ColMajor；定义 `CONV2D_TMUL_VERIFY` 后输出切换为 RowMajor，用于排除 ColMajor `TSTORE` 的干扰。

```cpp
#include <common/pto_tileop.hpp>

namespace supernpu::conv2d {

using namespace pto;

template <typename Element, int Rows, int Cols,
          int ValidRows = Rows, int ValidCols = Cols>
using AccTile = Tile<Location::Vec, Element, Rows, Cols,
                     BLayout::RowMajor, ValidRows, ValidCols>;

template <typename DType,
          int InChannels, int InHeight, int InWidth, int OutChannels,
          int TileM, int TileN, int TileK>
void conv2d_1x1(float *output, DType *input, DType *weight) {
    constexpr int GlobalM = InHeight * InWidth;
    constexpr int GlobalN = OutChannels;
    constexpr int GlobalK = InChannels;

    using GlobalInput = global_tensor<DType, ColMajor<GlobalM, GlobalK>>;
    using GlobalWeight = global_tensor<DType, ColMajor<GlobalK, GlobalN>>;
#ifdef CONV2D_TMUL_VERIFY
    using GlobalOutput = global_tensor<float, RowMajor<GlobalM, GlobalN>>;
#else
    using GlobalOutput = global_tensor<float, ColMajor<GlobalM, GlobalN>>;
#endif

    using TileA = TileLeft<DType, TileM, TileK>;
    using TileB = TileRight<DType, TileK, TileN>;
    using TileC = AccTile<float, TileM, TileN>;
    using InputIterator = global_iterator<GlobalInput, TileA>;
    using WeightIterator = global_iterator<GlobalWeight, TileB>;
    using OutputIterator = global_iterator<GlobalOutput, TileC>;

    InputIterator input_iter(input);
    WeightIterator weight_iter(weight);
    OutputIterator output_iter(output);

    constexpr int MBlocks = GlobalM / TileM;
    constexpr int NBlocks = GlobalN / TileN;
    constexpr int KBlocks = GlobalK / TileK;

    for (int m = 0; m < MBlocks; ++m) {
        for (int n = 0; n < NBlocks; ++n) {
            auto global_c = output_iter(m, n);
            TileC tile_c;

            auto global_a = input_iter(m, 0);
            auto global_b = weight_iter(0, n);
            TileA tile_a;
            TileB tile_b;
            TLOAD(tile_a, global_a);
            TLOAD(tile_b, global_b);
            TMATMUL(tile_c, tile_a, tile_b);

#pragma clang loop unroll(full)
            for (int k = 1; k < KBlocks; ++k) {
                auto next_global_a = input_iter(m, k);
                auto next_global_b = weight_iter(k, n);
                TileA next_tile_a;
                TileB next_tile_b;
                TLOAD(next_tile_a, next_global_a);
                TLOAD(next_tile_b, next_global_b);
                TMATMUL_ACC(tile_c, tile_c, next_tile_a, next_tile_b);
            }

            TSTORE(global_c, tile_c);
        }
    }
}

}  // namespace supernpu::conv2d
```

### 3.2 测试数据

固定 shape：

```text
Input NCHW:  1×16×4×4
Weight OIHW: 16×16×1×1
Output:      1×16×4×4
Tile M/N/K:  16/16/16
```

支持两种数据模式。

模式一，`PATTERN=off`：输入和权重全部为 1，256 个输出的预期值全部是 `16.0`。

```cpp
for (int i = 0; i < kInputElements; ++i) {
    input[i] = 1.0f;
}
for (int i = 0; i < kWeightElements; ++i) {
    weight[i] = 1.0f;
}
```

模式二，`PATTERN=on`：输入使用非均匀数据，权重为单位矩阵，用于检查 ColMajor `TLOAD`。

```cpp
for (int channel = 0; channel < 16; ++channel) {
    for (int spatial = 0; spatial < 16; ++spatial) {
        input[channel * 16 + spatial] = channel * 100 + spatial;
    }
}

for (int out_channel = 0; out_channel < 16; ++out_channel) {
    for (int in_channel = 0; in_channel < 16; ++in_channel) {
        weight[out_channel * 16 + in_channel] =
            out_channel == in_channel ? 1.0f : 0.0f;
    }
}
```

在单位权重下，RowMajor 验证输出的预期公式为：

```text
output[m, n] = input[channel=n, spatial=m] = n×100+m
```

## 4. 复现步骤与结果

### 4.1 实验 A：ColMajor `TSTORE` 地址重叠

使用全 1 数据，输入读取次序不会改变数值，因此可以直接观察输出地址覆盖范围。

编译：

```bash
cd "$SUPERNPUBENCH/benchmark/one-level-arch/test/kernel/conv2d"

make -B \
  TESTCASE=v300_conv2d VERIFY=off PATTERN=off \
  IN_C=16 IN_H=4 IN_W=4 OUT_C=16 \
  tM=16 tN=16 tK=16 \
  COMPILER_DIR="$COMPILER_DIR" \
  diss
```

ELF：

```bash
ELF="$SUPERNPUBENCH/benchmark/one-level-arch/output/kernel/conv2d/elf/kernel_conv2d_v300_conv2d_verifyoff_patternoff_IC16_H4_W4_OC16_tM16_tN16_tK16.elf"
```

定位输出数组：

```bash
"$COMPILER_DIR/llvm-nm" -S "$ELF" | grep output
```

```text
0000000000016000 0000000000000400 b _ZN12_GLOBAL__N_16outputE
```

执行并导出 1024 字节输出：

```bash
"$GFRUN" -f "$ELF" \
  --dump-memory 0x16000:1024:/tmp/v300_conv2d_colmajor_output.bin

od -An -v -t f4 /tmp/v300_conv2d_colmajor_output.bin \
  | tr -s ' ' '\n' \
  | sed '/^$/d' \
  | sort \
  | uniq -c
```

`gfrun` 正常结束：

```text
Thread:0Total Block number = 525
Thread:0Total Inst number = 4146
Suaccelss to Reach the End of Benchmark! R2 = 0
```

但数值结果错误：

```text
实际：
225 0.000000e+00
 31 1.600000e+01

预期：
256 1.600000e+01
```

现象不是随机错误。`31 = 16 + 16 - 1`，正好对应 16×16 二维数据使用 `row_stride=1` 并按 `row×stride+col` 寻址时只能覆盖的地址数：

```text
最小 offset = 0×1+0  = 0
最大 offset = 15×1+15 = 30
总共只覆盖 offset 0..30，即 31 个位置
```

### 4.2 实验 B：RowMajor `TSTORE` 对照组

```bash
make -B \
  TESTCASE=v300_conv2d VERIFY=on PATTERN=off \
  IN_C=16 IN_H=4 IN_W=4 OUT_C=16 \
  tM=16 tN=16 tK=16 \
  COMPILER_DIR="$COMPILER_DIR" \
  diss

ELF="$SUPERNPUBENCH/benchmark/one-level-arch/output/kernel/conv2d/elf/kernel_conv2d_v300_conv2d_verifyon_patternoff_IC16_H4_W4_OC16_tM16_tN16_tK16.elf"

"$GFRUN" -f "$ELF" \
  --dump-memory 0x16000:1024:/tmp/v300_conv2d_rowmajor_output.bin

od -An -v -t f4 /tmp/v300_conv2d_rowmajor_output.bin \
  | tr -s ' ' '\n' \
  | sed '/^$/d' \
  | sort \
  | uniq -c
```

结果正确：

```text
256 1.600000e+01
```

这个对照证明 RowMajor 输出的 `TSTORE` 路径可用，但全 1 输入不能证明 ColMajor `TLOAD` 正确，因为错误的读取次序仍然会得到 1。

### 4.3 实验 C：非均匀数据检查 ColMajor `TLOAD`

输出保持 RowMajor，排除实验 A 中 ColMajor `TSTORE` 的干扰；输入和权重仍为 ColMajor。

```bash
make -B \
  TESTCASE=v300_conv2d VERIFY=on PATTERN=on \
  IN_C=16 IN_H=4 IN_W=4 OUT_C=16 \
  tM=16 tN=16 tK=16 \
  COMPILER_DIR="$COMPILER_DIR" \
  diss

ELF="$SUPERNPUBENCH/benchmark/one-level-arch/output/kernel/conv2d/elf/kernel_conv2d_v300_conv2d_verifyon_patternon_IC16_H4_W4_OC16_tM16_tN16_tK16.elf"

"$GFRUN" -f "$ELF" \
  --dump-memory 0x16000:1024:/tmp/v300_conv2d_pattern_output.bin

od -An -v -t f4 /tmp/v300_conv2d_pattern_output.bin \
  | tr -s ' ' '\n' \
  | sed '/^$/d' \
  | awk 'BEGIN { i=0; bad=0 }
         {
           m=int(i/16); n=i%16;
           expected=n*100+m; actual=$1+0;
           if (actual != expected) {
             bad++;
             if (bad <= 12)
               printf("index=%d (m=%d,n=%d): actual=%g expected=%g\n",
                      i,m,n,actual,expected);
           }
           i++;
         }
         END { printf("total=%d mismatches=%d\n",i,bad); }'
```

结果：

```text
index=1  (m=0,n=1):  actual=0  expected=100
index=2  (m=0,n=2):  actual=15 expected=200
index=3  (m=0,n=3):  actual=14 expected=300
...
total=256 mismatches=240
```

即使输出改为正确工作的 RowMajor，仍有 240/256 个结果错误，说明问题不只存在于 `TSTORE`；ColMajor 输入和权重的 `TLOAD` 同样没有得到正确表达。

## 5. 反汇编现象

### 5.1 ColMajor 输出

```asm
addi zero, 1, ->a1
...
BSTART.TLSU TSTORE, FP32
B.DIM a0, 0, ->lb0       # 16 columns
B.DIM a0, 0, ->lb1       # 16 rows
C.B.DIMI 16, ->lb2       # tile physical columns
B.IOT u#1, mask=1111, last
B.IOR [a2,a1],[]          # GM stride = 1
```

### 5.2 RowMajor 输出

```asm
addi zero, 16, ->a0
...
BSTART.TLSU TSTORE, FP32
B.DIM a0, 0, ->lb0
B.DIM a0, 0, ->lb1
C.B.DIMI 16, ->lb2
B.IOT u#1, mask=1111, last
B.IOR [a1,a0],[]          # GM stride = 16
```

除了 `B.IOR` stride 数值外，没有指令字段告诉模型 GM 是 RowMajor 还是 ColMajor。ColMajor `TLOAD` 也使用同样的 NORM 块形式，并把 `src.GetStride(3)=1` 放入 `B.IOR`。

## 6. 源码级根因分析

### 6.1 Layout 类型本身是正确的

TileOP 的布局定义为：

```cpp
// include/common/layout.hpp
using RowMajor = MatrixLayout<Row, Col, kStride, 1>;
using ColMajor = MatrixLayout<Row, Col, 1, kStride>;
```

所以 16×16 ColMajor tensor 的：

```text
RowStride = 1
ColStride = 16
```

### 6.2 Linx TileOP 目标实现丢失了 ColMajor 信息

目标端 `TLOAD`：

```cpp
"B.IOR [%[s0],%[GmStride]], []\n"
...
[GmStride] "r"(src.GetStride(3))
```

目标端 `TSTORE`：

```cpp
"B.IOR [%[d0],%[GmStride]], []\n"
...
[GmStride] "r"(dst.GetStride(3))
```

两者都只传 `GetStride(3)`，也就是 row stride。对于 ColMajor，它等于 1；`GetStride(4)=16` 和 `global_tensor::isRowMajor=false` 都没有进入指令。

同一 TileOP API 的 CPU backend 能正确区分两种布局，因为 CPU 路径仍然持有 C++ 模板类型：

```cpp
// ColMajor CPU simulation
idx_gm = col * gm_shape::ColStride + row;

// RowMajor CPU simulation
idx_gm = row * gm_shape::RowStride + col;
```

因此，CPU backend 支持 ColMajor，而 Linx 目标端没有表达出等价语义，两个 backend 的行为不一致。

### 6.3 LLVM/Clang 没有改错 inline asm

TileOP 请求 `B.IOR [base, GetStride(3)]`，反汇编也确实得到：

```text
ColMajor -> B.IOR [base, 1]
RowMajor -> B.IOR [base, 16]
```

所以当前证据不支持“LLVM 后端错误 lowering”这一判断。错误在进入 LLVM 后端之前已经形成：TileOP wrapper 没有把 ColMajor 的布局转换信息编码进去。

### 6.4 gfrun 按 NORM row-strided 语义执行

`gfrun` 的 NORM `TLOAD` 使用：

```cpp
srcAddrRow = srcAddrStart + (row * stride) * elementSize;
load(srcAddrRow, col);
```

NORM `TSTORE` 使用：

```cpp
dstAddrRow = dstAddrStart + (row * stride) * elementSize;
store(dstAddrRow, col, data);
```

即统一按：

```text
element_offset = row × stride + col
```

这与模型源码注释和 TileOP TLSU 文档中的“`B.IOR` 携带 logical row stride”一致。模型无法仅根据 stride=1 判断调用者原本声明的是 ColMajor。

但模型当前缺少保护：对于 NORM 16×16 访问，显式 `stride=1 < valid_col=16` 会造成行重叠，模型仍继续执行。至少应该给出 assertion/error，避免返回“运行成功但数据静默错误”。

## 7. 问题归属

### 主要责任：Linx-TileOP-API

公开 API 允许以下合法 C++ 类型参与 `TLOAD/TSTORE`：

```cpp
global_tensor<float, ColMajor<16,16>>
```

CPU backend 也实现了 ColMajor 语义，但 Linx 目标实现生成的是不携带 GM layout 的普通 NORM 指令。因此，API 的类型能力与目标端实际可表达能力不一致。

TileOP API 应选择以下一种方案：

1. 如果当前 ISA NORM `TLOAD/TSTORE` 只支持 RowMajor/row-strided GM，使用 `static_assert(gm_shape::isRowMajor)` 明确拒绝 ColMajor。
2. 如果 ISA 支持 ND↔DN layout conversion，TileOP 应生成对应 layout attribute/转换指令，并传递正确的 leading dimension；不能只传 `GetStride(3)`。
3. 如果需要显式转置，应由 TileOP 展开为受支持的 load/store + tile transpose 路径。

### 次要责任：gfrun

模型当前对收到的 NORM 指令按 row-strided 语义执行是可解释的，因为指令中没有 ColMajor 信息。但建议增加以下检查：

```text
NORM TLOAD/TSTORE with explicit stride:
stride >= valid_col
```

不满足时直接报错，避免 31 个地址重叠后仍报告 benchmark 成功。

如果 ISA 最终规定了 ColMajor/layout-conversion 编码，则模型还需要实现对应的新语义；在编码契约明确之前，模型不能通过猜测 `stride==1` 来判断 ColMajor。

### LLVM/Clang 后端

本次反汇编表明 LLVM/Clang 正确保留了 TileOP inline asm 的寄存器和立即数，没有证据表明后端错误。因此暂不建议把此问题归到 LLVM codegen。

## 8. 建议验收标准

修复后至少需要通过以下三项：

1. 全 1、ColMajor 输出：256 个结果全部为 `16.0`，不能只有 31 个非零位置。
2. 非均匀 ColMajor 输入 + ColMajor identity weight + RowMajor 输出：256 个结果全部满足 `output[m,n]=n×100+m`。
3. CPU backend、编译后的 ELF+gfrun，以及独立 golden 三方一致。

如果当前 ISA 暂不支持 ColMajor GM，则验收标准改为：上述 ColMajor `TLOAD/TSTORE` 在编译期明确失败，并给出“Linx target only supports RowMajor GM for NORM TLOAD/TSTORE”的诊断，而不是生成可以运行但数值错误的 ELF。
