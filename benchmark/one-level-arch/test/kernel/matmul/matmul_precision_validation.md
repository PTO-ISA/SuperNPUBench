# MatMul 数值精度验证报告

> 更新日期：2026-09-22  
> 验证目标：`basic_op/matmul` 的 FP32、FP16、BF16、MXFP8、MXFP4、HiF4X2 计算正确性  
> 功能模型：`SuperScalarModel-asl/bin/gfrun`

## 1. 当前结论

小 shape 精度回归当前为：

```text
matmul precision summary: PASS=6 FAIL=0
```

| 数据类型 | Shape | 输入与 Golden | 结果 | max_abs | mismatch |
| --- | --- | --- | --- | ---: | ---: |
| FP32 | M64 N64 K128, tK64 | 随机 FP32；PyTorch FP32 MatMul | PASS | 8.1956387e-08 | 0/4096 |
| FP16 | M64 N64 K128, tK64 | 随机 FP32 转 FP16 后解码为 FP32；PyTorch MatMul | PASS | 0 | 0/4096 |
| BF16 | M64 N64 K128, tK64 | 随机 FP32 转 BF16 后解码为 FP32；PyTorch MatMul | PASS | 0 | 0/4096 |
| MXFP8 | M64 N64 K64, tK64 | 直接构造 E4M3 payload 和 E8M0 scale；解码实际 bytes 后做 PyTorch MatMul | PASS | 0 | 0/4096 |
| MXFP4 | M64 N64 K256, tK64 | 直接构造 E2M1x2 payload 和 E8M0 scale；解码实际 bytes 后做 PyTorch MatMul | PASS | 0 | 0/4096 |
| HiF4X2 | M64 N64 K64, tK64 | 直接构造 E1M2 payload 和 group-64 U32 scale；解码实际 bytes 后做 PyTorch MatMul | PASS | 0 | 0/4096 |

所有 cooperative basic-op MatMul 的输入布局已统一为 `A:[B,M,K]`、
`B:[B,N,K]`（FP4 carrier 为 `[B,N,K/2]`），Golden 计算 `A @ B.transpose(-1,-2)`。

FP32/FP16/BF16 使用两个 K tile，覆盖普通 `TMATMUL + TMATMUL_ACC` 链。MXFP4
使用四个 K tile，覆盖各 K block 的 group-32 scale 寻址以及
`TMATMUL_MX + TMATMUL_MX_ACC` 累加链。MXFP8/HiF4X2 当前使用单个 K tile，重点验证
payload 解码、scale 寻址、B 布局和单次 `TMATMUL_MX` 数值。

## 2. Golden 语义

验证脚本为 [`src/gfrun_matmul.py`](src/gfrun_matmul.py)。Golden 不使用全 0/全 1
作为唯一数据源。

### FP32、FP16、BF16

1. 使用固定 seed 生成随机 FP32 输入；
2. FP16/BF16 先转换成 DUT 实际输入格式；
3. 再把实际输入值解码为 FP32；
4. 使用 `torch.matmul` 计算 FP32 accumulator Golden；
5. 对比 gfrun 写出的 FP32 `res.bin`。

因此 FP16/BF16 的 Golden 包含输入格式舍入，但不把任意原始 FP32 直接当作 DUT
实际参与计算的值。

### MXFP8、MXFP4、HiF4X2

MX 验证只校验 MatMul consumer，不同时验证量化算法：

1. 直接生成可表示的 E4M3 或 E2M1 payload；
2. 直接生成多组有限 E8M0 scale（`0x7d..0x81`）；
3. 从将要写给 DUT 的 payload/scale bytes 解码出 FP32 A、B-major B；
4. 执行 `torch.matmul(A, B.transpose(-1,-2))`；
5. 与 gfrun FP32 输出比较。

这样能够把“量化策略是否正确”和“MX MatMul 是否正确”分开。

HiF4X2 使用 E1M2 payload：bit 3 为符号，bits 2:0 对应
`0,0.25,...,1.75`。每 64 个逻辑 K 元素使用一个 U32 scale word：低 8 bit
为 E6M2 base，bits 15:8 为每 8 lane 一个 E1 调整位，bits 31:16 为每 4 lane
一个 E1 调整位。校验输入会随机覆盖两个 E1 bitfield，而不是只使用恒等 scale。

## 3. B-major `[N,K]` 布局统一

### 3.1 原问题

原 basic_op 把 B carrier 保存为：

```text
[K/2, N]
```

并使用：

```cpp
SharedMatrixRight<dtype, K, N>
fixp::keep_acc().transpose_b()
```

但 RowMajor packed FP4 的两个 nibble 对应最内层维度的相邻逻辑元素。原布局让
`Col=N`，却把两个相邻 K 元素装入同一 carrier，打包轴和 RowMajor 的最内层轴
不一致。

### 3.2 修复后的统一约定

所有 cooperative basic-op MatMul 的 B 都改为 B-major：

```text
FP32/FP16/BF16/FP8/HiF8/MXFP8 B: [N, K]
MXFP4/HiF4X2 payload B:           [N, K/2]
MXFP8/MXFP4/HiF4X2 scale B:       [N, K/group]
```

所有路径都直接使用 `SharedMatrixRight<N,K>` 语义，不再设置
`transpose_b()`：

```cpp
global_tensor<dtype, RowMajor<N, K/PackedFactor>>
SharedMatrixRight<dtype, N, K>
fixp::keep_acc()  // no TransposeB
```

相关实现：

- [`../../../kernels/basic_op/matmul/matmul_shared_lowp.hpp`](../../../kernels/basic_op/matmul/matmul_shared_lowp.hpp)
- [`src/matmul_lowp.cpp`](src/matmul_lowp.cpp)
- [`src/gfrun_matmul.py`](src/gfrun_matmul.py)

### 3.3 确定性判别用例

使用：

```text
A payload = 0x22  # low/high E2M1 均为 1.0
B payload = 0x22
A/B scale = 0x7f # E8M0 1.0
M=N=K=64
```

每个输出应严格等于：

```text
sum(k=0..63, 1.0 * 1.0) = 64.0
```

修复前后：

| 版本 | 输出 unique | mismatch |
| --- | --- | ---: |
| 修复前 | `[32.0]` | 4096/4096 |
| 修复后 | `[64.0]` | 0/4096 |

修复后的反汇编为：

```text
TMATMUL_MX <M=64, N=64, K=64, e2m1x2>, S0, S1, ->T<4KB>
```

不再携带 `TransposeB`。

除六类主精度组合外，额外验证 `matmul_reuseB`、FP8 和 HiF8，也均为
`run_status=PASS, chk_status=PASS`。`matmul_hif4_l1_quantize` 使用有效配置
`M=tM=128,N=K=64` 编译通过；`matmul_quantize` 仍受已有的
`mxquant::InputTile` API/TCVT layout 编译问题阻塞，与本次 B 布局修改无关。
其中 `matmul_reuseB` 另以 `M256,N128,K128,tM128,tN64,tK64` 覆盖多 M/N/K
分块及跨 M-block 的 B 复用，32768 个输出元素全部匹配，`max_abs=8.94e-08`。

## 4. 一键精度回归

必须使用主 `linx-toolchain-build` checkout：

```bash
cd /Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench

export COMPILER_DIR=/Users/blacktraker/Programming/gitproj/DV4/linx-toolchain-build/output/linx_blockisa_llvm_musl/bin

make -C benchmark/one-level-arch/test/kernel/matmul \
  precision-check \
  COMPILER_DIR="$COMPILER_DIR" \
  OBJ_ROOT=/tmp/supernpubench_matmul_precision
```

`precision-check` 会依次编译并运行：

```text
FP32   M64 N64 K128 tM64 tN64 tK64
FP16   M64 N64 K128 tM64 tN64 tK64
BF16   M64 N64 K128 tM64 tN64 tK64
MXFP8  M64 N64 K64  tM64 tN64 tK64
MXFP4  M64 N64 K256 tM64 tN64 tK64
HIF4X2 M64 N64 K64  tM64 tN64 tK64
```

脚本默认从模型工作树根目录运行：

```text
/Users/blacktraker/Programming/gitproj/DV4/SuperScalarModel-asl
```

并调用：

```bash
bin/gfrun -s softcore.multiThreadNum=4 -f <ELF>
```

注意：这是功能/数值验证，不使用 `gfsim`。

## 5. 单用例验证

### MXFP4

```bash
cd /Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench
export COMPILER_DIR=/Users/blacktraker/Programming/gitproj/DV4/linx-toolchain-build/output/linx_blockisa_llvm_musl/bin

make -C benchmark/one-level-arch/test/kernel/matmul golden-check \
  TESTCASE=matmul_lowp LP_MODE=MXFP4 \
  B=1 M=64 N=64 K=256 tM=64 tN=64 tK=64 \
  COMPILER_DIR="$COMPILER_DIR" \
  OBJ_ROOT=/tmp/matmul_mxfp4_precision
```

### MXFP8

```bash
make -C benchmark/one-level-arch/test/kernel/matmul golden-check \
  TESTCASE=matmul_lowp LP_MODE=MXFP8 \
  B=1 M=64 N=64 K=64 tM=64 tN=64 tK=64 \
  COMPILER_DIR="$COMPILER_DIR" \
  OBJ_ROOT=/tmp/matmul_mxfp8_precision
```

### HiF4X2

`LP_MODE=HIF4` 与 `LP_MODE=HIF4X2` 都会映射到 `__fp4_hif4x2`；精度组合使用
名称更明确的 `HIF4X2`。

```bash
make -C benchmark/one-level-arch/test/kernel/matmul golden-check \
  TESTCASE=matmul_lowp LP_MODE=HIF4X2 \
  B=1 M=64 N=64 K=64 tM=64 tN=64 tK=64 \
  COMPILER_DIR="$COMPILER_DIR" \
  OBJ_ROOT=/tmp/matmul_hif4_precision
```

也可以直接检查已有 ELF：

```bash
python3 benchmark/one-level-arch/test/kernel/matmul/src/gfrun_matmul.py \
  -d /absolute/path/to/matmul.elf
```

## 6. 输出与判定

每个 case 的输入、结果和诊断日志写入：

```text
benchmark/one-level-arch/compare/<ELF basename>/
```

主要文件：

```text
src0.bin
src1.bin
src0_scale.bin / src1_scale.bin  # MX
golden.bin
res.bin
debug_compare.log
```

汇总日志：

```text
benchmark/one-level-arch/compare/matmul_result_check.log
```

输出指标包括：

- `mse`
- `max_abs`
- `mean_abs`
- `rel_l2`
- `mismatches`
- `elements`

## 7. 当前边界与后续项

1. MXFP4 已覆盖四个 K tile 的 `TMATMUL_MX + TMATMUL_MX_ACC`，当前编译结果不再在
   MatMul 路径生成 `c.movr t#3, ->zero`。MXFP8/HiF4X2 仍只覆盖单个 K tile；其多
   K-tile 累加链应继续补充。历史 `C.MOVR RegDst=0` 模型问题由
   `LinxISA/SuperScalarModel#796` 跟踪。
2. 本报告验证的是 MatMul consumer，不验证从任意 FP32 输入选择 MX/HiF4 scale 和 payload
   的量化算法。
3. `kernels/basic_op/matmul/gfrun_mx_scale_issues.md` 是旧模型版本的历史记录；当前
   ASL 模型已经能通过标准 MXFP4 control 和本报告中的修复后用例，不应直接使用旧结论
   判断当前 gfrun 行为。

## 8. 关联记录

- GitHub issue：<https://github.com/PTO-ISA/SuperNPUBench/issues/184>（已关闭）
- A/B 复现材料：<https://gist.github.com/lvhao7896/fc44f551e086f53d78817f7fab11eab3>
- 多 K-tile 模型问题：<https://github.com/LinxISA/SuperScalarModel/issues/796>
- `C.MOVR -> zero` 完整证据：<https://gist.github.com/lvhao7896/90c2b7615d822df043e318a486e2b334>
