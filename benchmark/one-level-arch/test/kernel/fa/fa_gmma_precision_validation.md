# FA 数值精度验证报告

> 验证日期：2026-09-24
>
> 算子：`fa_2d_unroll_gmma` / `fa_lowp`
>
> 功能模型：`SuperScalarModel-asl/bin/gfrun`

## 1. 结论

`Sq=128, Skv=1024, QD=VD=128, Tm=Tk=128` 下，FP32、BF16、FP16、FP8
均通过 host golden 数值校验：

| Cube / Vector 类型 | KV blocks | atol / rtol | 判定 | max_abs | MSE | mismatch |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| FP32 / FP32 | 8 | 1e-5 / 1e-5 | PASS | 1.8160790e-08 | 6.6770790e-18 | 0/16384 |
| BF16 / FP32 | 8 | 1e-4 / 1e-4 | PASS | 1.3742130e-05 | 1.3180262e-11 | 0/16384 |
| FP16 / FP32 | 8 | 1e-5 / 1e-5 | PASS | 2.0416919e-06 | 2.0961903e-13 | 0/16384 |
| FP8 E4M3 / FP32 | 8 | 1e-3 / 1e-3 | PASS | 3.0529127e-04 | 5.8589954e-09 | 0/16384 |

四个用例均正常到达 benchmark 终点、报告 `R2 = 0`，输出全部为有限值且不是
全零。`Skv/Tk = 8`，因此测试覆盖 online-softmax 跨 KV block 的 running max、
旧输出重缩放、running sum 和 `TMATMUL_ACC` 累加。

MXFP4 路径的当前结果：

| 算子 | Sq / Skv | KV blocks | atol / rtol | 判定 | max_abs | MSE | mismatch |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| `fa_lowp` MXFP4/BF16 | 128 / 8192 | 64 | 5e-2 / 5e-2 | **PASS** | 2.2988558e-03 | 2.5459772e-07 | 0/16384 |
| `fa_lowp` MXFP4/BF16 | 128 / 128 | 1 | 5e-2 / 5e-2 | **PASS** | 1.4704738e-02 | 9.1743954e-06 | 0/16384 |

两个用例均正常到达终点，输出全部有限且非全零。DUT 与 FP32 golden 的
相关系数分别为 `0.9910`（`Skv=8192`）和 `0.9946`（`Skv=128`），排除了
长序列输出幅度较小而被固定绝对容差掩盖的假阳性。

## 2. 校验脚本

验证脚本为 [`src/gfrun_fa.py`](src/gfrun_fa.py)，支持从 ELF 文件名解析：

```text
CubeFP32_VectorFP32
CubeBF16_VectorFP32
CubeFP16_VectorFP32
CubeFP8_VectorFP32
```

脚本执行以下步骤：

1. 使用固定 seed 123 生成非平凡随机 FP32 源数据，分布为 `N(0, 0.1)`，并裁剪到
   `[-1, 1]`；
2. 编码成 DUT 实际输入格式；
3. 从实际 payload 解码回 FP32，避免把量化前 FP32 当作低精度 DUT 输入；
4. 使用 PyTorch FP32 计算：

   ```python
   golden = torch.softmax((q @ k.T) / sqrt(QD), dim=-1) @ v
   ```

5. 从 `SuperScalarModel-asl` 根目录运行四 PE `gfrun`；
6. 检查退出码、`Reach the End of Benchmark` 和 `R2 = 0`；
7. 使用 `numpy.allclose` 比较 `res.bin` 与 `golden.bin`，并报告 MSE、最大绝对误差、
   mismatch 数量、有限值和全零检查。

`R2 = 0` 只代表功能模型正常结束，最终 PASS 由独立 host golden 比较决定。

## 3. 各类型输入语义

| 类型 | 写入 DUT 的 payload | Golden 使用的值 |
| --- | --- | --- |
| FP32 | IEEE FP32 | 原始 FP32 payload |
| BF16 | PyTorch FP32→BF16 后的原始 U16 bits | BF16 payload 解码为 FP32 |
| FP16 | NumPy FP16 bytes | FP16 payload 解码为 FP32 |
| FP8 | PyTorch E4M3FN 原始 U8 bits | E4M3 payload 解码为 FP32 |

Q、K、V 的逻辑 RowMajor shape 分别为 `[Sq,QD]`、`[Skv,QD]`、
`[Skv,VD]`。输出和 host golden 均为 `[Sq,VD]` FP32。

低精度 kernel 会在 PV 边界把 softmax probability 转成相应 Cube 输入类型；host
golden 计算标准 FP32 attention，因此各类型使用独立容差评估最终输出误差。

## 4. MXFP8 不在本轮校验范围

MXFP8 应使用具备 scale tile 和 `TMATMUL_MX` 的 lowp 实现。当前
`fa_2d_unroll_gmma` 不消费 E8M0 scale，而当前 `fa_lowp` 又是硬编码的 MXFP4
specialization；因此暂不校验 MXFP8，也不再把 unity-scale FP8 smoke 计作 MXFP8
数值结果。

## 5. 复现

必须使用主 `linx-toolchain-build` checkout：

```bash
cd /Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench

export COMPILER_DIR=/Users/blacktraker/Programming/gitproj/DV4/linx-toolchain-build/output/linx_blockisa_llvm_musl/bin

make -C benchmark/one-level-arch/test/kernel/fa \
  precision-check \
  COMPILER_DIR="$COMPILER_DIR" \
  OBJ_ROOT=/tmp/fa_gmma_precision
```

`precision-check` 会依次强制重编译并运行四种类型，最终打印：

```text
FA GMMA precision summary: PASS=4 FAIL=0
```

也可以只编译并验证单个配置：

```bash
make -C benchmark/one-level-arch/test/kernel/fa \
  golden-check \
  TESTCASE=fa_2d_unroll_gmma FA_MODE=BF16_VECFP32 \
  Sq=128 Skv=1024 QD=128 VD=128 Tm=128 Tk=128 X_dim=1 Y_dim=1 \
  COMPILER_DIR="$COMPILER_DIR" \
  OBJ_ROOT=/tmp/fa_gmma_bf16_precision
```

或直接校验已经带 `res_check=on` 编译的 ELF：

```bash
python3 benchmark/one-level-arch/test/kernel/fa/src/gfrun_fa.py \
  --elf /tmp/fa_gmma_numeric_BF16_VECFP32_sq128_skv1024/kernel/fa/elf/\
kernel_fa_fa_2d_unroll_gmma_Sq128_Skv1024_Tm128_Tk128_X1_Y1_CubeBF16_VectorFP32.elf
```

脚本默认使用：

```text
/Users/blacktraker/Programming/gitproj/DV4/SuperScalarModel-asl/bin/gfrun
```

可以用 `--gfrun-root` 指定另一个 ASL checkout，用 `--atol/--rtol` 覆盖类型默认
阈值。

## 6. 环境与 ELF 校验和

```text
linx-toolchain-build: e6a31efb4cfb
SuperScalarModel-asl: 3ddb05f9b0cb

FP32:  8819633174e2d0cc98fd27694f4a96b54d8870e2f8a149adb196bf6ff2d635f2
BF16:  76d44cf1df17ce64bfafaf88a6a6f6b76e4e9b3d1c490220ba5509c0c8f77c03
FP16:  8d68ee4f23873f40cd424ad0e9c6ea1f84d6e9102b1746b1689d6c575cb6b92e
FP8:   4de126bd5f7f3642244e2abc893383ca2436b460e85b8c4bff1e1557d4a9dcb4
```

2026-09-24 MXFP4 复测环境：

```text
SuperNPUBench baseline: bed7c3951459 + V carrier/layout working-tree change
linx-toolchain-build:   e6a31efb4cfb
Linx-TileOP-API:        0c02666c8350
SuperScalarModel:       94e3a4e6cbb0 + SuperScalarModel #848 local fix

MXFP4 Skv=128 ELF:  9bb04d2d7dfe98690156a5f2d1378ea5c3a381f8bc2ac79802ebc3d257f53922
MXFP4 Skv=8192 ELF: c4174fb884aa3ff36a07414f70c3909798f8d14f32db43711ff4f2cf52437a78
```

每次运行的输入、golden、结果和日志写入：

```text
benchmark/one-level-arch/compare/<ELF basename>/
```

主要文件为 `srcq.bin`、`srck.bin`、`srcv.bin`、`golden.bin`、`res.bin`、
`gfrun.log` 和 `golden_compare.log`。这些运行产物不提交到仓库。

## 7. 各精度数值校验思路

所有类型遵循同一个原则：**Golden 必须从实际写给 DUT 的 bytes 解码，而不是直接
使用量化前的 FP32 源数据**。否则输入舍入误差会被错误归因到 kernel。

| 类型 | 输入构造 | Host golden | 输出读取 | 校验重点 |
| --- | --- | --- | --- | --- |
| FP32 | 随机 FP32 bytes | 标准 FP32 attention | FP32 | 布局、online-softmax、跨块累加 |
| BF16 | FP32→BF16，保存原始 U16 bits | BF16 payload 解码后做 FP32 attention | FP32 | BF16 输入舍入和 BF16 PV probability |
| FP16 | FP32→FP16 bytes | FP16 payload 解码后做 FP32 attention | FP32 | FP16 输入舍入和 FP16 PV probability |
| FP8 | FP32→E4M3FN，保存原始 U8 bits | E4M3 payload 解码后做 FP32 attention | FP32 | E4M3 编码、符号、PV probability 量化 |
| MXFP4 | E2M1x2 payload + E8M0 scale | payload×scale 解码后做 FP32 attention | BF16 bits→FP32 | nibble 顺序、group-32 scale、动态 P 量化、BF16 输出 |

判定分成两层：

1. 功能层要求 gfrun 退出码为 0、到达 `Reach the End of Benchmark` 且 `R2=0`；
2. 数值层再检查输出元素数量、有限值、非全零，并按类型容差执行 `allclose`，同时记录
   `max_abs`、MSE 和 mismatch 数量。

低精度误差不只来自 Q/K/V 输入。`fa_2d_unroll_gmma` 的 BF16、FP16、FP8 路径还会
在 PV 边界把 softmax probability 转成 Cube 输入类型；`fa_lowp` 则会把每个 32 列
probability group 动态量化成 MXFP4。因此 host 侧使用标准 FP32 attention 作为算法
oracle，再用分类型阈值评价最终误差，而不是要求 bit-exact。

## 8. MXFP4 校验脚本与当前状态

独立脚本为 [`src/gfrun_fa_mxfp4.py`](src/gfrun_fa_mxfp4.py)，目标是实际使用
`TMATMUL_MX` 的 `fa_lowp`/`fa_lowp_recip`，不是通用 GMMA 路径中的名称 smoke test。

脚本会：

- 从 `{±0.5, ±1.0, ±1.5}` 直接抽取 E2M1 code；
- Q/K 沿 QD 轴、V 沿 Skv/reduction 轴把相邻 code 打包进 E2M1x2 的低/高 nibble；
- 生成 `0x7b..0x7e` 的非均匀 E8M0 scale，即 `2^-4..2^-1`；
- 使用 Q scale `[Sq,QD/32]`、K scale `[Skv,QD/32]`、V ScaleB
  `[VD,Skv/32]`；
- 从这些确切 bytes 解码 Q/K/V，计算 PyTorch FP32 attention golden；
- 将 DUT 的 BF16 `res.bin` 解码为 FP32后比较，默认暂定
  `atol=rtol=5e-2`。

调用方式：

```bash
python3 benchmark/one-level-arch/test/kernel/fa/src/gfrun_fa_mxfp4.py \
  --elf /absolute/path/to/kernel_fa_fa_lowp_..._CubeMXFP4_VectorBF16.elf
```

只检查 payload、scale、golden 的生成和文件尺寸，不启动 gfrun：

```bash
python3 benchmark/one-level-arch/test/kernel/fa/src/gfrun_fa_mxfp4.py \
  --elf /absolute/path/to/kernel_fa_fa_lowp_..._CubeMXFP4_VectorBF16.elf \
  --prepare-only
```

当前 `fa_lowp.hpp` 已完成以下修正：

1. PV 的 Shared B 保持物理 `[K,N] + TransB`，E2M1X2 GM carrier 按
   RowMajor 相邻列配对存为 `[K,N/2]`；校验脚本同样沿 N 轴打包 V；
2. V ScaleB 使用 `[VD,Skv/32]` GM 布局，Shared 有效 shape 为
   `[N,K/group]=[128,4]`；
3. P 的四个 group-32 E8M0 scale code 通过 `TCVT + TPACK` 紧凑打包为
   `[32,4]`，避免将四个 128 B padding fragment 拼成非紧凑矩阵；
4. `TROWEXPANDMUL` 直接消费 `TPARTVIEW` 的 `[32,32]` subview，删除冗余
   `TMULS(...,1.0)` 物化；
5. gfrun 按 [SuperScalarModel #848](https://github.com/LinxISA/SuperScalarModel/issues/848)
   将 row-expand 的最终 shape 检查延迟到
   `B.SUBVIEW` 生效后，不再用 `[32,128]` parent 提前拒绝合法 `[32,32]` view。

`Sq=128, Skv=8192, QD=VD=128, Tm=Tk=128, X=Y=1` 的实测结果：

```text
PASS
elements:   16384
mismatches: 0
max_abs:    0.00229885580483824
mse:        2.545977156876918e-07
all_finite: true
all_zero:   false
```

复现命令：

```bash
export COMPILER_DIR=/Users/blacktraker/Programming/gitproj/DV4/linx-toolchain-build/output/linx_blockisa_llvm_musl/bin

make -B -C benchmark/one-level-arch/test/kernel/fa \
  TESTCASE=fa_lowp FA_MODE=MXFP4_VECBF16 \
  Sq=128 Skv=8192 QD=128 VD=128 Tm=128 Tk=128 X_dim=1 Y_dim=1 \
  res_check=on COMPILER_DIR="$COMPILER_DIR" \
  OBJ_ROOT=/tmp/fa_mxfp4_sq128_skv8192 -j4 diss

python3 benchmark/one-level-arch/test/kernel/fa/src/gfrun_fa_mxfp4.py \
  -d /tmp/fa_mxfp4_sq128_skv8192/kernel/fa/elf/\
kernel_fa_fa_lowp_Sq128_Skv8192_Tm128_Tk128_X1_Y1_CubeMXFP4_VectorBF16.elf \
  --gfrun-root /Users/blacktraker/Programming/gitproj/DV4/SuperScalarModel-asl
```

`Skv=128` 在同一 seed 和阈值下同样 PASS：最大绝对误差
`0.0147047378`、MSE `9.1743954e-06`、`0/16384` mismatch。
