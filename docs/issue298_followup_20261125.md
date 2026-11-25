# Issue 298：剩余 12 个配置修复与功能检查

基线 main：3b6b611672ddf4c26aa8e6dd414006040c20f1be。
本轮归属为算子/微基准实现缺口；没有修改工具链或模拟器源代码。

| 类别 | 配置数 | 修复 | 本轮验证 |
|---|---:|---|---|
| HiF4 FA | 9 | 原生 Matrix-MX，Q/K/V U32 scales，合法 mixed BF16×HiF4 PV；修正 packed stride；概率和用 FP32 累加 | 原 issue 九项全部编译、gfrun、独立 FP64 softmax 参考通过 |
| S8+RowMax | 2 | FP32 Matmul/RowMax → clamp [-128,127] → 显式 RNE S8 TCVT | Local/Shared 均通过非零数值检查，覆盖正负 tie、饱和、量化前 RowMax |
| HiF4 L1 quantize | 1 | 保留四 PE FP32 matmul，显式 BF16/E6M2/S1P2 标量量化及 nibble packing | 原尺寸编译；随机、边界、零输入共三组逐字节通过，每组 4096 字节 payload + 128 字节 scale |

额外 KEEP_ACC 控制编译和 gfrun 通过。上述功能运行均为四 PE，正常记录 `formal_End=1 formal_Abort=0 status=0`。

## HiF4 FA 的输入与数值契约

源码：`/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/kernels/basic_op/fa/fa_hif4.hpp`。
驱动：`/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/test/kernel/fa/src/fa_hif4.cpp`。

- Q/K payload 分别为 packed `[Sq,QD/2]`、`[Skv,QD/2]` 字节，V 为 B-major packed `[VD,Skv/2]` 字节。
- U32 scales 分别为 `[Sq,QD/64]`、`[Skv,QD/64]`、`[VD,Skv/64]`；每个 word 包含 E6M2 base 和两层逐组指数位，参考数据包含非均匀 scale 及非零高位。
- 文件名为 `srcq.bin`、`srck.bin`、`srcv.bin` 和对应 `srcq_scale.bin`、`srck_scale.bin`、`srcv_scale.bin`。它们位于复现命令指定的绝对输出目录。
- GM Tensor 的 shape/stride 按逻辑 4-bit lanes 定义，只有 C++ 指针偏移按 packed 字节计数。
- QK、softmax 概率和输出为 BF16；概率和、PV 累加及在线分母为 FP32。P 保持 BF16，PV 使用 ScaleMask=2，不将 P 非法 TCVT 为 HiF4。
- 三个入口分别保留 full-QD QK、64-wide QK chain、fused RowMax 的差别。新路径是正确性基线，`X_dim/Y_dim` 暂不驱动旧版预取/流水优化，不宣称其原有优化性能。
- 目前限定 Tm=128、Sq 为 128 的倍数，Skv 为 Tk 的倍数，QD/Tk 为 64 的倍数。编译期检查完整 Q/K/V 和三个 scale carrier 总容量不超过 256 KiB。

独立参考：`/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/test/kernel/fa/src/check_hif4_fa.py`。
解码实际 HiF4 payload/scales 后使用 FP64 全量 softmax，不复用设备在线算法。验收阈值 `atol=0.002, rtol=0.01`。

| Sq / Skv / Tk | 每组入口数 | 最大绝对误差 | RMSE |
|---|---:|---:|---:|
| 1024 / 1024 / 128 | 3 | 0.000958037 | 0.000160316 |
| 128 / 8192 / 128 | 3 | 0.000270690 | 0.0000597591 |
| 256 / 512 / 256 | 3 | 0.001275760 | 0.000231970 |

Tk=256 的 BF16 概率求和初版未通过；改 FP32 求和后最大误差从约 0.02052 降至 0.001276，未放宽阈值。

## S8 与 L1 的语义边界

S8+RowMax 现在是两阶段算法，不再声称 ISA 支持“最终 S8 D 的 fused RowMax”。RowMax 是量化前 FP32 行最大值，量化为单位 scale、零 offset、饱和 RNE。两个模式已回到 active sweep/coverage；FPATR 原始编码验证为 `0x00042023`（NoQuant + RowMax）。

L1 使用标量量化作为正确性基线，不是向量性能优化。数学精度步骤和 E6M2 下溢/饱和规则写在
`/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/kernels/basic_op/matmul/matmul_hif4_l1_quantize.hpp`
及 `/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/kernels/basic_op/matmul/hif4_s1p2.hpp`。
只输出一级 E6M2 scale，不生成完整三层 HiF4 U32 scale。

## 复现

固定编译器 checkout `/Users/blacktraker/Programming/gitproj/DV4/linx-toolchain-build`，commit `e6a31efb4cfb17f1f1c33265cbf6dbb61bbba156`。
固定 gfrun `/Users/blacktraker/Programming/gitproj/DV4/SuperScalarModel/bin/gfrun`，model commit `09ed72caa858f62e0261fa7c402bd5dc7a387473`。现有依赖工作区差异、二进制 SHA256 另存 identity.json。

```bash
export COMPILER_DIR=/Users/blacktraker/Programming/gitproj/DV4/linx-toolchain-build/output/linx_blockisa_llvm_musl/bin
make -C /Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/test/kernel/matmul hif4-l1-check OBJ_ROOT=/tmp/issue298-replay-l1
make -C /Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/test/kernel/fa golden-check TESTCASE=fa_fixpipe FA_MODE=HIF4_VECBF16 Sq=256 Skv=512 Tm=128 Tk=256 X_dim=1 Y_dim=1 OBJ_ROOT=/tmp/issue298-replay-fa
make -B -C /Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/microbenchmark/fixp FIXP_MODE=S8_ROWMAX res_check=on OBJ_ROOT=/tmp/issue298-replay-s8
cd /Users/blacktraker/Programming/gitproj/DV4/SuperScalarModel
bin/gfrun -s softcore.multiThreadNum=4 -f /tmp/issue298-replay-s8/microbenchmark/fixp/elf/fixp/fixp_tmatmul_s8_rowmax_M32_N32_K32_tM32_tN32_tK32.elf
```

Shared S8 将 mode 改为 SHARED_S8_ROWMAX、ELF 后缀改为 shared_s8_rowmax。FA 另外两个入口为 fa_2d_unroll_gmma、fa_gmma_kchains；上表三个尺寸交叉构成九项。

日志、身份和九项结果保存在 `/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/outputs/issue298-followup-20261125`。

## 尚未证明的范围

- 这是剩余 12 项的针对性验证，不是重新对全仓所有算子跑功能回归；此前 50 项的 BUILD_PASS 仍只表示当时编译/链接通过。
- 合并两轮记录可计为原清单 62 项有效配置编译通过、18 项超容量配置已删除；不能说 80 项全部运行通过。
- 未跑 gfsim，不提供性能结论。
- 当前 llvm-objdump 的部分 Matrix RowMax 宏反汇编仍失败；`--disassembler-options=no-tile-macros` 可以导出物理指令。该工具链限制不等于 ELF 编译或 gfrun 失败；本轮未改它，也不声称完整 `compile.all` 的宏反汇编报告已通过。

## main 后续审计修复

Packed matmul（含 reuseA）、HiF4 FA 增量构建、旧 matmul hosted checker 和宏反汇编兼容处理已补修。最新范围和针对性验证见 `/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/docs/issue298_main_audit_fixes_20261125.md`。上游宏渲染限制仍在，构建现在显式使用物理 fallback，不掩盖其他错误。
