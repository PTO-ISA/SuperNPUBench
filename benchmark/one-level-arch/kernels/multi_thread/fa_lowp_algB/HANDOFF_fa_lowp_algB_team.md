# FA lowp Algo B/C 交接说明

## 结论

- `Sq=128, Skv=8192, bctrl.vec_cell_sched_enable=false` 下，Tk=128 的 Algo B/C 已能跑通：gfrun 数值检查通过，gfsim 也已完成。
- Algo C 已使用最终 Cube 分母路径：`TMATMUL_MX` / `TMATMUL_MX_ACC` 累加 `P×V'`，不再用 Vector 计算分母。
- 当前交付仍卡在 **Tk=256**。第一处阻塞是 LLVM 的 Tk=256 `B.ASSEMBLE` 编译 ICE（[llvm-project #112](https://github.com/LinxISA/llvm-project/issues/112)）；编译器问题解决后，预计还会撞到模型的多 writer assembled-parent 问题（[SuperScalarModel #839](https://github.com/LinxISA/SuperScalarModel/issues/839)）。因此“Tk=256 尚未完成”是正确判断。

## 当前可交付结果

代码开关在 `benchmark/benchmark/one-level-arch/kernels/basic_op/fa/fa_lowp_algB.hpp`：

- `FA_ALGB_ALGO_C=0`：Algo B；
- `FA_ALGB_ALGO_C=1`：Algo C（默认）；
- `FA_ALGB_GM_FUSED=0`：默认路径，不依赖 CUBE subview TCVT。

Tk=128 的一次基准结果（`bctrl.vec_cell_sched_enable=false`）：

| 配置 | Total Cycles |
|---|---:|
| Algo B | 64,940 |
| Algo C，默认 vector issue queue | 37,913 |
| Algo C，`vecIssueQDepth=64` | 29,127 |

完整 SwimLane 产物在：
`benchmark/benchmark/one-level-arch/test/kernel/fa/perf_runs/20260927_sq128_skv8192_vecsched0*`。

## 如何复现实验

下面命令以 `Sq=128, Skv=8192, Tm=128, Tk=128` 为例。切换 `FA_ALGB_ALGO_C`
或 `FA_ALGB_GM_FUSED` 时，先删除对应目标文件，避免 `make` 复用旧的 `.o`。

```bash
export ROOT=/data/xxr/feishu/workspace/superscalar_v5
export BM=$ROOT/benchmark/benchmark/one-level-arch
export COMPILER_DIR=$ROOT/linx-toolchain-build/output/linx_blockisa_llvm_musl/bin
export MODEL=$ROOT/model_pto339_342
cd $BM/test/kernel/fa

# Algo B：FA_ALGB_ALGO_C=0
rm -f $BM/output/kernel/fa/src/fa_lowp_algB.o
make all TESTCASE=fa_lowp_algB COMPILER_DIR=$COMPILER_DIR \
  FA_MODE=MXFP4_VECBF16 Sq=128 Skv=8192 Tm=128 Tk=128 X_dim=1 Y_dim=2 \
  CFLAGS="-DFA_ALGB_ALGO_C=0 -DFA_ALGB_GM_FUSED=0"

# Algo C：FA_ALGB_ALGO_C=1
rm -f $BM/output/kernel/fa/src/fa_lowp_algB.o
make all TESTCASE=fa_lowp_algB COMPILER_DIR=$COMPILER_DIR \
  FA_MODE=MXFP4_VECBF16 Sq=128 Skv=8192 Tm=128 Tk=128 X_dim=1 Y_dim=2 \
  CFLAGS="-DFA_ALGB_ALGO_C=1 -DFA_ALGB_GM_FUSED=0"
```

生成的 ELF 在 `$BM/output/kernel/fa/elf/`，文件名包含
`fa_lowp_algB_Sq128_Skv8192_Tm128_Tk128`。用 gfrun 做功能和数值检查：

```bash
cd $MODEL
./bin/gfrun -s softcore.multiThreadNum=4 -f <ELF>
```

用 gfsim 做时序检查时也必须在模型目录运行，因为它需要相对路径下的
`configs/fourpe.conf`：

```bash
cd $MODEL
./bin/gfsim -s bctrl.vec_cell_sched_enable=false -f <ELF>
```

每次性能实验都用 SwimLane helper，并保留三个产物：

```bash
RUN=$BM/test/kernel/fa/perf_runs/<run_id>
python3 $BM/test/common/run_swimlane.py \
  --gfsim $MODEL/bin/gfsim --elf <ELF> --outdir $RUN --name algC \
  --conf fourpe
```

输出包括 `algC_swim.log`、`algC.json` 和可在 Perfetto 打开的
`algC_nocounters.json`；从 `algC_swim.log` 记录 `Total Cycles`。

测试 Vector issue queue 加深到 64：

```bash
cd $MODEL
./bin/gfsim -s bctrl.vec_cell_sched_enable=false \
  -s bctrl.vecIssueQDepth=64 -f <Algo-C ELF>
```

目标数据流（fixp GroupMaxOut 直接作为 `TROWEXPANDEXPDIF` 输入）用
`-DFA_ALGB_GM_FUSED=1` 编译；它目前用于验证 ISA/TileOP/model 缺口，不能代替默认
`FA_ALGB_GM_FUSED=0` 的可交付 fallback。

Tk=256 只需把上述编译命令中的 `Tk=128` 改为 `Tk=256`；当前预期首先在 LLVM
issue #112 处失败，不能把失败误判成 gfsim 运行问题。

## Algo B/C 的算法和实现意图

两条路径共享同一段 FP4 quantization：QK 得到 `S` 后，先得到每个 32 列 group 的
`G=groupmax(S)`，计算 `P3=G-ln4`、`P4=exp(S-P3)`，再把 `P4` 转成 FP4；同时用
`P5=exp(P3-R)` 生成每个 group 的 E8M0 `PScale`。这里的设计意图是直接消费 fixp/CUBE
的 `GroupMaxOut`：每个 group 的值作为 `TROWEXPANDEXPDIF` 的 row-broadcast 输入，
不再重新执行 `TROWMAX`。当前 `FA_ALGB_GM_FUSED=0` 只是模型可跑的 fallback；
`FA_ALGB_GM_FUSED=1` 才是目标数据流，但暂时需要 BF16→FP32→BF16 的物化来绕过
BF16x2 CellReg/subview 限制。

- **Algo B**：`P×V'` 的分母在 Vector 侧完成。每个 group 对 `P4` 做 `TROWSUM`，乘上
  `P5` 后在 Vector 中按 online row-max 做缩放累加；`P×V` 的 numerator 仍由 Cube 完成，
  最后用 Vector 分母归一化。
- **Algo C**：分母也交给 Cube。`TMATMUL_MX` 计算首个 KV block；后续 block 先用
  `TROWEXPANDMUL` 施加旧分母的 online scale，再用 `TMATMUL_MX_ACC` 累加。PV numerator
  同样用 `TMATMUL_MX_ACC` 和 CScale 做 Cube 内重标定，最后只做一次 Vector 除法归一化。

因此，当前真正待补的是 GroupMaxOut 的 BF16x2 直接消费能力；不是重新设计 Algo B/C，
也不是再增加一条数学 `TMULS` 指令。

## 不同组件情况说明

| 负责人 | 问题 | 当前状态 / 下一步 |
|---|---|---|
| **FA kernel** | B/C 算法实现、P-scale 组装、fixp GroupMaxOut → `TROWEXPANDEXPDIF` 数据流 | Tk=128 已验证。最终意图是直接使用 fixp/CUBE 产生的 GroupMaxOut，不额外执行 `TROWMAX`；当前 `FA_ALGB_GM_FUSED=0` 只是 gfsim 可跑的 fallback。剩余 `TMULS` 是数学缩放（`tQ/tRg/tDiff`），不是分母未实现。 |
| **gfrun** | Tk=128 ELF 执行与数值检查 | 当前无阻塞；B/C 均已通过。 |
| **gfsim / SuperScalarModel** | #866 aux effective-D；#868 assembled-parent descriptor；#869 Shared-B/no-ScaleB 的 cscale source；#839 Tk=256 多 writer；#873 TPARTVIEW 的 row-expand descriptor；#874 RAS sparse restore | #873、#874 已新提 issue，并已有本地修复；#866/#868/#869 是真实模型问题，本地有 workaround；#839 是 Tk=256 的后续硬阻塞。未提交的本地模型修改不能丢。 |
| **LLVM / compiler** | #112：Tk=256 活跃 `B.ASSEMBLE` parent 被 KV loop 重用时 ICE | 这是 Tk=256 的第一处阻塞。现有独立最小 TU 在当前 clang 上可编译，不能据此宣布 full-kernel 问题已修复；需用完整 FA Tk=256 case 回归。#109 的 compiler-side TROWEXPAND 问题当前已由版本修复，不再重复改 issue。 |
| **TileOP API** | #224：CUBE subview → TCVT | issue 仍 open，本地 API 已是最新 commit；probe 仍失败。只影响 `FA_ALGB_GM_FUSED=1` 的可选路径，不阻塞默认交付。 |
| **PTO ISA** | #207：允许 BF16/FP16 x2 的 row-broadcast source 以 `validCol=2` 携带一个 CellReg，并按 slot 选择其中的逻辑值；扩展到 `TROWEXPAND*` | 这是一个真实的 ISA 表达能力缺口，但不阻塞 Tk=128，也不是当前 Tk=256 第一阻塞；建议在现有 #207 下补充 `TROWEXPANDEXPDIF`、MXFP4 FA 和 HiF4/MXQuant 用例。这里不是放宽到 `[M,G]`，而是只接受 BF16/FP16 x2 的两列物理载体。 |

## Issue 完整性

当前本地发现的真实问题均已有归属：已有 issue 使用原 issue，新发现的问题已补提：

- 新提：[#873](https://github.com/LinxISA/SuperScalarModel/issues/873)、[#874](https://github.com/LinxISA/SuperScalarModel/issues/874)。
- 已有且仍相关：#859、#839、#866、#868、#869、#112、#224、#207。
- #109 按约定不修改；当前 compiler-side 已不再是主阻塞。
- #869 保留为本地说明，不再重复提报；Algo C 分母不带 `cscale`，但 PV 累加仍需保留该模型 workaround。

## 建议交接顺序

1. 先交付 Tk=128 B/C 分支，并保留已验证的本地模型 workaround 和 SwimLane 结果。
2. LLVM 先处理 #112；用完整 Tk=256 FA case 验证，不能只看独立 TU。
3. #112 解决后，立即回归模型 #839；同时确认 #873/#874 的本地修复是否能进入模型版本。
4. #224、#207、STG_A/带宽属于后续路径，不作为当前交付前置条件。
