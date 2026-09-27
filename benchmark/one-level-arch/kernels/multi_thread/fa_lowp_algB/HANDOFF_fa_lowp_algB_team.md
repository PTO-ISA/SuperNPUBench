# FA lowp Algo B/C 交接说明

## 结论

- `Sq=128, Skv=8192, bctrl.vec_cell_sched_enable=false` 下，Tk=128 的 Algo B/C 已能跑通：gfrun 数值检查通过，gfsim 也已完成。
- Algo C 已使用最终 Cube 分母路径：`TMATMUL_MX` / `TMATMUL_MX_ACC` 累加 `P×V'`，不再用 Vector 计算分母。
- **Tk=256**：B 编译不过，因 live `B.ASSEMBLE` parent 拷贝触发 LLVM #112；C 编译及 gfsim 能过（含本地模型修复，50,907 cycles），数值待验证。
- **Vector 指令尚未如期减少到仅 `TROWEXPANDEXPDIF/TCVT`**：fixp GroupMax 的 BF16x2 广播 slot 选择尚缺（ISA #207），默认路径仍用 `TROWMAX`；online max/CScale、`G-ln4`、C 分母重标定和最终归一化尚未融合，仍需 `TMAX/TSUB(S)/TMULS/TROWEXPANDMUL/DIV`。B 另保留 Vector 分母归约与累加。

## 当前可交付结果

代码开关在 `benchmark/benchmark/one-level-arch/kernels/multi_thread/fa_lowp_algB/fa_lowp_algB.hpp`：

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

从仓库根目录执行。`COMPILER_DIR` 指向 Linx 工具链 bin，`MODEL` 指向含本地修复的模型目录。
测试入口、Makefile、输入生成和 B/C golden checker 均在 `test/kernel/multi_thread/fa_lowp_algB/`。

```bash
export COMPILER_DIR=/data/xxr/feishu/workspace/superscalar_v5/linx-toolchain-build/output/linx_blockisa_llvm_musl/bin
export MODEL=/data/xxr/feishu/workspace/superscalar_v5/model_pto339_342
TEST=benchmark/one-level-arch/test/kernel/multi_thread/fa_lowp_algB

# Sq=Skv=256、Tk=128：有输入 readback 的数值检查，错误返回非零。
python3 "$TEST/run.py" --algorithm B --run-id check_b
python3 "$TEST/run.py" --algorithm C --run-id check_c

# Sq=128、Skv=8192：性能实验均关闭 vec_cell_sched 并输出 SwimLane。
python3 "$TEST/run.py" --mode perf --algorithm B --sq 128 --skv 8192 --run-id perf_b
python3 "$TEST/run.py" --mode perf --algorithm C --sq 128 --skv 8192 --run-id perf_c
python3 "$TEST/run.py" --mode perf --algorithm C --sq 128 --skv 8192 --vecq 64 --run-id perf_c_q64

# fixp GroupMaxOut 路径：仍含 BF16→FP32→BF16 workaround，并非零额外指令目标。
python3 "$TEST/run.py" --algorithm C --gm-fused 1 --run-id check_c_fused

# Tk=256 编译回归：当前 C 已通过；改 --algorithm B 仍复现 #112。
python3 "$TEST/run.py" --mode build --algorithm C --sq 128 --skv 8192 --tk 256 --run-id build_c_tk256
```

每次使用新的 `--run-id`。脚本固定 `make -j4`，删除旧算子对象文件后重新编译，
将 ELF、参数和原始日志保存到测试目录的 `perf_runs/<run-id>/`。
`check` 模式生成非零随机输入并逐字节核对六个输入的 readback，检查对应算法的 golden；
默认容差为 `atol=0.05, rtol=0.05`。`perf` 模式不做数值判断，调用公共 SwimLane helper，
保存 `algoC_swim.log`、`algoC.json`、`algoC_nocounters.json`（B 同理）；打开最后一个文件，
同时记录日志中的 `Total Cycles`。历史 cycle 属于当时的工具链/模型及原测试入口，
新版本的 cycle 以实际复测为准。

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
| **LLVM / compiler** | #112：Tk=256 活跃 `B.ASSEMBLE` parent 被 KV loop 重用时 ICE | 完整 Tk=256 复测：B 仍 ICE，当前 C 已编译通过；上游/本地编译器均为 af743c28。#112 仍真实，不能再把它列为当前 C 的编译阻塞。#109 的 compiler-side TROWEXPAND 问题当前已由版本修复，不再重复改 issue。 |
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
2. LLVM 处理 Algo B 的 #112；当前 Algo C 已可进入 Tk=256 运行验证。
3. 用 Algo C Tk=256 ELF 验证功能/时序并回归历史模型 #839；确认 #873/#874 的本地修复是否进入模型版本。
4. #224、#207、STG_A/带宽属于后续路径，不作为当前交付前置条件。

## 独立分支验证（2026-09-27）

测试入口迁入 multi_thread 后，Sq=Skv=256、Tk=128、GM_FUSED=0：
Algo B/C 均通过数值检查（各 65536 字节输出，bad=0/32768），六个输入 readback 均匹配。
将 Algo C 输出首元素改为 infinity 后，checker 报 bad=1/32768 并返回 1；原始结果已恢复。
数值执行命令由 run.py 记录，实际使用 `gfrun -s softcore.multiThreadNum=4 -f <check ELF>`。
`compile.all` 可顺序执行 B/C 数值检查及非检查版编译。

## Algo C Tk=256 gfsim 复测

2026-09-27：`run.py --mode perf --algorithm C --sq 128 --skv 8192 --tk 256 --run-id algc_tk256_gfsim_recheck`
正常完成，Total Cycles=50,907。ELF 与完整/去计数器 SwimLane、原始日志在测试目录
`perf_runs/algc_tk256_gfsim_recheck/`。这是 GM_FUSED=0 fallback，使用含本地修复的
model_pto339_342；不代表未经修补的上游模型或 fixp GroupMax 直连已验证。
