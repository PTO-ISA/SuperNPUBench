# AlgoD handoff (2026-10-10)

> 历史快照：下文的prefix-only preload、变体互斥和未提交状态仅反映当日。当前持续环形preload与dblQ可组合；最新Tk256结果及实验设计见 [2026-10-13交接](HANDOFF_algoD_20261013.md)。

## 当前结论

- AlgoD 的 `base`、`preloadN`、`dblQ` 编译入口已存在。`dblQ` 已改为两个**不同**的 128 行 Q tile 共享每个 KV tile；不是复制同一个 Q。
- `dblQ` 在 `Sq=256, Skv=256/512, Tk=128` 的 gfrun 输出与 `base` **逐字节相同**，两半 Q 输出也不同。但两者均未通过 golden 校验，因此不能称 AlgoD 功能正确。
- `Sq=256, Skv=8192, Tk=128, GM2, vecIssueQDepth=64, ROB=128, bandwidth=256` 下，`dblQ` 相对 `base` 的 gfsim Total Cycles 从 59447 降到 42901（resident 关），从 56902 降到 42799（resident 开）。这些是未通过 golden 功能验证的性能数据。
- 当前 `preloadN` 只把前 N 个 K/KScale tile 提到计算循环前；后续 K/KScale 仍在循环内加载。它**不能**证明已解决循环内 scalar 阻挡后续 TLOAD 下发的问题。此前这组中间 perf 数据已清理，不应引用旧 preload 数字。
- 两种调度共有的错误可能与 gfrun 的 `TROWEXPANDEXPDIF`（特别是 packed broadcast / byte offset）有关，但当前证据尚未定因；也需排查其它共有计算路径。不能把“输出一致”当作模型 bug 的证明。

## 代码与环境

- Kernel: `fa_lowp_algD.hpp`；入口: `test/kernel/multi_thread/fa_lowp_algB/src/fa_lowp_algB.cpp`。
- 编译参数入口: `test/kernel/multi_thread/fa_lowp_algB/Makefile`；构建/校验/性能入口: 同目录 `run.py` 和 `verify_fa_lowp.py`。
- 工作分支 `fa_lowp_handoff_branch`: `feat/fa-lowp-gm-direct`，相对上游 ahead 6；AlgoD kernel、Makefile、run.py、入口文件均有**未提交修改**。工作树另有 C、topk、histogram 等用户修改，不要清理或覆盖。
- 本轮性能模型为独立 worktree `model_algod_l0ares_20261010`，分支 `experiment/algod-schedules-20261010`，commit `6dd3e88d9e4088c231a9873c721859f91a6d79ae`（来自 `fix/l0ares-resident-tag-uaf`），工作树干净。原 `model_pto339_342` 保留五处本地修改，没有覆盖。
- Makefile 用 `-O2`、`-enable-misched=false`；`make -j4`。C/D 只保留 `FA_ALGB_GM_FUSED=2`；当前 Tk 限定 128。`run.py` 要求 `MODEL` 和 `COMPILER_DIR` 环境变量指向模型与编译器目录。

## 算法与调度

QK 使用 MX matmul 的 BF16 score、融合 RowMax/GroupMaxOut。每个 32 列 group 计算 `groupLogReference = groupMax - ln(4)`，`TROWEXPANDEXPDIF` 生成 FP4 输入和 E8M0 scale；`TROWSUM` 后维护每组分母，最后合并。PV 使用 FP32 accumulator；中间块通过 `keep_acc().raw_acc().acc_hint().cscale(cscale)` 驻留 L0C 并重标定，末块通过 `keep_acc().acc_hint().cscale(cscale)` 发布至 CellReg，随后归一化与存储。

宏：`FA_ALGB_D_PRELOAD`、`FA_ALGB_D_PRELOAD_N`（2/3/4/5）、`FA_ALGB_D_DBLQ`。preload 与 dblQ 互斥。`dblQ` 外层按两个 Q tile 迭代，各自保存 `weightedValueSum`、`runningMaxGrid`、`runningSum`、`groupSums`、`previousGroupLogReference`；内层每个 kb 加载一次 K/KScale 和 V/VScale，依次计算 Q0、Q1，最后分别归一化并存储。`dblQ` 要求 `Sq` 为 256 的正整数倍。

## 功能验证

保留的结果见 `test/kernel/multi_thread/fa_lowp_algB/perf_runs/algd_schedules_20261010/`：

| Case | 输入回读 | 输出与 base | Golden |
| --- | --- | --- | --- |
| Sq256/Skv256 base、dblQ | 六个 buffer 均一致 | 65536 字节逐字节一致，Q0/Q1 两半不同 | 两者 FAIL；bad=24361/32768，max_abs=0.02747，mean_abs=0.003736 |
| Sq256/Skv512 base、dblQ | 六个 buffer 均一致 | 65536 字节逐字节一致，Q0/Q1 两半不同 | 两者 FAIL；bad=22308/32768，max_abs=0.01941，mean_abs=0.002703 |

原始数据：`schedule_output_comparison.json`、`base256_check/check.log`、`base512_check/check.log`、`dblQ512_check/check.log`；Skv256 dblQ 日志另在 `perf_runs/algd_dblQ_distinct_check_20261010/check.log`。`Skv=8192` **未**完成 golden 校验。比较仅说明 dblQ 没有观察到额外的输出差异；不证明共享路径正确。

## 性能（gfsim，均有 SwimLane）

目录：`test/kernel/multi_thread/fa_lowp_algB/perf_runs/algd_schedules_20261010/`。各 run 子目录保留 `algoD_swim.log`、`algoD.json`、`algoD_nocounters.json`、ELF、配置 wrapper 和 manifest。打开 `algoD_nocounters.json`；以日志中的 `Total Cycles` 为准。

| Run 子目录 | Sq | resident | bandwidth bytes | Total Cycles |
| --- | ---: | ---: | ---: | ---: |
| `base128_resident0_bw256` | 128 | 0 | 256 | 28903 |
| `base128_resident1_bw256` | 128 | 1 | 256 | 28814 |
| `base256_resident0_bw256` | 256 | 0 | 256 | 59447 |
| `dblQ256_resident0_bw256` | 256 | 0 | 256 | 42901 |
| `base256_resident1_bw256` | 256 | 1 | 256 | 56902 |
| `dblQ256_resident1_bw256` | 256 | 1 | 256 | 42799 |

共同配置：`Skv=8192, Tk=128, GM2, vecIssueQDepth=64, speROBDepth=128, bctrl.vec_cell_sched_enable=false, cube.enable_internal_acc=true, fourpe, soc_random=false`。`resident` 控制 `cube.l0aSharetregResident`；bandwidth 控制 `core.sharetreg_bandwidth_bytes`。当前对比聚焦 bw256，未保留随机流量和 bw128 数据。Sq256 同配置下 dblQ 分别少 16546 cycles（27.8%）和 14103 cycles（24.8%）。Sq128 不适用于 dblQ，不能跨 Sq 比较。

样例 ELF：`perf_runs/algd_schedules_20261010/dblQ256_resident1_bw256/kernel_multi_thread_fa_lowp_algB_D1_dblQ_C0_GM2_Sq256_Skv8192_Tk128_checkoff.elf`。该目录 `manifest.json` 记录模型 commit、ELF SHA256、完整运行命令和配置。旧 `GM1` 的 36928 cycles 不可与这里的 GM2 数字直接比较。

## 复现和后续工作

在 `test/kernel/multi_thread/fa_lowp_algB/` 中，先设置 `MODEL`、`COMPILER_DIR`（均指向工作区内相应目录）。示例：

```bash
python3 run.py --mode check --algorithm D --d-variant dblQ \
  --sq 256 --skv 512 --tk 128 --run-id algd_dblq512_recheck

python3 run.py --mode perf --algorithm D --d-variant dblQ \
  --sq 256 --skv 8192 --tk 128 --vecq 64 \
  --l0a-resident 1 --sharetreg-bandwidth 256 \
  --run-id algd_dblq256_perf_recheck
```

`run.py` 的 perf 路径会调用 `test/common/run_swimlane.py` 并保存三件 trace 产物；`--run-id` 必须是新目录。优先事项：

1. 用可控的 `TROWEXPANDEXPDIF` 最小用例或带修复的 gfrun 比对 packed group 的 byte offset 和每个 group 的 exp 输出，再重跑 base/dblQ golden。当前只是嫌疑，不能预设根因。
2. 若仍失败，利用 `--debug` 输出检查 score、GroupMax、group exp、group sum、PV 各阶段，从第一个分歧点定位共享路径。
3. 重新设计 preload 为循环内持续的 K/KScale 前瞻加载（清楚界定 N 的距离与 tile 生命周期），然后再测吞吐。当前 prefix-only 代码不要当成这项优化已完成。
4. 功能通过后，重跑 `Sq=256, Skv=8192` 的 base/dblQ 对照；每次 perf 均保存 SwimLane、ELF、配置和 Total Cycles。
