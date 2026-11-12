# r2：256-lane 自适应 2+6（SoA，`hashtable_lookup_soa_wide.hpp`）

- **result_id**：`r2_256l_2p6_rwdb48`；角色：adopted（**硬件假设**：RWDB=48；
  另有 `H-WAY` = TAG_WAY=7 变体）
- **workload**：hashtable_lookup（batch-map + SoA），6144 query，256 lane/batch，
  24 batch（每 PE 6），2.55M slot 表；keys u32 / values i32。
- **算法**：自适应 2+6 —— A 阶段 2 槽（全部 active lane），B 阶段只对 A 未命中 lane
  读 6 槽；value 在 batch 末尾 gather；batch 由 24 项 batch map 分配。
  源码：`benchmark/one-level-arch/kernels/basic_op/hashtable_lookup/hashtable_lookup_soa_wide.hpp`
  （main `7d3503bc`）。
- **构建**（opt-in，与主工具链不同）：PR146 clang + PR303 API +
  `-mllvm -linxv5-force-element-tile-mask=true`；fixture 由归档 256 版改 include
  到 soa_wide（`build/build_kernel.sh` 有完整命令）。
  ELF sha256 `35218ed00c5f34cd2a30d0c8ef7fea66ecfd5047611575d397246813383f1e2f`
  （当前只在 `<R2_ELF_DIR>/perf.elf`，`GAP-EXT-1`）。
- **性能（random300，`run_*.log`）**：

| run | RWDB | TAG_WAY | Total Cycles | kernel span | trace |
|---|---:|---:|---:|---:|---|
| `run_rw8.log` | 8 | 4 | 169,868 | 168,240（1,461–169,701） | `A-R2-RW8-TRACE` |
| `run_rw48.log` | 48 | 4 | **82,256** | 80,669（1,461–82,130） | `A-R2-TRACE` |
| `run_rw48_way7.log` | 48 | 7 | **76,959** | 75,351（1,461–76,812） | `A-R2-WAY7-TRACE` |

三个 run 的 trace 重跑周期与日志完全一致；均以 `linx_test_finisher ... pass` 结束、
退出码 0，trace 覆盖完整运行（无截断）。PipeView 诊断运行（`A-PIPEV`）周期不变
（r2 = 82,256），宿主实测 15.1 s、峰值 RSS 2.34 GB。

- **正确性**：同结构（64-lane 2+6）check 构建 PASS（`errors=0`）；**256-lane 2+6 本身
  未单独做 check 构建**（`GAP-R2-CHECK`）。数值结论以 r1/算子 check 为准。
- **结论边界**：RWDB=48 / WAY=7 是 §3.7 的 hardware_assumption（`H-RWDB`/`H-WAY`），
  不得与默认配置成绩混用；默认 RWDB=8 时该实现是 169,868，慢于 r1 的 135,819。
- **证据**：`run_*.log`；矩阵与 soa_wide 对比见 `../../analysis/PERFORMANCE_MATRIX.md`。

## 性能口径（NDF §3.9 / WORK-ARCHIVE-006）

- 原始行（`run_rw48.log`）：`Total Cycles = 82,256`；配置为 random300 +
  `sl2.RWDB_ENTRY_COUNT=48`、TAG_WAY=4。
- kernel 边界（`A-R2-TRACE` 重跑，周期数与 `run_rw48.log` 一致）：首个 TLOAD
  cycle 1,461 → 末 TSTORE 结束 82,130，kernel span = **80,669** cycles
  （占 Total 的 98.1%）。
- 计入/不计入：计入 kernel、线程同步与 finisher 写；不计入 host 时间与 checker
  （perf 构建跳过数值检查）。
- 指标公式：只声明延迟（cycles），不声明 MFU/有效带宽峰值，无峰值分母。
- 重复性：单次确定性仿真；不声明统计稳定性。
- trace：完整 SwimLane/nocounters 已保存（`A-R2-TRACE`，本机
  `outputs/hashtable_lookup_archive_assets/r2_rand300_rw48/`）；PipeView 尚未采集
  （`GAP-PV-1`）。

## ISA / 模型假设与 trace 完整性

- 目标三元组 `linx64v5-unknown-linux-musl`；计时用 gfsim（timing model），
  ASL 功能模型不适用；数值验证由同结构 64-lane check 构建在 gfsim 上完成
  （`GAP-R2-CHECK`：256-lane 本身未单独 check）。
- trace 为完整运行（`--swimlane 1`，无窗口/截断），事件范围从首个 TLOAD 到末
  TSTORE 之后；日志末行 `linx_test_finisher ... pass` 证明排空与完成，退出码 0。
- 发布包中的 `swim.log`/`meta.json` 已脱敏（`<REPO>`/`<GFSIM_WORKTREE>`/
  `<R2_ELF_DIR>` 占位），JSON trace 内容未改；原件与派生 hash 见
  `release_r3/sanitization.json` 与 `validation/PATH_SCAN.md`。
- 负控与失败路径见 `validation/negative_control/`（同一 ELF 翻转 1 个 golden 字节
  即 `val=0x3333 fail`、rc=70）。
