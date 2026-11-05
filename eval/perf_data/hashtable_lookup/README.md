# hashtable_lookup：6144-key 数值验证与执行轨迹

更新日期：2026-11-02。本目录默认材料已切换到6144-key数值校验版本。原139-key小测例单独保存在 [smoke_139](https://github.com/PTO-ISA/SuperNPUBench/blob/main/eval/perf_data/hashtable_lookup/smoke_139)，不能用它的27995 cycles解释本测例。

## 配置与验证结果

- 6144个U64查询，全部命中；表容量2550000，插入1785000项，装载率70%。4PE，每PE1536查询，即48批×32。
- 复用2026-10-31数值验证的完全相同ELF：SHA256 `960ba161b8c662a74bf3f93ed5ad96a75f3375d4a91eee4607bc870b9fa48bd4`。ELF内含原始table/queries/golden，无需重新生成随机数据。
- 本轮模型退出码0，四PE finisher均0x5555；fixture逐项检查6144个结果及首尾sentinel。res_check=off仅关闭通用框架检查，**不关闭fixture显式校验**；未链接libc。
- 新运行总周期 **421564**；原运行 **424487**。ELF不变，但gfsim二进制SHA256不同，少2923拍（约0.69%），不能称为算子优化或直接归因于trace开关。完整模型版本见versions.json，历史记录见historical_20261031.json。
- 使用指定SuperScalarModel工作树，4PE、读写各128 B/cycle、256KiB/64SSB、soc_random和vec_cell_sched开启。不使用废弃SoC参数。

## 文件

|材料|链接|
|---|---|
|完整反汇编|[hash_6144_PE4.diss](https://github.com/PTO-ISA/SuperNPUBench/blob/main/eval/perf_data/hashtable_lookup/hash_6144_PE4.diss)|
|数值校验版本ELF|[hash_6144_PE4.elf](https://github.com/PTO-ISA/SuperNPUBench/blob/main/eval/perf_data/hashtable_lookup/hash_6144_PE4.elf)|
|SwimLane（gzip，解压后Perfetto JSON）|[hash_6144_PE4.swimlane.json.gz](https://github.com/PTO-ISA/SuperNPUBench/blob/main/eval/perf_data/hashtable_lookup/hash_6144_PE4.swimlane.json.gz)|
|PipeView（gzip，解压后Konata）|[hash_6144_PE4.pipeview.log.out.gz](https://github.com/PTO-ISA/SuperNPUBench/blob/main/eval/perf_data/hashtable_lookup/hash_6144_PE4.pipeview.log.out.gz)|
|时间瓶颈分析|[BOTTLENECK_ANALYSIS.md](https://github.com/PTO-ISA/SuperNPUBench/blob/main/eval/perf_data/hashtable_lookup/BOTTLENECK_ANALYSIS.md)|
|完整PMU|[pmu.log](https://github.com/PTO-ISA/SuperNPUBench/blob/main/eval/perf_data/hashtable_lookup/pmu.log)|
|四PE数值PASS证据|[evidence.log](https://github.com/PTO-ISA/SuperNPUBench/blob/main/eval/perf_data/hashtable_lookup/evidence.log)|

压缩只改变存储形式，解压后分别用Perfetto/Konata打开。checksums.json给出归档文件和解压后轨迹的SHA256。command.json保留原运行路径，供审计；reproduce.py从当前归档目录读取ELF，可用环境变量MODEL覆盖模型绝对路径。

原全量调试日志保留在 `/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/outputs/hash_6144_trace_20261102/gfsim.log.gz`，约445MiB；远端存完整PMU和关键PASS日志，未把全量调试文本放入Git。原始未压缩轨迹也在 `/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/outputs/hash_6144_trace_20261102`。

## 优化版本（2026-11-04）

109,193-cycle 版本（256-lane、两槽并行、槽号寻址、empty-first、2 KB no-wrap guard）
归档在 [hash256_s4g_par_20261104](https://github.com/PTO-ISA/SuperNPUBench/tree/main/eval/perf_data/hashtable_lookup/hash256_s4g_par_20261104)：
追踪配置（`core.soc_random=false`、`core.soc_enable=false`）下 kernel span
**109,193** cycles、整个 fixture 171,646；规范配置（`core.soc_random=true`）
整个 fixture 400,331。

32 位 key gather 版（kernel **73,223**）归档在
[lo32_o3_73223_20261104](https://github.com/PTO-ISA/SuperNPUBench/tree/main/eval/perf_data/hashtable_lookup/lo32_o3_73223_20261104)：
追踪配置下 kernel span 73,223、整个 fixture 135,299；规范配置 280,218。
相对 109,193 版本 kernel -32.9%、整个 fixture -21.2%（规范配置 -30.0%）。
该版本只比较 key 低 32 位且假定所有查询命中（无空槽终止检查），通用语义请回退到
64 位比较 + 空槽检查的版本。

## 批次重排 + SoA 版（2026-11-05，kernel 63,983）

在 `lo32_o3_73223_20261104` 基础上做了三件事：key/value 拆成 SoA 数组、24 批到
4 PE 的最优 6+6+6+6 分配、以及性能模式开关 `HASH_SKIP_NUM_CHECK`。归档在
[batchmap_63983_20261105](https://github.com/PTO-ISA/SuperNPUBench/tree/main/eval/perf_data/hashtable_lookup/batchmap_63983_20261105)。

- 确定性配置（`soc_random=false`）：kernel span **63,983**；校验构建整个
  fixture 126,213（PASS），性能构建 64,399（跳过校验）。
- 规范配置（`soc_random=true`）：校验构建 277,675，性能构建 155,177。
- 相对 73,223 版本：kernel span -12.6%，确定性 fixture -6.7%，规范配置
  （校验构建）280,218 → 277,675（-0.9%）。

数值结论仍以校验构建 `hash_6144_PE4_batchmap.elf` + `check_hashtable_fixture.py`
为准；性能构建的 0x5555 是无条件写入的。
