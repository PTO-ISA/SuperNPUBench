# hashtable_lookup 性能与执行轨迹

归档日期：2026-11-02。源码：PR #272，main合并提交21439690fb2b4bdcd6a4117a74251237cc78da95。

配置：4 PE、139个查询、表容量37、初始插入29项；res_check=off，无libc。不是此前6144-key/70%装载率的测例。gfsim正常退出，Total Cycles=27995；未进行数值校验，不表示尾批次数值通过，也不是严格纯kernel性能。

模型：/Users/blacktraker/Programming/gitproj/DV4/SuperScalarModel/bin/gfsim。读写各128 B/cycle，SharedTile 256 KiB/64SSB，开启soc_random、vec_cell_sched和internal_acc。

## 文件

- [hash_PE4.elf](https://github.com/PTO-ISA/SuperNPUBench/blob/main/eval/perf_data/hashtable_lookup/hash_PE4.elf)
- [hash_PE4.diss](https://github.com/PTO-ISA/SuperNPUBench/blob/main/eval/perf_data/hashtable_lookup/hash_PE4.diss)
- [hash_PE4.swimlane.json](https://github.com/PTO-ISA/SuperNPUBench/blob/main/eval/perf_data/hashtable_lookup/hash_PE4.swimlane.json)
- [hash_PE4.pipeview.log.out](https://github.com/PTO-ISA/SuperNPUBench/blob/main/eval/perf_data/hashtable_lookup/hash_PE4.pipeview.log.out)
- [gfsim.log](https://github.com/PTO-ISA/SuperNPUBench/blob/main/eval/perf_data/hashtable_lookup/gfsim.log)
- [command.json](https://github.com/PTO-ISA/SuperNPUBench/blob/main/eval/perf_data/hashtable_lookup/command.json)
- [results.json](https://github.com/PTO-ISA/SuperNPUBench/blob/main/eval/perf_data/hashtable_lookup/results.json)

SwimLane使用Perfetto查看；PipeView为Konata格式。command.json和results.json保留原运行路径，作为原始记录。原outputs目录中的材料保留。
