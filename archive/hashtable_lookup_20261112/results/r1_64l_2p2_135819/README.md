# r1：64-lane 自适应 2+2（默认模型参数下的 kernel 侧最优）

- **result_id**：`r1_64l_2p2_135819`；角色：baseline（默认 RWDB=8、WAY=4）
- **workload**：hashtable_lookup（batch-map + SoA），6144 query，64 lane/batch，
  96 batch（每 PE 24），2.55M slot 表，70% 装载；key 为 u32（64-bit key 低 32 位），
  value 为 i32。
- **算法**：每轮 A 阶段读 slot o、o+1（全部 active lane），B 阶段只对 A 未命中 lane
  读 o+2、o+3；value 在 batch 末尾 gather 一次。count 为 Lanes 整数倍。
- **构建**：PR146 clang + PR303 API + `-mllvm -linxv5-force-element-tile-mask=true`；
  源码 `outputs/hash64_batchmap_20261105/slots_sweep/adaptive/kernel_adaptive_2_2.hpp`，
  fixture64/map64 在 `.../repro_matrix/`。重建 ELF 与归档 ELF md5 一致
  （`f5d5a0cb935efe851504a17559fe1ba1`，sha256 `2324ea4c…e3b7d6`）。
- **正确性**：同结构 check 构建（`HASH_SKIP_NUM_CHECK=0` + checker）
  `status=PASS`、`output_matches=true`、`errors=0`（check 运行 150,516 cycles，
  含同步/校验，非 kernel 性能）。
- **性能**：random300 `Total Cycles = 135,819`（`run.log`）；确定性（soc 关闭）= 89,048。
- **证据/复现**：`run.log`；`build/` 脚本；ELF 在
  `outputs/hash64_batchmap_20261105/slots_sweep/elfs/adaptive_2p2_perf.elf`
  （`GAP-EXT-1`：未发布）。
