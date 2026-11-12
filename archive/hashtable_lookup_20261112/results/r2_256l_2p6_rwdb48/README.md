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
  （当前只在 `/tmp/soa_op/perf.elf`，`GAP-EXT-1`）。
- **性能（random300，`run_*.log`）**：

| run | RWDB | TAG_WAY | Total Cycles |
|---|---:|---:|---:|
| `run_rw8.log` | 8 | 4 | 169,868 |
| `run_rw48.log` | 48 | 4 | **82,256** |
| `run_rw48_way7.log` | 48 | 7 | **76,959** |

- **正确性**：同结构（64-lane 2+6）check 构建 PASS（`errors=0`）；**256-lane 2+6 本身
  未单独做 check 构建**（`GAP-R2-CHECK`）。数值结论以 r1/算子 check 为准。
- **结论边界**：RWDB=48 / WAY=7 是 §3.7 的 hardware_assumption（`H-RWDB`/`H-WAY`），
  不得与默认配置成绩混用；默认 RWDB=8 时该实现是 169,868，慢于 r1 的 135,819。
- **证据**：`run_*.log`；矩阵与 soa_wide 对比见 `../../analysis/PERFORMANCE_MATRIX.md`。
