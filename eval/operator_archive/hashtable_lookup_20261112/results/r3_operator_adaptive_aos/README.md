# r3：算子默认实现（32-lane AoS 自适应 2+6）——诊断，未计时

- **result_id**：`r3_operator_adaptive_aos`；角色：operator default（诊断）
- **源码**：`benchmark/one-level-arch/kernels/basic_op/hashtable_lookup/hashtable_lookup_tileop_adaptive.hpp`
  （main `7d3503bc`；默认 `HASH_ADAPTIVE_TAIL=6`，guard 112 B）。
- **构建**：主 `linx-toolchain-build` clang，`hash_mode=smoke/perf/verify` 三种均通过
  （0 error）；perf ELF（6144 dataset，AoS 表）sha256
  `38a5f34af9b0ab65d690d4c2766319aef5bde991f72933a4a5fa31933214476b`
  （构建于 2026-11-10；ELF 见 `ASSETS.md` 的 `A-R3-ELF`，`GAP-EXT-1`）。
- **性能：未获得**。gfsim（pr1177 clean）在该算子的 64 位 key gather 返回路径触发既有断言：

  ```
  ASSERTION FAILED: responseReturnBeatId == geo.mrobEntryIdx
  ```

  2026-11-12 复现；adaptive / stride4 / masked 三版同样触发。这是模型限制，不是本次
  算子改动引入；在 issue 中以诊断状态记录（`GAP-R3-1`）。
- **正确性**：`hash_mode=verify` 构建通过；没有运行期数值 PASS（受同一模型限制）。
- **与 r2 的关系**：同一 2+6 探测结构，但 r3 是 32-lane AoS（8 B key、16 B 表项），
  r2 是 256-lane SoA；性能数字不可直接互推。
