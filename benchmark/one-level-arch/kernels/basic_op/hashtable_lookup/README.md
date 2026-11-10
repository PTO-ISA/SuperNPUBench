# TileOP hash + element-if lookup

默认实现：`hashtable_lookup_tileop_adaptive.hpp`（经 `hashtable_lookup.hpp` 转发），
自适应 2+2 probe。旧的四槽/轮实现保留在 `hashtable_lookup_tileop_stride4_guard.hpp`，
可用 `-DHASH_KERNEL_HEADER='"..."'` 选择。

接口：

```cpp
supernpu::hash_lookup::lookup<Capacity>(table, queries, output, count);
supernpu::multi_thread::hashtable_lookup<Capacity, 4>(table, queries, output, count);
```

第一种由调用者分配任务；第二种按 PE ID 划分连续 query 区间，要求启动对应数量的 PE。支持非 32 倍数的 count，count=0 不访存。输入与输出不可重叠。

## 自适应 2+6 probe（2026-11-10 起的默认实现）

每轮 probe 拆成两个阶段：

- **A 阶段**：所有仍 active 的 lane gather slot o、o+1（两条 gather 背靠背发出）；
- **B 阶段**：只有 A 里两个 slot 都没命中的 lane 才 gather slot o+2 … o+7
  （6 条 gather 一起发出，再按顺序比较）。

旧 stride4 实现把 4 条 gather 都挂在轮首 mask 下，命中较早的 lane（约 80% 在前两个
slot 就结束）仍要为后面的 slot 付 gather 请求；自适应版把 B 阶段 mask 收紧到 A 之后的
`activeB`，直接省掉这部分投机请求，同时用更宽的 B 组减少总轮数。

B 宽度由 `HASH_ADAPTIVE_TAIL` 选择（默认 6；设 2 即回到 2+2）。guard 契约随 tail 变化：
Capacity-1 之后需要 `HASH_ADAPTIVE_TAIL + 1` 个可读的 16 B 项（镜像 slot 0..tail），
即 tail=6 → 7 项 / 112 B，tail=2 → 3 项 / 48 B；要求 Capacity >= 4。
测试 harness 的 `hash_table_guard` 默认已改为 112。

### 性能（SoA perf kernel，gfsim random300；非本 AoS 实现的直接测量）

| RWDB | 64-lane 2+6 | 256-lane 2+6 |
|---:|---:|---:|
| 8（默认） | 143,096 | 170,132 |
| 16 | 118,207 | 136,217 |
| 24 | 117,524 | 95,444 |
| 32 | 116,635 | 86,331 |
| **48** | **114,348** | **81,317** |
| 64 | 117,209 | — |

- 256-lane 2+6 + `TAG_WAY=7` + RWDB=48 → **77,312**（random300）。
- 对照：2+2 在默认 RWDB=8 下 64-lane 135,819；256-lane 在 RWDB=48 下 2+2 为 90,483。
- 环境：`SuperScalarModel-pr1177/bin/gfsim`（head `18b24f6ca`），
  `core.soc_random=true`、`core.mean_read_latency=300`、`core.mean_write_latency=300`、
  读写带宽 128 B/cycle、`core.sharetreg_size_bytes=262144`、`core.sharetreg_ssb_count=64`、
  `TAG_WAY_COUNT=4`（除注明外）；RWDB 用 `-s sl2.RWDB_ENTRY_COUNT=N` 覆盖。
  kernel 为 batch-map + SoA、64/128/256 lane；perf kernel 每次 gather 返回 256 B，
  与本算子的 32 lane × 8 B gather 相同（都是 2 beat），所以 RWDB 结论可直接沿用。

### 为什么 RWDB 用 48

1. 2+6 每轮每个 PE 发 8 条 gather（2 A + 6 B），每条返回 256 B = 2 个 beat。
   RWDB=8 只能同时放 `floor(8/2)=4` 条 gather，6 条 B gather 被拆成多波串行；
   RWDB=48 可放 24 条在飞 = 3 个完整 2+6 轮，足以覆盖 gather 返回延迟
   （entry 驻留 p50 约 86 cycle，整条 gather 273~290 cycle）。
2. 实测扫描 64-lane 2+6：8→143,096、16→118,207、24→117,524、32→116,635、
   **48→114,348**、64→117,209。48 是最优点；再加深没有收益（返回队列/延迟交互），
   反而略退化。
3. 256-lane 2+6 同样在 48 见顶（81,317），不是某个 lane 宽度的偶然。
4. 注意：RWDB 是硬件配置项。在默认 RWDB=8 下，2+2 仍优于 2+6（135,819 vs 143,096），
   所以 2+6 作为默认的前提是目标配置 RWDB ≥ 16（推荐 48）；否则请用
   `-DHASH_ADAPTIVE_TAIL=2` 并把 `hash_table_guard` 设回 48。

### 验证

- 编译（主 `linx-toolchain-build`）：`hash_mode=smoke/perf/verify` 三种构建均通过。
- 计时验证受限：gfsim 在该算子的 64 位 gather 返回路径触发既有断言
  `responseReturnBeatId == geo.mrobEntryIdx`，2+6/2+2/stride4 三版同样触发，与本改动
  无关；gfrun 侧此前记录的 INT64 TLOAD 限制也仍未解。

- 表项为 16 字节：int64 key、int32 value、int32 padding；空槽 key=-1，miss 输出 -1。无删除标记，key=-1 不可插入。
- MurmurHash3 x86_32、seed=0，低 32 位先混合，再高 32 位，最后按 8 字节长度 finalize；与用户附件和 two-level 算法一致。
- 哈希使用 TMULS/TADDS/TXOR/TSHLS/TSHRS/TOR 等 TileOP。无符号取模采用 two-level 的两次有符号 TREM 恒等式，不要求 capacity 为 2 的幂。
- 查找使用 ElementTile / TPARTELEMENT / `#pragma pto element for`，内部 if 判断 live、空槽、完整 64 位 key 命中及回绕。结果与状态先在局部标量中合并，每轮每个 view 只写一次，避免当前编译器拒绝同一 lane 多次条件写入。
- 元素区域外按 active 状态动态循环：每轮通过 `TCMPS<NE>(active, 0)` 直接生成 GPR 掩码，并与 `(uint64_t{1} << valid) - 1` 相与，忽略尾块的补齐 lane；结果非零才继续探测。退出判断不再写入/读回 GM。Capacity 仅作为满表 miss 的安全上限，不是固定执行次数。
- `0 < Capacity <= INT32_MAX / 16`，确保槽位字节偏移和模运算中间值不溢出。S32 输出使用 M322ND TSTORE，lb1 设置为运行时 valid 行数，直接从 result Tile 写最终 GM。当前 ElementTile 固定完整有效形状，且与动态 CubeTile 的寄存器载体不兼容，因此在算子内复用 TSTORE_CUBE 的动态行内联汇编编码，不使用 scratch、标量回读或 MSCATTER。32-query/4-PE 测例编译与存储执行通过，trace 确认每 PE lb1=8；最终数值校验仍 FAIL。

参考接口：

- `/Users/blacktraker/Programming/gitproj/DV4/linx-toolchain-build/src/Linx-TileOP-API/docs/tileop-usage/element-if.md`
- `/Users/blacktraker/Programming/gitproj/DV4/pto-spec/docs/status/decisions/ADR-TILE-0008-tile-elementwise-and-irregular-operations.md`
- `/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/two-level-arch/kernels/control/hashtable_lookup_simd.hpp`

## 验证（2026-10-26）

已用指定主工具链成功编译、链接三种配置：37 槽约 80% 装载、37 槽满表、1 槽满表。测试各包含 139 个 query，4 PE，不整除的 PE 分区及 tile 尾块；用独立逐表扫描 golden 和首尾 guard 检查输出。输入包含负数、相同低 32 位但高位不同的 key、命中和 miss。

**运行校验尚未通过**：默认配置的 gfrun 在首次 `TLOAD INT64 / ND2M32.normal` 处触发 `CUBE TLOAD requires a fitting CELL destination`，尚未执行输出比对。已进一步定位：当前 ASL 模型的 `CubeCellElementBits(INT64)` 返回 0，`CubeCellGeometry` 随即拒绝；其 64 位 CUBE 路径仅支持 N8/U64，尚未接入 ISA #371 的 M32 double-CELL。ELF 分配 256B，与 32×1×8B 一致。当前 ISA 可执行合同明确允许 M32 S64 的 TLOAD/TSTORE，因此这次断言的直接原因是指定 ASL 模型版本不支持该布局/类型组合，而非目标容量不足。没有修改外部依赖，也没有宣称功能 PASS。

定位源码：
- `/Users/blacktraker/Programming/gitproj/DV4/SuperScalarModel-asl/isa/ISACommon/CubeCell.h`（CubeCellElementBits / CubeCellGeometry）
- `/Users/blacktraker/Programming/gitproj/DV4/pto-spec/asl/tile/model/shape/cube-double-cell.asl`（M32 b64 两个 128B CELL）
- `/Users/blacktraker/Programming/gitproj/DV4/pto-spec/asl/block/model/dispatch/tlsu-layout-conversion.asl`（BundleCubeTransportDataTypeSupported）

注意部分 ISA TLOAD 文字摘要仍保留旧的 non-64-bit 描述，与新增可执行合同不一致；上述结论依据 #371 已接受修订及 ASL 代码。

复现：

```bash
python3 /Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/test/kernel/hashtable_lookup/run_checks.py
```

脚本使用主工具链，运行指定 ASL worktree 的 gfrun；遇到非零退出、超时或缺少明确 PASS 即失败，输出实际绝对 artifact 路径。

本轮编译日志、ELF、完整反汇编、gfrun 日志、关键 trace 和精确版本保存在：
`/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/outputs/hash_lookup_element_20261026/`

## main gfrun 对照（用户指定，2026-10-26）

模型 checkout：`/Users/blacktraker/Programming/gitproj/DV4/SuperScalarModel-main`，commit `c668a6388907f503e80954fdfd4d013395d8775d`。

- 同一 ELF、4 PE：加载器拒绝旧 hosted PE0/worker ABI，要求 common-entry SPMD runtime，未执行算子。
- 主工具链重编译 `CC_OPTS=-DHASH_PE_COUNT=1`，1 PE 处理全部 query 并跳过四 PE 完成屏障：启动阶段在 PC `0x13250` 的 `acrc 1` 报 `illegal architecture manager request`，仍未进入 main 的算子验证。

因此 main 源码有 M32/INT64 支持不等于这份 hosted ELF 已能运行；目前还存在运行时 ABI/系统调用契约的版本不兼容。没有修改外部依赖或绕过模型校验。

日志：
- `/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/outputs/hash_lookup_element_20261026/gfrun_main.log`
- `/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/outputs/hash_lookup_element_20261026/gfrun_main_single.log`
- `/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/outputs/hash_lookup_element_20261026/kernel_main_single.diss`

## 探测循环修订

原版本固定执行 Capacity 轮已修正为按活跃 lane 动态退出。尾块无效 lane 不参与退出判断，避免它们初始化的 active=1 导致循环跑满。单 lane 命中或遇空槽即停，整批全部完成即退出；满表 miss 最多遍历一圈。此次修改仍受上述 runtime/model 阻塞影响，编译成功不代表运行数值验证通过。

## 无 libc 穿刺（2026-10-26）

按用户要求，默认 compile.all 已改为 res_check=off。该路径用 -ffreestanding 编译、-nostdlib -nostartfiles 链接仓库最小入口，不生成 printf/golden 校验代码；保留输入构造及必要的 PE 同步。run_checks.py 显式指定 res_check=on，避免误将无校验路径当作数值 PASS。

构建命令：
```bash
/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/test/kernel/hashtable_lookup/compile.all OBJ_ROOT=/tmp/hash_element_smoke
```

在 main worktree 用 gfrun、softcore.multiThreadNum=4 跑穿刺：已越过 hosted ABI 检查，进入 main；在 PC 0x11330 的旧 ssrget 0x0802 处失败，原因是它位于 STD 块，新模型要求 SYS 块且使用 THREAD_ID 0x0026。因此尚未执行 hash，也没有声称穿刺完成。未修改外部编译器或模型。

证据：
- `/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/outputs/hash_lookup_element_20261026/kernel_smoke.elf`
- `/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/outputs/hash_lookup_element_20261026/kernel_smoke.diss`
- `/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/outputs/hash_lookup_element_20261026/gfrun_main_smoke.log`

## PE ID 修订后穿刺（2026-10-26）

经用户授权，在指定主工具链 checkout 移植 0.59 / fc39fb07b 的 THREAD_ID=0x0026 修改，并补充 SSR_GET 的 SYS 属性及独立块隔离；未切换整个 0.59 分支，未合入无关 SYNC.CORE 特性，修改保留为本地 diff。增量构建并安装 clang、llc、llvm-objdump；PE ID 回归覆盖直接返回和读取后普通算术，通过。

重新构建无 libc、res_check=off 的 4 PE ELF，main gfrun 已越过原 PE ID 失败以及 INT64 TLOAD，后续在 PC 0x11994 的普通 S64 TSTORE 触发 `ordinary TSTORE requires PadValue=Zero`。仍未跑完整个算子，不构成功能 PASS；没有修改模型或继续修改这个新问题的编译器路径。

本轮证据目录：`/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/outputs/hash_lookup_element_20261026/`，包含 compiler_peid.patch、toolchain_peid_build.log、toolchain_peid_install.log、peid_fixed_versions.txt、kernel_peid_fixed.elf、kernel_peid_fixed.diss 和 gfrun_main_peid_fixed.log。

## 默认实现（2026-11-10 更新）

公共入口 `hashtable_lookup.hpp` 转发到 `hashtable_lookup_tileop_adaptive.hpp`：
自适应 2+6 probe（A 阶段读 2 槽；B 阶段只对 A 未命中的 lane 读 6 槽），
表尾 guard 112 B（7 项，镜像 slot 0..6）。性能、环境与 RWDB=48 的取舍见上文
“自适应 2+6 probe” 一节。

- 2026-11-03 记录（历史，已被上面的默认替换）：stride4 每轮探测 4 个连续槽位、
  48 字节 guard；6144 query / 4 PE / 70% 装载 **140,166 cycles**（首 TLOAD → 末
  TSTORE），相对 177,474 基线 -21.0%。
- 表契约（当前）：调用方必须在 slot `Capacity-1` 之后提供 7 个可读的 16 字节 entry
  （镜像 slot 0..6），且 `Capacity >= 4`；`HASH_ADAPTIVE_TAIL=2` 或 stride4 回退时为 3 个。
- 测试夹具：`hash_table_guard` 默认 112（Makefile / dataset fixture / smoke 表 /
  `run_dataset.py --table-guard`）；smoke 表按 `HASH_TABLE_GUARD/16` 扩展并填充 guard。
- 不带 guard 的实现仍保留在 `hashtable_lookup_tileop_masked.hpp`；如需回退，
  用 `HASH_KERNEL_HEADER=<...>/hashtable_lookup_tileop_masked.hpp
  hash_table_guard=0` 编译。

128-lane（32x4）尝试与工具链阻塞记录见
`/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/outputs/hash_lane128_20261103/README.md`；
上游 issue：https://github.com/LinxISA/linx-toolchain-build/issues/24
