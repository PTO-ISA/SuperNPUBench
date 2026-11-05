# hashtable_lookup：SoA + 批次重排版（kernel 63,983 cycle）

更新日期：2026-11-05。在 `lo32_o3_73223_20261104`（kernel 73,223）基础上继续：

1. **SoA 布局**：key 低 32 位与 value 拆成两个连续数组，探测只读 key 数组
   （地址步长 16 B → 4 B），value 在批次收尾单独 gather。
2. **批次重排**：24 个批次到 4 个 PE 的分配由连续切分改成离线搜到的
   6+6+6+6 最优分配（最慢 PE 的探测轮数 94 → 85）。
3. **性能模式开关** `HASH_SKIP_NUM_CHECK`：跳过 fixture 的 6,144 项 golden
   比对，用于纯计时；默认关闭，数值结论仍用校验构建。

## 结果

| 配置 | 构建 | kernel span | 整个 fixture | 数值 |
|---|---|---:|---:|---|
| 确定性（`soc_random=false`、`soc_enable=false`） | 校验 | **63,983** | 126,213 | PASS（6144/6144） |
| 确定性 | 性能（`HASH_SKIP_NUM_CHECK=1`） | 63,983 | **64,399** | 跳过（finisher 0x5555） |
| 规范（`core.soc_random=true`） | 校验 | — | 277,675 | PASS |
| 规范 | 性能 | — | **155,177** | 跳过 |

kernel span 明细：首个输入 TLOAD cycle 298，最后结果 TSTORE 结束 64,281；各 PE
最后一个 TSTORE：PE0 63,544、PE3 63,610、PE1 63,678、PE2 64,281。

与上一版对比（确定性、校验构建）：kernel span 73,223 → **63,983（-12.6%）**，
整个 fixture 135,299 → 126,213（-6.7%）。规范配置下（校验构建）277,675 vs
280,218（-0.9%）：随机访存延迟占主导，VEC 侧收益被摊薄。

## 数据与产物

- `data/keys32.bin`：2,550,000 × u32（64 位 key 的低 32 位）+ 128 项 guard。
- `data/values.bin`：2,550,000 × i32 value + 128 项 guard。
- `data/batch_map.bin`：24 × u32，PE t 处理 `batch_map[t*6..t*6+5]`。
- 两个数据数组由 `gen_soa.py` 从归档 AoS 表离线生成；`gen_batch_map.py`
  记录批次分配的计算方式并复核每 PE 的轮数和。
- 校验 ELF：`hash_6144_PE4_batchmap.elf`（SHA256 `3344129c…`），保留 6,144 项
  逐项校验；性能 ELF：`hash_6144_PE4_batchmap_perf.elf`（`5c9eccb1…`），跳过校验。
- 轨迹：`hash_6144_PE4_batchmap.swimlane.json.gz`（校验构建，126,213）、
  `hash_6144_PE4_batchmap_perf.swimlane.json.gz`（性能构建，64,399）、
  `hash_6144_PE4_batchmap.pipeview.log.out.gz`（校验构建）。

## 环境

| 组件 | 版本 | 路径 |
|---|---|---|
| 编译器 | PR #146 `a5f03b38a` + issue #151 修复 `71351b61`（工作树 `75987bcc`） | `/Users/blacktraker/Programming/gitproj/DV4/llvm-project-pr146/build-hash/bin/clang++` |
| API 头 | PR #303 `decac4f` | `/tmp/api-pr303/include` |
| 计时模型 | PR #1177 `18b24f6ca` | `/Users/blacktraker/Programming/gitproj/DV4/SuperScalarModel-pr1177/bin/gfsim` |
| 校验脚本 | `scripts/check_hashtable_fixture.py` | 同模型工作树 |

## 复现

编译（校验构建；性能构建加 `-DHASH_SKIP_NUM_CHECK=1`）：

```sh
CLANG=/Users/blacktraker/Programming/gitproj/DV4/llvm-project-pr146/build-hash/bin/clang++
REF=/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/eval/perf_data/hashtable_lookup/hash_6144_PE4.elf
$CLANG --target=linx64v5-unknown-linux-musl -mlxbc -fenable-matrix -O3 -std=c++20 \
  -D__linx -DENABLE_TENSOR_INSTR -ffreestanding -fno-exceptions -fno-rtti \
  -fno-vectorize -fno-slp-vectorize \
  -mllvm -linxv5-enable-generic-element=true \
  -mllvm -enable-all-vector-as-tilereg=true \
  --sysroot=/Users/blacktraker/Programming/gitproj/DV4/linx-toolchain-build/output/linx_blockisa_llvm_musl/sysroot \
  -nostdinc++ -isystem /tmp/lane128-prcfg \
  -isystem /Users/blacktraker/Programming/gitproj/DV4/llvm-project-pr146/libcxx/include \
  -I/tmp/api-pr303/include -I/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench \
  -DHASH_DATA_ELF_PATH="\"$REF\"" \
  -DHASH_KEY_DATA="\"$PWD/data/keys32.bin\"" \
  -DHASH_VALUE_DATA="\"$PWD/data/values.bin\"" \
  -DHASH_BATCH_MAP_DATA="\"$PWD/data/batch_map.bin\"" \
  -DHASH_QUERY_REPEATS=1 -DHASH_WIDE_LANES=256 \
  -mllvm -linxv5-force-element-tile-mask=true -c fixture.cpp -o /tmp/hash_bm.o
/Users/blacktraker/Programming/gitproj/DV4/llvm-project-pr146/build-hash/bin/ld.lld \
  -T link.ld /tmp/hash_bm.o -o /tmp/hash_bm.elf
```

运行（确定性）与校验：

```sh
cd /Users/blacktraker/Programming/gitproj/DV4/SuperScalarModel-pr1177
./bin/gfsim --test-finisher 1 -f /tmp/hash_bm.elf --conf fourpe \
  -s softcore.multiThreadNum=4 -s cube.enable_internal_acc=true \
  -s bctrl.vec_cell_sched_enable=true -s core.soc_random=false \
  -s core.soc_enable=false -s core.read_bandwidth_limit=128 \
  -s core.write_bandwidth_limit=128 -s core.sharetreg_size_bytes=262144 \
  -s core.sharetreg_ssb_count=64 -t 1
python3 scripts/check_hashtable_fixture.py --elf /tmp/hash_bm.elf \
  --reference-elf ../SuperNPUBench/eval/perf_data/hashtable_lookup/hash_6144_PE4.elf \
  --model gfsim --output /tmp/check_bm
```

规范配置把 `core.soc_random` 改成 `true`（去掉 `core.soc_enable=false`）。

## 文件

| 文件 | 说明 |
|---|---|
| `hash_6144_PE4_batchmap.elf` / `.diss` | 校验构建 ELF 与反汇编 |
| `hash_6144_PE4_batchmap_perf.elf` | 性能构建 ELF（跳过校验） |
| `kernel.hpp` / `fixture.cpp` / `link.ld` | 源码（`fixture.cpp` 含开关） |
| `data/keys32.bin` / `data/values.bin` / `data/batch_map.bin` | 布局与批次分配数据 |
| `hash_6144_PE4_batchmap.swimlane.json.gz` | 校验构建 SwimLane（126,213） |
| `hash_6144_PE4_batchmap_perf.swimlane.json.gz` | 性能构建 SwimLane（64,399） |
| `hash_6144_PE4_batchmap.pipeview.log.out.gz` | 校验构建 PipeView |
| `check.log` / `final-metrics.txt` | 数值校验与 kernel span 证据 |
| `results.json` / `command.json` / `versions.json` | 结构化结果、命令、版本 |
| `checksums.json` | 本目录文件 SHA256 |
| `gen_soa.py` / `gen_batch_map.py` | 数据与批次分配生成/复核脚本 |

## 已确认的上游阻塞（后续优化前提）

- element scatter legalization（`PredicateCell unordered scatter legalization is
  not implemented`）：阻挡"按探测长度排序 query + 输出 scatter 写回"，
  离线分析显示 24 批 rounds 之和可从 338 降到 62。
- element atomic-add legalization（`PredicateCell atomic-add legalization is not
  implemented`）：阻挡未命中 lane 的 compaction（稠密化尾部）。
- TLOAD 出的 tile 作为 gather index 的契约问题：阻挡离线预计算 hash。
- element-region 宽 per-lane 加载（8 B pair gather）：阻挡一次 gather 取两个槽。
- 两阶段/乒乓：issue #151 已修 ownership，模型侧 TCI 挂死与 tile descriptor
  PHI join 仍是剩余阻塞。
