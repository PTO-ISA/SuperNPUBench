# hashtable_lookup 256-lane 优化版（109,193 cycle 构建）

更新日期：2026-11-04。本目录归档 256-lane 线性探测 lookup 的最终优化构建，
即 `outputs/hash256_s4g_par_20261104` 中 kernel span 为 **109,193 cycles**
的 ELF（SHA256 `eb4c3bae829e44db631e7892069561b67c05d6bdc2654aaefdb28334f3a3d563`）。
父目录的 6144-key 基准材料保持不变。

## 测例与算法

- 6,144 个 U64 查询全部命中；表容量 2,550,000、插入 1,785,000 项（70% 装载率），
  4 PE，每 PE 1,536 查询 = 6 批 × 256 lane。
- 表项 16 B（key U64 + value I32 + padding）。fixture 在 slot Capacity-1 之后
  追加 2 KB guard，镜像 slot 0..127；本测例最长探测 63 槽，因此探测过程中
  无需对偏移做回绕。
- 每轮并行发出两个相邻槽的 key gather（同一个 round-start mask），两次 load
  的 TLSU 延迟相互重叠；命中后只记槽号，批次结束时一次性 gather value。

## 结果

| 配置 | kernel span | 整个 fixture | 数值 |
|---|---:|---:|---|
| 追踪配置（`core.soc_random=false`、`core.soc_enable=false`、`--conf fourpe`） | **109,193** | 171,646 | PASS |
| 规范配置（`core.soc_random=true`、读写 128 B/cycle、256 KiB/64 SSB） | — | 400,331 | finisher PASS |

kernel span 定义为第一个输入 TLOAD（cycle 290）到最后一个结果 TSTORE
（cycle 109,483）。追踪配置关闭了 SoC 的随机访存延迟，因此 fixture 总周期显著
低于规范配置；与仓库其他算子横向比较时应使用规范配置的数字。

同口径（追踪配置）对比：

| 版本 | kernel span | 整个 fixture |
|---|---:|---:|
| PR `wide_mask` 256 lane，1 槽/轮 | 187,851 | 250,057 |
| 256 lane，4 槽/轮 + guard（s4g 基线） | 130,577 | 193,010 |
| **本版本：2 槽/轮并行 + 槽号 + empty-first + no-wrap** | **109,193** | **171,646** |

- 相对 s4g 基线：kernel -16.4%，整个 fixture -11.1%。
- 相对 PR stride-1：kernel -41.9%，整个 fixture -31.4%。

随机 SoC 口径（`--soc-random`，同一 PR 模型 `18b24f6ca`）：

| 版本 | 整个 fixture |
|---|---:|
| PR `wide_mask` 256 lane，1 槽/轮 | 467,361 |
| s4g 基线 | 404,765 |
| **本版本** | **400,331** |

随机延迟放大访存时间后，VEC 侧收益被摊薄：本版本相对 s4g 为 -1.1%，相对
PR stride-1 为 -14.3%。两个口径都保留在归档中，避免把确定性追踪数字当作
规范配置结论。

## 优化要点

1. **两槽并行**：每轮两次 key gather 在同一 round-start mask 下背靠背发出，
   intra-round gather spacing 由 350-430 cycles 降到 79 cycles。
2. **槽号代替字节偏移**：省掉 `>>4`/`x16` 的地址往返，每四次 load 少约 17 TCI。
3. **合并判断、去掉恒真的 `element < valid`**：减少每槽的 compare/select。
4. **empty-first 分支顺序**：先判 `found == -1` 再判等值；该顺序在 stride-2
   形态下更快（stride-3 形态下相反，故单独测量后弃用）。
5. **无回绕**：2 KB guard 让 2-槽块的基址直接 `+2` 槽，不再每轮取模/回绕。

被否掉的变体（均已实测，非编译消除）：stride-3 + no-wrap 174,588、
stride-4 + no-wrap 183,681、512 lane 197,408、128 lane stride-2 212,360
（以上均为追踪配置下的整个 fixture）。

## 复现

构建（PR 编译器 #146 `a5f03b38` + API #303 `decac4f`）：

```sh
CLANG=/tmp/llvm-pr146/build-hash/bin/clang++
SRC=/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/outputs/hash256_s4g_par_20261104/src
REF=/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/eval/perf_data/hashtable_lookup/hash_6144_PE4.elf
$CLANG --target=linx64v5-unknown-linux-musl -mlxbc -fenable-matrix -O2 -std=c++20 \
  -D__linx -DENABLE_TENSOR_INSTR -ffreestanding -fno-exceptions -fno-rtti \
  -fno-vectorize -fno-slp-vectorize \
  -mllvm -linxv5-enable-generic-element=true \
  -mllvm -enable-all-vector-as-tilereg=true \
  --sysroot=/Users/blacktraker/Programming/gitproj/DV4/linx-toolchain-build/output/linx_blockisa_llvm_musl/sysroot \
  -nostdinc++ -isystem /tmp/lane128-prcfg -isystem /tmp/llvm-pr146/libcxx/include \
  -I/tmp/api-pr303/include -I/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench \
  -DHASH_DATA_ELF_PATH="\"$REF\"" -DHASH_QUERY_REPEATS=1 -DHASH_WIDE_LANES=256 \
  -mllvm -linxv5-force-element-tile-mask=true -c $SRC/fixture.cpp -o /tmp/hash.o
/tmp/llvm-pr146/build-hash/bin/ld.lld -T $SRC/link.ld /tmp/hash.o -o /tmp/hash.elf
```

运行（PR 模型 #1177 `18b24f6ca`，工作树 `/tmp/ssm-pr1177`）：

规范配置（`pmu.log` 即此命令输出）：

```sh
cd /tmp/ssm-pr1177
./bin/gfsim --test-finisher 1 \
  -s softcore.multiThreadNum=4 -s cube.enable_internal_acc=true \
  -s bctrl.vec_cell_sched_enable=true -s core.soc_random=true \
  -s core.read_bandwidth_limit=128 -s core.write_bandwidth_limit=128 \
  -s core.sharetreg_size_bytes=262144 -s core.sharetreg_ssb_count=64 \
  -f ./hash_6144_PE4_wide256.elf -t 1
```

追踪配置（生成 `hash_6144_PE4_wide256.swimlane.json.gz` 与 kernel span）：

```sh
./bin/gfsim --test-finisher 1 -f ./hash_6144_PE4_wide256.elf --conf fourpe \
  -s softcore.multiThreadNum=4 -s cube.enable_internal_acc=true \
  -s bctrl.vec_cell_sched_enable=true -s core.soc_random=false \
  -s core.soc_enable=false -s core.read_bandwidth_limit=128 \
  -s core.write_bandwidth_limit=128 -s core.sharetreg_size_bytes=262144 \
  -s core.sharetreg_ssb_count=64 -t 1 --swimlane 1 \
  --swimfile /tmp/wide256.swimlane.json \
  -s dfx.swimCounterSampleInterval=128
```

数值校验（`final-result.json` 即此命令输出）：

```sh
python3 /tmp/ssm-pr1177/scripts/check_hashtable_fixture.py \
  --elf ./hash_6144_PE4_wide256.elf \
  --reference-elf ../hash_6144_PE4.elf --model gfsim --output /tmp/hashcheck
```

`reproduce.py` 用本目录中的 ELF 重跑规范配置，`MODEL` 环境变量可覆盖模型路径。

## 材料

| 材料 | 文件 |
|---|---|
| ELF | [hash_6144_PE4_wide256.elf](https://github.com/PTO-ISA/SuperNPUBench/blob/main/eval/perf_data/hashtable_lookup/wide256_109193_20261104/hash_6144_PE4_wide256.elf) |
| 反汇编 | [hash_6144_PE4_wide256.diss](https://github.com/PTO-ISA/SuperNPUBench/blob/main/eval/perf_data/hashtable_lookup/wide256_109193_20261104/hash_6144_PE4_wide256.diss) |
| kernel 源码 | [kernel.hpp](https://github.com/PTO-ISA/SuperNPUBench/blob/main/eval/perf_data/hashtable_lookup/wide256_109193_20261104/kernel.hpp) |
| fixture | [fixture.cpp](https://github.com/PTO-ISA/SuperNPUBench/blob/main/eval/perf_data/hashtable_lookup/wide256_109193_20261104/fixture.cpp) |
| SwimLane（gzip，Perfetto） | [hash_6144_PE4_wide256.swimlane.json.gz](https://github.com/PTO-ISA/SuperNPUBench/blob/main/eval/perf_data/hashtable_lookup/wide256_109193_20261104/hash_6144_PE4_wide256.swimlane.json.gz) |
| PipeView（gzip，Konata） | [hash_6144_PE4_wide256.pipeview.log.out.gz](https://github.com/PTO-ISA/SuperNPUBench/blob/main/eval/perf_data/hashtable_lookup/wide256_109193_20261104/hash_6144_PE4_wide256.pipeview.log.out.gz) |
| 规范配置 PMU | [pmu.log](https://github.com/PTO-ISA/SuperNPUBench/blob/main/eval/perf_data/hashtable_lookup/wide256_109193_20261104/pmu.log) |
| 运行与校验证据 | [evidence.log](https://github.com/PTO-ISA/SuperNPUBench/blob/main/eval/perf_data/hashtable_lookup/wide256_109193_20261104/evidence.log) / [final-result.json](https://github.com/PTO-ISA/SuperNPUBench/blob/main/eval/perf_data/hashtable_lookup/wide256_109193_20261104/final-result.json) |
| 运行命令 | [command.json](https://github.com/PTO-ISA/SuperNPUBench/blob/main/eval/perf_data/hashtable_lookup/wide256_109193_20261104/command.json) |
| 版本 | [versions.json](https://github.com/PTO-ISA/SuperNPUBench/blob/main/eval/perf_data/hashtable_lookup/wide256_109193_20261104/versions.json) |
| 校验和 | [checksums.json](https://github.com/PTO-ISA/SuperNPUBench/blob/main/eval/perf_data/hashtable_lookup/wide256_109193_20261104/checksums.json) |

swimlane / pipeview 来自追踪配置；`pmu.log` 来自规范配置。压缩只改变存储形式，
解压后分别用 Perfetto / Konata 打开。

本地产物（构建与原始运行目录）：
`/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/outputs/hash256_s4g_par_20261104`。
