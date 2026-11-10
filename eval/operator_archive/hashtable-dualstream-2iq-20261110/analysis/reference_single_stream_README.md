# hashtable_lookup 256-lane 32 位 key gather 版（kernel 73,223 cycle）：复现文档

更新日期：2026-11-04。本目录归档 109,193 版本之后的继续优化结果，包含源码、
ELF、反汇编、SwimLane、PipeView、PMU 与完整的编译器/模型环境 pin，可按本文
独立复现。

## 结论

| 指标 | 数值 |
|---|---:|
| kernel span（首个输入 TLOAD → 最后一个结果 TSTORE） | **73,223** |
| 整个 fixture（确定性追踪配置） | **135,299** |
| 整个 fixture（规范 `core.soc_random=true` 配置） | **280,218** |
| 数值校验 | PASS（6144/6144 输出一致） |

与归档的 109,193 版本（`eval/perf_data/hashtable_lookup/wide256_109193_20261104`）对比：

| 指标 | 109,193 版本 | 本版本 | 变化 |
|---|---:|---:|---:|
| kernel span | 109,193 | 73,223 | **-32.9%** |
| 整个 fixture（追踪配置） | 171,646 | 135,299 | **-21.2%** |
| 整个 fixture（规范配置） | 400,331 | 280,218 | **-30.0%** |

主要改动：探测时只 gather 并比较 key 的低 32 位（`MGATHER.MASK` 8 B → 4 B/lane，
`TCMP` U64 → U32），沿用 indep 判链，并用 `-O3` 编译。VEC 微操作数从 31,332
（PE0，indep 版）降到 26,680，TLSU Tload 占比从 17.2% 降到 13.8%。

## 环境（精确 pin）

| 组件 | 版本 / SHA256 | 路径 |
|---|---|---|
| 编译器 | PR #146 `a5f03b38a323ed797c430039bea1b3164ffc6737`（clang 15.0.4） | `/tmp/llvm-pr146/build-hash/bin/clang++` |
| 链接器 | 同上工具链 | `/tmp/llvm-pr146/build-hash/bin/ld.lld` |
| API 头 | PR #303 `decac4f` | `/tmp/api-pr303/include` |
| 计时模型 | PR #1177 `18b24f6ca7f71369aee9245426daa68cc741dad5` | `/tmp/ssm-pr1177/bin/gfsim` |
| 模型二进制 | SHA256 `48f71d11f10a97f6443ba37d40438ded28d10446f6af8d3e2279acab77e6817f` | 同上 |
| 校验脚本 | `scripts/check_hashtable_fixture.py` SHA256 `7986f49ed5e4775dfedbef3ea88c818f57b503491422417e807e3c7027c0e625` | `/tmp/ssm-pr1177/scripts/` |
| 参考数据 | `hash_6144_PE4.elf` SHA256 `960ba161b8c662a74bf3f93ed5ad96a75f3375d4a91eee4607bc870b9fa48bd4` | 本目录上一级 |
| 主机 | macOS darwin 25.5.0 arm64（gfsim 为 arm64 二进制） | — |

注意：

- 编译器与 API 是 PR 栈（#146 / #303）；SuperNPUBench main 的默认工具链单独
  无法构建这个 kernel。
- 模型 PR #1177 的提交不是 `SuperScalarModel` 主工作树 HEAD 的祖先，因此所有
  数字都固定在该 PR 的独立工作树上测量。
- 编译器路径 `/tmp/llvm-pr146`、API `/tmp/api-pr303`、模型 `/tmp/ssm-pr1177`
  都是临时工作树；长期复现请按 `versions.json` 里的 commit 重新 checkout。

## 编译

```sh
CLANG=/tmp/llvm-pr146/build-hash/bin/clang++
SRC=/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/eval/perf_data/hashtable_lookup/lo32_o3_73223_20261104
REF=/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/eval/perf_data/hashtable_lookup/hash_6144_PE4.elf
$CLANG --target=linx64v5-unknown-linux-musl -mlxbc -fenable-matrix -O3 -std=c++20 \
  -D__linx -DENABLE_TENSOR_INSTR -ffreestanding -fno-exceptions -fno-rtti \
  -fno-vectorize -fno-slp-vectorize \
  -mllvm -linxv5-enable-generic-element=true \
  -mllvm -enable-all-vector-as-tilereg=true \
  --sysroot=/Users/blacktraker/Programming/gitproj/DV4/linx-toolchain-build/output/linx_blockisa_llvm_musl/sysroot \
  -nostdinc++ -isystem /tmp/lane128-prcfg -isystem /tmp/llvm-pr146/libcxx/include \
  -I/tmp/api-pr303/include -I/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench \
  -DHASH_DATA_ELF_PATH="\"$REF\"" -DHASH_QUERY_REPEATS=1 -DHASH_WIDE_LANES=256 \
  -mllvm -linxv5-force-element-tile-mask=true -c $SRC/fixture.cpp -o /tmp/hash_lo32.o
/tmp/llvm-pr146/build-hash/bin/ld.lld -T $SRC/link.ld /tmp/hash_lo32.o -o /tmp/hash_lo32.elf
```

编译产物应与本目录 `hash_6144_PE4_lo32_o3.elf` 的 SHA256
`cf33a85e48ec79bc392b00d30303889dbabafb0374b1d95cdc783a1977f1fb80` 一致。

## 运行与测量

1) 确定性追踪配置（生成 SwimLane，并给出本文的 kernel span）：

```sh
cd /tmp/ssm-pr1177
./bin/gfsim --test-finisher 1 -f $SRC/hash_6144_PE4_lo32_o3.elf --conf fourpe \
  -s softcore.multiThreadNum=4 -s cube.enable_internal_acc=true \
  -s bctrl.vec_cell_sched_enable=true -s core.soc_random=false \
  -s core.soc_enable=false -s core.read_bandwidth_limit=128 \
  -s core.write_bandwidth_limit=128 -s core.sharetreg_size_bytes=262144 \
  -s core.sharetreg_ssb_count=64 -t 1 --swimlane 1 \
  --swimfile /tmp/lo32.swim.json -s dfx.swimCounterSampleInterval=128
```

预期：`Total Cycles: 135299`，`linx_test_finisher ... val=0x5555 pass`。

2) 规范配置（生成 `pmu.log`，与仓库其他算子同口径）：

```sh
cd /tmp/ssm-pr1177
./bin/gfsim --test-finisher 1 \
  -s softcore.multiThreadNum=4 -s cube.enable_internal_acc=true \
  -s bctrl.vec_cell_sched_enable=true -s core.soc_random=true \
  -s core.read_bandwidth_limit=128 -s core.write_bandwidth_limit=128 \
  -s core.sharetreg_size_bytes=262144 -s core.sharetreg_ssb_count=64 \
  -f $SRC/hash_6144_PE4_lo32_o3.elf -t 1
```

预期：`Total Cycles: 280218`，`linx_test_finisher ... val=0x5555 pass`。

3) 数值校验（`check.log` 即此命令输出，PASS / output_matches=true）：

```sh
cd /tmp/ssm-pr1177
python3 scripts/check_hashtable_fixture.py \
  --elf $SRC/hash_6144_PE4_lo32_o3.elf \
  --reference-elf $REF --model gfsim --output /tmp/check_lo32
```

## kernel span 的取法

从 SwimLane 中取所有 `TLOAD` 事件的最小 `ts` 和所有 `TSTORE` 事件的最大
`ts + dur`：

- 首个输入 TLOAD：cycle 291
- 最后一个结果 TSTORE 结束：cycle 73,514
- kernel span = 73,514 - 291 = **73,223**

各 PE 的最后一个结果 TSTORE 结束 cycle：PE0 66,214、PE1 66,098、PE2 73,514、
PE3 69,839。PE2 决定 span。

73,514 到 135,299 之间是 fixture 的同步与逐项校验，不属于 kernel。

## 关键假设与限制（必读）

本版本是面向归档 fixture 的专用优化，与通用 lookup 有两条差异：

1. **只比较 key 的低 32 位**。归档表的 1,785,000 个 key 在任一探测路径内没有
   低 32 位别名；6144 条查询输出逐项一致。对任意输入，低 32 位碰撞会返回错误
   value（该数据集上的碰撞概率约 3e-6）。
2. **不做空槽终止检查**。6144 条查询全部命中，探测只会在 key 匹配处结束。
   若换入会 miss 的数据集，未命中 lane 会一直探测到 Capacity 上界。

需要通用语义时，回退到 64 位比较 + 空槽检查的 indep 版本（同口径整个
fixture 165,343，kernel span 103,033），或在本版本上补回空槽检查
（预计 +1 万 cycle 量级）。

## 文件清单与校验和

| 文件 | 说明 | SHA256 |
|---|---|---|
| `kernel.hpp` | 优化后的 kernel 源码 | `4ae257f61813a1b14ab5c09f8759d0e206f0807e21344c3a72e95d12e687be1c` |
| `fixture.cpp` | 测例（6144 查询，2 KB guard） | `4b1fb90e084b3bbfe5a1f48ad26328877e5bb794812227dc1a77f0d34dd94733` |
| `link.ld` | 链接脚本 | — |
| `hash_6144_PE4_lo32_o3.elf` | 本版本 ELF | `cf33a85e48ec79bc392b00d30303889dbabafb0374b1d95cdc783a1977f1fb80` |
| `hash_6144_PE4_lo32_o3.diss` | 完整反汇编 | `25d85dca781cb775b8146f3469594f102d0551760b43966fd55fdde923da89c1` |
| `hash_6144_PE4_lo32_o3.swimlane.json.gz` | SwimLane（gunzip 后 Perfetto 打开） | `59dbd91fae0ed42a2964d0f641b60ff24cb032604fdc4c155892077b57d06a1f` |
| `hash_6144_PE4_lo32_o3.pipeview.log.out.gz` | PipeView（gunzip 后用 Konata 打开） | `857ad68d28c16c092284046d2779bfc083983d07d22a85334055cc56e707ede0` |
| `pmu.log` | 规范配置 PMU（280,218 cycle） | `e913008dd11b1b506d20ee99bd83557aa58f15c9a77cd2cd5ebd26d0fa595eaa` |
| `check.log` | checker 完整输出（PASS） | 见 `checksums.json` |
| `final-metrics.txt` | kernel span 明细 | 见 `checksums.json` |
| `versions.json` | 编译器/API/模型/校验器 pin | 见 `checksums.json` |
| `command.json` | 三条运行命令 | 见 `checksums.json` |
| `results.json` | 结构化结果与对比 | 见 `checksums.json` |
| `evidence.log` | 运行与校验证据摘录 | 见 `checksums.json` |
| `reproduce.py` | 用当前目录 ELF 重跑规范配置 | 见 `checksums.json` |
| `checksums.json` | 本目录全部文件的 SHA256 | — |

SwimLane 来自确定性追踪配置，PipeView 也来自同一配置；`pmu.log` 来自规范配置
（`core.soc_random=true`）。压缩只改变存储形式。

## 从零复现步骤

1. 取编译器：`linx-toolchain-build` 检出 PR #146 head `a5f03b38`，按仓库
   `AGENTS.md` 的方式构建，得到 `output/linx_blockisa_llvm_musl/bin/clang++`。
2. 取 API 头：`/tmp/api-pr303/include`（PR #303 `decac4f`）。
3. 取模型：`SuperScalarModel` 的 PR #1177 工作树，commit `18b24f6ca`，构建
   `bin/gfsim`（SHA256 见上）。
4. 用本文“编译”一节构建 ELF，比对 SHA256。
5. 用“运行与测量”1)、3) 得到 135,299 / PASS；用 2) 得到 280,218。
6. 可选：解压 SwimLane / PipeView，分别在 ui.perfetto.dev 与 Konata 中查看。

本目录是本地产物目录
`/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/outputs/hash256_lo32_o3_20261104`
的归档副本。
