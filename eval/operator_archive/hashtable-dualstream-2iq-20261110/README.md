# hashtable_lookup 双流实现归档（2026-11-10）

**范围**：把一个 `hashtable_lookup` 算子（256-lane、2 槽/轮 linear probing）拆成两个 ELF，用双流
（`feat/dualstream`）同时跑，做 A/B 对照。**不替换库上默认算子**，也不改变 `eval/perf_data`
里的任何归档。

**结论（一句话）**：双流在本例上不再是回归——**每 (stream, PE) 一条 VEC_IQ + 容量守恒**
（8 分区 × 16 深度 = 4 × 32）把共享队列的 144,826 收回 **134,596**（−7.1%），与单流归档
135,299 基本持平（−0.5%）。**派发侧按流仲裁全线为负**，见 `cases/arb-variants/`。

Git 目录只放源码、配置、索引与小型日志；**ELF 与完整 swimlane 在 Release**：
[work-archive-hashtable-dualstream-20261110](https://github.com/PTO-ISA/SuperNPUBench/releases/tag/work-archive-hashtable-dualstream-20261110)（见 `delivery_index.json`）。

| case | Total Cycles | kernel span | 说明 |
|---|---:|---:|---|
| `single-ref-73223` | 135,299 | 73,223 | 归档单流基线（`eval/perf_data/hashtable_lookup/lo32_o3_73223_20261104`，本包只引用） |
| `dual-sharediq-2slot-3072x2` | 144,826 | 79,056 | 双流 + **共享** VEC_IQ（`b04f3ae50` 原状），对照 |
| **`dual-2iq-2slot-3072x2`** | **134,596** | **72,566** | **本次主例**：8 分区 × 16 深度（容量守恒） |
| `arb-variants` | 139,316 / 135,676 / 137,210 / 155,492 | — | 四个被否决的流仲裁方案（负结果） |

所有 case 的数值校验：`linx_test_finisher write addr=0x10009000 val=0x5555 pass`，退出码 0。

## 复现（不需要编译器）

1. 从 Release 取 `hashtable-dualstream-operator-r1.tar.gz` 并解压，得到 `elf/htA.elf`、`elf/htB0.elf`。
2. 准备指定的模型（`LinxISA/SuperScalarModel`，基线 `feat/dualstream@98a44667c` 合入 PR #1177
   `18b24f6ca`，本包结果提交 `45cda61b1`/`79c7ef226`），构建出 `bin/gfsim`
   （本包实测二进制 sha256 `eb3e6a41…`）。
3. 用 `cases/dual-2iq-2slot-3072x2/runs/run-01-134596/command.json` 里的 argv 运行；
   入口会检查 ELF、模型与配置身份，输出必须写到新目录。

```bash
./bin/gfsim --test-finisher 1 -f htA.elf --elf2 htB0.elf --conf dualstream \
  -s cube.enable_internal_acc=true -s bctrl.vec_cell_sched_enable=true \
  -s core.soc_random=false -s core.soc_enable=false \
  -s core.read_bandwidth_limit=128 -s core.write_bandwidth_limit=128 \
  -s core.sharetreg_size_bytes=262144 -s core.sharetreg_ssb_count=64
# 期望：Total Cycles = 134596，finisher 0x5555
```

> `GFSIM_VEC_STREAM_ARB` 默认 0（不仲裁）。仲裁变体的复现命令与计数在
> `cases/arb-variants/results.json`；那些是**负结果**，不建议作为默认配置。

## 源码、构建与身份

- `cases/dual-2iq-2slot-3072x2/source/`：产出被测 ELF 的完整源码（`kernel.hpp`、`fixture.cpp`、
  `link.ld`、`build_dual.sh`），逐文件哈希见 `identity.json`。
- `build/elf_build.log`：用 **PR #146 编译器**（`LinxISA/llvm-project@a5f03b38`，clang 15.0.4，
  target `linx64v5-unknown-linux-musl`）+ **PR #303 API 头**（`LinxISA/Linx-TileOP-API@decac4f`）
  重新编译；重建产物与测量时**逐字节相同**：
  - `htA.elf` = `407250c7…c8fc2`
  - `htB0.elf` = `5571df1d…f0116`
- 两条流各 3072 query（每 PE 768 = 3 batch × 256 lane），表整份嵌在两份 ELF 里；
  参考数据集 `hash_6144_PE4.elf` 的 sha256 = `960ba161…`。

## 假设与适用范围

- 本用例的 key 只比较**低 32 位**、且 **不判空槽终止**（见 `analysis/reference_single_stream_README.md`）；
  这些是归档 fixture 的既有假设，本包未改变。
- 两条流**完全独立**（各查各的 key、各写各的输出、各自校验），没有跨流同步。
- 双流相关结论只对"两条流都在等自己数据"的负载成立；`analysis/DUALSTREAM_EXPERIMENT.md`
  记录了三类不支持的情形（相位锁、共享拆分槽、负载不均）。

## 缺项与已知限制

| 缺项 | 原因 / 影响 |
|---|---|
| `-t 1` 原始周期日志（约 2 GB/次） | 未入 Git 也未入 Release；按规范属于可重建中间物，重跑命令见上 |
| `_nocounters.json` | 未生成；swimlane 原文件在 Release，去掉 `ph:"C"` 即可得到 |
| PipeView 原始日志 | 未采集；本包的性能结论不依赖流水线级证据 |
| 独立复核人 | 待指定 |
| 双流 canonical（`soc_random=true`） | 未跑；单流归档值 280,218 仅作参考 |

## 目录

```
hashtable-dualstream-2iq-20261110/
├── README.md                    # 本文件
├── delivery_index.json          # Release 资产与交付状态
├── operator_identity.json       # 各 case 身份与期望周期
├── SHA256SUMS                   # 本目录文件清单
├── analysis/                    # 双流实验记录、单流分析、归档单流 README
├── configs/                     # dualstream / fourpe 的有效配置快照
└── cases/
    ├── dual-2iq-2slot-3072x2/   # 主例（134,596）
    ├── dual-sharediq-2slot-3072x2/
    ├── single-ref-73223/
    └── arb-variants/            # 负结果汇总
```

关联问题：[LinxISA/SuperScalarModel#1206](https://github.com/LinxISA/SuperScalarModel/issues/1206)；
架构记录：`DUALSTREAM_HANDOFF.md §23`（`ds-pr1177` 分支）。
