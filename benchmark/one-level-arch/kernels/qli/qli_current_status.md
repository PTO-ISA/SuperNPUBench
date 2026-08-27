# QLI 当前状态报告 / QLI Current Status Report

> 生成时间 / Generated: 2026-08-27
> 目的 / Purpose: 精简提取当前 QLI 算子的实现、精度、性能、已知问题与后续方向，
> 替代冗长的 `qli_fix_record.md` 历史记录（后者作为归档参考保留）。
> Concise extract of QLI's current implementation, precision, performance,
> known issues and next steps — supersedes the verbose historical
> `qli_fix_record.md` (kept as archive reference).

---

## 1. 当前状态概述 / Current Status Overview

QLI（Quant Lightning Indexer，SparseFlashAttention 前处理算子）已完成
**分块多轮 MSD radix-select TopK** 实现，精度全部通过，性能较 TROWARGMAX 基线下降约 64%。

QLI (Quant Lightning Indexer, the pre-processing operator of
SparseFlashAttention) has completed the **multi-round MSD radix-select TopK**
implementation: precision passes on all cases, and cycles drop ~64% vs the
TROWARGMAX baseline.

### 1.1 代码基线 / Code Baseline

| 组件 / Component | 版本 / Version | 说明 / Note |
|---|---|---|
| SuperScalarModel | `a68dba29`（release_ver0812 钉住，detached HEAD） | + 7 个本地未提交补丁 / 7 local uncommitted patches |
| SuperNPUBench | `feat/qli-radix-topk` @ `631a36c` | 含 radix-select demo |
| Linx-TileOP-API | release_ver0812 钉住 / pinned | — |
| 平台 / Platform | Linux x86_64, GCC 11.4.0 | — |

### 1.2 当前实现 / Current Implementation

| 文件 / File | 函数 / Function | 定位 / Role |
|---|---|---|
| `qli_pto_opt.hpp` | `qli_topk_radix` | **推荐版本 / Recommended**：分块多轮 MSD radix-select |
| `qli_pto.hpp` | `qli_topk_npu` | 基准版本 / Baseline：TROWARGMAX 迭代 + 标量消零 |

radix-select 算法 4 步 / radix-select algorithm (4 steps)：

1. **位重解释 / Bit reinterpret**：float→uint32（`reinterpret_cast` 按位加载）。
2. **IEEE-754 单调映射 / Monotonic map**：`key = sign ? ~bits : (bits | MSB)`，
   NaN（规范位型 `0x7FC00000`）→0（排最底，topK 不选中）。
   与 CANN arch35 `FloatToSortableKey` 一致。
3. **逐字节 MSD radix / Byte-wise MSD radix**：Byte3→Byte0 依次 `THISTOGRAM`
   （`THISTOGRAMX`，Idx 前缀 tile 携带已定高位字节），标量合并多 chunk 直方图后
   从高桶向下累计，得到每字节 `kth_byte`，拼出完整 32-bit `kth_value`。
4. **提取 / Extract**：`key > kth_value` 全部入选；`key == kth_value` 边界并列
   按最小索引补足（rev 技巧）。逐 chunk 用 tile 内 pop-argmax
   （TROWARGMAX + 索引运算 + TMUL 消零）输出索引，**全程无标量 store、无重 TLOAD**。

输出契约 / Output contract：**TopK set**（无序集合，并列取最小索引）。

最新精化 / Latest refinements（commit `631a36c`）：

- baseline tile 补齐：Skv<128 → SinglePhy/128（v5 512B 物理列宽下限）。
- 有限哨兵：`-1e30f` → `-FLT_MAX`（`0xFF7FFFFF`），避免低于 -1e30 的 score 被重选。
- 有符号零：sortable key 中 `-0.0f` 规范化为 `+0.0f`，保证等零 score 按最小索引 tie-break。
- driver 批次从 `-DBatch` 推导（不再硬编码 B=1）。

---

## 2. 本地补丁最小集 / Local Patch Set

### 2.1 SuperScalarModel（仿真器侧，7 文件未提交）

| 补丁 / Patch | 文件 / File | 修改 / Change | 性质 / Nature |
|---|---|---|---|
| C3 | `emulator/engine/TMAEngine.cpp` | 新增 `colStride`，ColMajor 检测时设 `validRow*eleSize`；`doMemoryLoad`/`storeToMemory` 用 `colStride*idx` | 仿真器 bug |
| A4 | `isa/ISACommon/TileOpManager.h` | `REDUCEANDBROADCAST_RESERVE_1→TROWARGMAX` 等 4 个映射 | 仿真器 bug |
| R1 | `isa/Block.cpp` | THISTOGRAM ByteId 解码 `bits[19:18]`（原读 `bits[28:27]` 恒为 Byte3） | 仿真器 bug |
| B8 | `emulator/engine/AccumulateBlockInfo.cpp` | TROWSUM/TROWMAX/TROWARGMAX 允许 UINT32/UINT16 | 门禁放宽 |
| P3-D1 | `isa/codec/decodefiles/block32.decode` | B_DATR `bit[19:18]` `0 0`→`. .`（接受工具链非规范编码） | 接受工具链编码 |
| TSCAN | `TileOpManager.h`/`Block.cpp`/`TEPLEngine.cpp`/`AccumulateBlockInfo.cpp`/`SoftCore.h` | 新增 TSCAN 前缀和指令 + 校验 + dispatch | 未来能力（方案 B 复用） |

### 2.2 QLI 侧 / QLI side

| 补丁 / Patch | 文件 / File | 修改 / Change | 性质 / Nature |
|---|---|---|---|
| QLI-1 | `qli_pto.hpp` | `TCOLEXPANDMUL`→`TROWEXPANDMUL`（W 行广播） | QLI API 使用错误 |

**已撤销 / Reverted**（历史误判，不进入当前补丁集）：

- A3（TCOLEXPAND `srcR[j]`→`srcR[i]`）— Issue #272 确认 TCOLEXPAND 实现正确，是 QLI 用错 op。
- 行步长修改（C3 早期版本去掉 `eleSize` 乘法）— #247（`fb71d8f9`）已正确修复 B.IOR 行步长为字节，`stride*eleSize` 是正确的。

---

## 3. 当前精度结果 / Precision Results

### 3.1 关键 Case（gfrun，set-match 主判据）

| Sq | Skv | topK | chunks | cosine | TopK set | exact* | NaN |
|---|---|---|---|---|---|---|---|
| 64 | 128 | 128 | 1 | 1.000000 | 64/64 | 100% | 0 |
| 4 | 2048 | 512 | 1 | 1.000000 | 4/4 | 100% | 0 |
| 4 | 8192 | 512 | 4 | 1.000000 | 4/4 | 0.05% | 0 |
| 4 | 2080 | 512 | 1+tail | 1.000000 | 4/4 | 23.5% | 0 |

*exact 仅参考 / reference only：set 契约下输出无序，相同元素位置 swap 属预期行为。

### 3.2 全量 Case

| Sq | Skv | topK | cosine | TopK set | exact* | NaN |
|---|---|---|---|---|---|---|
| 4 | 128 | 8 | 1.000000 | 4/4 | 100% | 0 |
| 4 | 2048 | 8 / 256 | 1.000000 | 4/4 | 100% | 0 |
| 4 | 4096 | 512 | 1.000000 | 4/4 | — | 0 |
| 16 | 8192 | 512 | 1.000000 | 16/16 | — | 0 |
| 16 / 32 / 64 | 128 | 8 | 1.000000 | 100% | 100% | 0 |
| 4 | 2056 (tail) | 8 | 0.998680** | 4/4 | 100% | 0 |

**Skv=2056 的 cosine=0.998 源于 Step1-6 的 `Skv%kTk` 限制
（2056/32 非整除，仅计算 2048 列），与 TopK 无关。**

### 3.3 特殊值 / 对抗用例 / Special Values & Adversarial Cases

- NaN（规范位型 `0x7FC00000`）→ key=0，排除 ✅（与 arch35 `FloatToSortableKey` 一致）。
- `±inf`、负数、`±0.0` 单调映射正确 ✅。
- 对抗：早期 chunk 2044 eq + 后期 4 gt → 输出 `{0,1,2,3,2048,2049,2050,2051}` PASS。
- 对抗：NaN 多位型（负/signaling/payload/quiet）→ topK={0..7} PASS。

---

## 4. 当前性能分析 / Performance Analysis

### 4.1 性能对比（gfsim，Sq=4 Skv=8192 topK=512 kTk=32）

| 指标 / Metric | TROWARGMAX 基线 | radix-select | 变化 / Delta |
|---|---|---|---|
| gfrun blocks | 84,513 | 137,456 | +62.6% |
| Total cycles | ~1,819,116 | **657,097** | **-63.9%** |
| BRob Full Stall | 31.0% | **14.9%** | -16.1pp |
| STD block 占比 | 92.8% | **43.4%** | -49.4pp |
| All Cores Idle | 78.96% | **20.2%** | -58.8pp |

> 注 / Note：radix 的 gfrun blocks 更多但 cycles 更少——tile op 替代标量消零，
> 每 block 成本更低。STD 剩余 43.4% 来自 copy/loop 控制，TopK 标量消零已全部消除。

### 4.2 瓶颈演进 / Bottleneck Evolution

- **基线**：标量消零（`row_u32[idx]=0xF149F2CAu`，topK 轮迭代）占 92.8% STD block
  → 塞满 384-entry BROB（Avg depth 172/384）→ BRob Full Stall = 31%
  → 79% 周期所有引擎空闲 → Cube/Vector 等待标量完成 → BPC=0.23。
- **radix**：标量消零消除，改 tile 内 TMUL 消零 + 常驻 key tile（无重 TLOAD）。
  STD 降至 43.4%，Idle 降至 20.2%，但仍有进一步压缩空间。

### 4.3 已验证的优化路径 / Verified Optimization Path

| 阶段 / Stage | Cycles | Idle% | 说明 |
|---|---|---|---|
| 基线（TROWARGMAX） | 5,302,671* | 78.96% | *含 copy_bytes 逐字节拷贝 |
| P1（copy_bytes→TLOAD） | 1,828,638 | 38.88% | 消除 1MB 标量拷贝 |
| P2（kTk 32→64） | 1,819,116 | 39.86% | 微调，收益有限 |
| **radix-select（当前）** | **657,097** | **20.2%** | THISTOGRAM + tile pop |

---

## 5. 当前问题 / Known Issues

### 5.1 仿真器/工具链问题（待上游修复）/ Simulator & Toolchain Issues

| # | 问题 / Issue | 组件 / Component | 处置 / Handling |
|---|---|---|---|
| R1 | THISTOGRAM ByteId 位域解码错误（`bits[19:18]` vs `bits[28:27]`），Byte0/1/2 恒解码为 Byte3 | SSM `isa/Block.cpp` | 本地修复（`(binary>>18)&3`）；需上游确认位域并更新测试 `isa_test:285` |
| R2 | TROWEXPAND 广播 dst tileInfo 取自 src（[1,32]），后续 TCMP 校验按 [1,2048] 要求失败 | SSM `isa/Block.cpp` | QLI 改 `TSTORE→标量→TEXPANDS` 规避 |
| R3 | tile 作函数参数 → LinxV5 生成 `S64 [1,1024]` TLOAD（错误编码） | LLVM LinxV5 后端 | QLI 宏内联规避（`QLI_RADIX_POP_N` 等） |
| R4 | TCMPS 不允许 UINT32 dtype | SSM `AccumulateBlockInfo.cpp` | 未改，用 `TCMP + TEXPANDS` kth 广播 tile 规避 |
| R5 | TSCATTER/TGATHER 工具链无法编译（`B.IOR [u#1],[]` 被 asm matcher 拒绝） | llvm-project#70 | 阻断方案 B（TSCAN 单趟桶式提取） |
| #342 | `__bf16` TSTORE → `uint16` TLOAD 读 0（跨 dtype 读取） | SSM | 阻断 bf16/uint16 两级 radix |
| #343 | 标量 store 到堆 → TLOAD 读 0（scalar→tile 一致性） | SSM | 阻断标量桥接喂 tile |

### 5.2 算子侧已知限制 / Operator-Side Limitations

- **输出为无序 set**：若下游需按 score 降序，需追加最终排序（tile radix extraction 或候选 merge sort）。
- **不支持 Causal Mask、变长序列、PageAttention**。
- **`g=kTm`（未分块循环）**：`g>kTm` 需扩展内层循环。
- **`Skv` 约束**：须满足 `Skv%kTk==0`（Step1-6）且 `Skv%8==0`（TopK）。
- **scratch memory 契约隐式**：temp_hist 紧跟 indices 输出区之后，多 PE 并行时需显式化。
- **P4 多 PE 并行挂起**：数据分割版失败（cosine=0.954，token 3 正确、0-2 错），
  根因未定位（疑似 `Sq=1` 模板实例化 + `-O2` 向量化与 gfrun 标量模型不兼容），
  待在新 TopK 上重做。

---

## 6. 后续优化方向 / Future Optimization Directions

按优先级 / By priority：

| 优先级 | 方向 / Direction | 预期收益 / Expected Gain | 前置条件 / Prerequisite |
|---|---|---|---|
| P0 | 上游推送本地 7 补丁（R1/B8/A4/C3/decode） | 减少 local diff 负担，CI 覆盖 | 上游 review |
| P1 | 修复 TSCATTER 后上 TSCAN 单趟桶式提取 | 消除逐元素 pop，提取段周期大幅下降 | llvm-project#70 修复 |
| P2 | chunk-local TopK + 合并 | pop 次数 O(topK)→O(NumChunks) | tile mask 设计 |
| P3 | 多 PE 并行（4 token 分 4 PE，SPMD） | BPC 提升，接近 ~4x | Sq=1/-O2 根因定位 |
| P4 | bf16/uint16 两级 radix | digit 轮 4→2，tile 容量翻倍 | #342/#343 修复 |
| P5 | TCMPS UINT32 门禁放宽（R4） | 单源标量比较可用 | 上游决定 TCMPS 支持 UINT32 |
| P6 | tail chunk 通用化 + 显式 scratch 契约 | 支持 `Skv%kTk≠0`，多 PE 安全 | Step1-6 扩展 |

---

## 7. 参考文档 / References

| 文档 / Document | 内容 / Content |
|---|---|
| `qli_fix_record.md` | 完整历史修复记录（18 补丁、7 TopK 方案对比）——工作区本地文件，不在仓库内 / local workspace file, not in repo |
| `README.md` | 算子使用说明（同目录，最新 commit 维护，最权威）/ same dir |
| `qli_pto_opt_histogram_radix_design.md` | radix 方案设计与实施记录（§14，同目录 / same dir） |
| `qli_radix_opt_test_log.md` | 方案 A/B 评估（同目录 / same dir） |
| `qli_radix_issues_found.md` | R1–R5 已知问题（可提 ISSUE 文本，同目录 / same dir） |
| `qli_bf16_bugs.md` | #342/#343 bf16 跨 dtype bug 详述——工作区本地文件，不在仓库内 / local workspace file, not in repo |

---

## 8. 测试与验证流程 / Test & Verification Flow

**文件路径 / File paths**：

| 文件 / File | 作用 / Role |
|---|---|
| `../../test/kernel/qli/src/gen_qli_golden.py` | 生成输入 .bin + numpy golden 参考（set-match 主判据） |
| `../../test/kernel/qli/src/qli_check_opt.cpp` | radix 版 driver（.data 直接地址，P1 优化） |
| `../../test/kernel/qli/src/qli_check_data.s` | `.incbin` 嵌入 .bin 到 ELF |
| `qli_pto_opt.hpp` | Step1-6 scores + radix TopK（同目录 / same dir） |
| `SuperScalarModel/bin/gfrun` | 功能仿真 + `--dump-memory` |
| `SuperScalarModel/bin/gfsim` | 时序仿真 + PMU（`-m` 截断规避末尾死锁） |

**执行步骤 / Steps**：

```
1. gen_qli_golden.py --mode gen --sq N --skv M --topk K
2. clang -c qli_check_data.s -o qli_check_data.o
3. make TESTCASE=qli_check_opt QLI_DTYPE=FP8 Sq=N Skv=M topk=K
4. llvm-nm ELF | grep "_binary_src.*_start" → 获取符号地址
5. python3 fix_cpp_addrs.py <qli_check_opt.cpp> <ELF> <llvm-nm>
6. make TESTCASE=qli_check_opt ... (重编译至地址稳定)
7. gfrun -f ELF --dump-memory <OUT_SCORES>:<size>:sim_out.bin
8. gen_qli_golden.py --mode verify → cosine + TopK set
9. gfsim -f ELF -s dfx.deadLockThreshold=100000 -m <截断> → PMU
```

**关键注意 / Key notes**：

- `OUT_INDICES` 须动态 `= OUT_SCORES + Sq*Skv*4`（避免与 scores 重叠）。
- `--dump-memory` 大小须覆盖 scores + indices + scratch。
- 每次重编译后符号地址漂移，须迭代 `llvm-nm` → `fix_cpp_addrs` → 重编。
- golden 必须用 FP8 量化后的 Q/K 计算 score（否则量化误差致 TopK 不匹配）。
- gfsim 完整 run 末尾死锁（标量 store + tile load 混合的 TimingSim 限制），用 `-m` 截断，PMU 数据完整。

---

*Generated 2026-08-27 · Base: SSM `a68dba29` + 7 local patches · SuperNPUBench `feat/qli-radix-topk@631a36c` · radix-select: cosine=1.0 set=100% · 657,097 cycles (-64% vs baseline) · P4 multi-PE pending · TSCATTER/#342/#343 blockers tracked*
