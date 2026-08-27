# QLI — Quant Lightning Indexer (Radix-Select TopK)

## 功能 / Overview

QLI 是 SparseFlashAttention (SFA) 的前处理算子，从全量 K/V 序列中选出最关键的
topK 个 token 索引，供后续稀疏注意力使用。

QLI is the pre-processing operator of SparseFlashAttention (SFA). It selects the
topK most important token indices from the full K/V sequence for sparse attention.

本目录的 `qli_pto_opt_simple.hpp`（`qli_topk_radix`）是 **精简版 MSD radix-select
TopK**（推荐），`qli_pto_opt.hpp` 为完整功能版（含 THISTOGRAMX 自定义汇编、
NaN/-0 防御等增强特性），`qli_pto.hpp`（`qli_topk_npu`，TROWARGMAX）为基准版本。

This directory's `qli_pto_opt_simple.hpp` (`qli_topk_radix`) is the **simplified
MSD radix-select TopK** (recommended). `qli_pto_opt.hpp` is the full-featured
version (with THISTOGRAMX custom asm, NaN/-0 handling, etc.).
`qli_pto.hpp` (`qli_topk_npu`) is the TROWARGMAX baseline.

| 特性 / Feature | `qli_pto.hpp` (baseline) | `qli_pto_opt_simple.hpp` (精简版) | `qli_pto_opt.hpp` (完整版) |
|---|---|---|---|
| TopK 算法 | TROWARGMAX 迭代 | THISTOGRAM Byte3→Byte0 逐字节收窄 + tile pop-argmax | 同左 + NaN/-0 防御 + rev 最小索引 |
| 排序位宽 | — | float32 → uint32 sortable key | 同左 + NaN 隔离 |
| 输出契约 | 有序（argmax 顺序） | **TopK set**（无序） | **TopK set**（无序，最小索引优先） |
| 标量参与 | 每轮标量消零 + 重 TLOAD | 提取阶段全 tile 内消零，无重 TLOAD | 同左 |
| 代码量 | ~150 行 | **~235 行**（TopK 部分） | ~376 行（TopK 部分） |

## 计算公式 / Formula

与参考实现 `quant_lightning_indexer_v2` 一致：

```
score[s1, s2] = scale_k[s2] * Σ_g ( W[s1,g] * scale_q[s1,g] * ReLU(QK[g,s1,s2]) )
```

7 个步骤 / 7 steps:

| Step | 计算 / Computation | tile op |
|---|---|---|
| 1 | S = Q @ K^T | TMATMUL |
| 2 | S = ReLU(S) | TMAX |
| 3 | W *= scale_q | TMUL |
| 4 | S *= W*scale_q (broadcast) | TROWEXPANDMUL |
| 5 | out = Σ_g S (ReduceG) | TCVT + TCOLSUM |
| 6 | out *= scale_k | TMUL |
| 7 | indices = TopK_RadixSelect(out) | 多轮 MSD radix-select（见下） |

## Radix-Select TopK 算法 / Multi-Round MSD Radix-Select

对每个 token 的一行 score `[1, Skv]`（内部按 2048 列分 chunk）：

1. **位重解释 / Bit reinterpret**：float → uint32，`reinterpret_cast` 按位加载。
2. **IEEE-754 单调映射 / Monotonic map**：score 可为负（W 可为负），符号位需翻转
   保证「更大 uint32 key ↔ 更大 float 值」单调（与 CANN arch35 `FloatToSortableKey`
   一致）：
   `key = (sign) ? ~bits : (bits | MSB)`，其中 NaN（规范位型 `0x7FC00000`）
   先映射为 `ALL_ONE` 再按符号翻转，最终 `key = 0`（排最底，topK 不选中）。
3. **逐字节 MSD radix / Byte-wise MSD radix**：对 Byte3→Byte0 依次
   `THISTOGRAM`（`THISTOGRAMX`，Idx 前缀 tile 携带已定高位字节）统计 active 集合，
   标量合并多 chunk 直方图后从高桶向下累计找到每个字节的 `kth_byte`，
   得到 `kth_value`（完整 32-bit 阈值）。
4. **提取 / Extract**：`key > kth_value` 全部入选；`key == kth_value` 边界并列
   按最小索引补足（rev 技巧）。逐 chunk 用 tile 内 pop-argmax
   （TROWMAX+TCMP+索引运算+TMUL 消零）输出索引，**全程无标量 store、无重 TLOAD**。

输出契约为 **TopK set**（无序集合），与 golden 的 `TopK set match` 对齐。

## 模板参数 / Template Parameters

### `qli_pto<dtype, Sq, Skv, D, g, kTm, kTk>`（Step1-6）

| 参数 | 含义 / Meaning |
|---|---|
| `dtype` | Q/K 数据类型：`__fp8_e4m3` / `int8_t` |
| `Sq` | Query 序列长度（token 数） |
| `Skv` | Key 序列长度（须为 kTk 的倍数） |
| `D` | Head 维度（固定 128） |
| `g` | Head 数 N（须为 kTm 的倍数） |
| `kTm` | tile M 维度 = g 分块大小 |
| `kTk` | tile K/N 维度 = K 分块大小（如 32） |

### `qli_topk_radix<Sq, Skv, topK>`（Step7）

| 参数 | 含义 / Meaning |
|---|---|
| `Sq` | token 数 |
| `Skv` | 每 token 候选数（scores 列数，须为 8 的倍数） |
| `topK` | 每个 token 选取的索引数（≤ Skv） |

## 数据布局 / Data Layout

| 张量 / Tensor | 形状 / Shape | 类型 / Type | 说明 |
|---|---|---|---|
| Q | [Sq*g, D] | dtype | BSND，同一 token 的 g 个 head 连续 |
| K | [Skv, D] | dtype | 所有 head 共享 |
| W | [Sq, g] | float | weight（可为负） |
| scale_q | [Sq, g] | float | per-token per-head 反量化 scale（≥0） |
| scale_k | [Skv] | float | per-token 反量化 scale（≥0） |
| scores | [Sq, Skv] | float | 输出 score 矩阵 |
| indices | [Sq, topK] | int32 | TopK 索引输出（set，无序） |

## tile 尺寸约束 / Tile Constraints

- 每个 tile 活跃尺寸须在 512B..8KB（v5 TLOAD/TSTORE 编码约束）。
- Step 1-6: `tileQ [kTm,D]`, `tileK [kTk,D]`, `tileS [kTm,kTk]`, `tileSum [1,kTk]`。
- Step 7: 每 chunk `[1,2048]` key tile + `[4,8]` Idx 前缀 tile + `[1,256]` 直方图。
- tail chunk（Skv 非 2048 倍数）：物理列宽 TailPhy≥128（512B 下限），valid 用实际列数。

## 精度与性能 / Precision & Performance (verified)

### 关键 Case（gfrun, set-match 主判据）

| Sq | Skv | topK | chunks | cosine | TopK set | exact* | NaN |
|---|---|---|---|---|---|---|---|
| 64 | 128 | 128 | 1 | 1.000000 | 64/64 | 100% | 0 |
| 4 | 2048 | 512 | 1 | 1.000000 | 4/4 | 100% | 0 |
| 4 | 8192 | 512 | 4 | 1.000000 | 4/4 | 0.05% | 0 |
| 4 | 2080 | 512 | 1+tail | 1.000000 | 4/4 | 23.5% | 0 |

*exact 仅参考：set 契约下输出无序，相同元素位置 swap 属预期。

### 性能（gfsim, Sq=4 Skv=8192 topK=512, kTk=32）

| 指标 | 基线(TROWARGMAX) | radix-select |
|---|---|---|
| gfrun blocks | 84,513 | **137,456** |
| Total cycles | ~1.82M | **657,097** |
| BRob Full Stall | 31.0% | **14.9%** |
| STD block 占比 | 92.8% | **43.4%** |

## 已知限制 / Known Limitations

- 不支持 Causal Mask、变长序列、PageAttention。
- g=kTm（未分块循环），g>kTm 需扩展内层循环。
- 输出为无序 set；若下游需要按 score 降序，需追加最终排序。
- `Skv` 须满足 `Skv%kTk==0`（Step1-6 限制）且 `Skv%8==0`（TopK）。

## Known Issues

本实现过程中发现的仿真器/工具链上层问题（本地修复/规避后验证通过）整理在
[`qli_radix_issues_found.md`](qli_radix_issues_found.md)：
THISTOGRAM ByteId 位域解码、TROWEXPAND 广播 tileInfo、tile 作函数参数编码、
TCMPS UINT32 门禁等 R1–R4。

## 测试 / Test

- 驱动：`test/kernel/qli/src/qli_check_opt.cpp`
- 编译：`make TESTCASE=qli_check_opt QLI_DTYPE=FP8 Sq=64 Skv=128 topk=128`
- golden：`test/kernel/qli/src/gen_qli_golden.py --mode gen|verify`（set-match 主判据）
- 验证流程详见 `test/kernel/qli/README.md`

## 源文件 / Source Files

| 文件 / File | 说明 / Description |
|---|---|
| `qli_pto.hpp` | Step1-6 基准 + TROWARGMAX TopK（baseline） |
| `qli_pto_opt.hpp` | Step1-6 + 完整版 MSD radix-select TopK（含 THISTOGRAMX/NaN/-0/rev） |
| `qli_pto_opt_simple.hpp` | **Step1-6 + 精简版 MSD radix-select TopK（本 README 主题，推荐）** |
| `qli_pto_opt_histogram_radix_design.md` | radix 直方图方案设计与实施记录 |
| `qli_radix_issues_found.md` | 已知问题罗列（R1–R4，可提 ISSUE） |

> 注：历史实验变体（bucket、fused、tail、bf16 等）未随本 demo 提交。