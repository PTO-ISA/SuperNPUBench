# SuperNPUBench 算子 gfrun 验证报告

## 仓库版本

| 组件 | 仓库 | 分支 | Commit |
|---|---|---|---|
| gfrun 模型 | https://github.com/LinxISA/SuperScalarModel | `feat/pto-v058-adaptation` | `319294ffdde0304e12c530746b7f63b5ce4083d9` |
| llvm-project | https://github.com/LinxISA/llvm-project | `temp/shared-tload-integration-20260811` | `eb64de8afcbda043aec7e56dae346905dc982039` |
| Linx-TileOP-API | https://github.com/LinxISA/Linx-TileOP-API | `temp/shared-tload-integration-20260811` | `72f8255ca610eae1542dfb633709ce2b18b49955` |

工具链：`llvm-project temp/shared-tload-integration-20260811 @ eb64de8af` + `Linx-TileOP-API temp/shared-tload-integration-20260811 @ 72f8255c`

## 执行方式

- 串行执行（非并行，避免并行负载导致的假阳性）
- 每配置 120s 超时上限
- 平台：macOS arm64，clang

## 统计

| 状态 | 数量 | 占比 |
|---|---|---|
| **通过** | **20** | **69%** |
| **失败** | **9** | **31%** |
| 超时 | 0 | 0% |

## 全量结果

| # | 类别 | 配置 | dtype | 结果 | blocks/insts | 断言位置 |
|---|---|---|---|---|---|---|
| 1 | broadcast | vec_07 | half | **FAIL** | — | AccumulateBlockInfo.cpp:622 |
| 2 | broadcast | vec_019 | half | **FAIL** | — | AccumulateBlockInfo.cpp:64 |
| 3 | element_wise | gelu | bf16 | **FAIL** | — | AccumulateBlockInfo.cpp:64 |
| 4 | gather | gather | fp32 | **FAIL** | — | AccumulateBlockInfo.cpp:64 |
| 5 | fa | 2d_unroll Tm16 Tk16 | float | **PASS** | 9583 / 48350 | — |
| 6 | concat | concat_gather | int32 | **PASS** | 14511 | — |
| 7 | concat | concat_scatter | int32 | **FAIL** | — | TMAEngine.cpp:381 |
| 8 | control | hashtable_lookup | — | **FAIL** | — | AccumulateBlockInfo.cpp:64 |
| 9 | norm | rms_norm | — | **PASS** | 4424 | — |
| 10 | reduction | reducesum_row | float | **PASS** | 269 | — |
| 11 | reduction | reducesum_col | float | **PASS** | 268 | — |
| 12 | reduction | reducesum_col | int32 | **PASS** | 268 | — |
| 13 | reduction | reducesum_col | half | **PASS** | 268 | — |
| 14 | reduction | reducemax_row | float | **PASS** | 269 | — |
| 15 | reduction | reducemax_row | int32 | **PASS** | 269 | — |
| 16 | reduction | reducemax_col | float | **PASS** | 153 | — |
| 17 | reduction | reducemax_col | int32 | **PASS** | 153 | — |
| 18 | sort | topk | — | **FAIL** | — | AaccelssMemoryEngine.cpp:12 |
| 19 | transpose | unroll | half | **PASS** | 2902 | — |
| 20 | deepseek | fused_weight | — | **PASS** | 20 | — |
| 21 | deepseek | rms_norm | — | **FAIL** | — | Memory.cpp:336 |
| 22 | matmul | 256² tK32 | float | **PASS** | 2269 | — |
| 23 | matmul | 256² tK64 | float | **PASS** | 1245 | — |
| 24 | matmul | 512² tK64 | float | **PASS** | 9005 | — |
| 25 | matmul | 256×2048² tK64 | float | **PASS** | 67101 | — |
| 26 | multi_thread | vec/tadd | — | **PASS** | 269 | — |
| 27 | multi_thread | vec/trowsum | — | **PASS** | 267 | — |
| 28 | flashMLA | Sq64 Dk512 | — | **PASS** | 5174 | — |

> 注：#28 multi_thread/matmul（`kernel_multi_thread_matmul_B1_M256_N256_K256_tM32_tN32_tK32.elf`）未在 `kernel_elf_list.md` 中列出，单独执行结果为 **FAIL**（AccumulateBlockInfo.cpp:711，implicit-ACC CUBE 目标编码），不计入上表 28 项。

## 失败分类（9 个）

| 失败类型 | 数量 | 断言 | 影响算子 |
|---|---|---|---|
| TSTORE source 契约不匹配 | 4 | `ValidateLocalTlsu` AccumulateBlockInfo.cpp:64 | gelu, gather, control, broadcast vec_019 |
| Text-store 被拒绝 | 2 | `AssertNotTextStore` AaccelssMemoryEngine.cpp:12 / Memory.cpp:336 | topk, deepseek rms_norm |
| COPY expansion 源契约 | 1 | `ValidateReduceAndExpandTepl` AccumulateBlockInfo.cpp:622 | broadcast vec_07 |
| MSCATTER index dtype | 1 | `ExecuteMSCATTER` TMAEngine.cpp:381 | concat_scatter |
| implicit-ACC CUBE 目标 | 1 | `AccumulateBlockInfo` AccumulateBlockInfo.cpp:711 | multi_thread matmul（表外） |

## 失败断言详情

### 1. TSTORE source 契约不匹配（4 个）

```
ASSERTION FAILED: inst->srcs.size() == 2 && inst->dsts.empty() &&
IsCompatibleDataTile(inst->srcs[1], ...) &&
"Local TSTORE requires one compatible source Tile"
```
位置：`emulator/engine/AccumulateBlockInfo.cpp:64`，`ValidateLocalTlsu`

影响：gelu(bf16)、gather(fp32)、control、broadcast vec_019

### 2. Text-store 被拒绝（2 个）

```
// topk:
ASSERTION FAILED: false
位置: AssertNotTextStore, AaccelssMemoryEngine.cpp:12

// deepseek rms_norm:
ASSERTION FAILED: gAllowTextStore || !is_text_region(address, width)
位置: Store, Memory.cpp:336
```

影响：topk、deepseek rms_norm

### 3. COPY expansion 源契约（1 个）

```
ASSERTION FAILED: inst->srcs.size() == 3 && inst->dsts.size() == 1 &&
inst->srcs[1] && inst->srcs[2] &&
IsCompatibleDataTile(inst->srcs[1], ...) &&
"PTO v0.58 COPY expansion requires source0 and broadcast-source"
```
位置：`emulator/engine/AccumulateBlockInfo.cpp:622`，`ValidateReduceAndExpandTepl`

影响：broadcast vec_07

### 4. MSCATTER index dtype（1 个）

```
ASSERTION FAILED: block->srcTile.size() >= 2u &&
block->srcTile[1]->tileInfo &&
(block->srcTile[1]->tileInfo->dataType == DataType::INT32 ||
 block->srcTile[1]->tileInfo->dataType == DataType::UINT32) &&
"MSCATTER index must be S32/U32 (MSCATTER.md dtypes)"
```
位置：`emulator/engine/TMAEngine.cpp:381`，`ExecuteMSCATTER`

影响：concat_scatter

### 5. implicit-ACC CUBE 目标（1 个）

```
ASSERTION FAILED: (inst->dsts.empty() || sharedLocalToShared) &&
"implicit-ACC CUBE operations must not encode an ordinary destination"
```
位置：`emulator/engine/AccumulateBlockInfo.cpp:711`

影响：multi_thread matmul

