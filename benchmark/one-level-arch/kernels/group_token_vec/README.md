# Group Token Old (PTO one-level-arch)

## 功能

MoE Token 分组算子（"old" 变体，含 SIMD 排序），是 MoE Dispatch 的完整三阶段实现。
逻辑参考 `cann-samples/Samples/2_Performance/group_token_story/src/group_token_old_main.asc`。

三个阶段：

1. **CalTokenPerExpertCnt** — 统计每个 expert 收到的 token 数量（直方图）
2. **GroupToken** — 把 token 按"本地 expert"scatter 分组（SIMT 原子写指针）
3. **SortKernel** — FloorFunc 计算 per-token minLocalExpId，再按 minLocalExpId 排序，
   生成连续分区的 token id 列表（SIMD counting sort）

数据流：
```
topkIndex [bs, k]
  ├──→ CalTokenPerExpertCnt  →  tokenPerExpertCnt [expertNum]
  ├──→ GroupToken            →  groupedTokenIds [expertPerRank, bs]
  │                             expertSectionTokenCnt [expertPerRank]
  └──→ FloorFunc + Sort      →  sortedTokenIds [bs]  (连续分区)
                                 sectionStarts [expertPerRank+1]
```

## 与 group_token 的区别

| 维度 | group_token | group_token_old |
|------|-------------|-----------------|
| 阶段数 | 2（直方图 + scatter） | 3（直方图 + scatter + 排序） |
| Phase 3 | 无 | FloorFunc + counting sort |
| 输出格式 | 分区散布（`[expert][bs]`） | 连续排序（`[bs]` 紧凑排列） |
| 对应参考 | `group_token.asc` | `group_token_old_main.asc` |

## 输入输出

| 参数 | 类型 | 说明 |
|------|------|------|
| `topkIndex` | `uint32_t*` | 输入 [bs × k]，每个元素是全局 expert id |
| `tokenPerExpertCnt` | `uint32_t*` | 输出 [expertNum]，每个 expert 的 token 计数 |
| `groupedTokenIds` | `uint32_t*` | 输出 [expertPerRank × bs]，scatter 分组结果 |
| `expertSectionTokenCnt` | `uint32_t*` | 输出 [expertPerRank]，各分区 token 数 |
| `sortedTokenIds` | `uint32_t*` | 输出 [bs]，排序后的连续 token id |
| `sectionStarts` | `uint32_t*` | 输出 [expertPerRank+1]，各分区起始偏移 |

## MoE 拓扑参数

| 参数 | 值 | 说明 |
|------|------|------|
| `kBS` | 512 | batch size（token 总数） |
| `kTopK` | 16 | 每个 token 选中的 top-k expert 数 |
| `kExpertPerRank` | 4 | 每个 rank 的本地 expert 数 |
| `kExpertNum` | 128 | 全局 expert 总数 |

## 实现方式

### Phase 1: CalTokenPerExpertCnt（直方图）
标量路径遍历 topkIndex 计数；SIMT 路径每 lane 负责一个 expert。

### Phase 2: GroupToken（scatter）
对每个 token 取 `min(expertId % expertPerRank)`，用写指针 scatter 到分区。

### Phase 3: SortKernel（排序）
- **FloorFunc**：计算每个 token 的 minLocalExpId
- **SortByLocalExpId**：counting sort 按 minLocalExpId 稳定排序，
  生成连续分区的 token id 数组 + 分区边界

## 源文件

| 文件 | 说明 |
|------|------|
| `group_token_old.hpp` | 算子实现（标量 + 可选 SIMT/SIMD） |

## 参见

- [cann-samples group_token_old_main.asc](../../../../../cann-samples/Samples/2_Performance/group_token_story/src/group_token_old_main.asc)
- [group_token.hpp](../group_token/group_token.hpp) — 2 阶段变体（无排序）
- [sort/topk.hpp](../sort/topk.hpp) — 类似的直方图 + 标量模式
