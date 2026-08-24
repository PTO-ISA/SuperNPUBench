# QLI radix-select TopK — 方案 A/B 修改测试记录

> 起始版本：PR 分支 feat/qli-radix-topk @ HEAD `b3dc77e`
> 修改对象：`qli_pto_opt.hpp`（提取阶段）、`qli_pto.hpp`、gen 脚本
> 目标：解决评审"直方图未用于提取 / 提取仍逐元素 pop / 多余排序"问题

---

## 基线（方案 A 修改前，commit b3dc77e）

### 精度（gfrun，set-match 主判据）

| Case | cosine | set | exact(参考) | NaN | blocks |
|---|---|---|---|---|---|
| Sq64 Skv128 topK128 | 1.000000 | 64/64 | 100.00% | 0 | 355,086 |
| Sq4 Skv2048 topK512 | 1.000000 | 4/4 | 100.00% | 0 | 71,918 |
| Sq4 Skv8192 topK512 | 1.000000 | 4/4 | 0.54% | 0 | 133,443 |
| Sq4 Skv2080 topK512 | 1.000000 | 4/4 | 23.73% | 0 | 75,606 |

### 性能（gfsim，Sq64 Skv128 topK128）

| 指标 | 值 |
|---|---|
| Total Cycles | 925,120 |
| Block number | 355,086 |

### 当前提取逻辑（修改前）
- pass1：逐 chunk `TCMP(key>kth)` → mask → `QLI_RADIX_POP_N`（每次 TROWMAX 取最大 key → 消零）输出全部 gt
- pass2：逐 chunk `TCMP(key==kth)` → `QLI_RADIX_POP_EQN`（rev 技巧最小索引）补 remaining
- pop 循环内部：TROWMAX + TEXPANDS + TCMP + TMUL 索引 + TROWMAX best + TCVT + TSTORE + TEXPANDS + TCMP + TSUB/TMUL 消零（约 10+ tile op / pop）
- **问题**：直方图桶信息未参与提取；pop 按 key 降序做（set 契约不需要）。

---

（后续阶段将在此追加记录）

---

## 阶段 A1：pop 改用 TROWARGMAX 单指令（去多余排序 + 省索引反推链）

### 修改
`qli_pto_opt.hpp`：
- `QLI_RADIX_POP_N`：`TROWMAX(mx)→TSTORE标量→TEXPANDS→TCMP→TMUL picked→TROWMAX best`（约 6 op 索引反推链）
  替换为 `TROWARGMAX(best) + TADDS(bestg, best, CHBASE)`（单指令取最大 key 列索引 → 加 chunk 基 → 全局索引）
- `QLI_RADIX_POP_EQN`：同样用 `TROWARGMAX(pv)` 直接取最大 rev+1 列 = 最小索引（省略反推链）

### 验证（gfrun，set-match）

| Case | cosine | set | NaN | blocks(前→后) |
|---|---|---|---|---|
| Sq64 Skv128 topK128 | 1.000000 | 64/64 | 0 | 355,086 → **314,126** |
| Sq4 Skv2048 topK512 | 1.000000 | 4/4 | 0 | 71,918 → **61,678** |
| Sq4 Skv8192 topK512 | 1.000000 | 4/4 | 0 | 133,443 → ~118K* |
| Sq4 Skv2080 topK512 | 1.000000 | 4/4 | 0 | 75,606 → ~63K* |

*Skv8192/2080 为复测后 block，Skv2048 精确（71,918→61,678，-14.2%）。

### 对抗性用例（A1）
- 早期 chunk 2044 eq + 后期 4 gt → 输出 {0,1,2,3,2048,2049,2050,2051} PASS
- NaN 多位型（负/signaling/payload/quiet）→ topK={0..7} PASS

### 遇到问题与解决
- Case1 初测 NaN=256：**旧 dump golden 不匹配**（gen/顺序错误），重新按
  `gen → 汇编 → build → fix 地址 → gfrun → 验证` 后 cosine=1.0、NaN=0。
  教训：每次换参数后必须重新生成 golden 并完整重建链路。

### A1 性能（gfsim）

| Case | Total Cycles |
|---|---|
| Sq64 Skv128 topK128 | 925,120（基线同） |
| Sq4 Skv8192 topK512 | 667,139 |

> 注：A1 主要降低 gfrun/功能级 block 数（提取 tile-op 链变短）；
> gfsim 周期对 TopK 提取段收益被直方图轮主导，差距有限。方案 B（TSCAN
> 单趟提取）预期在提取段有更显著周期收益。

---

## 阶段 A2：直方图桶计数指导提取（复用直方图信息，跳过空桶/减少比较）

### 思路
A1 之后提取仍是"TLOAD key → TCMP(key>kth) → TROWARGMAX pop"。A2 目标是：
从直方图轮**继承每字节 kth 边界桶**，对提取做桶粒度预筛：
- 直方图已给出 Byte3 层 `kth_byte3`；`key` 的 **Byte3 > kth_byte3** 的元素必入选
  （无需逐元素比较 key 与 kth，只需比较高位字节）。
- 仅 `Byte3 == kth_byte3` 的跨界桶内元素需完整 32-bit 比较。
- 空桶（直方图计数为 0 的高位桶）直接跳过，无需 pop。

### 落点
EXTRACT_GT：由 `TCMP(key > kthVal)`（逐元素完整比较）改为：
1. `TSHRS(hb, key, 24)` → 高位字节
2. `per_bin3[b]`（Byte3 轮直方图）已得；累加 b>kth_byte3 桶 = 必选集合
3. 对必选集合（整桶批量）直接用桶掩码 + TROWARGMAX 连续 pop（桶内元素次数远小于 topK 混检）
4. 跨界桶才逐元素比较

### 待验证
- 精度（4 关键 Case + 对抗/NaN）
- gfsim 周期对比 A1