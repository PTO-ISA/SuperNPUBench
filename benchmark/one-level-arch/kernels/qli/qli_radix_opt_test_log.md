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

## 阶段 A2：复用直方图桶计数（分析后结论：不引入，A 止于 A1）

### 分析
原计划利用 Byte3 直方图 `per_bin3` 做"整桶必选"预筛。逐项核查后**不实施**：

| 候选优化 | 结论 |
|---|---|
| 跳过空桶 | 已天然实现：`RadixCountOf`(TROWSUM) 计数，`QLI_RADIX_POP_N` 循环次数=cntGt，空桶 0 次 |
| 提取 TLOAD 2→1 次/chunk（GT+EQ 合一） | ❌ 违反两遍语义：跨 chunk 需先全部 gt 再 eq；同 chunk 内先 eq 会复现已修复的越界 bug |
| 复用直方图轮 key/per_bin | ❌ 直方图轮已完成且局部变量释放；同桶内元素部分 >kth 部分 <kth，无法桶级批量决定 |

**结论**：无 scan 指令下 A2 无可落地收益。A 方案止于 A1（TROWARGMAX 去排序 + 单指令提取）。
真正消除逐元素 pop 依赖方案 B（TSCAN）。

---

## 阶段 B：新增 TSCAN 前缀和指令（单趟桶式提取）

### B1/B2：指令落地与单测
- TileOpManager.h：`TileOp::TSCAN`（TINVALID 前）+ `ParFunctionTEPL::TEPL_TSCAN`（EXCEPTION func21）
  + `GetTeplTileOp` 映射
- Block.cpp：`function_masks[3]` 增加 func21（0x001ffdfd → 0x002ffdfd）
- TEPLEngine.cpp：`ExecuteTSCAN`（行内 inclusive prefix-sum）+ dispatch
- AccumulateBlockInfo.cpp：`IsReduceAndExpandTepl`/`IsReduceAndExpandTeplDataType` 允许 TSCAN（UINT32 等）
- template_asm.hpp：`TSCAN(dst,src)` asm（`BSTART.TEPL 117`，单源单目同尺寸）
- gfrun 重新编译通过

**独立 spike（/tmp/opencode/test_tscan.cpp）**：

| 输入 | 预期 scan | 实际 | 结论 |
|---|---|---|---|
| [1,1,0,1,0,0,1,1] | [1,2,2,3,3,3,4,5] | ✗ 首版 [33,33,33,34,...] → **volatile 初始化后 [1,2,2,3,3,3,4,5]** | ✅ 指令正确 |
| [3,1,4,1,5,0,0,0] | [3,4,8,9,14,...] | [3,4,8,9,14,...] | ✅ |

> 首版失败根因：LinxV5 `-O2` 把**静态 0/1 数组**标量写向量化为 `(v<<5)|v` tile op
> （同 P3-D6/Spike A 现象）；非 volatile 数据保存入 tile 前先经 volatile 拷贝规避。

### B3：QLI 单趟提取（TSCAN+TSCATTER）—— TSCATTER 工具链阻塞，不可落地

实现并于 QLI 尝试接入 `qli_topk_radix_scan`（TSCAN 求 rank + TSCATTER 按
`outBase+rank-1` 散写）后，**TSCATTER 无法编译**：

```
/template_asm.hpp:6600:6: error: Match Instruction Error!
    "B.IOR [%7],[]\n"
     ^
```
`B.IOR [u#1],[]`（TSCATTER 的 off tile 绑定）被 LinxV5 工具链 asm matcher 拒绝；
独立 probe（/tmp/opencode/test_tscatter.cpp）同样失败。**TSCATTER/TGATHER 在当前
LLVM 版本未实现**（尽管头文件有定义）。独立 spike 验证：
- TSCAN 指令：✅ 正确（[1,2,2,3,3,3,4,5] / [3,4,8,9,14]）
- TSCATTER：❌ 工具链编译失败

**结论**：方案 B 的"TSCAN 前缀和后单趟 scatter"在**无 TSCATTER 工具链支持**下
不可落地为 QLI 提取。TSCAN 指令层已验证可行（保留为未来能力），但 QLI 提取只能
走方案 A（pop 式）。**方案 B 评估：不采用（TSCATTER blocker）**。

### B 阶段回归确认（排除破坏）
- SSM 改动（TSCAN: TileOpManager/Block.cpp function_masks/TEPLEngine/校验）后：
  - spike_thistogram_radix R2=0 ✅
  - qli_check（TROWARGMAX 传统路径）R2=0 ✅
  - qli_topk_radix（方案 A）cosine=1.0 set=4/4 ✅
- QLI 主文件已恢复为仅方案 A（`git checkout HEAD -- qli_pto_opt.hpp`），
  B 的 `qli_topk_radix_scan` 实验代码不进入 A 提交。

### 最终状态（本轮）
| 方案 | 状态 | 说明 |
|---|---|---|
| **A（TROWARGMAX pop 提取）** | ✅ 采用 | 已 commit `0718cb5`；4 Case cosine=1.0 set=100% |
| **B（TSCAN+TSCATTER 单趟）** | ❌ 暂缓 | TSCAN 指令可行；TSCATTER 工具链未支持 → 无法落地 |
| SSM TSCAN 指令 | 保留本地 | 供未来工具链支持 scatter 后复用