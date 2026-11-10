# hashtable_lookup `lo32_o3` 性能分析（基于本次复现的 swimlane）

数据源：`out/lo32.swim.json`（本次 deterministic 运行，与归档 sha256 一致的代码路径）。
分析脚本：`out/derive_span.py`、`out/agg_swim2.py`、`out/agg_swim4.py`、`out/agg_swim5.py`。

## 1. 你的理解对不对：对，但要补两处

**主链路确实是「VEC 算 hash → TLSU 按地址 gather → VEC 比价 → 下一轮」**，源码逐行对得上：

| 阶段 | 谁执行 | 落在哪些指令（kernel.hpp 行号） |
|---|---|---|
| 取 query | TLSU | `TLOAD(keys, queries + base, 256)`（:188） |
| hash | VEC | `TCVT` 取低/高 32 位（:90-92）；murmur3 两轮 `TMULS`+`TSHLS/TSHRS/TOR`(rotl)+`TXOR`+`TADDS`（:53-59, :94-105）；`% 2550000` 用 `TCVT/TREM/TMULS/TADD/TAND` 拼出来（:70-79） |
| 探测取表 | **TLSU** | `probeRound4` 里两次 `table[nextOffset].key`、`table[nextOffset+1].key`（:131-134）→ 降成 `MGATHER_MASK`，地址来自 VEC 算出的 offsets |
| 比价/选新掩码 | VEC | 低 32 位 `TCMP` + `TAND/TOR/TSEL/TADD`（:142-155），`TROWMAX`+`TCMPS` 判 `anyActive`（:160-166） |
| 取 value | TLSU | 「deferred value gather」一次 `MGATHER_MASK`（:201-210） |
| 写回 | TLSU | `TSTORE` 256×i32（:211-214） |

**CUBE 完全没参与**：trace 里 `CUBE_0..3` 注册了进程但**零事件**；只有 `SPE_0 / VECTOR_0..3 / TLSU_0 / CellReg_0` 有事件。

两处要补：

1. **不是「TLSU 去访问存储」这么简单**——它是 **masked gather**，每次只取仍在探测的 lane；而且控制流是**每轮重新算掩码再发下一轮**，严格串行。
2. **每轮的实际开销大头在 VEC，不在访存**（见第 3 节）。

## 2. 规模（trace 实测）

| 量 | 值 |
|---|---|
| kernel span | 73,223（291 → 73,514，PE2 决定） |
| 每 PE batch 数 | 6（256 lane/批 × 6 = 1536 query/PE） |
| TLOAD / TSTORE | 每 PE 各 6 次（与 batch 数一致） |
| MGATHER_MASK 总数 | 700（PE0 166 / PE1 162 / PE2 194 / PE3 178） |
| 每 batch 探测轮数 | ≈ (gathers−1)/2，典型 **10–16 轮**，最长 33 轮（PE2 最后一批 66 gather） |
| 平均探测深度 | ≈ 26–32 槽/query（fixture 最长 63 槽；每轮只走 2 槽） |
| gather 时延 | p50 **121** / p90 **568** / max 1,571 拍；平均 275 |
| 同轮两笔 gather 间隔 | **67 拍**（固定，说明是背靠背发射） |
| 相邻轮起点间隔 | 中位 **450 拍**，均值 ~400，冷启动轮 1.2k–1.7k |
| 每 batch 耗时 | ≈ 12.2k 拍 |

## 3. 一轮 450 拍花在哪（关键）

抽 PE2 的一个热轮（gather 4765/4832 → 数据 4945 回 → 下一轮 5332 发）：

```
4945 数据回来
4946-4962  TCMP ×3, TAND, TOR, TSEL ×5, TADD ×2, TCI, TLEA ...   （比价+更新 hit/live/offset）
4962-4963  TROWMAX + TCMPS                                        （anyActive 归约，跨 lane）
5046-5190  TCMP/TLEA/TMUL/TSEL/TADD 第二串（下一轮的地址生成已开始）
5332       下一轮 gather 发射
```

即：**gather 命中时只占 ~70–150 拍，剩下 ~300–390 拍是 VEC 的元素级比价 / 选中 / 归约 / 地址生成**。

对应到引擎占用（区间 [291, 73514]）：

| 资源 | 占用 | 说明 |
|---|---|---|
| VEC（每个 PE） | **62.5% / 61.3% / 72.8% / 66.7%** | 四条 VECTOR 进程，有块在飞的时间占比 |
| TLSU gather（每个 PE） | 41–45%（union 30.3k–32.8k） | 单 TLSU 被 4 PE 共用，全局 union 150% ⇒ 事件是"请求寿命"不是独占占用 |
| CellReg_0 | **87%** | tile 寄存器读写几乎连续，不是突发 |
| TLOAD / TSTORE | 每笔 200–700 拍，6 次/PE | 占比小 |

**结论：这是「每轮串行 + VEC 元素运算」型瓶颈，不是访存带宽/延迟本身。** 一旦表被 cache 热起来，gather 只剩 62–98 拍，而每轮 VEC 侧固定 ~390 拍；轮数 × 每轮 VEC 时间决定了 73,223。

## 4. 可下手的三个点（都还没实测，只是从 trace + 源码推出来）

1. **每轮只探 2 个槽，但文件头注释写的是 4 个槽**（kernel.hpp:7-9 说 `o, o+1, o+2, o+3`；实际 :131-134 只读 `nextOffset` 和 `nextOffset+1`，:152 是 `+= 2`，函数名却叫 `probeRound4`）。
   探针深度 63 槽 ⇒ 现在 32 轮。若真做 4 槽/轮，轮数减半；但每轮 VEC 比价也接近翻倍，净收益取决于"gather 等待 vs VEC 比价"的比例——按本轮的 100:390，估算能砍掉 20–30%，不是 50%。
2. **`anyActive()` 每轮都做一次跨 lane 归约**（`TROWMAX`+`TCMPS`，kernel.hpp:195）。PE0 有 573 个 TROWMAX 块、~80 轮，说明归约在关键路径上重复出现。可以每 2–4 轮判一次，或用标量掩码位。
3. **首轮冷缺失很贵**：第一个 gather 1.4k–1.7k 拍（表 40 MB，L2 装不下），后面降到 62–98。4 个 PE 的冷启动轮几乎同时发生 → 可考虑按 PE 错开或预取。

## 5. 负载不均

| PE | 最后 TSTORE end | 相对最慢 |
|---|---|---|
| PE0 | 66,214 | −9.9% |
| PE1 | 66,098 | −10.1% |
| PE2 | **73,514** | — |
| PE3 | 69,839 | −5.0% |

四 PE 固定切 1536 query/份，但**探测深度是数据相关的**（PE2 194 次 gather vs PE0 166），最后 `res_check_wait_for_all` 等最慢的 PE2。**span 由最慢 PE 卡住，约 10% 是纯负载不均**——这个用例里 4 PE 的固定切分不如按 batch 动态抢。

## 6. 与归档的对照

- kernel span 73,223、per-PE 结束拍点与归档 `results.json` **逐个相同**。
- 归档 `results.json` 提到的对比基线（109,193 / 103,033 版）本目录没有，未复核。
- 本文所有百分比都是**本次 trace 实测**，不是从归档文档抄的。

## 7. 4 槽/轮实测（2026-11-04 补做）

用本目录自建的 clang（PR #146 `a5f03b38`）重编算子，改了 `probeRound4` 为每轮探 4 个槽（`out/src_s4p/kernel.hpp`，未改模型）：

| 版本 | Total Cycles | kernel span | finisher |
|---|---|---|---|
| 2 槽/轮（归档版） | 135,299 | 73,223 | 0x5555 PASS |
| **4 槽/轮** | **140,506（+3.85%）** | **78,275（+6.9%）** | 0x5555 PASS |

swimlane 证实 4 槽确实每轮发 4 条 gather（同轮间隔 92/70/21 拍），**模型没有 stall、没有断言**，跑完且数值正确。

原因：轮数确实减半（每 PE gather 174 次 → 约 43 轮，2 槽是 166 次 → 约 82 轮），但每轮周期从 ~450 拍涨到 ~880 拍（VEC 判定链翻倍），加上命中在 `o+1` 也仍要读 `o+2/o+3` 的过度探测，净亏。

归档 `hash256_s4g_par_20261104/README.md` 里 "4-slot parallel: model stalls with 16 wide gathers in flight" 这条在本模型（PR #1177）+ 本编译器（PR #146）上**不复现**；但"stride-4 更慢"的方向与其 stride 扫描（183,681 vs 171,646）一致。

