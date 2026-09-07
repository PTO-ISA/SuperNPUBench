# 已退休直接 Tile op —— 从活跃看护隔离

这些 op 已由 PTO-SPEC 从**活跃直接 Tile 指令集**退休（对应 tile ASL 已删）。它们**不应再作为活跃接口看护**：
正确闭环是 API/Bench 改用软件序列或 fail-closed，模型侧拒绝退休 selector（`tile-fault-retired-encodings-001.asl`）。
demo 移到此处保留作历史，`run_guard.sh` 只遍历各域 `src/*.cpp`，故不再拾取这些 case。

判据：当前 pto-spec `asl/tile/` 下已无对应活跃 ASL（对比 TPACK/TPERMUTE 仍在=未退休）。

## 清单（15 个）

### ADR-TILE-0013 / 提交 `edcbd8b`（发布 0.58.5.1，"software-replaceable tile operations"，Tile op 126→118 / TEPL selector 86→78）
| 算子 | 类别 |
|---|---|
| TCONCAT | 拼接 |
| TEXTRACT | 抽取 |
| TINSERT | 插入 |
| THISTOGRAM | 直方图 |
| TQUANT | 量化 |
| TDEQUANT | 反量化 |
| TSORT | 排序 |
| TMRGSORT | 归并排序 |

### ADR-CUBE-0013（private cube/vector/cell rearrangement；reserved-illegal，no alias/replacement）
| 算子 | 退休 selector |
|---|---|
| TPARTADD | 0x071..74 六个旧 selector 之一 |
| TPARTMUL | 0x071..74 |
| TPARTMAX | 0x071..74 |
| TPARTMIN | 0x071..74 |
| TFILLPAD | 0x065 |
| TTRANS | 0x06E |

### ADR-BLOCK-0018（Issue #99）
| 算子 | 说明 |
|---|---|
| TIMG2COL | **Local-Tile 直接形式退休**，改由块形式 `BSTART.TIMG2COL` 承载 |

## 影响修正
此前这些被当活跃接口看护，其中 tconcat/tdequant/tsort/ttrans/tfillpad/tpart* 还"PASS"着——实为在验废弃指令；
tquant/tinsert 曾被误报为模型缺口（#560 gfrun-6/7，已作废）。隔离后活跃看护不再包含退休 op。
