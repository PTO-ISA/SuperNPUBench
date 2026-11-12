# RWDB 瓶颈：机制与证据

## 机制（从模型源码）

- 一条 gather 按 128 B beat 切分，beat 数 = 目的地数据量 / 128 B：
  64 lane × 4 B = 2 beat，256 lane × 4 B = 8 beat，32 lane × 8 B = 2 beat。
- `l2_agu.cpp::PrepareInstruction()`：`m_waveBeatCount = min(waveTotalBeats, RWDB_ENTRY_COUNT)`。
- `l2_rwdb.cpp::ProcessWaveReservation()`：`freeEntries < request.beatCount` 时**整条 wave 拒绝**，
  不能部分开工（全有或全无）。
- entry 按 beat 逐个释放，但下一条 wave 要一次凑齐 beatCount 个，因此效果近似“整条完成才释放”。
- 在飞 gather 并发 = `floor(RWDB_ENTRY_COUNT / beats_per_gather)`：
  RWDB=8 时 64 lane 4 条、128 lane 2 条、256 lane 1 条、512 lane 一波都放不下。

## 直接证据

| 配置 | `l2_rwdb_entry_full_cycles` | Total |
|---|---:|---:|
| 64-lane 2+2，RWDB=8 | 51,328（37.8%） | 135,819 |
| 64-lane 2+2，RWDB=32 | 0 | 124,107 |
| 64-lane 2+6，RWDB=16 | 10,828 | 118,207 |
| 64-lane 2+6，RWDB=32 | 52 | 116,635 |
| 512-lane 2+2，RWDB=8（诊断） | 114,464（58.2%） | 196,638 |

- tag pipe 只用到 0.18/2 lookup/cycle（9%），GM 带宽 12/128 B/cycle（10%），
  BHU 峰值 134/360；因此返回通路（RWDB）而不是 tag/带宽是结构瓶颈。

## 关联

- 上游 issue：<https://github.com/LinxISA/SuperScalarModel/issues/1210>
  （8-entry wave 全有或全无；建议改为动态/部分授权）。
- TLSU 组件职责与数据流：`outputs/hash64_batchmap_20261105/slots_sweep/TLSU_ARCH.md`
  （InstBuffer / PriorBuffer / Arbiter / AGU / TagArray / MROB / BYPQ / BHU /
  DataSlotManager / DataBankArbiter / RWDB / InflightQ）。

## 结论边界

- RWDB/ways 是**硬件配置假设**（`H-RWDB`、`H-WAY`），不得与默认配置成绩混用。
- `RWDB=48 + TAG_WAY=7` 的结果（76,959）是“该硬件假设下”的最好成绩；
  默认参数下的最好成绩是 64-lane 2+2 的 135,819。
