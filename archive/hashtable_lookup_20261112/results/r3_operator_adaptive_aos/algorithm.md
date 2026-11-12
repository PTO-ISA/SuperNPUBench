# 算法定义（r3：basic_op 算子默认实现，32-lane AoS 2+6）

## 目标运算

同 r1/r2 的线性探测；表是 AoS `TableEntry{int64 key; int32 value; int32 padding}`，
空槽 key = -1（UINT64_MAX），命中用完整 64 位比较，遇空槽即停止。

## 实现

- 32-lane ElementTile；hash/取模同 r1；
- 每轮 A 阶段 2 槽（masked inline-asm gather）、B 阶段 6 槽（只对 A 未命中 lane）；
- 命中/空槽都使 lane 退出（空槽表示查找失败），hitOffset 记录命中槽；
- 全部完成后一次 value gather（未 mask，miss lane 读 table[8] 后用 TSEL 覆盖为 -1），
  再按 32-lane 写回。

## 数据流

```
queries --TLOAD--> hash --> offsets --A(2)+B(6)--> hitOffset/active
      --value gather--> result --TSTORE--> output
```

## 成立前提 / 差异

- A-003：Capacity-1 之后有 7 个可读 16 B guard；count 为 32 的倍数（当前验证范围）。
- 与 perf kernel 的差异：AoS vs SoA、完整 64 位比较 vs 低 32 位、32 vs 64/256 lane。
- 计时状态：gfsim 在 64 位 key gather 上断言（GAP-R3-1），**没有性能结论**。
