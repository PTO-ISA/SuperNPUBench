# 算法定义（r1：64-lane 自适应 2+2）

## 目标运算

线性探测哈希查找：`slot = murmur3_x86_32(key) % Capacity`（seed=0），
沿 slot, slot+1, … 顺序查找 `key`（32 位低半），命中返回 `value`，未命中返回 -1。

## 实现

- 输入：`queries[base .. base+Lanes-1]`（int64），`keys[slot]`（u32，key 低 32 位），
  `values[slot]`（i32）。
- 每批 Lanes=64 个 query；hash 与取模在 64-lane tile 上算（两次有符号 TREM 恒等式）。
- 每轮：A 阶段读 `keys[o]`、`keys[o+1]`（全部 active lane）；B 阶段只对 A 未命中 lane
  读 `keys[o+2]`、`keys[o+3]`；命中即停并记录 hitOffset；`anyActive` 为假时退出。
- 所有 lane 完成后，对 `values[hitOffset]` 做一次 masked gather，再 TSTORE 写回
  `output + base`。

## 数据流

```
queries --TLOAD--> keys(u64 tile) --TCVT/移位--> low32/high32 --murmur--> hash
hash --%Capacity--> offsets --A/B rounds--> hitOffset --value gather--> result --TSTORE--> output
```

## 成立前提 / 差异

- A-001：所有 query 的 key 都在表中；count 为 Lanes 整数倍；表尾有可读 guard。
- 与 64 位完整比较的差异：只比较低 32 位。fixture 的 key 是唯一的 64 位哈希，
  低半别名概率约 1e-8，且 check 构建已验证 6144 输出。
- guard 读越界是只读的，不会改变结果（guard 镜像 slot 0..7，不含 query key）。
