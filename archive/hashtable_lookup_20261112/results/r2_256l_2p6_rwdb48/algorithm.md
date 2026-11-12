# 算法定义与成立条件（r2：256-lane 自适应 2+6，SoA）

## 目标运算

与 r1 相同（MurmurHash3 x86_32 + 线性探测）。差异在并行粒度与投机宽度：
Lanes=256（每 batch 256 query，共 24 batch），每轮 A 阶段 2 槽、B 阶段 6 槽。

## 数据与接口

- **表**：`Capacity = 2,550,000` 槽，装载率约 70%。SoA 布局：
  `keys32[i]`（u32，64-bit fixture key 的低 32 位）、`values[i]`（i32）。
  槽 `i` 的 value 与 key 同下标；空槽由“key 不等于任何查询 key”体现。
- **查询/输出**：6144 个 i64 key；`output[q] = values[hit]`，未命中为 `-1`。
- **batch map**：24 项，每项是一个 batch 的 query 基址（以 Lanes 为单位），
  4 个 PE 各 6 个 batch。
- **表尾 guard**：单轮 2+6 最多读到 `Capacity-1+7`；fixture 在表尾追加 128 项
  （512 B）镜像槽 0..127。多轮探测时 offset 不回绕，因此 guard 长度必须覆盖
  最大探测长度（本 fixture 实测 63 槽），7 项只是单轮窗口的下界。
- **batch 大小约束**：query 数为 Lanes 整数倍（6144 = 24×256）。

## 哈希

MurmurHash3 x86_32，seed 0，对 8 字节 key 的两个 32-bit 块各做一轮：

```text
h = 0
for block in (key_low32, key_high32):
    k = block * 0xcc9e2d51;  k = rotl32(k, 15);  k *= 0x1b873593
    h ^= k;  h = rotl32(h, 13);  h = h*5 + 0xe6546b64
h ^= 8
h ^= h >> 16;  h *= 0x85ebca6b
h ^= h >> 13;  h *= 0xc2b2ae35
h ^= h >> 16
offset = h mod 2,550,000
```

取模用“半字 + 余数”实现：`h mod C = ((h>>1 mod C)*2 + (h&1)) mod C`，避免 64-bit
除法；`C` 为编译期常量。

## 自适应探测（2+6）

```text
for each batch:
    load 256 keys, compute offsets, active = all ones, hit = -1
    while any(active):
        A: for lanes with active, read slots o, o+1
            若命中，记录 hit = o 或 o+1，active = 0，o += 2
        B: for lanes still active, read slots o+2 .. o+7
            若命中，记录 hit = 首个命中槽，active = 0，o += 8
    gather: 对 hit >= 0 的 lane 读 values[hit]
    store 256 results
```

要点：

1. **A 阶段全 lane 读取**（2 槽），先覆盖命中率最高的两个槽；
2. **B 阶段只对未命中 lane 读 6 槽**，使用 A 阶段结束时的 active 掩码，不重复读
   已命中 lane 的槽；
3. **value 延迟 gather**：探测阶段只读 key，命中偏移全部确定后按 lane 掩码 gather
   一次，把随机读合并为一个 batch 一次发出；
4. **offset 不回绕**：依赖表尾 guard（见上）与 fixture 的最大探测长度。

## 数据流

```
queries --TLOAD(256 lane)--> hash --> offsets --A(2)+B(6) rounds--> hitOffset
      --masked value gather--> result --TSTORE--> output
```

## 正确性条件

| 条件 | 违反后果 | 本归档的保证 |
|---|---|---|
| key 唯一 | 命中不唯一，value 可能取到别名 | fixture key 为唯一 64-bit hash；只比较低 32 位，冲突概率约 1e-8/query |
| guard 覆盖最大探测长度 | 越界读 | fixture 追加 128 项 guard，实测最长 63 槽；单轮窗口仅需 7 项 |
| query 数为 Lanes 整数倍 | 尾 batch 越界 | 6144 = 24×256，成立 |
| 空槽不等于任何 query key | 未命中 lane 无法终止 | fixture 空槽为 0，query key 非 0 |
| 命中 lane 不再被写 | 结果被后续轮覆盖 | 命中后立即清 active |

`probes < Capacity` 是终止兜底；表已满时返回未命中而不是死循环。

数值验证：同结构 64-lane 2+6 check 构建 PASS（`errors=0`，
`output_matches=true`）；256-lane 2+6 本身未单独 check（`GAP-R2-CHECK`），
正确性依赖与 64-lane 相同的 element 级语义、仅 Lanes 不同。

## 性能成立条件

- **RWDB=48**（`sl2.RWDB_ENTRY_COUNT`）：wave 授权全有或全无；2+6 的一次探测
  每个 256-lane batch 需要 8 个 128 B beat，48 entry 允许 6 个 gather 并发，
  掩盖 300-cycle 随机延迟。默认 RWDB=8 时同一实现为 169,868 cycles（慢于
  64-lane 2+2 的 135,819），因此 **82,256 只在 RWDB=48 的硬件假设下成立**。
- **TAG_WAY=4**（`H-WAY` 变体为 7）：影响 L2 tag 冲突；76,959 是 WAY=7 的结果，
  不与 WAY=4 混用。
- **SharedTReg 256 KiB / 64×4 KiB**：容纳 batch 的 key/offset/mask tile。
- **soc_random + 300-cycle 均值延迟 + 128 B/cycle 读写带宽**：延迟结论只在该
  随机模型下有效；r1 的确定性 89,048 不可与 random300 混比。
- **单次确定性仿真**：同一 ELF/配置产生同一周期数，不声明统计稳定性。

## 数字来源与差异

`hashtable_lookup_soa_wide.hpp` 头注释与提交信息里的 **81,317 / 77,312** 来自更早的
独立 sweep（`outputs/hash256_adaptive_2p6_rwdb_sweep_20261110/runp_rwdb48.log`，
显式 `--conf fourpe`，ELF `/tmp/adaptive256/kp.elf`）。本归档 r2 使用后续构建的
`/tmp/soa_op/perf.elf` + ELF auto-detect 的 fourpe 配置，同一组 flags 下得到
**82,256 / 76,959**。两者相差约 1.2%，差异来源是配置入口与 ELF 版本，不是模型参数；
本归档以 `run_rw48*.log` 的 82,256 / 76,959 为准，81,317 仅作为历史旁证。
