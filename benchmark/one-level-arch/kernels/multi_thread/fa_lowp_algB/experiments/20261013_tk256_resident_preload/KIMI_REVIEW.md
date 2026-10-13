# Resident tag 4→8 审查报告（纯理论，未跑实验）

日期 2026-10-13。依据：combined_main_20261012 模型 CubeCore.cpp:3429-3895/5060-5106、CubeCore.h:962-997；算子 fa_lowp_algD.hpp；tk256_compare_20261013/report.md。

## 1. tag key 与 Q0/Q1 实际需求量

key = (physSlot, isScale, windowOffset)。shared A 时 `AUsesPackedM32` 恒 false（CubeCore.cpp:3434 被 `src0IsShared` 门断），故 **packed M32 不适用、window 非 0**：uopsPerXfer = cellRows32/tileSizeM16 = 2，每个 Q tile 每线程需 2 个数据 tag（window 0 和 512B）。dblQ 下 Q0+Q1 = **4 数据 tag/线程**，恰好占满 4 槽。Qscale 走 scale tag：record/hit 均硬编码 `l0aResidentTags[0]`（:3788, :5102），即 2 个 scale tag 只进 thread0 池 → thread0 共 6 key 抢 4 槽，**结构性 LRU thrash**；thread1-3 刚好 4/4 无余量。指纹吻合：Tk128 miss 354 ≈ 64kb×6+冷启动，Tk256 192 ≈ 32×6，per-kb 减半关系成立。旧注释"4 路×4KB"的容量推断不可沿用，6>4 才是病根。

## 2. scale lifetime 未被 resident 保护

`L0AResidentTagRecord` 只对 `!isScale` 置 L0A entry `resident`（:3872）；leftScaleBuffer entry 从不置位，尽管 `FreeLeftScaleBuffer` 会尊重 resident（:4622）。leftScaleBuffer 仅 8 项 FIFO，且每 kb 每个 qi 的 pScale（本地 Scaling，:5107 非共享分支）也占该缓冲 → FIFO 压力可 free 掉 Qscale entry → tag 见 `entry.free` 作废 → **tag 加到 8 仍治不了 scale miss**。另：HitScale/HitA 均无比对 generation，entry free 后再分配他者存在陈旧命中（正确性隐患，非性能）。

## 3. PV 的 P 与 L0A pin

P 为本地 CellReg 装配（`src0IsShared=false`），不产生 tag。pin 的 L0A entry 只增不减？本 kernel 不同：shared-A 窗口恒为 4/线程，8 tag 后 pin 数仍 4（scale 不 pin），L0A 32 项中有效容量不变；但 pin 由"churn"变"稳定"，对 P/K 流的 LRU 干扰反而下降。风险仅在未来更多 shared 窗口的 kernel，需加 pin 高水位观测。

## 4. miss 计数语义

`l0aTagMisses` 在 Record 时 +1：含冷启动与淘汰插入，data/scale 不分、thread 不分、无字节量纲。miss ⇒ 确实产生重读（genL0ALoad），可作重读事件数代理，但不能直接当带宽字节；换算需配合 ShareTReg TILE_A 读字节计数。

## 5. 可证伪预测

- A1（8 tag 混合）：miss 应降至冷启动量级（<40，Tk128）；若仍 per-kb 量级 → 证伪"纯容量"，指向 scale FIFO/flush。
- A2（4 数据 + 4 scale 分池）：若 ≈A1，thread0 混合非主因；若残余 miss 集中 scale，分池定位。
- A3（8 tag + scale entry 亦置 resident）：若 scale miss 再降 → 证实第 2 节机制。
- 所有配置 Tk256/Tk128 miss 比 ≈ kb 数比（per-kb 机制指纹），不成立即假说有误。

实验矩阵：A0 现状（已有 354/192）、A1 最小改动（kL0AResidentTagNum=8）、A2 分池、A3 scale pin；A2/A3 为必要的 A/scale 分离对照。每配置跑 Tk128+Tk256。

## 6. 需补观测字段

miss/hit 按 (thread × data/scale) 拆分；tag 淘汰计数；l0aTagFlushes 输出；leftScaleBuffer FIFO free 事件及 Qscale entry 被 free 次数；L0A resident pin 高水位与分配 retry；ShareTReg src0（数据/scale 分开）读字节。

## 7. 证据不足处

- uopsPerXfer=2、window 实际取值分布未在 trace 直接验证（推断自模型常量与注释）。
- Qscale entry 是否真被 pScale FIFO 挤出，缺 dependenceCount 时序证据。
- generation 不比对的陈旧命中是否被其他路径挡住，未核实。
- miss→cycle 无直接换算：L0 in-flight 2679/1294 为排他类别，不能全归 resident miss。
- Tk256 仅放宽 static_assert 的实验副本、无 res_check，所有数值不含量值正确性。
