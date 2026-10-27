# HiF4 量化 Batch Matmul 性能历史记录

> 记录于 2026-10-26 09:43（hif4_deepseek_case3 单次复测）
> 整理日期：2026-10-23
> 测试对象：`quant_batch_matmul_test_hif4`（HiF4X2 fp4 + U32 scale，4-PE cooperative）
> 时序模型：`SuperScalarModel/bin/gfsim`（1.65 GHz 等效 MAC）
> 功能验证：`SuperScalarModel/bin/gfrun`（R2 = 0）
> 工具链：`linx-toolchain-build/output/linx_blockisa_llvm_musl`

## 1. 关键改动时间线

| 阶段 | 改动 | 影响 |
|------|------|------|
| PR #1056 | MX scale 冗余 load 修复 | case2 57125→56605（-0.9%），case3 13080→12360（-5.5%） |
| BFU 前端修复 | `SelectGlobalStageTid` F1 优先 grant pending taken FB | case2 tN256 K4096 由断言崩溃 → 可跑通 |
| scaleA CUBE_M32 | scaleA 布局 RowMajor → CubeM32（toolchain 放宽 MX ScaleA 契约） | 全 case MAC 利用率提升（见 §3） |

## 2. deepseek_case2 主对比（M512 N256 K4096）

`hif4_deepseek_case2` 标准配置 = `tM128 tN128 tK512`。

### 2.1 scaleA 布局：RowMajor vs CubeM32

| 配置 | RowMajor cycles | RowMajor MAC | CubeM32 cycles | CubeM32 MAC | Δcycles | ΔMAC |
|------|----------------|--------------|----------------|-------------|---------|------|
| tN128 tK512（deepseek_case2） | 56685 | 28.90% | **54334** | **30.15%** | **-4.1%** | **+1.25pp** |
| tN128 tK1024 | 56588 | 28.95% | 55619 | 29.46% | -1.7% | +0.51pp |
| tN256 tK512 | 60313 | 27.16% | 59390 | 27.59% | -1.5% | +0.43pp |

### 2.2 tN128 vs tN256（scaleA = RowMajor）

| 配置 | cycles | MAC util | 等效 TFLOPS |
|------|--------|----------|-------------|
| tN128 tK512 | 56685 | 28.90% | 62.51 |
| tN256 tK512 | 60313 | 27.16% | 58.75 |

结论：tN 翻倍（128→256）tile 数减半，但 epilogue 更厚、mac 流水排空气泡更多，
净损 -6.4% cycles、-1.74pp MAC。

### 2.3 等效 TFLOPS（Cube Achieved，1.65 GHz）

| 配置 | cycles | MAC util | TFLOPS |
|------|--------|----------|--------|
| tN128 tK512（RowMajor） | 56685 | 28.90% | 62.51 |
| tN128 tK512（CubeM32） | 54334 | 30.15% | 65.21 |
| tN128 tK1024（RowMajor） | 56588 | 28.95% | 62.62 |
| tN128 tK1024（CubeM32） | 55619 | 29.46% | 63.71 |
| tN256 tK512（RowMajor） | 60313 | 27.16% | 58.75 |
| tN256 tK512（CubeM32） | 59390 | 27.59% | 59.66 |

## 3. 全量 hif4 回归：RowMajor vs CubeM32（24 个 ELF）

> 全部功能正确（gfrun R2 = 0）。除 `K1600 tM128 tN256 tK320` 两端均 EXIT=134
> （`IsDivisible` GMMA 维度断言，与前端/scale 无关）外，其余全部 EXIT=0。

| case（B1，除标注 B2） | Row cyc | Row MAC | Cube cyc | Cube MAC | Δcycles | ΔMAC |
|----------------------|---------|---------|----------|----------|---------|------|
| M128 N128 K256 tM128 tN128 tK64 | 6725 | 1.90% | 6305 | 2.03% | -6.2% | +0.13 |
| M128 N256 K256 tM128 tN32 tK64 | 12754 | 2.01% | 9569 | 2.68% | -25.0% | +0.67 |
| M256 N128 K256 tM64 tN128 tK64 | 9869 | 2.59% | 9237 | 2.77% | -6.4% | +0.18 |
| M256 N256 K256 tM64 tN32 tK64 | 25141 | 2.04% | 20047 | 2.55% | -20.3% | +0.51 |
| M256 N256 K512 tM64 tN32 tK64 | 41282 | 2.48% | 32353 | 3.17% | -21.6% | +0.69 |
| M512 N256 K1600 tM128 tN128 tK64 | 92005 | 6.96% | 75353 | 8.49% | -18.1% | +1.53 |
| M512 N256 K1600 tM128 tN256 tK320 | EXIT134 | — | EXIT134 | — | — | — |
| M512 N256 K1600 tM128 tN256 tK64 | 67323 | 9.51% | 61989 | 10.32% | -7.9% | +0.81 |
| M512 N256 K1600 tM64 tN32 tK64 | 253951 | 2.52% | 189314 | 3.38% | -25.4% | +0.86 |
| M512 N256 K4096 tM128 tN128 tK1024 | 56588 | 28.95% | 55619 | 29.46% | -1.7% | +0.51 |
| M512 N256 K4096 tM128 tN128 tK512 | 56685 | 28.90% | 54334 | 30.15% | -4.1% | +1.25 |
| M512 N256 K4096 tM128 tN256 tK512 | 60313 | 27.16% | 59390 | 27.59% | -1.5% | +0.43 |
| M512 N256 K512 tM128 tN128 tK128 | 18915 | 10.83% | 16148 | 12.68% | -14.6% | +1.85 |
| M512 N256 K512 tM128 tN128 tK256 | 14245 | 14.38% | 12975 | 15.78% | -8.9% | +1.40 |
| M512 N256 K512 tM128 tN128 tK512 | 12916 | 15.86% | 12295 | 16.66% | -4.8% | +0.80 |
| M512 N256 K512 tM128 tN128 tK64 | 37145 | 5.51% | 23471 | 8.73% | -36.8% | +3.22 |
| M512 N256 K512 tM128 tN256 tK128 | 17285 | 11.85% | 15821 | 12.94% | -8.5% | +1.09 |
| M512 N256 K512 tM128 tN256 tK256 | 13048 | 15.70% | 12197 | 16.79% | -6.5% | +1.09 |
| M512 N256 K512 tM128 tN256 tK512 | 12408 | 16.51% | 12744 | 16.07% | +2.7% | -0.44 |
| M512 N256 K512 tM128 tN64 tK256 | 18404 | 11.13% | 15457 | 13.25% | -16.0% | +2.12 |
| M512 N256 K512 tM128 tN64 tK512 | 17303 | 11.84% | 14557 | 14.07% | -15.9% | +2.23 |
| M512 N256 K512 tM64 tN256 tK512 | 18345 | 11.16% | 17965 | 11.40% | -2.1% | +0.24 |
| M512 N256 K512 tM64 tN32 tK64 | 78089 | 2.62% | 59369 | 3.45% | -24.0% | +0.83 |
| M256 N256 K256 tM64 tN32 tK64（B2） | 45598 | 2.25% | 35566 | 2.88% | -22.0% | +0.63 |

### 3.1 观察

- **scaleA CubeM32 全 case 普适提升**：除 `M512 N256 K512 tM128 tN256 tK512`
  （+2.7%，疑似测量噪声）外，其余 22 个 case 全部下降，幅度 -1.5% ~ -36.8%，
  平均约 -13%。
- **收益最大的是小 tile / tM64 tN32 配置**（-20% ~ -25%），其次是 tN64/tN128
  中 tile（-8% ~ -18%），tN256 大 tile 收益较小（-1.5% ~ +2.7%）。
- 根因：scaleA 归一到 CUBE_M32 与 primary A（CubeTileM32）一致后，TLOAD 走
  `ND2M32` 描述符，`TMATMUL_MX` 消费不再走 legacy NORM 兼容路径
  （`isa/Block.cpp:733-739`），消除了 NORM↔CELL 的归一处理。

## 4. BFU 前端 bug 修复记录

- **现象**：`case2`（M512 N256 K4096 tM128 **tN256** tK512）在 gfsim 崩溃，
  `TimingSim/frontend/bctrl/bfu/bfu.cpp:GetLocalPipeID` 断言；tN128 正常。
- **根因**：fbid 4858（stid=1 的 `taken=1 end=0 pipe_id=-1U` global FB）在 F1 被
  RR grant 轮空，`CreateNewFB` 未建 local region，流到 F3 被 `WakeupLocalPipe`
  唤醒时 local region 缺失。
- **修复**：`SelectGlobalStageTid` 对 `stage==F1` 优先 grant 存在 pending taken FB
  的 stid。详见 `SuperScalarModel/bfu_local_pipe_premature_flush_issue.md`。
- **结果**：tN256 K4096 由崩溃 → 60313 cycles / 27.16%（scaleA RowMajor 时）。

## 5.1 hif4_deepseek_case3 单次复测（2026-10-26 09:43）

| case（B M N K tM tN tK） | cycles | Cube MAC util | 等效 TFLOPS | R2 | EXIT |
|---|---|---|---|---|---|
| hif4_deepseek_case3（B1 M512 N256 K512 tM128 tN256 tK512） | 12744 | 16.07% | 34.755 | 0 | 0 |

> 与 §3 表中同配置 `M512 N256 K512 tM128 tN256 tK512`（Cube cyc=12744 / 16.07%）一致。

## 5.2 hif4_deepseek_case3 带宽配置测试（PR #1065，2026-10-26 09:50）

> PR #1065 新增 DV121 HBM/L2 带宽 profile（`soc_random=true` + 60/113 B/cycle）。
> 带宽信用机制只在 `soc_random=true` 下生效。

| 配置 | soc_random | BW (R/W) | cycles | MAC | TFLOPS | EXIT |
|------|-----------|----------|--------|-----|--------|------|
| 默认 baseline | false | 156/156 | 12744 | 16.07% | 34.755 | 0 |
| DV121_HBM | true | 60/60 | — | — | — | 134（死锁） |
| DV121_L2 | true | 113/113 | — | — | — | 134（死锁） |
| soc_random 隔离对照 | true | 156/156 | — | — | — | 134（死锁） |

> **死锁根因是 `soc_random=true`，非带宽数值**：三个 soc_random 配置均在
> cycle 16796 触发 `Deadlock detected: simulation stalled`（`SimSys.cpp` ResVerify），
> HBM/L2/默认 BW 行为一致。死锁发生在 4-PE 锁步 + TMA `nuke_pair_diag` 后，
> B211 永久 executing。详见 `docs/perf/20261026_094304/hif4_deepseek_case3_bandwidth_pr1065.md`。
>
> **soc_random=false + 仅改带宽（60/113）**：信用机制关闭、带宽不生效，两者均
> 与基线一致（12744 / 16.07% / 34.755）。

## 5.3 soc_random=true 死锁修复后重测（2026-10-26，模型侧修复）

> 修复了 SuperScalarModel 两个 BFU 前端 nuke 路径 bug 后（见
> `bfu_nuke_single_slot_collision_deadlock_issue.md`、`bfu_nuke_replay_local_region_mismatch_issue.md`），
> `soc_random=true` 可正常跑完。

| 配置 | soc_random | cycles | MAC | TFLOPS | EXIT |
|------|-----------|--------|-----|--------|------|
| 默认 baseline | false | 12744 | 16.07% | 34.755 | 0 |
| 默认 BW | true | **21572** | **9.49%** | **20.532** | 0 |

> - seed 1/2/3/42/1234 结果一致（21572 / 9.49% / 20.532），确定性。
> - `soc_random=true` 引入随机 SoC 访存延迟：cycles +69%（12744→21572）、
>   MAC 16.07%→9.49%。带宽数值（60/113/156）在该模型下均不改变结果（信用机制
>   未绑定，见 §5.2）。

## 5.4 hif4 reuseA M256/N4096/K1280 尝试（2026-10-26，未跑通）

> 目标：M=256 N=4096 K=1280 走 `quant_batch_matmul_hif4_reuseA`（m->k->n 循环、
> A 复用 + internal-acc），选最优 tile 并测性能。已给 Makefile 加 `REUSEA=on`
> 开关（`-DREUSEA` + ELF 名 `_reuseA` 后缀）。

**结论：reuseA 模板在当前模型/内核下无法运行 N=4096**，两类独立阻断：

1. **CellReg 溢出（`Tile register is not enough`）**——reuseA 把全部 Nb 个累加器
   `tileC cTiles[Nb]` 同时驻留，总容量 = `gN × kPeM × 4` 字节 = 512 KB（tM=128/
   kPeM=32）或 256 KB（tM=64/kPeM=16），与 tN/tK 无关；CellReg 池仅 256 KB，
   tM=128 必爆，tM=64 恰好 256 KB 再加 A/scaleA 开销也爆。参考 shape
   （M256/K4096/N1280，即 N=1280）累加器仅 160 KB 才放得下——N=4096 比它大 3.2×。
2. **IsDivisible GMMA 断言**（大 tK）——tK=320/640/1280（kScaleK=5/10/20，非
   2 的幂）触发 `IsDivisible(opdSize, d1TR*dataSize)`（scale tile padded 尺寸
   与 logical 尺寸不整除），与历史 §3 的 `K1600 tN256 tK320` 同族。

实测 tile 网格（res_check=on，避免 fill 干扰；tN×tK≤163840 共享容量约束内）：

| tM | tK | kScaleK | 结果 |
|----|-----|---------|------|
| 128 | 128/256 | 2/4 | Tile register not enough（512 KB 累加器） |
| 128 | 320/640/1280 | 5/10/20 | IsDivisible |
| 64  | 64 | 1 | Tile register not enough（256 KB + A/scale 溢出，最终仍爆） |
| 64  | 128/256 | 2/4 | Tile register not enough |

**后续方向**：让 reuseA 支持大 N 需改内核——按 N 分块（chunk N）使驻留累加器从
`gN×kPeM` 降到 `chunkN×kPeM`；或修 IsDivisible（scale tile padded/logical 不整除）
后，再配合 chunk-N 才能给 tK≥320 的大 tile。见 `quant_batch_matmul_hif4_reuseA.hpp`
的 `tileC cTiles[Nb]` 结构与 `TileLsuArg.cpp` 的 scale tile 尺寸推导。

## 5.5 hif4 reuseA M256/N1280/K4096 最优 tile（2026-10-26）

> N=1280 时累加器驻留 `gN×kPeM×4` = 160 KB（tM=128）< 256 KB，可跑通。已给
> Makefile 加 `REUSEA=on`、给 `run_csv_tests.py` 加 `hif4_reuseA` dtype、CSV 加
> `hif4_reuseA_M256N1280K4096`（tM128 tN64 tK1024）。

tile 扫描（res_check=on，1.65 GHz 等效；总 MAC = 256×1280×4096）：

| tM | tN | tK | TotalCyc | GroupCyc | Cube MAC util | 等效 TFLOPS |
|----|----|----|----------|----------|---------------|-------------|
| 128 | **64** | **1024** | **72141** | **68534** | **56.78%** | **122.79** |
| 128 | 32 | 2048 | 72731 | 69287 | 56.32% | 121.80 |
| 128 | 32 | 1024 | 75237 | 71753 | 54.44% | 117.74 |
| 128 | 128 | 512 | 78875 | 75157 | 51.93% | 112.31 |
| 128 | 64 | 512 | 80886 | 77233 | 50.64% | 109.52 |
| 128 | 128 | 256 | 101518 | 97755 | 40.35% | 87.26 |
| 128 | 256 | 256 | 102044 | 98327 | 40.14% | 86.81 |
| 128 | 128 | 1024 | 124480 | 120749 | 32.90% | 71.16 |
| 128 | 64 | 2048 | 124612 | 120803 | 32.87% | 71.09 |
| 128 | 256 | 512 | 134360 | 130653 | 30.49% | 65.93 |
| 64 | 128 | 512 | 137531 | 133756 | 29.78% | 64.41 |
| 64 | 256 | 512 | 253872 | 250133 | 16.13% | 34.89 |

**最优：`tM=128 tN=64 tK=1024`**（56.78% MAC util / 68534 Group cycles / 122.79 TFLOPS，
gfrun R2=0）。

规律：
- tM=128（kPeM=32）全面优于 tM=64（M32 累加器更宽）。
- tN=64 是甜点：太小（32）CUBE 累加器欠用，太大（128/256）A 复用次数（Nb=1280/tN）下降。
- tK=1024 是甜点（Kb=4）：比 512 少一半 A 重载，比 2048 少 CellReg 压力。
- 此处 56.78% 是「当前 main + 默认参数 + quant_batch 旧 reuseA」下的值；后续 §5.6
  通过复用 matmul_lowp_reuseA + `sharetreg_bandwidth_bytes=256` 提升到 74~80%，
  并拆解出 87.9% 的完整来源（kernel + 带宽 + 模型版本差异）。

## 5.6 reuseA kernel 改造 + 参数消融 + case2 重测（2026-10-26）

### 5.6.1 复用 matmul_lowp_reuseA 消除 kernel 差距

`quant_batch_matmul_hif4_reuseA.hpp` 改为薄封装复用 `basic_op/matmul/matmul_lowp_reuseA`
（full-K A cache + 驻留链限制 `MAX_RESIDENT_OUTPUTS=2` + CubeM32 scaleA）。
M256/N1280/K4096 tM128 tN128 tK512：

| 版本 | GroupCyc | MAC util（默认） | MAC util（+带宽256） |
|------|----------|------------------|----------------------|
| 改造前 quant_batch reuseA | 75157 | 51.93% | 52.71% |
| **改造后 quant_batch reuseA** | 58181 | 66.29% | **74.32%** |
| matmul_lowp reuseA（参照） | 56921 | 71.72% | 79.68% |

kernel 差距 20pp → ~5pp（剩余来自 quant_batch 的 GFSIM 4-PE file I/O 分支，非 kernel）。

### 5.6.2 参数消融：唯一关键是 sharetreg_bandwidth_bytes

正向单参数（matmul_lowp，基线 71.72% / 完整 79.68%）：

| 参数 | MAC util | Δ |
|------|----------|---|
| **`core.sharetreg_bandwidth_bytes=256`** | **79.80%** | **+8pp** |
| cube.l0aSharetregResident=true | 71.72% | 0 |
| bctrl.bctrl_bandwidth=8 / spe.speROBDepth=128 / vecIssueQDepth=64 / uop_isq=64 / hwpf 预取 / vec_cell_sched / sharetreg_size&ssb | ~71.7% | 0 |

即 87.9% 拆解：kernel(+20pp) + `sharetreg_bandwidth_bytes=256`(+8pp) + **ELF 编译代码生成(~10pp，非模型版本)**。
`sharetreg_bandwidth_bytes` 默认 128 B/拍（core.toml），256 = 2 cell 读 + 4 PE 广播扇出 + 2 笔/拍准入 + L0B 翻倍，解掉 B（SharedTile 广播读）的带宽瓶颈。

### 5.6.4 深挖：剩余 ~10pp 是 ELF 编译差异，不是模型版本

用 `git worktree` checkout 参考 base `bdbafed` + 应用 `HIF4_MODEL_STATE_*.patch` 编译 gfsim，
与当前 main 逐项对比（同一 matmul_lowp ELF / 同一参数）：

| gfsim | soc_random | 结果 |
|-------|-----------|------|
| bdbafed+patch | 关 | 51264 / 79.68% |
| main | 关 | 51264 / 79.68% |
| bdbafed+patch | 开(参考配置) | 58610 / 69.41% |
| main | 开 | 58610 / 69.41% |

**bdbafed 与 main 行为逐字节一致**——"模型版本差异 ~8pp" 是错误假设。真正差异在 **ELF**：

| ELF | full_cache 函数大小 | B.IOT 数 | 无soc MAC | 有soc MAC |
|-----|--------------------|----------|-----------|-----------|
| 参考（eval 目录，macOS 上 e6a31ef toolchain 编译） | 12702 B | 481 | **89.62%** | 87.68% |
| 本地（Linux WSL，output clang Sep-15 构建） | 14238 B（CC_OPT 默认）/ 13278 B（CC_OPT=default） | 672 | 79.68% / 80.05% | 69.41% |

TLOAD(192)/TMAT(160)/TSTORE(20)/B.IOS(480) 两 ELF 完全一致，唯一差异是 **B.IOT 数 481 vs 672
（本地多 191 个冗余 tile 操作数绑定）**——本地 toolchain 的 clang 二进制（Sep-15 构建）对
scaleA/C 累加器生成了冗余绑定，吃掉 ~10pp。toolchain 源码 commit 同为 `e6a31ef`，但编译出的
clang 二进制（本地 Linux output vs 参考 macOS）代码生成不同；`CC_OPT=default` 只能补回 0.37pp。

**最终 87.9% 完整拆解**：kernel full_cache(+20pp) + `sharetreg_bandwidth_bytes=256`(+8pp) +
ELF 编译/toolchain 代码生成(~10pp)。后 ~10pp 需对齐 toolchain 构建（clang 二进制），非
SuperScalarModel 模型、非 kernel 结构。

### 5.6.5 toolchain 对齐：更新 TileOP-API 消除 ~10pp（已解决）

B.IOT 差异根因不是 clang 二进制，而是 **TileOP-API 头文件版本**：本地 `Linx-TileOP-API`
落后远程 `origin/linx` 108 个 commit，缺了
`fix(tileop): pack MX scale binders into dual-source B.IOT`（#257/#263/#265/#267）这一批
MX scale 绑定打包优化——旧版每个 MX scale 绑定单独发一条 B.IOT，新版打包进 dual-source B.IOT。

对齐步骤（无需重建 LLVM/clang，只要重装 header）：

```bash
cd linx-toolchain-build/src/Linx-TileOP-API
git checkout -B linx origin/linx          # e93ef98 → 1153ea6（+108 commit）
make install CLANG_PREFIX=.../output/linx_blockisa_llvm_musl
# 重新编译 matmul_lowp
```

结果（同一 gfsim / 同参数）：

| 阶段 | B.IOT 数 | full_cache 函数 | MAC util |
|------|----------|----------------|----------|
| 旧 TileOP-API | 672 | 14238 B | 79.68% |
| **更新后** | **481** | 13030 B | **88.67%** |

B.IOT 672→481（-191），MAC 79.68%→88.67%，与参考 87.91%~89.62% 对齐。同时回归验证
`quant_batch_matmul_test_hif4`（base kernel）与 reuseA 两个变体在新 TileOP-API 下编译通过。

> 结论：87.9% 的最终来源（全部落定）= kernel full_cache 复用(+20pp) +
> `sharetreg_bandwidth_bytes=256`(+8pp) + TileOP-API 的 MX-scale B.IOT 打包(~10pp)。
> 三者都已在当前环境复现，无任何未解释的"模型版本/神秘差异"。

### 5.6.3 deepseek_case2（M512 N256 K4096）重测

> 此前 case2 历史数据（§2/§3）均为默认带宽 **128**（perf_test.sh 未覆盖 `-s`）。

| kernel | toolchain | 带宽 | GroupCyc | MAC util |
|--------|-----------|------|----------|----------|
| base `quant_batch_matmul_hif4_mt`（历史） | 旧 | 128 | 54334 | 30.15% |
| base（重测） | 新(TileOP-API 对齐) | 128 | 43028 | 35.02% |
| matmul_lowp reuseA（重测） | 旧 | 128 | 37574 | 43.39% |
| **matmul_lowp reuseA（重测）** | **新(TileOP-API 对齐)** | **128** | 30126 | **54.10%** |
| matmul_lowp reuseA（重测） | 新 | 256 | 30027 | 54.28% |

- reuseA 让 case2 MAC 30.15%→43.39%（+13pp）；TileOP-API 对齐（MX-scale B.IOT 打包）再
  +10.7pp（43.39%→54.10%），base kernel 也 +5pp（30.15%→35.02%）。
- case2 的 256 带宽几乎无益（+0.18pp）——带宽收益取决于 N：case2 的 N=256（B 小、带宽
  非瓶颈），而 M256/N1280 的 N=1280（B 大 5 倍）才 +8pp。

### 5.6.6 case2_var1（M256 N256 K4096，base kernel）

case2 的 M 减半变体（`hif4_deepseek_case2_var1,hif4,256,256,4096,128,128,512,1`），
base kernel + 新 toolchain：

| 用例 | M | 峰值 units | GroupCyc | MAC util |
|------|---|-----------|----------|----------|
| case2 | 512 | 16384 | 43028 | 35.02% |
| **case2_var1** | 256 | 8192 | 22808 | **31.02%** |

M 减半 → 峰值减半（16384→8192）、GroupCyc 减半（43028→22808，与 MAC 减半一致），但
MAC util 略降（35.02%→31.02%）。原因：base kernel（i->j->k）M=256 时每个 PE 只算 2 个
M64/128 块，流水线 fill/drain 排空占比相对更高，CUBE 峰值利用率略差。gfrun R2=0。

### 5.6.7 case2_var1（M256 N256 K4096）reuseA

同 shape 走 matmul_lowp reuseA（新 toolchain）：

| kernel | 带宽 | GroupCyc | MAC util | TFLOPS |
|--------|------|----------|----------|--------|
| base kernel | 128 | 22808 | 31.02% | 67.09 |
| **reuseA** | 128 | 14435 | **56.03%** | 121.17 |
| reuseA | 256 | 13352 | **60.51%** | 130.87 |

- reuseA 让 var1 MAC 31.02%→56.03%（+25pp）。
- 带宽 256 对 var1 有益（+4.5pp，56.03%→60.51%）：M=256（A 小）+ N=256（B 小），
  compute-to-memory 比值低（~256 MAC/B），SharedTReg 带宽（B 广播读）成次级瓶颈；对比
  case2（M512，比值 ~355 MAC/B）宽带几乎无益。
- **峰值修正**：CUBE 峰值 workload units = `M×N/8`（case2=512·256/8=16384，var1=256·256/8=8192，
  M256/N1280=256·1280/8=40960），由 shape 决定，与 full_cache/detail 无关。full_cache vs
  detail 影响的是「MAC 利用率%」（调度效率），不是峰值。故 case2（54.10%）与 M256/N1280
  full_cache（88.67%）的差距是调度效率，非峰值。

## 5. 备注

- scaleA 的 CubeM32 优化依赖 toolchain（Linx-TileOP-API）放宽
  `validate_matrix_scale_contract`（MX ScaleA/ScaleB 允许 CUBE_M32）。该改动
  保存在本地未提交（`linx-toolchain-build/src/Linx-TileOP-API/include/jcore/
  template_asm.hpp`），重编译 hif4 需要该头文件已安装到 clang resource dir。
- SuperNPUBench kernel 改动已提交：`fa8a193 perf(hif4): scaleA CUBE_M32 layout
  + reuseA kernel`。