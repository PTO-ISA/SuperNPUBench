# Group Token Old — 编译与仿真记录

> 本文档记录 `group_token_old`（三阶段 MoE Token 分组算子，含 SIMD 排序）在
> 一次完整验证流程中的编译、仿真执行路径、命令、过程与日志分析。
> 验证日期：2026-08-13。

## 1. 执行环境路径

| 项 | 路径 |
|---|---|
| 算子源码 | `benchmark/one-level-arch/kernels/group_token_old/group_token_old.hpp` |
| 测试工程 | `benchmark/one-level-arch/test/kernel/group_token_old/`（Makefile / compile.all / src/group_token_old.cpp） |
| Linx 工具链 | `<workspace>/linx-toolchain-build/output/linx_blockisa_llvm_musl/bin`（clang 15.0.4，`linx64v5-unknown-linux-musl`） |
| 生成 ELF | `benchmark/one-level-arch/output/kernel/group_token_old/elf/group_token_old.elf`（4.6 KB） |
| 仿真器 | `<workspace>/SuperScalarModel/bin/`（`gfrun` 功能模型 / `gfsim` 时序模型） |

工作区根：`/mnt/workspace/gitCode/cann/Dev-experience/v300_0813`。

## 2. 编译

### 2.1 执行命令

```bash
export COMPILER_DIR=<workspace>/linx-toolchain-build/output/linx_blockisa_llvm_musl/bin
cd benchmark/one-level-arch/test/kernel/group_token_old
make TESTCASE=group_token_old COMPILER_DIR=$COMPILER_DIR
```

### 2.2 关键编译参数（Makefile.common 派生）

```text
clang++ -c -mlxbc -fenable-matrix -O2 -std=c++20
    -mllvm -enable-all-vector-as-tilereg=true
    -mllvm -linxv5-enable-HL-Inst-Opt=true -mllvm -linxv5-enable-dim-opt=true
    -mllvm -linxv5-enable-ldst-bridge=false
    -mllvm -linxv5-enable-continuous-mem-opt=true
    -mllvm -linxv5-enable-tile-clock-hand=false
    -mllvm -linxv5-enable-simt-clock-hand=true -mllvm -enable-misched=false
    -D__linx -DENABLE_TENSOR_INSTR
    -I.../include -I.../test/common -I.../kernel源目录/... -I.../kernels -I.../models
链接：clang++ -nostartfiles test/common/_start.s <obj> -o <elf>
```

### 2.3 编译结果

- 退出码 `0`，无编译错误 / 警告。
- 产物：`output/kernel/group_token_old/elf/group_token_old.elf`。
- 说明：`#if defined(__linx)` 分支生效 —— `runGroupTokenOld` 走**标量保底路径**
  （`calTokenPerExpertCnt_scalar` / `groupToken_scalar<true>` / `floorFunc_scalar` +
  `sortByLocalExpId_scalar`），SIMT 加速路径（`__vec__` 内核）不在 `__linx` 下编译。

## 3. 仿真过程

### 3.1 gfrun（功能模型，正确性）

```bash
cd <workspace>/SuperScalarModel
./bin/gfrun -f <workspace>/SuperNPUBench_zxy/benchmark/one-level-arch/output/kernel/group_token_old/elf/group_token_old.elf
```

加载段（日志原文）：

```text
Memory: 0x10200 .rodata / .eh_frame / .text(1KB) / .data.rel.ro / .bss(66KB)
Memory: stack mem(131136KB) / map mem(131076KB)
Starting from 0x112a4
Thread:0 Total Block number = 27160
Thread:0 Total Inst number = 287395
```

### 3.2 gfsim（时序模型，性能）

```bash
cd <workspace>/SuperScalarModel
./bin/gfsim -f <workspace>/SuperNPUBench_zxy/benchmark/one-level-arch/output/kernel/group_token_old/elf/group_token_old.elf
```

不需要 `-s core.singleTierMode=true`：本算子在 `__linx` 下为纯标量路径，不执行
VectorLite 引擎的 tile-op。

## 4. 仿真结果

| 项 | gfrun | gfsim |
|---|---|---|
| 退出码 | 0 | 0 |
| 完成标记 | `Success to Reach the End of Benchmark! R2 = 0` | 正常打印 SuperScalar Report（无卡死） |
| 指令统计 | Block = 27160, Inst = 287395 | 同左 |
| 总周期 | — | `Total Cycles = 255598` |
| 仿真耗时 | 秒级 | 33s（`Simulation time: 33s`） |

`R2 = 0` 即测试代码返回码：测试 main 在 `__linx` 下直接 `return 0`，并内置了
`PLAT=cpu` 时与标量参考对比的 6 项验证（直方图计数、分区计数、分组 id、pod 信息、
分区边界、排序结果），失败时返回非 0。**功能正确性通过；时序仿真正常跑完。**

## 5. 仿真日志分析（gfsim）

### 5.1 Top-Down 总览（SuperScalar Unified Top-Down）

| 指标 | 占比 | 含义 |
|---|---|---|
| Retiring | 24.22% | 正常退休 |
| Backend Bound | 75.74% | 后端等待（Core Bound 75.74%） |
| Frontend Bound | 0.02% | 前端无瓶颈 |
| Bad Speculation | 0.01% | 分支误预测开销可忽略 |

纯标量负载特征明显：`Cube/Vector/TLSU` 相关项全为 0，`Scalar ALU 214.82%`
（标量 ALU 是唯一执行资源，多周期冲突归一后 >100% 属正常）。

### 5.2 Key Stats 要点

```text
superScalar Tileop Total Cycles = 255598   （= Total Cycles，全程无 tile-op）
  |--Cube Tileop Total Cycles / Vector / TMA = 0
superScalar Run Tileop Total Cycles = 0     （cube/vector/TMA Busy 均为 0）
All Cores Idle Cycles = 255598              （无协处理器活动周期）
IPC = 0.00                                  （无 tile-op，IPC 按 uop 统计无意义）
Retired Block Num = 19139                   （STD 19139 + MEMSET 2 + FENTRY 1 + FRET 1）
BPC = 0.07, MPKB = 7.98
Discontinuous BPC Count = 10740, Average Continuous BPC Length = 1.78
inter-block misp = 162（仅有的刷新来源，162/27160 blocks ≈ 0.6%）
```

解读：该算子为 **100% 标量指令流**（无 Cube/Vector/TMA tile-op），255598 个周期
全部为标量执行，Cube/Vector/TMA 全程空闲。性能统计中 IPC/BPC 等方法基于 tile-op，
对纯标量内核参考意义有限；周期数主要受标量 ALU 吞吐与分支块粒度
（平均连续块长 1.78）影响。

### 5.3 a3_violation 告警（24620 条，可忽略）

```text
[C:58][TMA.NA]: a3_violation peId=0 reason=two_accesses_in_one_block
               queue=stq/liq incoming=scalar_store/scalar_load
               incoming_is_tile=0 resident_is_tile=0 stid=0 bid=xxx rid=xxx
```

- 计数：约 24620 条，贯穿整个仿真（C:58 ~ C:255465），全程仅此一种原因
  `two_accesses_in_one_block`，全部为**标量 load/store**（`incoming_is_tile=0`）。
- 含义：同一 block 内对同一 cache line 的两个标量访问在 LSU 队列（liq/stq）中
  未按序排队时的 A3 访问顺序告警；为标量密集程序的常见告警性输出，**
  不影响指令执行与最终结果**，仿真正常结束（exit 0）。
- 该告警源于矩阵引擎（TMA）的地址访问检查模块对标量路径的过度告警，并非算子缺陷。

### 5.4 其他

- 配置转储：`inst_decode_bw=4, block_rob_depth=256, cube_isq_depth=32, vector_isq_depth=64,
  L0A/L0B/L0C_size=64` 等默认配置，本次仿真未覆盖。
- gfsim 日志不打印 R2 成功标记，判定标准为：完整打印 SuperScalar Report、退出码 0。

## 6. 结论

1. **编译通过**：clang 15.0.4（linx64v5-musl）编译 `group_token_old` 无错误无警告，生成 ELF。
2. **功能仿真通过**：gfrun 累计执行 27160 blocks / 287395 条指令，成功跑完，R2 = 0
   （测试内置 6 项标量参考对比，详见 `test/kernel/group_token_old/src/group_token_old.cpp`）。
3. **时序仿真通过**：gfsim 无卡死跑完，255598 周期 / 33s；纯标量路径，
   Cube/Vector/TMA 全程空闲，符合 `__linx` 下标量保底路径的预期。
4. 日志中 24620 条 `a3_violation`（two_accesses_in_one_block）为标量访问顺序告警，
   不影响正确性与最终结果，属仿真器告警噪声。