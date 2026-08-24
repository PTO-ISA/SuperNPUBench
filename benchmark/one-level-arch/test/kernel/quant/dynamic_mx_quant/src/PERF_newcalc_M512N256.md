# probe_dynamic_mx_quant_tail_ocp_fp8_newcalc 性能分析（M=512 / N=256 / BlockSize=32）

数据来源：`gfsim` 打屏输出（完整日志 `/tmp/gfsim_full.log`）。
被测 ELF：`dynamic_mx_quant_probe_ocp_fp8_newcalc.elf`（PM=512, PN=256, BlockSize=32）。

本文第一部分**逐行**解释 gfsim 打屏的每一项（含所有子项，即使本例为 0 也说明它代表什么）；
第二部分是本 shape 的实测分析。

---

# 第一部分：打屏每一项指标的含义（逐行全覆盖）

## 0. 配置表（Config，仿真开始前打印的参数）

| 项 | 本例值 | 含义 |
|---|---|---|
| `inst_decode_bw` | 4 | 译码带宽：每周期最多译码 4 个 block header。 |
| `block_rob_depth` | 256 | 块重排序缓冲（BROB）容量 = 256 个块条目。后端能容纳的最大在飞块数上限。 |
| `BISQ:cube_isq_depth` | 32 | Cube 引擎块发射队列（Issue Queue）深度。 |
| `BISQ:vector_isq_depth` | 64 | Vector 引擎块发射队列深度。 |
| `CUBE uop_isq_depth` | 32 | Cube 内部 uop 发射队列深度。 |
| `CUBE tcvt_isq_depth` | 32 | Cube 内 TCVT（类型转换）uop 队列深度。 |
| `CUBE L0A_size / L0B_size / L0C_size` | 64 | Cube 本地缓冲 L0A（左矩阵）/L0B（右矩阵）/L0C（累加器）各 64 条目。 |

## 1. SuperScalar NPU Stats

| 项 | 含义 |
|---|---|
| **Total Cycles** | 端到端总仿真周期数（时钟拍），从首指令到末指令退休。端到端时延的硬指标。 |
| **Sim Total Cycles** | 仿真器实际推进的周期数；正常与 Total Cycles 相等。 |

## 2. Unified Top-Down（自顶向下瓶颈分解）

顶层四类互斥、相加≈100%，衡量「每个发射槽位浪费在哪」。子项是对四大类的进一步归因。

### 顶层四类

| 项 | 含义 |
|---|---|
| **Retiring** | 槽位被有效退休指令占用的比例，越高越好。 |
| **Bad Speculation** | 因预测错/机器清空作废的槽位。 |
| **Frontend Bound** | 前端（取指/译码）供不上指令的空槽。 |
| **Backend Bound** | 后端（执行/访存资源）跟不上的空槽。 |

### Retiring 细分

| 项 | 含义 |
|---|---|
| **LD / ST / BR / ALU / Other Retiring** | 退休槽位按指令类型拆分：加载 / 存储 / 分支 / 算术逻辑 / 其它 各自占比。 |

### Bad Speculation 细分

| 项 | 含义 |
|---|---|
| **Misprediction** | 分支预测错误浪费的槽位。 |
| **Machine Clear** | 因内存序违例/异常触发整条流水线清空浪费的槽位。 |

### Frontend Bound 细分

| 项 | 含义 |
|---|---|
| **Fetch Latency** | 取指延迟型前端空槽（取指流被打断，如 miss）。 |
| **Fetch Bandwidth** | 取指带宽不足型空槽（取指流连续但喂不满）。 |
| **Branch Resteering** | 分支解析后重定向取指的开销。 |
| **ITLB/BTLB miss** | 指令 TLB / 块 TLB 缺失惩罚。 |
| **BHC miss penalty** | 块历史缓存（Block History Cache）缺失惩罚。注：本项分母是事件次数，事件极少时会显示成离谱百分比（本例 52449%），**属该子项自身归一化产物，不影响顶层四类占比**，忽略即可。 |
| **Other fetch lat** | 其它取指延迟。 |
| **Decode Stall** | 译码级阻塞。 |
| **Rename Stall** | 改名级阻塞。 |
| **BROB Stall** | 因块 ROB 满导致前端无法继续分配的阻塞（顶到 BROB 容量）。 |
| **SYS Block Stall** | 系统块导致的阻塞。 |

### Backend Bound 细分

| 项 | 含义 |
|---|---|
| **Memory Bound** | 后端阻塞中因等访存的部分。 |
| **Core Bound** | 后端阻塞中因等计算单元的部分。 |
| **L1 / L2 / L3/DRAM Bound** | 访存受限进一步按命中层级细分。 |
| **Store Bound** | 因存储队列/写通道满导致的阻塞。 |
| **Scalar ALU** | 等标量 ALU 的压力（各执行域独立累计的占用率，可 >100%，非互斥切分）。 |
| **Cube / Vector** | 等 Cube / 等 Vector 引擎的压力。 |
| **TLSU Tload / Tstore / TMOV** | 等张量访存单元 加载 / 存储 / 片内搬运 的压力。 |
| **Idle** | 后端无事可做的空转。 |
| **Other Core** | 其它核心侧阻塞，细分：**Simple ALU pipe full**（简单 ALU 流水满）/ **Complex INT busy**（复杂整数单元忙）/ **Branch port busy**（分支端口忙）。 |

## 3. superScalar Key Stats（Tileop 引擎级）

| 项 | 含义 |
|---|---|
| **superScalar Tileop Total Cycles** | 有 tileop 在流水线窗口内的总周期。 |
| **Cube / Vector / TMA Tileop Total Cycles** | 三引擎各自累计占用周期（排队+执行贡献），看热点引擎。 |
| **superScalar Total Tileop Counter** | 退休 tileop 总条数。 |
| **Cube / Vector / TMA Execute Tileop Counter** | 各引擎执行的 tileop 条数。 |
| **superScalar Run Tileop Total Cycles** | 至少有一个引擎在真正执行 tileop 的周期（去掉全空转）。 |
| **Cube Busy Cycles** | Cube 真正在算的忙拍。 |
| **Vector Busy Cycles** | Vector 忙拍；按 uop×lane 累计（内部并行度展开），故可远大于 Total Cycles。 |
| **TMA Busy Cycles** | ⚠️ core 级 `CountStats` 每周期采样 `tileLsu->IsBusy()`（载入侧 BridgePairQ 非空）得到的执行单元瞬时占用。**本配置恒读 0，不代表 TMA 免费**——TMA 实际在 BCC/group 层记账。判 TMA 开销请看下方 §3 的 **TMA Tileop Total Cycles** 与 §5 的 **Average Outstanding TLoad/TStore**。细分（Tload/Tstore/…）同样恒 0。 |
| **Tload Busy Cycles** | TLOAD 忙拍。 |
| **write Blocked By Ring Cycles** | TLOAD 写目标被环形缓冲（Ring）反压阻塞的周期。 |
| **Tstore Busy Cycles** | TSTORE 忙拍。 |
| **Tmov Busy Cycles** | 片内搬运 TMOV 忙拍。 |
| **Mgather / Mscatter Busy Cycles** | 内存 gather / scatter 忙拍。 |
| **Vgather / Vscatter Busy Cycles** | 向量 gather / scatter 忙拍。 |
| **All Cores Idle Cycles** | 所有核心都无 tileop 可跑的空转周期。 |
| **Vector-Cube Cocurrent Cycles** | Vector 与 Cube 同一拍都在忙的重叠周期（并行度，越高越好）。 |
| **Vector-Cube Active Cycles** | Vector 与 Cube 至少一个在忙的周期（并集）。 |
| **Cube-TMA Cocurrent / Active Cycles** | Cube 与 TMA 的重叠 / 并集周期。 |
| **Vector-TMA Cocurrent / Active Cycles** | Vector 与 TMA 的重叠 / 并集周期。Concurrent÷Active = 重叠率，为 0 = 完全串行。 |

## 4. PROB Status Occupancy（块在各状态累计停留周期）

块调度记分板：每个块从分配到退休经过一系列状态，这里是**所有块在各状态累计停留周期**（累加量，非占比）。哪个状态数值大，块就主要卡在哪。

| 状态 | 含义 |
|---|---|
| **free** | 记分板条目空闲。 |
| **alloc** | 已分配、等改名/资源（数值大 = 大量块卡在入口排队）。 |
| **renamed** | 已改名，等派发。 |
| **dispatched** | 已派发到发射队列。 |
| **in_queue** | 在发射队列中等待就绪。 |
| **issued** | 已发射，等执行资源。 |
| **executing** | 正在执行。 |
| **writeback** | 结果写回中。 |
| **completed** | 已完成，等按序退休（数值大 = 完成后长时间等前面的块退休）。 |
| **retired** | 已退休。 |
| **fault** | 出错块，正常为 0。 |
| **need_flush** | 需清空块，正常为 0。 |

## 5. BCC Key Stats — Average Outstanding Block（在飞块数）

（total 与 SThread 0 两份，单线程下相同）

| 项 | 含义 |
|---|---|
| **Average Outstanding Block Number** | 平均同时在飞（已发射未退休）的块数，衡量实际块级并行度。 |
| **Average Outstanding Cube / Vector Block Number** | 按引擎细分的在飞块数。 |
| **Average Outstanding TLoad / TStore Number** | 平均在飞的加载 / 存储块数。 |

## 6. CubeCore 0 Statistics（Cube 引擎细粒度；本 kernel 不用 Cube，全 0/nan/inf）

| 项 | 含义 |
|---|---|
| **Cube core TFLOPS(1.65GHz)** | Cube 在 1.65GHz 下的实测吞吐（万亿次浮点/秒）。 |
| **Cube core alu utilization** | Cube ALU 利用率。 |
| **Cube core utilization(Run TileOp, header)** | 按 tileop header 计的 Cube 占用率。 |
| **Cube core L0CWR ratio** | L0C（累加器）写回比率。 |
| **Cube core load wait ratio** | Cube 等加载的比率。 |
| **Cube core load port data channel utilization** | 加载端口数据通道利用率。 |
| **Cube core uop load average used cycle** | 每 uop 加载平均耗周期。 |
| **Cube core load req average used cycle** | 每加载请求平均耗周期。 |
| **Cube core exe tileop count** | Cube 执行的 tileop 数。 |
| **Cube core average acc slot use** | 累加器槽平均占用。 |
| **Cube core average l0a / l0b entry use** | L0A / L0B 条目平均占用（`busy` 后缀 = 仅统计忙期内）。 |
| **Cube core exe uop count** | Cube 执行的 uop 数。 |
| **Cube core total load / store count** | Cube 总加载 / 存储次数。 |
| **Cube core cannot pick main chain uop ratio** | 无法挑选主链 uop 的比率（调度饥饿）。 |
| **Cube core pick cub chain uop ratio** | 挑到 cube 链 uop 的比率。 |
| **Cube busy cycle** | Cube 忙周期。 |
| **Cube core mmad / load / store busy cycle** | 矩阵乘加 / 加载 / 存储 各忙周期。 |
| **Cube core tileop mmad / load / store cycle range** | 各类 tileop 周期跨度。 |
| **Cube core load and mmad cycle overlap** | 加载与矩阵乘加的重叠周期（Cube 内部流水度）。 |

## 7. Retired block Type（退休块类型分布）

| 项 | 含义 |
|---|---|
| **Retired Block Num** | 退休块总数。 |
| **Retired STD Block Num** | 标量/控制（STD）块数。 |
| **Retired PARALLEL Block Num** | 并行张量块总数，细分 CUBE / VCALL（Vector 调用）/ TMA 及各自占比。 |
| **Retired SYS Block Num** | 系统块数。 |
| **Retired FP Block Num** | 浮点（标量 FP）块数。 |
| **Retired TEMPLATE Block Num** | 框架模板块，细分：**MEMSET / MEMCOPY**（初始化/拷贝模板）、**FENTRY / FEXIT / FRET**（函数入口/出口/返回帧）。 |
| **Retired OTHER Block Num** | 其它块。 |
| **Retired Inst Number** | 退休指令数（此口径下 tileop 探针可能为 0），细分 Cube / Vec Inst。 |
| **BPC** | 每周期退休的块数（Blocks Per Cycle）。 |
| **IPC** | 每周期退休指令数（此口径对纯 tileop 可能为 0，看 Vector PMU 的 IPC(UOP) 更准）。 |
| **Average BROB depth** | 块 ROB 平均占用深度（越接近容量 256 越说明后端堵）。 |
| **Effective window size** | 有效乱序窗口大小（实际能并行考察的指令窗）。 |
| **MPKB** | 每千块错误预测数（Mispredicts Per Kilo-Block）。 |
| **MPKI** | 每千指令错误预测数。 |
| **Simulation speed (Kilo Insts Per Second)** | 仿真器本身跑多快（工程指标，与被测 kernel 性能无关）。 |
| **Discontinuous BPC Count** | 取指不连续（跳转打断）次数。 |
| **Average Continuous BPC Length** | 平均连续取指长度（两次打断间的块数）。 |
| **Average fetch minsts/header per cycle** | 每周期取到的微指令/header 数。 |

## 8. BCC Detail Stats（Tile rename）— tile 寄存器改名与利用率

寄存器堆类别缩写：**T**=通用 tile、**U**=（第二类）通用 tile、**M/N**=矩阵左/右源、**ACC**=累加器、**STK**=栈、**DEP**=依赖、**MCALL_GPR/MCALL_STK**=函数调用相关。

| 项 | 含义 |
|---|---|
| **Tile register average tag num** | 平均使用的物理寄存器 tag 数（改名压力），按 T/U/M/N/ACC/STK/DEP/MCALL_* 细分。 |
| **Tile register average utilization(B)** | 平均占用字节数，按类别细分并给占比。 |
| **Tile register max size of T/U/M/N** | 各堆的容量上限（字节），`utilization` 子行 = 峰值占用字节与占比。 |
| **T/U/M/N/ACC/STK/DEP/MCALL_* MAPQ average occupied size** | 各类改名映射队列（MAPQ）平均占用；反映改名管线排队压力。 |

## 9. BCC Stall Stats（后端阻塞原因；total 与 SThread 0 两份）

| 项 | 含义 |
|---|---|
| **BRob Full Stall** | 因块 ROB 满而阻塞的周期数（后端堵的首要信号）。 |
| **RENAME Stall** | 改名阶段阻塞。 |
| **Block Rename Get / Set Stall** | 改名读取 / 写入映射时的阻塞。 |
| **BIsq Full Stall** | 因块发射队列满而阻塞的周期数。 |
| **TileReg Full Stall** | 因 tile 寄存器耗尽而阻塞（为 0 = 寄存器容量非瓶颈）。 |

## 10. BCC Detail Stats（Flush Statistics）— 清空来源

| 项 | 含义 |
|---|---|
| **intra-block ld-st conflict** | 块内加载-存储冲突触发的清空。 |
| **inter-block ld-st conflict** | 跨块加载-存储冲突触发的清空。 |
| **intra-block misp** | 块内分支误预测触发的清空。 |
| **inter-block misp** | 跨块分支误预测触发的清空。 |

## 11. SPE OOO & IEX stats（标量乱序/整数执行，按线程）

| 项 | 含义 |
|---|---|
| **decode inst bubble** | 译码气泡（无指令可译码）周期。 |
| **average decode bundle** | 平均每次译码打包的指令数。 |
| **Thread0-3 Pick Cnt** | 各线程被调度器选中派发的次数（只有 Thread0 非 0 = 单线程）。 |
| **Thread0-3 rob stall cnt** | 各线程因 ROB 满停顿次数。 |
| **decode pe rob stall** | 因 PE 侧 ROB 满导致译码停顿。 |
| **decode local gpr stall** | 因本地通用寄存器不足停顿。 |
| **inst buffer stall** | 指令缓冲满停顿。 |
| **predicate reg stall** | 谓词寄存器不足停顿。 |
| **vector reg ptag stall** | 向量寄存器物理 tag 不足停顿。 |
| **vector local scalar reg stall** | 向量本地标量寄存器不足停顿。 |
| **retired ld / st / scalar / vector / div-sqrt inst** | 退休的 加载/存储/标量/向量/除法开方 指令数。 |
| **MAPQ lg0 stall** | 改名映射队列（>0 占用）导致的停顿。 |
| **ROB lg0 stall** | ROB 导致的停顿。 |
| **Thread0 decode / alloc / rename cycle** | Thread0 在译码 / 分配 / 改名各阶段活跃周期。 |
| **Thread0 d2_s1 dispatch cnt** | 从译码-2 到发射-1 的派发计数。 |
| **Thread0 slots_total** | 总发射槽数（Top-Down 分母来源）。 |
| **Thread0 slots_rob_allocated** | 已被 ROB 分配的槽数。 |
| **Thread0 slots_fe_bubble** | 前端气泡槽数。 |
| **Thread0 slots_idle** | 空闲槽数。 |
| **Thread0 robStall** | Thread0 ROB 停顿累计。 |
| **Thread0 retired_load / store / innerJump** | 退休的加载/存储/内跳转数。 |
| **Thread0 retiredAluInst / retiredSAluInst / retiredVAluInst** | 退休 ALU / 标量 ALU / 向量 ALU 指令数。 |
| **Thread0 retiredScalar / retiredVector / retiredDivSqrt** | 退休标量 / 向量 / 除开方指令数。 |

## 12. Vector PMU Stats（向量引擎细粒度）

| 项 | 含义 |
|---|---|
| **TileopNum** | 向量引擎执行的 tileop 条数。 |
| **UopNum** | 这些 tileop 展开成的微操作（uop）总数（一条 tileop 按 tile 行/列展开成多个 uop）。 |
| **IPC(UOP)** | 每周期完成 uop 数 = UopNum / 有效周期，向量吞吐的真实度量（1.0 = 每拍一个 uop）。 |
| **RowExpand\*** 族 | 行广播乘（如 TROWEXPANDMUL / TROWEXPAND）内部计数：`RowExpandTileNum`（走该路径的 tile 数）、`UopSplit`（拆成的 uop 数）、`Src0CellRegReadNum`/`BroadcastCellRegReadNum`（源 0 / 广播源读 cell 次数）、`BroadcastBufferHitNum`（广播缓存命中）、`BroadcastOwnerDeliveryNum`/`BroadcastCacheFillNum`（广播投递/填充）、`BroadcastWaitCycles`/`ReadPortWaitCycles`/`SourceResponseWaitUopCycles`（等广播/读口/源响应的周期）、`CompletedTileNum`/`FlushedTileNum`/`StaleResponseNum`（完成/被清/过期响应 tile 数）。本例全 0 = 未走该内部路径统计。 |
| **RowReduce\*** 族 | 行归约（如 TROWMAX/TCOLMAX 的归约实现）内部计数：`TileNum`/`LayerNum`（归约 tile 数/分层数）、`UopSplit`/`UopDone`、`RootWriteNum`（根写次数）、`FlushedTileNum`、以及一系列 `*WaitTileCycles`/`*WaitUopCycles`（条目不可用/带宽/源缓冲满/层等待/读口/源响应/发射间隔/同组/执行流水满 各类等待周期）、`RootDstBufWaitTileCycles`/`RootWritePortWaitCycles`/`RootWriteResponseWaitCycles`（根目标缓冲/写口/写响应等待）、`CellBufPeak`（cell 缓冲峰值）。本例全 0。 |
| **ReadNum** | 读 cell 寄存器总次数。 |
| **ReadLat** | 读 cell 平均延迟（拍）。 |
| **LatLE5 / LE10 / LE15 / GT15** | 读延迟直方图分桶（≤5 / ≤10 / ≤15 / >15 拍各多少次）。 |
| **WriteNum / WriteLat + LatLE5/10/15/GT15** | 写 cell 的总次数、平均延迟与延迟直方图。 |

---

# 第二部分：M=512 / N=256 / BlockSize=32 实测分析

## 2.1 工作量与拆分

- 元素总数：512 × 256 = **131072** 个 half。
- Tiling：`TileM = clamp(M, 8, kTilemMax=64) = 64`（fp32 预算下上限 64），`full_m = 512/64 = 8`，`numKb = 256/32 = 8` → **64 个 tile-block**。
- 每 tile-block 发 12 Vector + 4 TMA = 16 tileop → 64 × 16 = **1024**（打屏 `Total Tileop Counter 1024`、`Vector 768 / TMA 256`、`Retired VCALL 768 / TMA 256` 全部吻合）。

## 2.2 核心结论

| 指标 | 数值 | 判读 |
|---|---|---|
| Total Cycles | 35236 | 0.269 cyc/元素（35236 / 131072） |
| Retiring | 6.89% | 有效工作占比极低 |
| Bad Speculation | 0.93% | 分支几乎不影响（inter-block misp 仅 23 次） |
| Frontend Bound | 0.13% | 前端无压力 |
| **Backend Bound** | **92.05%** | **压倒性后端受限** |
| ├ Memory Bound | 0.00% | 数据全在片上，无访存瓶颈 |
| └ Core Bound | 92.05% | 全部来自计算域 |

**结论：纯 Core-Bound（后端计算受限），瓶颈集中在单一 Vector 引擎。**

## 2.3 瓶颈定位：引擎完全串行

| 指标 | 数值 | 含义 |
|---|---|---|
| Vector Tileop Cycles | 26849（76.2%） | Vector 绝对热点 |
| TMA Tileop Cycles | 9075（25.8%） | 访存搬运次之 |
| Cube Tileop Cycles | 0 | Cube 引擎全程闲置（CubeCore 段全 0/nan） |
| **Vector-TMA Cocurrent Cycles** | **0** | **Vector 与 TMA 零重叠，完全串行** |
| Vector-TMA Active Cycles | 34913 | 二者并集≈总周期 |
| All Cores Idle | 323（0.9%） | 空转很少，忙但不重叠 |

26849 + 9075 ≈ 35924 几乎线性相加 = 总周期，印证零流水重叠。每块「TLOAD→Vector 计算→TSTORE」严格顺序。

## 2.4 向量吞吐偏低

| 指标 | 数值 | 判读 |
|---|---|---|
| Vector TileopNum | 768 | 与拆分一致 |
| Vector UopNum | 17344 | 每 tileop 平均展开 ≈22.6 uop |
| **IPC(UOP)** | **0.492** | 每两拍才完成一个 uop，向量流水未打满 |
| ReadLat / WriteLat | 14 / 14（全落 ≤15 桶） | cell 读写延迟固定 14 拍，未被在飞 uop 掩盖 |

## 2.5 后端排队证据

| 指标 | 数值 | 判读 |
|---|---|---|
| BRob Full Stall | 25009（71%） | 块 ROB 满是首要停顿源 |
| BIsq Full Stall | 16201（46%） | 发射队列也频繁打满 |
| TileReg Full Stall | 0 | tile 寄存器非瓶颈 |
| Average Outstanding Block | 13.07（Vector 11.64） | 在飞块被后端资源卡住 |
| Average BROB depth | 227.80 / 容量 256 | BROB 长期近满 |
| PROB alloc 占用 | 3612164（各状态之最） | 大量块卡在「已分配等资源」入口 |
| Cell register 峰值 | T 56.88% / U 17.50%（M/N/ACC 全 0） | 寄存器仍有余量 |

BRob/BIsq 满而 TileReg 不满 → 瓶颈是**块级调度带宽 + 串行依赖链**，非寄存器容量。

## 2.6 线程占用

SPE 仅 Thread0 活跃（Pick/decode/rename/retire 全在 Thread0，Thread1-3 全 0）= 单线程串行控制流，未利用多 PE-线程并行。

## 2.7 优化方向（按收益排序）

1. **双缓冲 TMA vs Vector（最大收益）**：Vector-TMA Concurrent=0。把下一块 TLOAD 与当前块 Vector 计算重叠，可把 TMA 的 9075 拍大部分藏进 Vector 的 26849 拍，端到端最多省 ≈25%（9075/35236）。
2. **消除冗余第二次 TLOAD**：data pass 的 `TLOAD(xh2, gx)` 重载了 scale pass 已载入的同一 `gx`。复用 `xh` 可减一次搬运与一条依赖，缩短串行链。
3. **提升向量 IPC**：IPC(UOP)=0.49，14 拍 cell 延迟未被掩盖。加深在飞流水（更多在飞 tile-block）或重排依赖，让更多 uop 并发摊薄延迟。
4. **（结构性）Cube 全闲**：本 kernel 无矩阵运算，Cube 天然用不上，非缺陷；该 shape 算力被单一 Vector 域瓶颈，扩展空间在 Vector/TMA 流水而非 Cube。

---

## 附：复现命令

```bash
# 编译（fp32 预算修复后 TileM 上限=64，M=512 可编译）
cd SuperNPUBench/benchmark/one-level-arch/test/kernel/quant/dynamic_mx_quant
rm -f build/*probe_ocp_fp8_newcalc*.o
make TESTCASE=dynamic_mx_quant TYPE=PROBE_OCP_FP8_NEWCALC CFLAGS="-DPM=512 -DPN=256"

# 时序仿真（打屏含上述全部指标）
cd SuperScalarModel
bin/gfsim -f <path>/dynamic_mx_quant_probe_ocp_fp8_newcalc.elf
```
