# DynamicMxQuant — 当前实现状态

本文件记录 DynamicMxQuant kernel 的**当前落地状态**。完整的预期设计（从 0 到全模板/全功能）见 [DESIGN.md](DESIGN.md)；实现过程中的约束与工具链缺口记录见 [RECORD.md](RECORD.md)。

## 状态总览

8 个目标配置 = {OCP-FP8, cuBLAS-FP8, OCP-FP4, DynRange-FP4} × {tail, nontail}；**另加 2 个大 BlockSize 专用模板**（方案 A 切分归约轴，规避非尾轴 TileN 上的对齐/TileSize 双重约束），见下文「大 BlockSize 变体」小节。

> **状态定义**（当前工具链不成熟，代码存在缺陷是必然的，故不以「零缺陷」为准，而以下述两态区分）：
> - **已调试**：代码计算逻辑**基本正确**（逐 op 对齐 AscendC），且**所有已知问题都记录在 RECORD 中**。允许存在待工具链/ISA 补齐的已记录缺口（如 fp32→fp4 cast 语义待确认、非尾轴 parity 交织缺失），只要它们被显式记录。
> - **未调试**：代码逻辑**完全错误 / 未经订正**——未逐 op review，或核心算法仍套用错误路径。
>
> **本轮已调试 5 个 kernel：`TAIL_CUBLAS_FP8`、`NONTAIL_CUBLAS_FP8`、`NONTAIL_OCP_FP4`、`TAIL_OCP_FP4`、`TAIL_OCP_FP8`（固定 SPMD 4-PE，位补求倒数 + boxed 尾块）**；其余 3 个（`NONTAIL_OCP_FP8` 未逐 op review，`TAIL_DYNRANGE_FP4`、`NONTAIL_DYNRANGE_FP4` 待改）为未调试。
> 注：两个 OCP-fp4（`tail` + `nontail` plain/bigbs）的 scale 路径均已迁移**值域归约新算法**（`TABS`+`TCOLMAX/TROWMAX` 求块 \|max\| → `TCVT`→bf16 → `reinterpret_tile`+`TANDS` 取指数位 → 内联 finalize），data 路径均为 fp32 域乘法 + fp32→fp4 直转，与 AscendC 的 bf16 域路径不同，其 cast round 语义待 ISA/编译器确认——已记录为 RECORD 问题6，故按上述定义仍算已调试。

| 配置 | axis | scaleAlg | dstType | 状态（review / 落地） |
|------|------|----------|---------|------|
| TAIL_OCP_FP8 | 尾轴 | OCP | FP8_E4M3 | ✅ 已调试（固定 SPMD 4-PE）：位补求倒数（`0x7F00-bits` 代 TRECIP）+ boxed 尾块（非旧递归尾块）；`dynamic_mx_quant_tail_ocp_fp8.hpp`。gfrun 4 线程非尾块配置（M=1024）`R2=0` 跑到底、4 线程负载均衡。**单PE 精度实测（M=256/N=32，gfrun 单线程 + res_check，仅 tid=0 前 64 行为真数据）**：scale 逐字节**全对**（MSE=0）；data 仅 **5/2048 偏差**，全部落在 emulator `fp32→e4m3` 饱和边界 bug 窗口——input≈±2.0 × recip(2^8) = \|积\|∈[496,512) 触发进位越 `exp_max` 的 clamp 漏判，吐 `0x78/0xf8`(±256) 而非饱和 `0x7e/0xfe`(±448)，即已立项的 **SuperScalarModel issue364 / RECORD 问题21**，非 kernel 缺陷（kernel 送入 TCVT 的 fp32 正确，是 TCVT 自身 clamp 错）。**残留**：boxed 尾块（`SubM%TileM≠0`）reduce→TCVT 撞模型侧 stride 缺陷（问题22 / `ISSUE_reduce_output_stride_tail.md`，本轮未修）；特殊值 inf/zero/special 三 guard 未纳入（C4，仅极简位补路径，零块巨 recip 见问题20）；**4-PE 全量精度已跑通（2026-08-31）**：gfrun 4 线程 full-tile（M=512 N=256 BS=32）过官方 res_check —— **`output=pass (MaxAE=0.011719) / scale=pass (MaxAE=0，逐字节精确)`**（真随机输入 seed=42）。三项前置：问题15 的 e8m0 TCVT 转换**本地恢复**（d8903938 基线提交 `ad288c24`；**当前 codex/pr-0.58.4 基线重港为 `60ce26fd`**，忠实重放官方 52f56d5，见问题15 regression。注：codex 上 fp8 4-PE res_check 亦已复现 pass，2026-09-01）+ 问题22 的 e8m0-面 TCVT 物理形状契约**临时放宽**（`Block.cpp`，未提交）+ 问题23 的 `GFRUN_FORCE_DIRECTBOOT_ABI=1`（QEMU 缺席时的 ABI 规避，未提交）。本环境 QEMU 不可用故未走 `run_precision_check.py`（其硬编码远程 QEMU 路径），改**手工 gen → 编 res_check → gfrun 4 线程 → 官方 `dynamic_mx_quant_data_compare.py`** 复现。single-PE 探针 `probe_dynamic_mx_quant_tail_ocp_fp8_newcalc.hpp` 保留 |
| TAIL_OCP_FP4 | 尾轴 | OCP | FP4_E2M1 | ✅ 已调试：scale 路径逐 op 对齐 AscendC（boxed `compute_ocp_scale_tail_boxed_pw`）；**列装箱补齐物理宽**方案——物理列宽补齐到 `PW=⌈BlockSize/64⌉×64`、每 op 列装箱到有效 BlockSize（`TROWMAX` 按 ValidCol 归约、不合并 block），fp4 输出 tile 物理 `PW/2` 字节（32B 对齐）/有效 `BlockSize/2` 字节，boxed 载入/落盘只搬 ValidCol 故尾块不越界，基址折叠定位每 block；**无 concat/配对/零块**，仅需 `N%BlockSize==0`（单 block/tile 亦可）；奇数 numKb 显式补 0x00 E8M0 到 padding scale 列；compact uint8 scale + boxed M_tail。**已知未决**：data 路径 fp32 域乘法 + fp32→fp4 直转（AscendC 走 bf16 域），cast round 语义待 ISA/编译器确认（RECORD 问题6）；**scale 路径 gfrun 跑通、scale=pass（MSE=0）**；但 **data 路径当前基线（codex/pr-0.58.4 `dmxq-ops-20260828` + 工具链 `d6a52b8` + llvm `0f878a8`）为运行期 `output=fail（MSE=7.83）`**。现状（2026-09-01）：**已能编译 + gfrun 4 线程跑通**——旧的 `template_asm.hpp:115` 编译期 static_assert 与 `Block.cpp` 物理形状崩，因 d6a52b8/codex 已原生解决 cmode/形状契约而**不再复现**。output=fail 的**根因在模型侧、非 kernel 缺陷**：gfrun 缺 fp4 **写侧**（`CubeEngine::DataFormatCvt` 的 `dstType==FP4` 值编码 + `TEPLEngine::ExecuteTCVT` nibble 打包），系 `31f7a8f`(#314) 被 `930d9981` revert 连坐删、官方未恢复。跟踪 `ISSUE_fp4_pack_tcvt_regression.md` → **SuperScalarModel issues454**；最小复现 `fp4_shape_probe.cpp WIDEN=on`（`6.0→码6 应码7`）；本地修复思路（先值编码后打包）见 RECORD「2026-09-01」节。旧的编译期形状契约（问题16 / `ISSUE_tcvt_fp4_shape_contract.md`）已随工具链升级解除 |
| TAIL_CUBLAS_FP8 | 尾轴 | cuBLAS | FP8_E4M3 | ✅ 已调试 + **正式方案迁移** + **gfrun 端到端逐字节验证** + **固定 SPMD 4-PE（2026-09-01，codex 基线 gfrun 4 线程精度 pass）**：逐 op 对齐 AscendC（含 boxed 尾块）。scale pass 已就地内联展开（规避问题8 tile-函数入参 S64 栈往返），换正式方案——v0.58 `reinterpret_tile` 零指令位重解释（替 scratch-HBM，问题4）+ 原生 `TCMPS<CmpMode::{LT,NE,GT,EQ}>`（替 min/max-EQ 模拟，问题3）；bf16/half/fp32 三输入 compile+diss 通过。**fp16 driver 变体 gfrun 端到端 `R2=0`：output 与 scale 对 golden 逐字节完全一致**，含**宽范围数据复验**（\|max\| 0.24→25.1 跨 7 档，scale e8m0 逐行按 `floor(log2(448/max))` 分档 117→123 全对）。**bf16 默认 driver 的 gfrun 被既有 emulator 缺陷2 挡住**（bf16 `TROWMAX` 不在白名单=FP16/FP32/INT32），故字节级校验走 fp16 路径。**4-PE SPMD 适配（`kPeNum` 模板参，默认 1=单线程零回归；`tail_cublas_fp8_4pe.cpp` TYPE=`TAIL_CUBLAS_FP8_4PE`，fp16 in）**：复用 `tail_ocp_fp8` 两级切分骨架（按 M 行切 4 PE 连续段 + 段内 TileM tiling），`process_tile<TileMv,ValidRows>` 去重 full/tail 双份计算体。**codex 基线 gfrun 4 线程（M=512 N=256 BS=32 seed=42）：`output=pass (MaxAE=0.009766) / scale=pass (MaxAE=0，逐字节精确)`**。两处改动：(1) reduce 下游列向量中间 tile 声明 **physical Cols=1**（匹配 rowReduce 无条件 col=1，否则撞 `IsCompatibleOperationDataTile` 的 `col==physicalCol`，同 `tail_ocp_fp8`，见 RECORD 问题22）；(2) **cuBLAS 守卫的复合条件（`&&`/`||`）从数据域 `TAND`/`TOR` 组合掩码改为嵌套 TSEL**——这是**修正 kernel 原写法的 PTO ISA 不合规**（非编译器/model 问题），详见下文「cuBLAS 守卫掩码的 PTO ISA 合规写法」小节。gfsim 因不读 `GFRUN_FORCE_DIRECTBOOT_ABI`（仅 gfrun）在退出期 syscall 号误解码崩,与 kernel 无关、precision 以 gfrun 为门 |
| TAIL_DYNRANGE_FP4 | 尾轴 | DynRange | FP4_E2M1 | ⏳ 待改：同尾轴 fp4 方案 + 同尾块问题 |
| NONTAIL_OCP_FP8 | 非尾轴 | OCP | FP8_E4M3 | ❓ 未调试：仍用广播版 `compute_ocp_scale_not_tail`（uint16 广播 scale，归约轴未 ÷BlockSize）、输出侧与 `nontail_cublas_fp8` 相同，但本身未逐 op review |
| NONTAIL_OCP_FP4 | 非尾轴 | OCP | FP4_E2M1 | ✅ 已调试：TileN=64 plain tile；scale 走**值域归约新算法**（`TABS`+`TCOLMAX` 在输入值域求块 \|max\| → `TCVT`→bf16 → `reinterpret_tile<uint16_t>`+`TANDS` 取指数位 → `TMULS` 乘 `2^-emax` → `TCVT`→e8m0），finalize（inf→0x7f81 / zero→0 / special→0x0040 三 `TSEL` + `TSUB(0x7f00-shared)`）**全内联**，消除 `compute_ocp_scale_not_tail_boxed` 函数调用（问题8）与 `reinterpret_u16_to_bf16` scratch-HBM（问题4），reinterpret 用零指令 `reinterpret_tile<>` 视图；1:1 对齐 `tail_ocp_fp4` 母本（TROWMAX→TCOLMAX、boxed valid row=1）；uint8 compact 平铺（每块 1 字节，归约轴 ÷BlockSize 偶数对齐，同 `nontail_cublas_fp8`）。**已知未决**：(1) data 路径 fp32 域乘法 + fp32→fp4 直转（同 `tail_ocp_fp4`），cast round 语义 + fp4 打包 TCVT 形状契约待确认（RECORD 问题6/16）；(2) 非尾轴 parity 交织仍缺（ISA 有 `TINTERLEAVE`/`TDEINTERLEAVE` 但 linx `-D__linx` 头未暴露，RECORD 问题5）。**2026-09-02 新增**：(a) fp4 输出 tile 从旧 `[.,TileN/2]` 字节列约定迁移到 **element-列形 `[.,TileN]`**（与源 fp32 同列，使 fp32→fp4 打包 TCVT 走打包 specialization 而非 ordinary 路径的 physical-Cols 校验；`_bigbs` 同步迁移），当前 codex/pr-0.58.4 基线单线程 + 4-PE 均可编译；(b) 增 `kPeNum` 模板参（默认 1=单线程零回归）做 **4-PE SPMD**（按块行 kb 运行期连续切分，同 `nontail_cublas_fp8`），新增 `nontail_ocp_fp4_4pe.cpp` + `TYPE=NONTAIL_OCP_FP4_4PE`。**单 PE（multiThreadNum=1）落盘写满全部块行、逐字节正确** → 切分与计算逻辑无误；**4-PE gfrun res_check 落盘出现规模相关的"仅 PE0 段有数据、其余块行为零"现象（Post≤192 挂 / 256 过），根因未定位，见 RECORD 问题24**（该现象在 `nontail_cublas_fp8` 上同样存在，非本 kernel 特有；已排除 kernel 代码差异/write 截断/标量内存竞争/单纯 tile 存储/工作量不均，方向待查）。fp4 data 路径 output 另受模型侧 fp4 写侧缺失影响（问题16/issues454，同 `tail_ocp_fp4`） |
| NONTAIL_CUBLAS_FP8 | 非尾轴 | cuBLAS | FP8_E4M3 | ✅ 已调试 + **gfrun 端到端验证** + **固定 SPMD 4-PE（2026-09-01，codex 基线 gfrun 4 线程精度 pass）**：plain 路径已把 `compute_cublas_core` 就地展开、换用正式方案（v0.58 `reinterpret_tile` 零指令位重解释 + 原生 `TCMPS<CmpMode>`，替代 scratch-HBM 往返 + min/max-EQ 模拟）。**4-PE SPMD 适配（`kPeNum` 模板参，默认 1=单线程零回归；`nontail_cublas_fp8_4pe.cpp` TYPE=`NONTAIL_CUBLAS_FP8_4PE`，fp16 in）**：按**块行 kb**（归约块索引，= 输出行块 / scale 行）运行期连续切 4 PE（块行切分不改任何 tile 形状，无需按 Pe 编译期展开）；**并同时把 cuBLAS 守卫的数据域 `TAND`/`TOR` 组合掩码改为嵌套 TSEL——修正 kernel 原写法的 PTO ISA 不合规**（旧仿真器纵容，codex 会崩，详见「cuBLAS 守卫掩码的 PTO ISA 合规写法」小节）。**codex 基线 gfrun 4 线程（Axis=512 Post=256 BS=32 seed=42）：`output=pass (MaxAE=0.011719)`（逐字节，e4m3 舍入边界）；scale 值经交织复原后逐字节等于金标（`interleave(planar)==golden`），仅 planar↔parity 交织布局差 = 问题5（TINTERLEAVE 未暴露，与 4-PE/合规无关）**。故正确性以 output 为门。ELF 用 env_test 工具链编译、工作目录 gfrun 执行到底 `R2=0`：**data 逐字节匹配 golden（32 行全对；e4m3 输出为 1B tile，B.IOR 元素步长==字节步长故本测点无需字节步长补丁，见问题19）、scale 值逐字节匹配**（仅 parity 交织布局差=问题5）。gfrun 依赖 5 处 emulator 反应式移植（RECORD 问题9/14/17/18）。**`_bigbs` 分支已同步迁移正式方案**（`reinterpret_tile` + 原生 `TCMPS<CmpMode>`，就地展开原 `compute_cublas_core` 调用以规避问题8，无 scratch-HBM）：bf16/half/fp32 三输入 BS=128 compile+diss 通过、发射原生 CmpMode（42 条 TCMPS，无 min/max-EQ 序列）。**budget 更正（实测 2026-08-20）**：cuBLAS bigbs 固有 fp32/uint32 中间量（physical `[R_sub,TileN]` 32b）经 8192B tile 律锁 `R_sub*TileN≤2048`（实测 2048 编过、4096 撞 TADDS TilesizeCode），故「formal 4096」不可达、assert 已收紧到 2048。**bigbs 亦已 gfrun 端到端验证（2026-08-20）**：独立 harness `nontail_cublas_fp8_bigbs.cpp`（`TYPE=NONTAIL_CUBLAS_FP8_BIGBS`，Axis=128/Post=32/BS=128→`R_sub=32/TileN=32` 自动路由 bigbs）用 env_test linx 编译、**工作目录 gfrun** 执行到底 `R2=0`：**data 逐字节匹配 golden（4096B 全对）、scale 值逐字节匹配**（32 个真实 E8M0 全对，仅 parity 交织布局差=问题5）。golden 由 BS 参数化生成器 `--block-size 128` 直出（不分叉）。注：**env_test 的 gfrun 撞 `ValidateCompareSelectTepl` 断言**（compare/select TEPL dtype 白名单不含当前 uint32 组合），工作目录 gfrun 白名单更全故跑通——两 gfrun 版本对 cublas bigbs 的 compare/select 支持有 skew。**bigbs 亦已 PTO ISA 合规修复（数据域 TAND/TOR → 嵌套 TSEL）并 codex 基线 gfrun 单线程验证（2026-09-01，Axis=128 Post=32 BS=128 bf16 seed=42）：`output=pass 逐字节 MaxAE=0`，scale 值 `interleave(planar)==golden`（仅 问题5 布局差）**（bigbs 未做 4-PE，大 BS 场景并行度有限） |
| NONTAIL_DYNRANGE_FP4 | 非尾轴 | DynRange | FP4_E2M1 | ⏳ 待改：套用 nontail OCP-FP4 的 TileN=64 plain tile |

**小结**：**已调试 5 个** = `TAIL_CUBLAS_FP8` + `NONTAIL_CUBLAS_FP8` + `NONTAIL_OCP_FP4` + `TAIL_OCP_FP4` + `TAIL_OCP_FP8`（计算逻辑基本正确、已知问题全部记录）。其中两个 cuBLAS-FP8 无未决项；两个 OCP-fp4 的 data 路径共享 fp32→fp4 直转的未决 cast 语义（RECORD 问题6），已记录故仍算已调试；`TAIL_OCP_FP8`（固定 SPMD 4-PE，位补求倒数 + boxed 尾块）gfrun 4 线程非尾块配置 `R2=0`，残留 boxed 尾块 stride 缺陷（问题22）+ 特殊值 guard（C4）待纳入。**未调试 3 个**：`NONTAIL_OCP_FP8`（能编译但未逐 op review）与 `TAIL_DYNRANGE_FP4`、`NONTAIL_DYNRANGE_FP4`（待改）。

**fp4 发射能力已验证**：探针 `test/kernel/quant/dynamic_mx_quant/src/fp4_probe.cpp`（`TYPE=FP4_PROBE`）证实 fp32→`__fp4_e2m1x2` 单步 `TCVT` + `TSTORE` 发射真实指令、无对齐断言。尾轴 fp4 的**输出 tile 切分**问题（RowMajor 每行需 ≥ 32 打包字节）已在 `TAIL_OCP_FP4` 落地解决（**列装箱补齐物理宽到 `PW=⌈BlockSize/64⌉×64` + 每 op 列装箱到有效 BlockSize**，`TROWMAX` 按 ValidCol 归约、不合并 block，无 concat/配对/零块）；待改的 2 个 DynRange-FP4 配置可复用同一方案；详见 DESIGN §7.4 与 RECORD 问题2。

## Tile 旋钮编译期推导 + 非尾轴 plain↔bigbs 自动路由

**`TileM`/`TileN`/`R_sub` 不再是调用方模板参数**，改由算子输入 + 输入 dtype 预算**编译期推导**（`dynamic_mx_quant_common.hpp` 的 constexpr helper：`max_tilem` / `pick_tilen` / `max_rsub`）：
- **尾轴**：函数顶部 `TileM = max_tilem<M, Contig, InT, IsCublas>()`——`cublas` Contig=`BlockSize`、`ocp-fp4` Contig=`PW=⌈BlockSize/64⌉×64`（每 tile 一个补齐块，绑定 `TileM*PW`），夹在 `[tilem_min(≥512B tile), budget/Contig]` 与 `M`。
- **非尾轴**：入口先 `TileN = pick_tilen<BlockSize, Post, OutT, InT, IsCublas>()`；`if constexpr (TileN >= 对齐下界)` 则走 plain 单遍路径，**否则（大 BS 无合法 TileN）自动路由到 `_bigbs` 方案 A**（`R_sub = max_rsub<...>()`）。`if constexpr` 保证未取分支不实例化，故 plain 的 `TileN=0` / bigbs 非法 `R_sub` 不触发 static_assert。
- **`InT` 现为真实数据路径**（`fp16(__half)` / `bf16(__bf16)` / `fp32(float)`，`if constexpr` 分派，镜像 AscendC `Compute()` line 920-940 的 `ComputeMaxExp{Ocp,Cublas}{Bf16,Half,Fp32}`）：`InT` 既作预算推导（更宽输入 dtype 缩小 tile），又贯通到 scale-归约与 data 两条路径。`static_assert(InT ∈ {__bf16,__half,float})`。类型差异集中在**输入正则化**一处（对齐 AscendC）：
  - **OCP** 有两条流：**值域归约新算法（`nontail_ocp_fp4` plain+bigbs、`tail_ocp_fp4` 已迁移）**——三类型均在输入值域 `TABS`+`TCOLMAX/TROWMAX` 求块 \|max\|（`TABS` 白名单仅 FP16/FP32，故 bf16 先 `TCVT→fp32`；half 走 half 域、fp32 走 fp32 域），归约**之后**才 `TCVT`→bf16 并 `reinterpret_tile<uint16_t>`+`TANDS` 取指数位，天然免输入正则化器与 scratch-HBM；**旧指数位域归约（仅剩 `nontail_ocp_fp8`）**：bf16 = 指针 reinterpret→uint16 直取指数位；half = `TCVT half→bf16`(TRUNC)→取指数位（`half_to_bf16bits`）；fp32 = `reinterpret f32→u32`→`TANDS(FP32_EXP_MASK)`→`TSHRS(16)`→narrow uint16（`f32_to_bf16expbits`）——先取指数位再 `TANDS`+`TCOLMAX`。
  - **cuBLAS**（归约统一到 fp32 amax）：三类型均在 InT 值域 `TABS`+`TROWMAX/TCOLMAX`；`if constexpr(InT==float)` 跳过 fp32 cast（已是 fp32），否则 `TCVT InT→fp32`。`compute_cublas_core` 不变。half 全程值域归约、无 half→bf16 前置 cast。
  - **data 路径三分支**（镜像 `ComputeData` line 785）：fp32 直接 fp32 域乘 recip（免前置 cast）；half/bf16 先 `TCVT→fp32` 再乘再 `TCVT→out`。
  - bigbs cuBLAS pass1：bf16 保持 uint16 abs-bit 域归约（对齐 AscendC、零回归、`TEXPANDS(0)` uint16 种子合法而 bf16 种子会 crash LinxV5）；half/fp32 用值域归约 + **剥离首子块做种子**（避免立即数种子）+ running-`TMAX`。
- 预算模型：绑定 tile 8192B；OCP 绑定宽 `sizeof(InT)`、cuBLAS 当前经 fp32 scratch-HBM 往返（问题4）绑定宽 4B（`kRegBitcast` 置 true 后回落 `sizeof(InT)`）。elem 预算：bf16-OCP=4096、bf16-cuBLAS(当前)=2048。对齐下界：fp8 `TileN%32`、fp4 `TileN%64`。
- **默认 `BS=32` 推导值与改造前一致**（tail-cublas TileM=8、tail-ocp-fp4 PW=64→TileM=8、nontail-cublas TileN=32、nontail-ocp-fp4 TileN=64），零行为回归。

## 大 BlockSize 变体（方案 A：切分归约轴）

大 BlockSize 覆盖由上述**非尾轴统一入口自动路由**到下面 2 个 `_bigbs` impl（**已删除独立 driver / Makefile TYPE**，改为在统一 nontail driver 里额外发一个大 BS 调用做编译期实例化）。两个 `_bigbs.hpp` 作为路由目标保留，模板签名仍带 `TileN`/`R_sub`（由 dispatcher 按当前预算算出后传入）。

**动因（代码结构变化）**：非尾轴 plain kernel 单遍载入整块 `[BlockSize, TileN]`，连续轴 TileN 同时背负**双重约束**——对齐**下界**（fp4 `TileN%64==0`、fp8 `TileN%32==0`）与 TileSize **上界**（`Rows*Cols*sizeof ≤ 8192` → `TileN ≤ 元素上限/BlockSize`），大 BlockSize 时「下界 > 上界」无合法 TileN（ocp-fp4 BS≥96、cublas-fp8 当前 BS≥96 / 正式 BS≥160 失效）。**方案 A** 把归约行（长 BlockSize）切成 `R_sub` 行子块（`R_sub | BlockSize`）、running-`TMAX` 跨子块累积，使 TileSize 改绑 `R_sub*TileN`（`R_sub` 为自由旋钮）而非 `BlockSize*TileN`，两约束就此**解耦**——任意 BlockSize 下 TileN 都能满足对齐。

| 模板 | 源文件 | 入口 | scaleAlg / dstType | 状态 |
|------|--------|------|--------------------|------|
| `dynamic_mx_quant_nontail_ocp_fp4_bigbs` | `dynamic_mx_quant_nontail_ocp_fp4_bigbs.hpp` | 由 `NONTAIL_OCP_FP4` 统一入口自动路由 | OCP / FP4_E2M1 | ✅ 已调试：值域归约新算法（fp32 统一累加器跨 R_sub 子块 running-`TMAX` + 全内联 finalize），逐 op 对齐 `tail_ocp_fp4` 母本 / AscendC `ComputeScaleOcp` |
| `dynamic_mx_quant_nontail_cublas_fp8_bigbs` | `dynamic_mx_quant_nontail_cublas_fp8_bigbs.hpp` | 由 `NONTAIL_CUBLAS_FP8` 统一入口自动路由 | cuBLAS / FP8_E4M3 | ✅ 已调试：逐 op 对齐 AscendC `ComputeScaleCuBlas` |

- **pass1 归约**：拆 `R_sub` 子块 + `TMAX` 累积，因 max 满足结合律故 `max-of-(子块 max) == 全行 max`，与 AscendC 单遍归约**等价**。**ocp-bigbs 走值域归约新算法**：fp32 统一累加器（种子 `TEXPANDS(0.0f)`，abs≥0 故 0 为幺元），三类型均 `TCVT→fp32`+`TABS`（`TABS` 白名单仅 FP16/FP32，bf16/half 先转 fp32，无损）+`TCOLMAX`+running-`TMAX`，累积后才 `TCVT`→bf16 取指数位；**cublas-bigbs 走 uint16 abs-bit 域**——与 AscendC `ComputeScaleCuBlas` bf16 输入分支**同域**（`And(BF16_ABS_MASK)` + `uint16 Reg::Max` 累积，`...large_tail.h:426-440`），累加器种子 `TEXPANDS(uint16 0)` 合法（bf16 seed 会崩 LinxV5 后端 `getCopyToParts`，但 bf16 逐元素 `TMAX` 本身不崩，已探针实测）。
- **finalize**：ocp-bigbs 与 plain 同走**全内联 finalize**（`reinterpret_tile<uint16_t>`+`TANDS` 取指数位 → `TMULS` 乘 `2^-emax` → `TCVT`→e8m0 直转 + `finalize_recip_u16` 三 `TSEL`：inf→0x7f81 / zero→0 / special→0x0040，镜像 AscendC `_ocp_new.h` `ComputeScaleOcp`），**已删 static helper `ocp_scale_from_maxexp_not_tail_boxed_bigbs`**（内联后无调用者）；cublas-bigbs 走**就地内联展开的 `compute_cublas_core`**（原生 `TCMPS<CmpMode>` + `reinterpret_tile<>` 视图，规避问题8，guard 的 `TOR` ≡ AscendC `MaskXor`）。pass2 数据路径与对应 plain kernel 一致。
- **共同残留**：(1) 问题5 scale parity 交织缺失（compact 平铺，非 AscendC 交织布局）；(2) runtime 实测：**cublas-bigbs 已 gfrun 端到端验证（2026-08-20，data 逐字节匹配、scale 值逐字节匹配，仅问题5 交织布局差；「仅布局差」由 data 逐字节匹配 + 单调判别实验坐实，非近常量值集判据，见 RECORD 问题5「验证」小节）**；ocp-bigbs 仍未逐个跑过（skew 已对 pinned 组合解除、可跑）。**cublas-bigbs 的 `static_assert` 已收紧到 `R_sub*TileN≤2048`（实测更正 2026-08-20）**：cuBLAS 固有 fp32 amax + uint32 位运算 + pass2 fp32 数据 cast 均为 physical `[R_sub,TileN]` 的 **32b tile**，经 8192B tile 律锁死 2048（2048 编过、4096 撞 TADDS `TilesizeCode`），此 32b 中间量是指数抽取固有、非可去 workaround，故「formal 4096」**不可达**；BS=128 用 `R_sub=32/TileN=32`（`R_sub=64/TileN=32` 亦可）。ocp-bigbs 值域归约的输入 sub-chunk 载入是 16b tile（绑定 `R_sub*TileN≤4096`），fp32 累加器/abs 为 32b 中间量但按 boxed valid row=1 记，不受 fp8 那样的 32b 全宽约束；另有 data 路径 fp32→fp4 直转 cast 语义待确认（问题6）。
- **BlockSize 范围**：任意 `R_sub` 的倍数（`R_sub | BlockSize`），专供 plain kernel 覆盖不了的大 BS；小 BS 也能跑但 plain 更省（无切分/重读）。默认 `R_sub=32`。

## 已实现要点

- **三种 scale 算法全保留**，按 AscendC 合法性配对输出 dtype（OCP↔FP8&FP4、cuBLAS↔FP8、DynRange↔FP4），逐 op 对齐 AscendC `ComputeScale{Ocp,Cublas,DynamicDtypeRange}`。
- **emax 由输出 dtype 派生**（`emax_field<OutT, Domain>()`），非调用方自由参数。
- **两遍 ComputeScale→ComputeData 结构** 降低寄存器压力，规避 LinxV5 <512B tile 溢出断言。
- **位重解释 / cuBLAS 比较：`nontail_cublas_fp8` 的 plain 路径已迁移正式方案**——`nontail_cublas_fp8_plain` 就地展开 scale pass（规避 RECORD 问题8 的 tile-函数入参 S64 栈往返），把 scratch-HBM 位重解释换成零指令 `reinterpret_tile<>`、把 min/max+EQ 模拟比较换成原生 `TCMPS<CmpMode::…>`；compile+diss 实测无 scratch-HBM 往返、发射原生 CmpMode（RECORD 问题3/4）。**`nontail_cublas_fp8_bigbs` 亦已同步迁移正式方案**（同法就地展开 + `reinterpret_tile<>` + 原生 `TCMPS<CmpMode>`，bf16/half/fp32 BS=128 compile+diss 通过）。**`tail_cublas_fp8` 亦已迁移正式方案 + gfrun 端到端逐字节验证**（同法就地展开 + `reinterpret_tile<>` + 原生 `TCMPS<CmpMode>`；三输入 compile+diss、fp16 路径 gfrun `R2=0` output/scale 逐字节匹配含宽范围复验）。**至此 3 个 cuBLAS kernel 全部迁移完毕，`common::compute_cublas_core` 已无 kernel 调用**（其仅由这 3 个 kernel 使用）。**ocp-fp4 系列（`tail`/`nontail` plain+bigbs）亦已迁移值域归约新算法，消除 `reinterpret_u16_to_bf16` scratch-HBM（问题4）与 `compute_ocp_scale_not_tail_boxed`/`ocp_scale_from_maxexp_not_tail_boxed_bigbs` 函数调用（问题8），全用零指令 `reinterpret_tile<>` 视图 + 内联 finalize**；仍用 scratch-HBM 的 `reinterpret` helper 仅剩旧流程 `nontail_ocp_fp8`（问题4）。scale 交织仍无正式方案（`TINTERLEAVE` 未暴露），全部 kernel 保持 planar 平铺（RECORD 问题5）。

## cuBLAS 守卫掩码的 PTO ISA 合规写法（嵌套 TSEL，非数据域 TAND/TOR）

> **这是 kernel 写法的规范合规修正，不属于编译器/model 问题，故不入 RECORD。**

cuBLAS scale core 的守卫式指数抽取需要复合布尔条件，例如
`(exp>0 && exp<254 && man>0) || (exp==0 && man>0x400000) ? exp+1 : exp`。

**曾用的写法（不合规）**：`compare → 数据域 TAND/TOR 组合掩码 → 单个 TSEL 选值`——把 compare 结果当作全宽
整数掩码（0/-1）用 `TAND`/`TOR` 按位组合。这只在**旧仿真器**上能跑（旧实现让 compare 吐全宽数据掩码），
**违反 PTO ISA 规范**。

**PTO ISA 规范（pto-spec，normative）明确**：
- `PTO-REQ-TEPL-COMPARISON-001: packed predicate compare and select semantics.`
- **TCMP/TCMPS 产出「packed predicate Tile」**（predicate-kind 存储，位打包：逻辑元素 i 占 `byte floor(i/8)`
  的 `bit i%8`，容量 `ceil(Row*Col/8)` 字节）——见 `docs/tile/.../logical/TCMP.md`。
- **TSEL 的 mask 源必须是「packed predicate Tile」**（bit0 选 false、bit1 选 true）——`.../TSEL.md`。
- **TAND 只作用于「integer elements」，且显式「packed and floating formats reject before effects」**
  ——`.../TAND.md`。即 TAND 从规范定义上就**不能**作用于 compare 产出的 packed predicate。
- 全 spec **不存在**「predicate 组合」指令（无 predicate→predicate 的逻辑运算）；复合条件只能经
  **TSEL 逐个 predicate 串联**。

**当前合规写法（嵌套 TSEL，每个 TSEL 只吃单个直接 compare predicate）**：
```
sel = exp                                                 // 默认
c1=(exp>0), c2=(exp<254), c3=(man>0)                      // 直接 TCMPS packed predicate
n3 = c3? exp+1 : exp ; n2 = c2? n3 : exp ; sel = c1? n2 : exp    // p0=c1&&c2&&c3 段（c3→c2→c1 gate）
c4=(exp==0), c5=(man>0x400000)
u5 = c5? exp+1 : sel ; sel = c4? u5 : sel                        // p1=c4&&c5 段（从含 p0 的 sel 起做 OR）
```
`TSEL(dst,cond,trueval)` 就地二拍 = `dst = cond? trueval : dst_prior`。`finite`/`nonzero` 本就是直接
compare predicate、原样喂 TSEL。逐值等价旧的 TAND/TOR+单 TSEL（掩码不承载数值，只做选择）。

**落地范围（整个 cuBLAS 系列已收口）**：
- `tail_cublas_fp8`（4-PE）：codex gfrun 4 线程 `output+scale=pass`（尾轴无交织）。
- `nontail_cublas_fp8` plain（4-PE）：codex gfrun 4 线程 `output=pass`，scale 值 `interleave(planar)==golden`（仅 问题5 布局差）。
- `nontail_cublas_fp8_bigbs`（单线程）：codex gfrun `output=pass` 逐字节（MaxAE=0），scale 值同 问题5 布局差。
- `common::compute_cublas_core`（+ `compute_cublas_scale_tail/not_tail`）：**dead reference（无 kernel 调用，各 kernel 就地内联）**，已改为合规嵌套 TSEL、删除旧的非合规 IDEAL 注释块；仅编译期解析验证（模板未实例化）。

至此 3 个 cuBLAS kernel + 共享 reference 全部合规，无残留数据域掩码组合。

## 当前覆盖范围

DESIGN 附录 A 是 AscendC 全量有效场景；下表是**当前代码实际落地**的子集：

| 维度 | 当前落地 | 全量目标（见 DESIGN 附录 A） |
|------|---------|------|
| 输入类型 | **FP16（`__half`）/ BF16（`__bf16`）/ FP32（`float`）** — `if constexpr` 分派，镜像 AscendC `ComputeMaxExp{Ocp,Cublas}{Bf16,Half,Fp32}`（6 个 kernel 全覆盖，三类型编译+反汇编通过；runtime 未逐个实测——skew 已对 pinned 组合解除、可跑）| （已全量落地） |
| 输出 dtype | FP8_E4M3、FP4_E2M1 | + FP4_E1M2、FP8_E5M2 |
| round_mode | rint | + round、floor（仅 FP4 合法） |
| blockSize | 32 | 32 的倍数、≤1024 |
| scale 输出布局 | 3 个已调试 kernel 用 uint8 E8M0 **compact 平铺**（每块 1 字节，归约轴 ÷BlockSize 后偶数对齐）：`tail_cublas_fp8` `[M, evenAlign(K/32)]`、`nontail_cublas_fp8` / `nontail_ocp_fp4` `[evenAlign(Axis/32), Post]`。未调试 kernel 仍 uint16 广播 | 尾轴 `[M, evenAlign(K/32)]`（无需交织）；**非尾轴 `[ceil(Axis/32/2), Post, 2]` 交织**（even/odd 块行按列 zip，parity 内层）——见「已知限制」 |
| K 维度 | 必须为 BlockSize 的倍数（无尾块 padding） | 任意 K（含尾块） |
| M 维度尾块 | 仅 `tail_cublas_fp8` 已改 boxed 部分 tile；其余 tail kernel 见「已知限制」 | 全 tail kernel boxed |

## 待扩展 / 已知限制

- [ ] **2 个 DynRange-FP4 配置的 tile 切分落地**：`nontail_dynrange_fp4` 套 `TileN=64` plain tile（同 `nontail_ocp_fp4`，2 行改动）；`tail_dynrange_fp4` 复用 `tail_ocp_fp4` 的**列装箱补齐物理宽**方案（物理宽补齐到 `PW=⌈BlockSize/64⌉×64` + 每 op 列装箱到有效 BlockSize，无 concat/配对/零块）。fp4 发射本身已验证可用。（`tail_ocp_fp4` 已落地此方案，见状态表。）
- [ ] **非尾轴 `[BlockSize, TileN]` 切分的 BlockSize 边界（大 BS 无合法 TileN）**：2 个已调试非尾轴 kernel 均单遍载入满 `[BlockSize, TileN]`，对齐下界（连续轴 TileN，最严输出 tile）与 TileSize 上界（`元素上限/BlockSize`，最宽 load/store tile）压在同一根 TileN 上，`下界 > 上界` 时无解。**两 kernel 成因不同，须分别看**：
  - **`nontail_cublas_fp8`**（fp8 输出，下界 `TileN%32==0`≥32）：**用 fp32(32b) HBM 往返规避问题4**（`compute_cublas_core` 的 `reinterpret_f32_to_u32`），故有两套上界。**kernel 的 `static_assert` 按正式方案（4096/BS）编写**，当前工具链缺口以注释形式记录（详见 kernel 头注释）——
    - **编译器补齐寄存器 reinterpret 后的正式方案（assert 采用此界）**：32b 往返消失、绑定回落 bf16 输入(2B)，上界 `4096/BS` → **BS ≤ 128**（BS=96→TileN=32；BS=128→仅 32；**BS≥160 无解**）。
    - **当前规避方案（未 assert，仅注释记录）**：32b tile 绑定，上界 `2048/BS` → 合法 ⟺ **BS ≤ 64**（BS=32→TileN∈{32,64}；BS=64→仅 32；**BS≥96 无解**）。故 `64 < BS ≤ 128` 时 assert 通过、但当前工具链仍停在 `IsValidActiveSize`，待问题4 补齐后自动放开。
  - **`nontail_ocp_fp4`**（fp4 打包输出，下界 `TileN%64==0`≥64）：值域归约**单遍载入整块 `[BlockSize, TileN]`**，绑定预算是 **16b 输入 tile**（`BlockSize*TileN*2≤8192` → `BlockSize*TileN≤4096`，fp32 abs/累加器为 boxed valid row=1 非全宽），上界恒 `4096/BS` → 合法 ⟺ **BS ≤ 64**（BS=32→TileN∈{64,128}；BS=64→仅 64；**BS≥96 无解**）。
  - **两者的 plain 单遍在大 BS 失效**（cublas 当前 BS≥96 / 正式 BS≥160；ocp-fp4 BS≥96）；覆盖大 BS 需**方案 A（切分归约轴 + 累积，通用解）**或**方案 B（fractal/Box 豁免对齐，fallback）**，详见 RECORD 问题2 / DESIGN §7.5。默认 BlockSize=32 在两者可用范围内。**方案 A 已落地（两个 kernel）**：
    - `dynamic_mx_quant_nontail_ocp_fp4_bigbs`（`TYPE=NONTAIL_OCP_FP4_BIGBS`，把归约行切成 `R_sub` 子块 + running-`TMAX` 累积），BS=128 编译/链接/反汇编通过（4×`TANDS`→`TCOLMAX`→`TMAX` 累积链 + fp4 cast 已发射，无对齐/TileSize 断言）。
    - `dynamic_mx_quant_nontail_cublas_fp8_bigbs`（`TYPE=NONTAIL_CUBLAS_FP8_BIGBS`，同结构：归约在 uint16 abs-bit 域 running-`TMAX` 累积——**此域与 AscendC 非尾轴 `ComputeScaleCuBlas`（bf16 输入分支：`And(BF16_ABS_MASK)` + `uint16 Reg::Max` 累积到 `maxU16`，`dynamic_mx_quant_not_tail_axis_optimize_high_perf_large_tail.h:426-440`）完全同域**，非负 bf16 位序与幅值单调、inf/NaN 由 core 的 `finite` 掩码兜住，同时规避 bf16 `TEXPANDS` seed 崩溃 LinxV5 后端（`getCopyToParts` illegal-type；改用 uint16 `TEXPANDS(0)` seed 合法。**注：bf16 逐元素 `TMAX` 本身不崩——已探针实测——故 bf16 值域 + peeled-seed 也能编译；选 uint16 位域是因它精确匹配 AscendC,非因 bf16 `TMAX` 不可用**）；累积后 `reinterpret_tile<__bf16>`→fp32 + **就地内联展开的 `compute_cublas_core`**（原生 `TCMPS<CmpMode>` + `reinterpret_tile<uint32_t>` 视图，规避问题8，无 scratch-HBM），pass2 fp32→fp8）。bf16/half/fp32 三输入 BS=128（`R_sub=32`、`numSub=4`、`TileN=32`）compile+diss 通过（`TANDS`→`TCOLMAX`→`TMAX` 累积链 + 42 条原生 TCMPS + fp8 cast 已发射，无对齐/TileSize 断言、零 scratch-HBM）。
    - **两个 bigbs 均已逐 op 对齐 AscendC**（详见「大 BlockSize 变体」小节）。cublas-fp8-bigbs 对齐 `ComputeScaleCuBlas`：归约域一致（uint16 abs-bit）、守卫 + recip 为**就地内联展开的 `compute_cublas_core`**（原生 CmpMode；`p0=(exp>0)&&(exp<254)&&(man>0)`、`p1=(exp==0)&&(man>HALF)`，二者互斥故 `TOR`≡AscendC `MaskXor`）、pass2 与已 review 的 plain `compute_cublas_scale_not_tail` 一致。ocp-fp4-bigbs 对齐 `_ocp_new.h` `ComputeScaleOcp`：pass1 值域归约（fp32 统一累加器种子 `TEXPANDS(0.0f)`、三类型 `TCVT→fp32`+`TABS`+`TCOLMAX`+running-`TMAX`），拆子块 `TMAX` 累积因结合律等价于单遍归约，累积后 `TCVT`→bf16 取指数位、finalize 与 plain 同走**全内联** `reinterpret_tile`+`TANDS`+`TMULS(2^-emax)`+`TCVT<bf16→e8m0>`+`finalize_recip_u16` 三 `TSEL`（已删 static helper `ocp_scale_from_maxexp_not_tail_boxed_bigbs`）。**残留缺口**：两者均缺问题5 scale parity 交织（compact 平铺）；ocp-bigbs 另有 data 路径 fp32→fp4 直转 cast 语义待确认（问题6）；**runtime 实测：cublas-bigbs 已 gfrun 端到端验证（2026-08-20，data 逐字节匹配 golden、scale 值逐字节匹配，仅问题5 交织布局差）**，ocp-bigbs 仍未逐个跑过。**预算（实测更正 2026-08-20）**：cublas-bigbs `static_assert` 已收紧到 `R_sub*TileN≤2048`——其 fp32/uint32 32b 中间量是指数抽取固有、经 8192B tile 律锁死（2048 编过、4096 撞 TADDS `TilesizeCode`），「formal 4096」不可达；ocp-bigbs 走 16b、不受此限（`≤4096`）。
- [ ] **尾块统一 boxed**：`tail_ocp_fp8` / `tail_dynrange_fp4` 仍是递归尾块——`M%TileM≠0` 时会无限模板递归导致编译失败；默认 `M=8` 时 `M_tail=0` 未触发，属潜在 bug，待统一改为 boxed 部分 tile（参照 `tail_cublas_fp8`）。（`tail_ocp_fp4` 已用 boxed M_tail，不在此列。）
- [x] **迁移 cuBLAS 比较到原生 CmpMode + 寄存器 reinterpret**：`nontail_cublas_fp8` 的 **plain 路径已完成**——就地展开 `compute_cublas_core`，比较全换成原生 `TCMPS<CmpMode::{LT,NE,GT,EQ}>`，位重解释全换成 v0.58 `reinterpret_tile<>`（零 scratch-HBM）。gfrun 端到端验证数值逐字节正确（见状态表 NONTAIL_CUBLAS_FP8）。**`nontail_cublas_fp8_bigbs` 亦已迁移**（同法就地展开 + `reinterpret_tile<>` + 原生 `TCMPS<CmpMode>`，bf16/half/fp32 BS=128 compile+diss 通过、无 scratch-HBM；budget 实测更正为 `R_sub*TileN≤2048`）。**`tail_cublas_fp8` 亦已迁移 + gfrun 端到端逐字节验证**（fp16 路径 `R2=0`、含宽范围复验，见状态表）。**至此使用 `common::compute_cublas_core` 的 3 个 kernel（nontail plain + cublas-bigbs + tail_cublas_fp8）全部迁移完毕，无剩余调用方**；RECORD 问题3/4/8 主线对 cuBLAS 路径已全部落地。（ocp-fp4 系列的 scratch-HBM `reinterpret` helper 与 cuBLAS core 无关，另计。）
- [ ] **FP4_E1M2 / FP8_E5M2** 输出 dtype（`emax_field` trait 已覆盖，仅缺 kernel/driver）。
- [x] **FP16 / FP32 输入**（已落地，`if constexpr` 分派镜像 AscendC `Compute()` 920-940）：6 个 kernel（4 debugged + 2 bigbs）放开 `static_assert(InT ∈ {__bf16,__half,float})`、`InT` 贯通 scale-归约与 data 两路径。正则化域映射：**OCP** half=`TCVT half→bf16`(TRUNC)+取指数位（inf/nan 靠下游 `eq_inf==0x7F80` 命中）、fp32=`f32→u32`+`TANDS(0x7F800000)`+`TSHRS(16)`+narrow（逐元素提取与 Max 归约可交换）；**cuBLAS** 三类型 InT 值域 `TABS`+归约，fp32 免 fp32-cast、half 全程值域无 half→bf16 前置 cast。**data 三分支**：fp32 直接 fp32 域乘、half/bf16 先 `TCVT→fp32`。bf16 走原指针-reinterpret 路径**零回归**。发射能力经 `dtype_probe.cpp` 反汇编确认（`TCVT half→bf16`/`half→fp32`/`fp32→uint16` narrow、uint32 `TANDS`/`TSHRS`/`TROWMAX`/`TCOLMAX`、fp32 `TABS` 均发射真实指令、无对齐/round 断言）。golden 生成器加 `--in-dtype {bf16,fp16,fp32}`（fp16 经 Python `struct 'e'`；scale 域三类型一致、data 域按输入精度）。**runtime 未逐个实测**（skew 已对 pinned 组合解除、可跑；现仅编译+反汇编+逐 op 复核）。
- [ ] **K 维度尾块** padding（K 非 BlockSize 倍数）。
- [ ] **多 round_mode**（round / floor，仅 FP4 合法）。
- [ ] **非尾轴 scale 交织 `[ceil(Axis/32/2), Post, 2]`**：当前 3 个已调试 kernel 输出 **compact 平铺**（归约轴 ÷BlockSize + 偶数对齐，每块 1 字节），但缺 AscendC 最终 mxScale 的 **parity 交织**（even/odd 块行按列 zip）。**阻塞在 linx 头封装缺口（非 ISA 能力缺口）**：`TINTERLEAVE`/`TDEINTERLEAVE` 在 LinxISA 0.57 有定义（`linxisa-0.57-intrinsics.txt:101-102`、`docs/content/intrinsics/{tinterleave,tdeinterleave}.md`），语义即所需 parity zip，但 `-D__linx` 头（仅 `template_asm.hpp`）未暴露其 lowering（详见 RECORD 问题5）——对应 AscendC `DataCopy<DIST_INTLV_B8>` / `Reg::Interleave`（standalone `dynamic_mx_quant_post.h` / swiglu `axis_not_last.h`）。**尾轴无需交织**（块行在行内已连续，平铺即等价，见 host tiling :415-416 / swiglu `axis_last.h:585-592`）。

## 构建

```bash
export COMPILER_DIR=<linx-toolchain>/output/linx_blockisa_llvm_musl/bin
export LINX_SYSROOT=<linx-toolchain>/output/linx_blockisa_llvm_musl/sysroot/usr

cd test/kernel/quant/dynamic_mx_quant
# 已落地配置（任选其一）
make TESTCASE=dynamic_mx_quant TYPE=TAIL_OCP_FP8 res_check=on diss
make TESTCASE=dynamic_mx_quant TYPE=NONTAIL_OCP_FP4 res_check=on diss
# fp4 发射探针
make TESTCASE=dynamic_mx_quant TYPE=FP4_PROBE diss
```

## 可跑通的 linx + gfrun/gfsim 版本（相对远程基线的补充改动）

> **更新（2026-08-31）——当前组合与下表（08-24）不同**：本轮 `tail_ocp_fp8` 4-PE 精度用的是
> **model 分支 `dmxq-ops-20260828`（基线 `d8903938`）**，含两个正式提交 `4cacc579`（问题14
> ValidateScalarLogicalTepl 位宽比较）+ `ad288c24`（**问题15 e8m0 TCVT 恢复**）与两个未提交临时项
> `Block.cpp`（问题22 e8m0-面契约放宽）/`main.cpp`（问题23 ABI 规避）。
> **⚠ 下表「`189e45f6`=52f56d5 官方已解决自带」一行已过时**：`52f56d5` 被 `930d9981` 连坐回退，
> `origin/main` 与 `d8903938` 现均无 e8m0 转换（见问题15 新口径 = **regression·本地恢复·SuperScalarModel
> issues439**），需本地重放 `ad288c24` 才有，**不再是「同步官方分支即自带」**。下表为 08-24 旧组合
> （linx `e6a31ef` / model `feat/pto-v058-adaptation @ 1fecf9e6`）的历史记录。

> 实测 2026-08-24。`probe_ocp_fp8_newcalc` 与 `tail_cublas_fp8_fp16` 在下述组合上 **gfrun `R2=0` + gfsim 跑到底**（newcalc 620 cyc、tail_cublas_fp8_fp16 1122 cyc，均无 assert/段错误）。

**① linx 工具链（编 ELF）**：工作目录 `linx-toolchain-build @ e6a31ef`（`output/linx_blockisa_llvm_musl`）。
- **无需本地源码改动**即可编本文的 scale 路径：TCVT 补发 `->lb2`（问题13）与 TCVT 打包-fp4 的 `TileLogicalShapeMatch` 编译期 `static_assert`（问题16）**均已在上游头** `template_asm.hpp`（实测 `TCVT_T:133` 已发 `B.DIM zero,%c7,->lb2` + `Cols` 操作数）。该仓唯一本地文件是 untracked 的 `.github/workflows/pr-gate.yml`（CI 门禁，与跑 kernel 无关）。
- **例外（非 1B 输出 dtype 的 data 逐字节校验）**：需给 installed 头打本地 `B.IOR` 字节步长补丁（问题19，未提交）；未打补丁的工具链 raw-asm 按元素步长发射，bf16/fp32 等非 1B 的 GM 载入半行错位、会出 16/32 行棋盘错位。cuBLAS 非尾轴 e4m3(1B) 输出本测点元素步长==字节步长故不受此限；scale 路径与 newcalc/tail_cublas 探针亦不受此限。

**② gfrun/gfsim（SuperScalarModel）**：分支 `feat/pto-v058-adaptation`，远程基线 `origin @ 63dbb5a2`，工作树 HEAD `1fecf9e6` — **领先基线 7 个 commit**（`git cherry -v origin/main HEAD` 实测：仅 1 个标 `-`=官方已有等价、6 个标 `+`=本地原创）：

| commit | author | 相对 origin/main | RECORD | 改动 |
|---|---|---|---|---|
| `189e45f6` | jialewang | **官方移植**（`-`，= `52f56d5f` #253，patch-id 完全相同，在 origin/main 等 6 分支） | 问题15【已解决·官方】 | 实现 TCVT bf16/fp32/fp16→e8m0(SF8) + ARGMAX/ARGMIN 选择器 |
| `1fecf9e6` | ziyang-cheng | 本地原创（`+`） | 问题14 | `ValidateScalarLogicalTepl` 源 dtype 相等→**位宽相等**（放行零指令 `reinterpret_tile` 视图）|
| `1f398190` | ziyang-cheng | 本地原创（`+`） | 问题18 | 就地 TSEL 从 dst 读 false-source（执行侧）|
| `ab822e7a` | ziyang-cheng | 本地原创（`+`） | 问题18 | 就地 TSEL dst 融进首个 select B.IOT（校验侧）|
| `c022a929` | ziyang-cheng | 本地原创（`+`） | 问题14 兄弟 | compare/select 源 tile 按**位宽**匹配（`IsCompatibleDataTile`）|
| `50afe316` | ziyang-cheng | 本地原创（`+`） | 问题17 | compare/select TEPL 接受 UINT32（TCMPS）|
| `2f87edc9` | ziyang-cheng | 本地原创（`+`） | 问题9 | 一元 TEPL 接受 BF16（TABS）|

- **官方 vs 本地的关键区分**：只有 **e8m0 TCVT（问题15）官方已修**，同步官方分支即自带；其余 **6 处是永久本地反应式移植**（patch-id 扫全 origin 无命中），**官方从未采纳**——每次跟随官方 force-push/rebase 后须重新反应式补齐（见 `reference_v058_branch_force_pushed`），否则 `reinterpret_tile`/UINT32-compare/in-place-TSEL/BF16-TABS 路径会重新撞断言。
- **newcalc 探针为何 gfsim 不崩**：objdump 实测 `ssrset` 计数=0（编译期定尺、循环全展开），不触发 `Decoder` 的 ssrsetOrderQ null-write 潜伏 bug（该 bug 仍在 Model 源码，只对真正发 ssrset 的 kernel 才崩）。

## 验证边界

- **正确性依据**：`TAIL_CUBLAS_FP8`、`NONTAIL_CUBLAS_FP8`、`NONTAIL_OCP_FP4`、`TAIL_OCP_FP4`、`TAIL_OCP_FP8` 这 5 个已逐 op 对齐 AscendC（已调试）。其中两个 cuBLAS-FP8 无未决项；两个 OCP-fp4 的 scale 路径 faithful，data 路径 fp32→fp4 直转的 cast 语义待 ISA/编译器确认（RECORD 问题6）；`TAIL_OCP_FP8` 逐元素算法逐 op 对齐 single-PE 探针，**已升级为 gfrun 4-PE full-tile 官方 res_check 精度实测（2026-08-31，`output=pass MaxAE=0.011719 / scale=pass MaxAE=0`，真随机输入）**，前置 = 问题15 e8m0 恢复 + 问题22 e8m0-面契约放宽 + 问题23 ABI 规避（见状态表脚注）；但特殊值 inf/zero/special 三 guard 未纳入（C4）、boxed 尾块仍撞问题22。其余 3 个配置存疑——编译+链接通过不构成验证，`NONTAIL_OCP_FP8` 的 OCP scale 核心与两个 DynRange 配置均未逐 op 复核。
  - **`NONTAIL_CUBLAS_FP8` 已从「逐 op 对齐」升级为「gfrun 端到端逐字节验证」（2026-08-20）**：plain 路径（正式方案 = `reinterpret_tile` + 原生 CmpMode）ELF 用 env_test 工具链编译、工作目录 gfrun 跑到底 `R2=0`，data 逐字节匹配 golden、scale 值逐字节匹配（仅 parity 交织布局差=问题5）。前提：gfrun 带 5 处 emulator 反应式移植（RECORD 问题9/14/17/18）。
  - **`TAIL_CUBLAS_FP8` 亦已升级为「gfrun 端到端逐字节验证」（2026-08-22，正式方案迁移后）**：fp16 driver 变体 gfrun `R2=0`，output 与 scale 对 golden 逐字节完全一致，含宽范围数据复验（\|max\| 0.24→25.1 跨 7 档、scale 逐行分档全对）。**bf16 默认 driver 因既有 emulator 缺陷2（bf16 `TROWMAX` 白名单）无法 gfrun**、字节级校验走 fp16 路径。**至此已有两个业务 kernel 带 golden 实测数值正确（两个 cuBLAS-FP8）**，其余仍停在「逐 op 对齐」静态审查。
- **scale 布局核对（本轮）**：4 个已调试 kernel 的归约轴 scale 均 ÷BlockSize + 偶数对齐（`tail_cublas_fp8` / `tail_ocp_fp4` `[M, evenAlign(K/32)]`、两个 nontail `[evenAlign(Axis/32), Post]`）。`nontail_ocp_fp4` 本轮从 uint16 广播改为 boxed compact 平铺，与已 review 的 `nontail_cublas_fp8` 一致——此改动为镜像既有 review 代码 + 逐 op 比对 AscendC 每块 E8M0 语义，编译+链接通过，**该改动尚未 runtime 回归**（skew 已对 pinned 组合解除、可在 gfrun 上跑，待补端到端回归确认）。**尾轴平铺即 AscendC 布局**；**非尾轴仍缺 parity 交织**（ISA 有 `TINTERLEAVE`/`TDEINTERLEAVE`，但 linx `-D__linx` 头未暴露，封装缺口，见「已知限制」/ RECORD 问题5），故两个 nontail kernel 的 scale 布局尚非 AscendC-faithful。
- **fp4 发射链路**：反汇编确认发射链路可用（归约 → bf16↔fp32 转换 → 广播乘 → 输出 cast → 存储），但发射能力≠实现正确，`NONTAIL_OCP_FP4` 与 `TAIL_OCP_FP4` 均已逐 op 落地（data 路径的 fp32→fp4 cast 语义仍待确认，RECORD 问题6）。
- **skew 已对当前 pinned 组合解除**：2026-08-18 实测**新编** `dynamic_mx_quant` OCP probe ELF 在当前 gfrun（`local_test @ 0e213a2c`）上**跑到底且精度正确**（scale=0x79、y=0x78，`R2=0`）——此前「fresh ELF 一律 crash、只能用 prebuilt」不再成立。故 scale 路径（含问题14 reinterpret、问题15 e8m0 TCVT 修复后）的 runtime 验证不再被 skew 阻塞。**唯一仍阻塞的是 `TAIL_OCP_FP4` 的 fp4 data 路径**：其输出 `fp32→fp4` TCVT 撞 emulator 解码期形状结构断言（实证崩在 physical `row==row`，详见 RECORD 问题16，emulator 建模缺陷、非 skew），需 emulator 放宽 physical row/col 校验后才能跑通 fp4 精度。其余 kernel 的端到端精度 harness（`run_precision_check.py`）已就绪，可在 pinned 组合上直接跑（换 kernel/toolchain-model 组合仍应实跑确认）。
