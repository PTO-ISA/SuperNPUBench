# DynamicMxQuant — 当前实现状态

本文件记录 DynamicMxQuant kernel 的**当前落地状态**。完整的预期设计（从 0 到全模板/全功能）见 [DESIGN.md](DESIGN.md)；实现过程中的约束与工具链缺口记录见 [RECORD.md](RECORD.md)。

## 状态总览

8 个目标配置 = {OCP-FP8, cuBLAS-FP8, OCP-FP4, DynRange-FP4} × {tail, nontail}；**另加 2 个大 BlockSize 专用模板**（方案 A 切分归约轴，规避非尾轴 TileN 上的对齐/TileSize 双重约束），见下文「大 BlockSize 变体」小节。

> **状态定义**（当前工具链不成熟，代码存在缺陷是必然的，故不以「零缺陷」为准，而以下述两态区分）：
> - **已调试**：代码计算逻辑**基本正确**（逐 op 对齐 AscendC），且**所有已知问题都记录在 RECORD 中**。允许存在待工具链/ISA 补齐的已记录缺口（如 fp32→fp4 cast 语义待确认、非尾轴 parity 交织缺失），只要它们被显式记录。
> - **未调试**：代码逻辑**完全错误 / 未经订正**——未逐 op review，或核心算法仍套用错误路径。
>
> **本轮已调试 4 个 kernel：`TAIL_CUBLAS_FP8`、`NONTAIL_CUBLAS_FP8`、`NONTAIL_OCP_FP4`、`TAIL_OCP_FP4`**；其余 4 个（`TAIL_OCP_FP8`、`NONTAIL_OCP_FP8` 未逐 op review，`TAIL_DYNRANGE_FP4`、`NONTAIL_DYNRANGE_FP4` 待改）为未调试。
> 注：两个 OCP-fp4 的 data 路径均为 fp32 域乘法 + fp32→fp4 直转（`tail` line 57-58/92、`nontail` line 92-95），与 AscendC 的 bf16 域路径不同，其 cast round 语义待 ISA/编译器确认——已记录为 RECORD 问题6，故按上述定义仍算已调试。

| 配置 | axis | scaleAlg | dstType | 状态（review / 落地） |
|------|------|----------|---------|------|
| TAIL_OCP_FP8 | 尾轴 | OCP | FP8_E4M3 | ❓ 未调试：`M%TileM≠0` 时递归尾块会无限模板递归（默认 M=8 → M_tail=0 侥幸绕过编译）；OCP tail scale 未逐 op review |
| TAIL_OCP_FP4 | 尾轴 | OCP | FP4_E2M1 | ✅ 已调试：scale 路径逐 op 对齐 AscendC（boxed `compute_ocp_scale_tail_boxed_pw`）；**列装箱补齐物理宽**方案——物理列宽补齐到 `PW=⌈BlockSize/64⌉×64`、每 op 列装箱到有效 BlockSize（`TROWMAX` 按 ValidCol 归约、不合并 block），fp4 输出 tile 物理 `PW/2` 字节（32B 对齐）/有效 `BlockSize/2` 字节，boxed 载入/落盘只搬 ValidCol 故尾块不越界，基址折叠定位每 block；**无 concat/配对/零块**，仅需 `N%BlockSize==0`（单 block/tile 亦可）；奇数 numKb 显式补 0x00 E8M0 到 padding scale 列；compact uint8 scale + boxed M_tail。**已知未决**：data 路径 fp32 域乘法 + fp32→fp4 直转（AscendC 走 bf16 域），cast round 语义待 ISA/编译器确认（RECORD 问题6）；**scale 路径已 gfrun 跑通**（问题15 打通 `TCVT bf16→e8m0`），但 **fp4 data 路径的输出 `fp32→fp4` TCVT 撞 TCVT 形状匹配契约**（TileLogicalShapeMatch：打包 fp4 与源无法同时满足 physical row/col 相等）。**当前工具链（PTO 0.58.1）此契约落在编译期 `static_assert`（`template_asm.hpp:115`）→ `TAIL_OCP_FP4` 连 `.o` 都编不出**（崩在 `Cols` 32≠64）；放宽编译期后同契约在 emulator 运行期 `Block.cpp:1039` 再崩（`Row` 8≠4）。复现仅需编译、无需 Model 仓、无需本地改动。待**工具链头 + emulator 双侧**放宽（删 physical `row==row`/`col==col`、只比 valid 维度），详见 RECORD 问题16 / `ISSUE_tcvt_fp4_shape_contract.md`。**备选方案并存**：同一 tile 切分问题另有「2-block scratch-HBM concat 配对 + 零块」方案（旧实现，保留在 `dynamic_mx_quant_tail_ocp_fp4.hpp.bak`），与当前列装箱方案行为等价但互斥；**选哪个待 skew 解除、两者可运行期比对后再决策**，当前默认列装箱方案 |
| TAIL_CUBLAS_FP8 | 尾轴 | cuBLAS | FP8_E4M3 | ✅ 已调试 + **正式方案迁移** + **gfrun 端到端逐字节验证**：逐 op 对齐 AscendC（含 boxed 尾块）。scale pass 已就地内联展开（规避问题8 tile-函数入参 S64 栈往返），换正式方案——v0.58 `reinterpret_tile` 零指令位重解释（替 scratch-HBM，问题4）+ 原生 `TCMPS<CmpMode::{LT,NE,GT,EQ}>`（替 min/max-EQ 模拟，问题3）；bf16/half/fp32 三输入 compile+diss 通过（实测 10 条 `reinterpret_tile` 视图、14 条原生 TCMPS、零 scratch-HBM）。**fp16 driver 变体 gfrun 端到端 `R2=0`：output 与 scale 对 golden 逐字节完全一致**，含**宽范围数据复验**（\|max\| 0.24→25.1 跨 7 档，scale e8m0 逐行按 `floor(log2(448/max))` 分档 117→123 全对；窄高斯数据下 8 行同档 120 属数据巧合非 bug）。**bf16 默认 driver 的 gfrun 被既有 emulator 缺陷2 挡住**（bf16 `TROWMAX` 不在 `IsReduceAndExpandTeplDataType` 白名单=FP16/FP32/INT32，`AccumulateBlockInfo.cpp`）——emulator 建模缺陷、与本次迁移正交，故字节级校验走 fp16 路径 |
| TAIL_DYNRANGE_FP4 | 尾轴 | DynRange | FP4_E2M1 | ⏳ 待改：同尾轴 fp4 方案 + 同尾块问题 |
| NONTAIL_OCP_FP8 | 非尾轴 | OCP | FP8_E4M3 | ❓ 未调试：仍用广播版 `compute_ocp_scale_not_tail`（uint16 广播 scale，归约轴未 ÷BlockSize）、输出侧与 `nontail_cublas_fp8` 相同，但本身未逐 op review |
| NONTAIL_OCP_FP4 | 非尾轴 | OCP | FP4_E2M1 | ✅ 已调试：TileN=64 plain tile；scale 本轮改为 boxed `compute_ocp_scale_not_tail_boxed` + uint8 compact 平铺（每块 1 字节，归约轴 ÷BlockSize 偶数对齐，同 `nontail_cublas_fp8`）。**已知未决**：(1) data 路径 fp32 域乘法 + fp32→fp4 直转（同 `tail_ocp_fp4`，line 92-95），cast round 语义待确认（RECORD 问题6）；(2) 非尾轴 parity 交织仍缺（ISA 有 `TINTERLEAVE`/`TDEINTERLEAVE` 但 linx `-D__linx` 头未暴露，RECORD 问题5） |
| NONTAIL_CUBLAS_FP8 | 非尾轴 | cuBLAS | FP8_E4M3 | ✅ 已调试 + **gfrun 端到端验证**：plain 路径已把 `compute_cublas_core` 就地展开、换用正式方案（v0.58 `reinterpret_tile` 零指令位重解释 + 原生 `TCMPS<CmpMode>`，替代 scratch-HBM 往返 + min/max-EQ 模拟）。ELF 用 env_test 工具链（含 B.IOR 元素步长修复 `f35d3aa`）编译、工作目录 gfrun 执行到底 `R2=0`：**data 逐字节匹配 golden（32 行全对）、scale 值逐字节匹配**（仅 parity 交织布局差=问题5）。gfrun 依赖 5 处 emulator 反应式移植（RECORD 问题9/14/17/18）。**`_bigbs` 分支已同步迁移正式方案**（`reinterpret_tile` + 原生 `TCMPS<CmpMode>`，就地展开原 `compute_cublas_core` 调用以规避问题8，无 scratch-HBM）：bf16/half/fp32 三输入 BS=128 compile+diss 通过、发射原生 CmpMode（42 条 TCMPS，无 min/max-EQ 序列）。**budget 更正（实测 2026-08-20）**：cuBLAS bigbs 固有 fp32/uint32 中间量（physical `[R_sub,TileN]` 32b）经 8192B tile 律锁 `R_sub*TileN≤2048`（实测 2048 编过、4096 撞 TADDS TilesizeCode），故「formal 4096」不可达、assert 已收紧到 2048。**bigbs 亦已 gfrun 端到端验证（2026-08-20）**：独立 harness `nontail_cublas_fp8_bigbs.cpp`（`TYPE=NONTAIL_CUBLAS_FP8_BIGBS`，Axis=128/Post=32/BS=128→`R_sub=32/TileN=32` 自动路由 bigbs）用 env_test linx 编译、**工作目录 gfrun** 执行到底 `R2=0`：**data 逐字节匹配 golden（4096B 全对）、scale 值逐字节匹配**（32 个真实 E8M0 全对，仅 parity 交织布局差=问题5）。golden 由 BS 参数化生成器 `--block-size 128` 直出（不分叉）。注：**env_test 的 gfrun 撞 `ValidateCompareSelectTepl` 断言**（compare/select TEPL dtype 白名单不含当前 uint32 组合），工作目录 gfrun 白名单更全故跑通——两 gfrun 版本对 cublas bigbs 的 compare/select 支持有 skew |
| NONTAIL_DYNRANGE_FP4 | 非尾轴 | DynRange | FP4_E2M1 | ⏳ 待改：套用 nontail OCP-FP4 的 TileN=64 plain tile |

**小结**：**已调试 4 个** = `TAIL_CUBLAS_FP8` + `NONTAIL_CUBLAS_FP8` + `NONTAIL_OCP_FP4` + `TAIL_OCP_FP4`（计算逻辑基本正确、已知问题全部记录）。其中两个 cuBLAS-FP8 无未决项；两个 OCP-fp4 的 data 路径共享 fp32→fp4 直转的未决 cast 语义（RECORD 问题6），已记录故仍算已调试。**未调试 4 个**：`NONTAIL_OCP_FP8`、`TAIL_OCP_FP8`（能编译但未逐 op review）与 `TAIL_DYNRANGE_FP4`、`NONTAIL_DYNRANGE_FP4`（待改）。

**fp4 发射能力已验证**：探针 `test/kernel/quant/dynamic_mx_quant/src/fp4_probe.cpp`（`TYPE=FP4_PROBE`）证实 fp32→`__fp4_e2m1x2` 单步 `TCVT` + `TSTORE` 发射真实指令、无对齐断言。尾轴 fp4 的**输出 tile 切分**问题（RowMajor 每行需 ≥ 32 打包字节）已在 `TAIL_OCP_FP4` 落地解决（**列装箱补齐物理宽到 `PW=⌈BlockSize/64⌉×64` + 每 op 列装箱到有效 BlockSize**，`TROWMAX` 按 ValidCol 归约、不合并 block，无 concat/配对/零块）；待改的 2 个 DynRange-FP4 配置可复用同一方案；详见 DESIGN §7.4 与 RECORD 问题2。

## Tile 旋钮编译期推导 + 非尾轴 plain↔bigbs 自动路由

**`TileM`/`TileN`/`R_sub` 不再是调用方模板参数**，改由算子输入 + 输入 dtype 预算**编译期推导**（`dynamic_mx_quant_common.hpp` 的 constexpr helper：`max_tilem` / `pick_tilen` / `max_rsub`）：
- **尾轴**：函数顶部 `TileM = max_tilem<M, Contig, InT, IsCublas>()`——`cublas` Contig=`BlockSize`、`ocp-fp4` Contig=`PW=⌈BlockSize/64⌉×64`（每 tile 一个补齐块，绑定 `TileM*PW`），夹在 `[tilem_min(≥512B tile), budget/Contig]` 与 `M`。
- **非尾轴**：入口先 `TileN = pick_tilen<BlockSize, Post, OutT, InT, IsCublas>()`；`if constexpr (TileN >= 对齐下界)` 则走 plain 单遍路径，**否则（大 BS 无合法 TileN）自动路由到 `_bigbs` 方案 A**（`R_sub = max_rsub<...>()`）。`if constexpr` 保证未取分支不实例化，故 plain 的 `TileN=0` / bigbs 非法 `R_sub` 不触发 static_assert。
- **`InT` 现为真实数据路径**（`fp16(__half)` / `bf16(__bf16)` / `fp32(float)`，`if constexpr` 分派，镜像 AscendC `Compute()` line 920-940 的 `ComputeMaxExp{Ocp,Cublas}{Bf16,Half,Fp32}`）：`InT` 既作预算推导（更宽输入 dtype 缩小 tile），又贯通到 scale-归约与 data 两条路径。`static_assert(InT ∈ {__bf16,__half,float})`。类型差异集中在**输入正则化**一处（对齐 AscendC）：
  - **OCP**（归约统一到 uint16 bf16-指数域）：bf16 = 指针 reinterpret→uint16 直取指数位（**原路径不变，零回归**）；half = `TCVT half→bf16`(TRUNC)→取指数位（`half_to_bf16bits`，inf/nan 经下游 `eq_inf==0x7F80` 自然命中，免 Select）；fp32 = `reinterpret f32→u32`→`TANDS(FP32_EXP_MASK)`→`TSHRS(16)`→narrow uint16（`f32_to_bf16expbits`；逐元素指数提取与 Max 归约可交换，故归约后即得 bf16 指数域 max_exp）。
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
| `dynamic_mx_quant_nontail_ocp_fp4_bigbs` | `dynamic_mx_quant_nontail_ocp_fp4_bigbs.hpp` | 由 `NONTAIL_OCP_FP4` 统一入口自动路由 | OCP / FP4_E2M1 | ✅ 已调试：逐 op 对齐 AscendC `ComputeScaleOcp` |
| `dynamic_mx_quant_nontail_cublas_fp8_bigbs` | `dynamic_mx_quant_nontail_cublas_fp8_bigbs.hpp` | 由 `NONTAIL_CUBLAS_FP8` 统一入口自动路由 | cuBLAS / FP8_E4M3 | ✅ 已调试：逐 op 对齐 AscendC `ComputeScaleCuBlas` |

- **pass1 归约**：拆 `R_sub` 子块 + `TMAX` 累积，因 max 满足结合律故 `max-of-(子块 max) == 全行 max`，与 AscendC 单遍归约**等价**；累加器种子 `TEXPANDS(uint16 0)` 合法（bf16 seed 会崩 LinxV5 后端 `getCopyToParts`，但 bf16 逐元素 `TMAX` 本身不崩，已探针实测）。ocp 走 uint16 **指数位域**、cublas 走 uint16 **abs-bit 域**——后者与 AscendC `ComputeScaleCuBlas` bf16 输入分支**同域**（`And(BF16_ABS_MASK)` + `uint16 Reg::Max` 累积，`...large_tail.h:426-440`）。
- **finalize**：ocp-bigbs 的 `ocp_scale_from_maxexp_not_tail_boxed_bigbs` 与 plain `compute_ocp_scale_not_tail_boxed` **同走** `ocp_scale_mulcast_from_maxexp`（bf16 乘 `2^-emax` + `Cast<bf16→e8m0>` 直转，镜像 AscendC `_ocp_new.h` `ComputeScaleOcp`）；cublas-bigbs 走**就地内联展开的 `compute_cublas_core`**（原生 `TCMPS<CmpMode>` + `reinterpret_tile<>` 视图，规避问题8，guard 的 `TOR` ≡ AscendC `MaskXor`）。pass2 数据路径与对应 plain kernel 一致。
- **共同残留**：(1) 问题5 scale parity 交织缺失（compact 平铺，非 AscendC 交织布局）；(2) runtime 实测：**cublas-bigbs 已 gfrun 端到端验证（2026-08-20，data 逐字节匹配、scale 值逐字节匹配，仅问题5 交织布局差；「仅布局差」由 data 逐字节匹配 + 单调判别实验坐实，非近常量值集判据，见 RECORD 问题5「验证」小节）**；ocp-bigbs 仍未逐个跑过（skew 已对 pinned 组合解除、可跑）。**cublas-bigbs 的 `static_assert` 已收紧到 `R_sub*TileN≤2048`（实测更正 2026-08-20）**：cuBLAS 固有 fp32 amax + uint32 位运算 + pass2 fp32 数据 cast 均为 physical `[R_sub,TileN]` 的 **32b tile**，经 8192B tile 律锁死 2048（2048 编过、4096 撞 TADDS `TilesizeCode`），此 32b 中间量是指数抽取固有、非可去 workaround，故「formal 4096」**不可达**；BS=128 用 `R_sub=32/TileN=32`（`R_sub=64/TileN=32` 亦可）。ocp-bigbs 走 16b、不受此限（仍 `≤4096`），另有 data 路径 fp32→fp4 直转 cast 语义待确认（问题6）。
- **BlockSize 范围**：任意 `R_sub` 的倍数（`R_sub | BlockSize`），专供 plain kernel 覆盖不了的大 BS；小 BS 也能跑但 plain 更省（无切分/重读）。默认 `R_sub=32`。

## 已实现要点

- **三种 scale 算法全保留**，按 AscendC 合法性配对输出 dtype（OCP↔FP8&FP4、cuBLAS↔FP8、DynRange↔FP4），逐 op 对齐 AscendC `ComputeScale{Ocp,Cublas,DynamicDtypeRange}`。
- **emax 由输出 dtype 派生**（`emax_field<OutT, Domain>()`），非调用方自由参数。
- **两遍 ComputeScale→ComputeData 结构** 降低寄存器压力，规避 LinxV5 <512B tile 溢出断言。
- **位重解释 / cuBLAS 比较：`nontail_cublas_fp8` 的 plain 路径已迁移正式方案**——`nontail_cublas_fp8_plain` 就地展开 scale pass（规避 RECORD 问题8 的 tile-函数入参 S64 栈往返），把 scratch-HBM 位重解释换成零指令 `reinterpret_tile<>`、把 min/max+EQ 模拟比较换成原生 `TCMPS<CmpMode::…>`；compile+diss 实测无 scratch-HBM 往返、发射原生 CmpMode（RECORD 问题3/4）。**`nontail_cublas_fp8_bigbs` 亦已同步迁移正式方案**（同法就地展开 + `reinterpret_tile<>` + 原生 `TCMPS<CmpMode>`，bf16/half/fp32 BS=128 compile+diss 通过）。**`tail_cublas_fp8` 亦已迁移正式方案 + gfrun 端到端逐字节验证**（同法就地展开 + `reinterpret_tile<>` + 原生 `TCMPS<CmpMode>`；三输入 compile+diss、fp16 路径 gfrun `R2=0` output/scale 逐字节匹配含宽范围复验）。**至此 3 个 cuBLAS kernel 全部迁移完毕，`common::compute_cublas_core` 已无 kernel 调用**（其仅由这 3 个 kernel 使用）；**其余仍用 scratch-HBM 的仅剩 ocp-fp4 系列的 `reinterpret` helper**（与 cuBLAS core 无关，问题4）。scale 交织仍无正式方案（`TINTERLEAVE` 未暴露），全部 kernel 保持 planar 平铺（RECORD 问题5）。

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
  - **`nontail_ocp_fp4`**（fp4 打包输出，下界 `TileN%64==0`≥64）：**不走 fp32 往返**（OCP 在 bf16/uint16 域，仅 16b reinterpret，与输入同宽、不加宽绑定）→ 上界恒 `4096/BS`、**不随 reinterpret 修复变化**（正式与当前同界，assert 即此界）→ 合法 ⟺ **BS ≤ 64**（BS=32→TileN∈{64,128}；BS=64→仅 64；**BS≥96 无解**）。
  - **两者的 plain 单遍在大 BS 失效**（cublas 当前 BS≥96 / 正式 BS≥160；ocp-fp4 BS≥96）；覆盖大 BS 需**方案 A（切分归约轴 + 累积，通用解）**或**方案 B（fractal/Box 豁免对齐，fallback）**，详见 RECORD 问题2 / DESIGN §7.5。默认 BlockSize=32 在两者可用范围内。**方案 A 已落地（两个 kernel）**：
    - `dynamic_mx_quant_nontail_ocp_fp4_bigbs`（`TYPE=NONTAIL_OCP_FP4_BIGBS`，把归约行切成 `R_sub` 子块 + running-`TMAX` 累积），BS=128 编译/链接/反汇编通过（4×`TANDS`→`TCOLMAX`→`TMAX` 累积链 + fp4 cast 已发射，无对齐/TileSize 断言）。
    - `dynamic_mx_quant_nontail_cublas_fp8_bigbs`（`TYPE=NONTAIL_CUBLAS_FP8_BIGBS`，同结构：归约在 uint16 abs-bit 域 running-`TMAX` 累积——**此域与 AscendC 非尾轴 `ComputeScaleCuBlas`（bf16 输入分支：`And(BF16_ABS_MASK)` + `uint16 Reg::Max` 累积到 `maxU16`，`dynamic_mx_quant_not_tail_axis_optimize_high_perf_large_tail.h:426-440`）完全同域**，非负 bf16 位序与幅值单调、inf/NaN 由 core 的 `finite` 掩码兜住，同时规避 bf16 `TEXPANDS` seed 崩溃 LinxV5 后端（`getCopyToParts` illegal-type；改用 uint16 `TEXPANDS(0)` seed 合法。**注：bf16 逐元素 `TMAX` 本身不崩——已探针实测——故 bf16 值域 + peeled-seed 也能编译；选 uint16 位域是因它精确匹配 AscendC,非因 bf16 `TMAX` 不可用**）；累积后 `reinterpret_tile<__bf16>`→fp32 + **就地内联展开的 `compute_cublas_core`**（原生 `TCMPS<CmpMode>` + `reinterpret_tile<uint32_t>` 视图，规避问题8，无 scratch-HBM），pass2 fp32→fp8）。bf16/half/fp32 三输入 BS=128（`R_sub=32`、`numSub=4`、`TileN=32`）compile+diss 通过（`TANDS`→`TCOLMAX`→`TMAX` 累积链 + 42 条原生 TCMPS + fp8 cast 已发射，无对齐/TileSize 断言、零 scratch-HBM）。
    - **两个 bigbs 均已逐 op 对齐 AscendC**（详见「大 BlockSize 变体」小节）。cublas-fp8-bigbs 对齐 `ComputeScaleCuBlas`：归约域一致（uint16 abs-bit）、守卫 + recip 为**就地内联展开的 `compute_cublas_core`**（原生 CmpMode；`p0=(exp>0)&&(exp<254)&&(man>0)`、`p1=(exp==0)&&(man>HALF)`，二者互斥故 `TOR`≡AscendC `MaskXor`）、pass2 与已 review 的 plain `compute_cublas_scale_not_tail` 一致。ocp-fp4-bigbs 对齐 `_ocp_new.h` `ComputeScaleOcp`：pass1 拆子块 `TMAX` 累积因结合律等价于单遍归约（uint16 指数位域，同 AscendC）、finalize 与 plain `compute_ocp_scale_not_tail_boxed` 同走 `ocp_scale_mulcast_from_maxexp`（bf16 乘 `2^-emax` + `Cast<bf16→e8m0>` 直转）。**残留缺口**：两者均缺问题5 scale parity 交织（compact 平铺）；ocp-bigbs 另有 data 路径 fp32→fp4 直转 cast 语义待确认（问题6）；**runtime 实测：cublas-bigbs 已 gfrun 端到端验证（2026-08-20，data 逐字节匹配 golden、scale 值逐字节匹配，仅问题5 交织布局差）**，ocp-bigbs 仍未逐个跑过。**预算（实测更正 2026-08-20）**：cublas-bigbs `static_assert` 已收紧到 `R_sub*TileN≤2048`——其 fp32/uint32 32b 中间量是指数抽取固有、经 8192B tile 律锁死（2048 编过、4096 撞 TADDS `TilesizeCode`），「formal 4096」不可达；ocp-bigbs 走 16b、不受此限（`≤4096`）。
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

## 验证边界

- **正确性依据**：`TAIL_CUBLAS_FP8`、`NONTAIL_CUBLAS_FP8`、`NONTAIL_OCP_FP4`、`TAIL_OCP_FP4` 这 4 个已逐 op 对齐 AscendC（已调试）。其中两个 cuBLAS-FP8 无未决项；两个 OCP-fp4 的 scale 路径 faithful，data 路径 fp32→fp4 直转的 cast 语义待 ISA/编译器确认（RECORD 问题6）。其余 4 个配置存疑——编译+链接通过不构成验证，`NONTAIL_OCP_FP8` / `TAIL_OCP_FP8` 的 OCP scale 核心与两个 DynRange 配置均未逐 op 复核。
  - **`NONTAIL_CUBLAS_FP8` 已从「逐 op 对齐」升级为「gfrun 端到端逐字节验证」（2026-08-20）**：plain 路径（正式方案 = `reinterpret_tile` + 原生 CmpMode）ELF 用 env_test 工具链编译、工作目录 gfrun 跑到底 `R2=0`，data 逐字节匹配 golden、scale 值逐字节匹配（仅 parity 交织布局差=问题5）。前提：gfrun 带 5 处 emulator 反应式移植（RECORD 问题9/14/17/18）。
  - **`TAIL_CUBLAS_FP8` 亦已升级为「gfrun 端到端逐字节验证」（2026-08-22，正式方案迁移后）**：fp16 driver 变体 gfrun `R2=0`，output 与 scale 对 golden 逐字节完全一致，含宽范围数据复验（\|max\| 0.24→25.1 跨 7 档、scale 逐行分档全对）。**bf16 默认 driver 因既有 emulator 缺陷2（bf16 `TROWMAX` 白名单）无法 gfrun**、字节级校验走 fp16 路径。**至此已有两个业务 kernel 带 golden 实测数值正确（两个 cuBLAS-FP8）**，其余仍停在「逐 op 对齐」静态审查。
- **scale 布局核对（本轮）**：4 个已调试 kernel 的归约轴 scale 均 ÷BlockSize + 偶数对齐（`tail_cublas_fp8` / `tail_ocp_fp4` `[M, evenAlign(K/32)]`、两个 nontail `[evenAlign(Axis/32), Post]`）。`nontail_ocp_fp4` 本轮从 uint16 广播改为 boxed compact 平铺，与已 review 的 `nontail_cublas_fp8` 一致——此改动为镜像既有 review 代码 + 逐 op 比对 AscendC 每块 E8M0 语义，编译+链接通过，**该改动尚未 runtime 回归**（skew 已对 pinned 组合解除、可在 gfrun 上跑，待补端到端回归确认）。**尾轴平铺即 AscendC 布局**；**非尾轴仍缺 parity 交织**（ISA 有 `TINTERLEAVE`/`TDEINTERLEAVE`，但 linx `-D__linx` 头未暴露，封装缺口，见「已知限制」/ RECORD 问题5），故两个 nontail kernel 的 scale 布局尚非 AscendC-faithful。
- **fp4 发射链路**：反汇编确认发射链路可用（归约 → bf16↔fp32 转换 → 广播乘 → 输出 cast → 存储），但发射能力≠实现正确，`NONTAIL_OCP_FP4` 与 `TAIL_OCP_FP4` 均已逐 op 落地（data 路径的 fp32→fp4 cast 语义仍待确认，RECORD 问题6）。
- **skew 已对当前 pinned 组合解除**：2026-08-18 实测**新编** `dynamic_mx_quant` OCP probe ELF 在当前 gfrun（`local_test @ 0e213a2c`）上**跑到底且精度正确**（scale=0x79、y=0x78，`R2=0`）——此前「fresh ELF 一律 crash、只能用 prebuilt」不再成立。故 scale 路径（含问题14 reinterpret、问题15 e8m0 TCVT 修复后）的 runtime 验证不再被 skew 阻塞。**唯一仍阻塞的是 `TAIL_OCP_FP4` 的 fp4 data 路径**：其输出 `fp32→fp4` TCVT 撞 emulator 解码期形状结构断言（实证崩在 physical `row==row`，详见 RECORD 问题16，emulator 建模缺陷、非 skew），需 emulator 放宽 physical row/col 校验后才能跑通 fp4 精度。其余 kernel 的端到端精度 harness（`run_precision_check.py`）已就绪，可在 pinned 组合上直接跑（换 kernel/toolchain-model 组合仍应实跑确认）。
