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
| TAIL_OCP_FP4 | 尾轴 | OCP | FP4_E2M1 | ✅ 已调试：scale 路径逐 op 对齐 AscendC（boxed `compute_ocp_scale_tail_boxed_pw`）；**列装箱补齐物理宽**方案——物理列宽补齐到 `PW=⌈BlockSize/64⌉×64`、每 op 列装箱到有效 BlockSize（`TROWMAX` 按 ValidCol 归约、不合并 block），fp4 输出 tile 物理 `PW/2` 字节（32B 对齐）/有效 `BlockSize/2` 字节，boxed 载入/落盘只搬 ValidCol 故尾块不越界，基址折叠定位每 block；**无 concat/配对/零块**，仅需 `N%BlockSize==0`（单 block/tile 亦可）；奇数 numKb 显式补 0x00 E8M0 到 padding scale 列；compact uint8 scale + boxed M_tail。**已知未决**：data 路径 fp32 域乘法 + fp32→fp4 直转（AscendC 走 bf16 域），cast round 语义待 ISA/编译器确认（RECORD 问题6）；运行期因 skew 未验。**备选方案并存**：同一 tile 切分问题另有「2-block scratch-HBM concat 配对 + 零块」方案（旧实现，保留在 `dynamic_mx_quant_tail_ocp_fp4.hpp.bak`），与当前列装箱方案行为等价但互斥；**选哪个待 skew 解除、两者可运行期比对后再决策**，当前默认列装箱方案 |
| TAIL_CUBLAS_FP8 | 尾轴 | cuBLAS | FP8_E4M3 | ✅ 已调试：逐 op 对齐 AscendC（含 boxed 尾块） |
| TAIL_DYNRANGE_FP4 | 尾轴 | DynRange | FP4_E2M1 | ⏳ 待改：同尾轴 fp4 方案 + 同尾块问题 |
| NONTAIL_OCP_FP8 | 非尾轴 | OCP | FP8_E4M3 | ❓ 未调试：仍用广播版 `compute_ocp_scale_not_tail`（uint16 广播 scale，归约轴未 ÷BlockSize）、输出侧与 `nontail_cublas_fp8` 相同，但本身未逐 op review |
| NONTAIL_OCP_FP4 | 非尾轴 | OCP | FP4_E2M1 | ✅ 已调试：TileN=64 plain tile；scale 本轮改为 boxed `compute_ocp_scale_not_tail_boxed` + uint8 compact 平铺（每块 1 字节，归约轴 ÷BlockSize 偶数对齐，同 `nontail_cublas_fp8`）。**已知未决**：(1) data 路径 fp32 域乘法 + fp32→fp4 直转（同 `tail_ocp_fp4`，line 92-95），cast round 语义待确认（RECORD 问题6）；(2) 非尾轴 parity 交织仍缺（ISA 有 `TINTERLEAVE`/`TDEINTERLEAVE` 但 linx `-D__linx` 头未暴露，RECORD 问题5） |
| NONTAIL_CUBLAS_FP8 | 非尾轴 | cuBLAS | FP8_E4M3 | ✅ 已调试：逐 op 对齐 AscendC |
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
- **finalize**：ocp-bigbs 的 `ocp_scale_from_maxexp_not_tail_boxed_bigbs` 与 plain `compute_ocp_scale_not_tail_boxed` **同走** `ocp_scale_mulcast_from_maxexp`（bf16 乘 `2^-emax` + `Cast<bf16→e8m0>` 直转，镜像 AscendC `_ocp_new.h` `ComputeScaleOcp`）；cublas-bigbs 复用 `compute_cublas_core`（guard 的 `TOR` ≡ AscendC `MaskXor`）。pass2 数据路径与对应 plain kernel 一致。
- **共同残留**：(1) 问题5 scale parity 交织缺失（compact 平铺，非 AscendC 交织布局）；(2) skew 下未 runtime 验证。cublas-bigbs 的 `static_assert` 按正式预算 `R_sub*TileN≤4096` 编写，当前 fp32 往返（问题4）实际限 `≤2048`，故 BS=128 用 `R_sub=32/TileN=32` 可编，正式方案后可放宽 `R_sub=64/TileN=64`。ocp-bigbs 另有 data 路径 fp32→fp4 直转 cast 语义待确认（问题6）。
- **BlockSize 范围**：任意 `R_sub` 的倍数（`R_sub | BlockSize`），专供 plain kernel 覆盖不了的大 BS；小 BS 也能跑但 plain 更省（无切分/重读）。默认 `R_sub=32`。

## 已实现要点

- **三种 scale 算法全保留**，按 AscendC 合法性配对输出 dtype（OCP↔FP8&FP4、cuBLAS↔FP8、DynRange↔FP4），逐 op 对齐 AscendC `ComputeScale{Ocp,Cublas,DynamicDtypeRange}`。
- **emax 由输出 dtype 派生**（`emax_field<OutT, Domain>()`），非调用方自由参数。
- **两遍 ComputeScale→ComputeData 结构** 降低寄存器压力，规避 LinxV5 <512B tile 溢出断言。
- **位重解释经 scratch-HBM 字节别名**（无寄存器 bitcast，`-D__linx` 无 TCAST）。

## 当前覆盖范围

DESIGN 附录 A 是 AscendC 全量有效场景；下表是**当前代码实际落地**的子集：

| 维度 | 当前落地 | 全量目标（见 DESIGN 附录 A） |
|------|---------|------|
| 输入类型 | **FP16（`__half`）/ BF16（`__bf16`）/ FP32（`float`）** — `if constexpr` 分派，镜像 AscendC `ComputeMaxExp{Ocp,Cublas}{Bf16,Half,Fp32}`（6 个 kernel 全覆盖，三类型编译+反汇编通过；runtime 待 skew 解除）| （已全量落地） |
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
    - `dynamic_mx_quant_nontail_cublas_fp8_bigbs`（`TYPE=NONTAIL_CUBLAS_FP8_BIGBS`，同结构：归约在 uint16 abs-bit 域 running-`TMAX` 累积——**此域与 AscendC 非尾轴 `ComputeScaleCuBlas`（bf16 输入分支：`And(BF16_ABS_MASK)` + `uint16 Reg::Max` 累积到 `maxU16`，`dynamic_mx_quant_not_tail_axis_optimize_high_perf_large_tail.h:426-440`）完全同域**，非负 bf16 位序与幅值单调、inf/NaN 由 core 的 `finite` 掩码兜住，同时规避 bf16 `TEXPANDS` seed 崩溃 LinxV5 后端（`getCopyToParts` illegal-type；改用 uint16 `TEXPANDS(0)` seed 合法。**注：bf16 逐元素 `TMAX` 本身不崩——已探针实测——故 bf16 值域 + peeled-seed 也能编译；选 uint16 位域是因它精确匹配 AscendC,非因 bf16 `TMAX` 不可用**）；累积后 reinterpret→fp32 复用既有 `compute_cublas_core`，pass2 fp32→fp8）。BS=128（`R_sub=32`、`numSub=4`、`TileN=32`）编译/链接/反汇编通过（`TANDS`→`TCOLMAX`→`TMAX` 累积链 + cublas core + fp8 cast 已发射，无对齐/TileSize 断言）。
    - **两个 bigbs 均已逐 op 对齐 AscendC**（详见「大 BlockSize 变体」小节）。cublas-fp8-bigbs 对齐 `ComputeScaleCuBlas`：归约域一致（uint16 abs-bit）、守卫 + recip 走 `compute_cublas_core`（`p0=(exp>0)&&(exp<254)&&(man>0)`、`p1=(exp==0)&&(man>HALF)`，二者互斥故 `TOR`≡AscendC `MaskXor`）、pass2 与已 review 的 plain `compute_cublas_scale_not_tail` 一致。ocp-fp4-bigbs 对齐 `_ocp_new.h` `ComputeScaleOcp`：pass1 拆子块 `TMAX` 累积因结合律等价于单遍归约（uint16 指数位域，同 AscendC）、finalize 与 plain `compute_ocp_scale_not_tail_boxed` 同走 `ocp_scale_mulcast_from_maxexp`（bf16 乘 `2^-emax` + `Cast<bf16→e8m0>` 直转）。**残留缺口**：两者均缺问题5 scale parity 交织（compact 平铺）；ocp-bigbs 另有 data 路径 fp32→fp4 直转 cast 语义待确认（问题6）；**两者均因 skew 未 runtime 验证**。`static_assert` 均按正式（补齐 reinterpret 后）预算 `R_sub*TileN≤4096` 编写；当前 fp32 往返使 cublas 实际 `R_sub*TileN≤2048`，故当前 `TileN=32`/`R_sub=32` 可编，正式方案后可放宽到 `R_sub=64`/`TileN=64`。
- [ ] **尾块统一 boxed**：`tail_ocp_fp8` / `tail_dynrange_fp4` 仍是递归尾块——`M%TileM≠0` 时会无限模板递归导致编译失败；默认 `M=8` 时 `M_tail=0` 未触发，属潜在 bug，待统一改为 boxed 部分 tile（参照 `tail_cublas_fp8`）。（`tail_ocp_fp4` 已用 boxed M_tail，不在此列。）
- [ ] **linx 4 参带 CmpMode 的 TCMP/TCMPS**：cuBLAS 现用 min/max + 默认-EQ 比较模拟 GT/LT/NE；补齐后可切 `compute_cublas_core` 末尾保留的 IDEAL 版（RECORD 问题3，已提 issue）。
- [ ] **FP4_E1M2 / FP8_E5M2** 输出 dtype（`emax_field` trait 已覆盖，仅缺 kernel/driver）。
- [x] **FP16 / FP32 输入**（已落地，`if constexpr` 分派镜像 AscendC `Compute()` 920-940）：6 个 kernel（4 debugged + 2 bigbs）放开 `static_assert(InT ∈ {__bf16,__half,float})`、`InT` 贯通 scale-归约与 data 两路径。正则化域映射：**OCP** half=`TCVT half→bf16`(TRUNC)+取指数位（inf/nan 靠下游 `eq_inf==0x7F80` 命中）、fp32=`f32→u32`+`TANDS(0x7F800000)`+`TSHRS(16)`+narrow（逐元素提取与 Max 归约可交换）；**cuBLAS** 三类型 InT 值域 `TABS`+归约，fp32 免 fp32-cast、half 全程值域无 half→bf16 前置 cast。**data 三分支**：fp32 直接 fp32 域乘、half/bf16 先 `TCVT→fp32`。bf16 走原指针-reinterpret 路径**零回归**。发射能力经 `dtype_probe.cpp` 反汇编确认（`TCVT half→bf16`/`half→fp32`/`fp32→uint16` narrow、uint32 `TANDS`/`TSHRS`/`TROWMAX`/`TCOLMAX`、fp32 `TABS` 均发射真实指令、无对齐/round 断言）。golden 生成器加 `--in-dtype {bf16,fp16,fp32}`（fp16 经 Python `struct 'e'`；scale 域三类型一致、data 域按输入精度）。**runtime 待 skew 解除**（现仅编译+反汇编+逐 op 复核）。
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
- **scale 布局核对（本轮）**：4 个已调试 kernel 的归约轴 scale 均 ÷BlockSize + 偶数对齐（`tail_cublas_fp8` / `tail_ocp_fp4` `[M, evenAlign(K/32)]`、两个 nontail `[evenAlign(Axis/32), Post]`）。`nontail_ocp_fp4` 本轮从 uint16 广播改为 boxed compact 平铺，与已 review 的 `nontail_cublas_fp8` 一致——此改动为镜像既有 review 代码 + 逐 op 比对 AscendC 每块 E8M0 语义，编译+链接通过，但因 skew 无法端到端回归，**改动待 skew 解除后再回归确认**。**尾轴平铺即 AscendC 布局**；**非尾轴仍缺 parity 交织**（ISA 有 `TINTERLEAVE`/`TDEINTERLEAVE`，但 linx `-D__linx` 头未暴露，封装缺口，见「已知限制」/ RECORD 问题5），故两个 nontail kernel 的 scale 布局尚非 AscendC-faithful。
- **fp4 发射链路**：反汇编确认发射链路可用（归约 → bf16↔fp32 转换 → 广播乘 → 输出 cast → 存储），但发射能力≠实现正确，`NONTAIL_OCP_FP4` 与 `TAIL_OCP_FP4` 均已逐 op 落地（data 路径的 fp32→fp4 cast 语义仍待确认，RECORD 问题6）。
- **端到端精度暂不可信**：工具链↔仿真器版本 skew，新编 ELF 当前无法在 gfrun/gfsim 稳定运行；精度 harness（`run_precision_check.py`）已就绪，待 skew 解除后再跑。
