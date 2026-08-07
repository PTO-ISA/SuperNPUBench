# DynamicMxQuant — 当前实现状态

本文件记录 DynamicMxQuant kernel 的**当前落地状态**。完整的预期设计（从 0 到全模板/全功能）见 [DESIGN.md](DESIGN.md)；实现过程中的约束与工具链缺口记录见 [RECORD.md](RECORD.md)。

## 状态总览

8 个目标配置 = {OCP-FP8, cuBLAS-FP8, OCP-FP4, DynRange-FP4} × {tail, nontail}。

> **状态定义**（当前工具链不成熟，代码存在缺陷是必然的，故不以「零缺陷」为准，而以下述两态区分）：
> - **已调试**：代码计算逻辑**基本正确**（逐 op 对齐 AscendC），且**所有已知问题都记录在 RECORD 中**。允许存在待工具链/ISA 补齐的已记录缺口（如 fp32→fp4 cast 语义待确认、非尾轴 parity 交织缺失），只要它们被显式记录。
> - **未调试**：代码逻辑**完全错误 / 未经订正**——未逐 op review，或核心算法仍套用错误路径。
>
> **本轮已调试 4 个 kernel：`TAIL_CUBLAS_FP8`、`NONTAIL_CUBLAS_FP8`、`NONTAIL_OCP_FP4`、`TAIL_OCP_FP4`**；其余 4 个（`TAIL_OCP_FP8`、`NONTAIL_OCP_FP8` 未逐 op review，`TAIL_DYNRANGE_FP4`、`NONTAIL_DYNRANGE_FP4` 待改）为未调试。
> 注：两个 OCP-fp4 的 data 路径均为 fp32 域乘法 + fp32→fp4 直转（`tail` line 57-58/92、`nontail` line 92-95），与 AscendC 的 bf16 域路径不同，其 cast round 语义待 ISA/编译器确认——已记录为 RECORD 问题6，故按上述定义仍算已调试。

| 配置 | axis | scaleAlg | dstType | 状态（review / 落地） |
|------|------|----------|---------|------|
| TAIL_OCP_FP8 | 尾轴 | OCP | FP8_E4M3 | ❓ 未调试：`M%TileM≠0` 时递归尾块会无限模板递归（默认 M=8 → M_tail=0 侥幸绕过编译）；OCP tail scale 未逐 op review |
| TAIL_OCP_FP4 | 尾轴 | OCP | FP4_E2M1 | ✅ 已调试：scale 路径逐 op 对齐 AscendC（boxed `compute_ocp_scale_tail_boxed`；每对相邻 block 各自子归约 → scratch-HBM concat 到 64 宽 fp32 → 单次 `TCVT` 打包 fp4，避开 <512B 溢出断言）；compact uint8 scale + boxed M_tail；K 须 ≥2 block（driver 用 K=64）。**已知未决**：data 路径 fp32 域乘法 + fp32→fp4 直转（AscendC 走 bf16 域），cast round 语义待 ISA/编译器确认（RECORD 问题6）；运行期因 skew 未验 |
| TAIL_CUBLAS_FP8 | 尾轴 | cuBLAS | FP8_E4M3 | ✅ 已调试：逐 op 对齐 AscendC（含 boxed 尾块） |
| TAIL_DYNRANGE_FP4 | 尾轴 | DynRange | FP4_E2M1 | ⏳ 待改：同尾轴 fp4 方案 + 同尾块问题 |
| NONTAIL_OCP_FP8 | 非尾轴 | OCP | FP8_E4M3 | ❓ 未调试：仍用广播版 `compute_ocp_scale_not_tail`（uint16 广播 scale，归约轴未 ÷BlockSize）、输出侧与 `nontail_cublas_fp8` 相同，但本身未逐 op review |
| NONTAIL_OCP_FP4 | 非尾轴 | OCP | FP4_E2M1 | ✅ 已调试：TileN=64 plain tile；scale 本轮改为 boxed `compute_ocp_scale_not_tail_boxed` + uint8 compact 平铺（每块 1 字节，归约轴 ÷BlockSize 偶数对齐，同 `nontail_cublas_fp8`）。**已知未决**：(1) data 路径 fp32 域乘法 + fp32→fp4 直转（同 `tail_ocp_fp4`，line 92-95），cast round 语义待确认（RECORD 问题6）；(2) 非尾轴 parity 交织仍缺（ISA 有 `TINTERLEAVE`/`TDEINTERLEAVE` 但 linx `-D__linx` 头未暴露，RECORD 问题5） |
| NONTAIL_CUBLAS_FP8 | 非尾轴 | cuBLAS | FP8_E4M3 | ✅ 已调试：逐 op 对齐 AscendC |
| NONTAIL_DYNRANGE_FP4 | 非尾轴 | DynRange | FP4_E2M1 | ⏳ 待改：套用 nontail OCP-FP4 的 TileN=64 plain tile |

**小结**：**已调试 4 个** = `TAIL_CUBLAS_FP8` + `NONTAIL_CUBLAS_FP8` + `NONTAIL_OCP_FP4` + `TAIL_OCP_FP4`（计算逻辑基本正确、已知问题全部记录）。其中两个 cuBLAS-FP8 无未决项；两个 OCP-fp4 的 data 路径共享 fp32→fp4 直转的未决 cast 语义（RECORD 问题6），已记录故仍算已调试。**未调试 4 个**：`NONTAIL_OCP_FP8`、`TAIL_OCP_FP8`（能编译但未逐 op review）与 `TAIL_DYNRANGE_FP4`、`NONTAIL_DYNRANGE_FP4`（待改）。

**fp4 发射能力已验证**：探针 `test/kernel/quant/dynamic_mx_quant/src/fp4_probe.cpp`（`TYPE=FP4_PROBE`）证实 fp32→`__fp4_e2m1x2` 单步 `TCVT` + `TSTORE` 发射真实指令、无对齐断言。尾轴 fp4 的**输出 tile 切分**问题（RowMajor NoneBox 每行需 ≥ 2 个 MX block）已在 `TAIL_OCP_FP4` 落地解决（每对 block 子归约 + scratch-HBM concat 到 64 宽再单次打包）；待改的 2 个 DynRange-FP4 配置可复用同一方案；详见 DESIGN §7.4 与 RECORD 问题2。

## 已实现要点

- **三种 scale 算法全保留**，按 AscendC 合法性配对输出 dtype（OCP↔FP8&FP4、cuBLAS↔FP8、DynRange↔FP4），逐 op 对齐 AscendC `ComputeScale{Ocp,Cublas,DynamicDtypeRange}`。
- **emax 由输出 dtype 派生**（`emax_field<OutT, Domain>()`），非调用方自由参数。
- **两遍 ComputeScale→ComputeData 结构** 降低寄存器压力，规避 LinxV5 <512B tile 溢出断言。
- **位重解释经 scratch-HBM 字节别名**（无寄存器 bitcast，`-D__linx` 无 TCAST）。

## 当前覆盖范围

DESIGN 附录 A 是 AscendC 全量有效场景；下表是**当前代码实际落地**的子集：

| 维度 | 当前落地 | 全量目标（见 DESIGN 附录 A） |
|------|---------|------|
| 输入类型 | BF16（16bit 域） | + FP16 / FP32（32bit 域，DESIGN §3.1 已设计路径，代码未落地） |
| 输出 dtype | FP8_E4M3、FP4_E2M1 | + FP4_E1M2、FP8_E5M2 |
| round_mode | rint | + round、floor（仅 FP4 合法） |
| blockSize | 32 | 32 的倍数、≤1024 |
| scale 输出布局 | 3 个已调试 kernel 用 uint8 E8M0 **compact 平铺**（每块 1 字节，归约轴 ÷BlockSize 后偶数对齐）：`tail_cublas_fp8` `[M, evenAlign(K/32)]`、`nontail_cublas_fp8` / `nontail_ocp_fp4` `[evenAlign(Axis/32), Post]`。未调试 kernel 仍 uint16 广播 | 尾轴 `[M, evenAlign(K/32)]`（无需交织）；**非尾轴 `[ceil(Axis/32/2), Post, 2]` 交织**（even/odd 块行按列 zip，parity 内层）——见「已知限制」 |
| K 维度 | 必须为 BlockSize 的倍数（无尾块 padding） | 任意 K（含尾块） |
| M 维度尾块 | 仅 `tail_cublas_fp8` 已改 boxed 部分 tile；其余 tail kernel 见「已知限制」 | 全 tail kernel boxed |

## 待扩展 / 已知限制

- [ ] **2 个 DynRange-FP4 配置的 tile 切分落地**：`nontail_dynrange_fp4` 套 `TileN=64` plain tile（同 `nontail_ocp_fp4`，2 行改动）；`tail_dynrange_fp4` 每 block 子归约 + scratch-HBM concat 到 64 宽（复用 `tail_ocp_fp4` 已落地方案）。fp4 发射本身已验证可用。（`tail_ocp_fp4` 已落地此方案，见状态表。）
- [ ] **尾块统一 boxed**：`tail_ocp_fp8` / `tail_dynrange_fp4` 仍是递归尾块——`M%TileM≠0` 时会无限模板递归导致编译失败；默认 `M=8` 时 `M_tail=0` 未触发，属潜在 bug，待统一改为 boxed 部分 tile（参照 `tail_cublas_fp8`）。（`tail_ocp_fp4` 已用 boxed M_tail，不在此列。）
- [ ] **linx 4 参带 CmpMode 的 TCMP/TCMPS**：cuBLAS 现用 min/max + 默认-EQ 比较模拟 GT/LT/NE；补齐后可切 `compute_cublas_core` 末尾保留的 IDEAL 版（RECORD 问题4）。
- [ ] **FP4_E1M2 / FP8_E5M2** 输出 dtype（`emax_field` trait 已覆盖，仅缺 kernel/driver）。
- [ ] **FP16 / FP32 输入**（DESIGN §3.1 32bit 域路径已设计，代码未落地）。
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
