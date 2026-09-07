# res_check 数值校验问题分类汇总 — 2026-09-01

> 数据源：`/tmp/res_check_run/summary_corrected.tsv`（492 ELF，271 PASS / 221 FAIL，55.1%）
> 编译器：`linx-toolchain-build` main（clang-15.0.4，`linx64v5-unknown-linux-musl`），`COMPILER_DIR=.../linx_blockisa_llvm_musl/bin`
> 模型：`SuperScalarModel/bin/gfrun`（多线程/共享模式 `-s softcore.multiThreadNum=4`）

---

## 1. 总览

校验分两个阶段，问题按 **编译器 / 算子 / gfrun 模型** 三类归因。

| 阶段 | 集合 | 数量 | 归因 |
|------|------|------|------|
| **编译阶段**（未产出 ELF，排除在 492 之外） | clang 前端崩溃 | 2 | **编译器** |
| 编译阶段 | CUBE 算子旧 API（`TileLeft/Right/Acc`→`CubeTile*`）+ HIF4X2 dtype | 若干 | **算子**（已单独记录，见 §3） |
| **运行阶段**（492 ELF 中的 FAIL） | 数值失配 rc=0 R2=1 | 178 | **gfrun 模型** |
| 运行阶段 | 非法指令断言 rc=1 | 40 | **gfrun 模型** |
| 运行阶段 | gfrun SIGABRT rc=134 | 1 | **gfrun 模型** |
| 运行阶段 | 日志截断，R2 不可定 | 2 | 待定 |

**关键结论**：运行阶段 221 个 FAIL 中，**编译器 = 0、算子 = 0**，219 个为 gfrun 模型问题（+ 2 待定）。此前分析把 `s_qf_s4/v_qf_s4` 与 `trem_fp16` 计入"算子"是误归因，本次据数据纠正（见 §4）。

---

## 2. 编译器问题（2）

### 2.1 clang-15 前端 SIGABRT（exit 134）— `trem_fp16_16x16` / `trems_fp16_16x16`

| 模式 | 普通 build | `-DRES_CHECK` build | 现象 |
|------|-----------|---------------------|------|
| `trem_fp16_16x16` | ✅ 编译通过 | ❌ exit 134 | clang-15 前端 `CC1Command::Execute` 崩溃 |
| `trems_fp16_16x16` | ✅ 编译通过 | ❌ exit 134 | 同上 |

- **触发条件**：仅 `res_check=on`（`-DRES_CHECK`）路径崩溃。该路径在 `src/trem_fp16_16x16.cpp` 用 `std::fmod((double)a[i],(double)b[i])` 构造参考值、并以 `(__half)verify_epsilon<__half>()` 传 eps 给 `verify()`；与 `-mlxbc -fenable-matrix` 的矩阵代码生成路径叠加时触发 clang 前端内部断言。
- **归属判定**：clang 崩溃回显 `PLEASE ATTACH THE FOLLOWING FILES TO A BUG REPORT`，属编译器前端 bug，非算子可修。两个模式未产出 ELF，故不在 492 之内、不计入 221。
- **处置**：记为编译器 issue；如需在 RES_CHECK 下暴露底层模型问题，需等 clang 修复或另觅绕过路径（不属本次算子修复范围）。

---

## 3. 算子侧问题（运行阶段 0；编译阶段已单独记录）

经逐项复核，运行阶段 221 个 FAIL **不含算子侧代码 bug**。此前列入"算子"的候选项纠正如下（详见 §4）：

| 候选项 | 原归因 | 纠正后归因 | 依据 |
|--------|--------|-----------|------|
| `s_qf_s4` / `v_qf_s4` | 算子（check_zero_result 对 S4 零点不适用） | **模型**（S4 4-bit dtype 处理） | 同结构的 `s_qf_s8`/`v_qf_s8`（`mk_desc(1,0,9)`，零点=0）均 PASS，仅 4-bit `__int4x2` 失败 → 非 check_zero_result 误报 |
| `trem_fp16`/`trems_fp16` | 算子/编译器 | **编译器**（见 §2） | 仅 `-DRES_CHECK` 崩溃，普通 build 通过 |

编译阶段真正的算子侧项（**不在 221 之内**，已单独建档）：

1. **CUBE 算子 API 迁移**：`one-level-arch/kernels/` 下部分头文件仍用旧 `TileLeft/TileRight/Acc`，命中 `static_assert(!IsCubeLayout || ...)` 编译失败。已记录于 `benchmark/one-level-arch/kernels/single_thread/deepseek/TileKernels迁移说明.md` 与 `multi_thread/matmul/hif4x2_compile_fail_issue.md`。属较大重构，不在本次快速修复范围。
2. **HIF4X2 dtype gap**：PTO 头文件 `pto_tile.hpp` 的 `matrix_numeric_class()` 未列 `__type_fp4_hif4x2`，见 `hif4x2_compile_fail_issue.md`（PTO 头/编译器侧）。
3. **`report_fixp.py` `mxacc_cscale` 校准（已修复 ✅）**：`.diss` 4 条 B.IOT 源、`MATH_SRC_BY_MODE["mxacc_cscale"]` 原值 `2` 得 `aux=2`，与期望 `aux=1` 不符。已据 ISA 语义改为 `3`：`TMATMULMX.ACC + CScaleEn`（`ScaleMask=0` 丢两条 MX 块尺度载体，留 A/B/C 三条 math 源，第 4 条为 CScale 后处理 aux）。修复后 `report_fixp.py` 由 `PASS=121 FAIL=1` → `PASS=122 FAIL=0`（余 `lrelu_only` BLOCKED 为已知 LLVM 后端 pattern gap，非本次可修）。

---

## 4. 归因纠正记录

本次据 `summary_corrected.tsv` 逐条复核，纠正此前分析的三处误归因：

1. **`s_qf_s4`/`v_qf_s4` → 模型，非算子**。`check_zero_result`（`fixp_tmatmul.cpp:39`）在 RES_CHECK 下：A/B 置 0、D 预填 0x01，正确结果为 0 则 D 被清零→PASS；D 残留非零→FAIL。S4 用 `mk_desc(1,0,5)`（零点=0），与 S8 的 `mk_desc(1,0,9)` 同构；S8 既 PASS，说明非"零点致 D 非零"的 check_zero_result 误报，而是 4-bit `__int4x2` 打包 dtype 在模型 TMATMUL 路径的专门缺陷。
2. **`trem_fp16`/`trems_fp16` → 编译器，非算子/非模型**。普通 build 通过，仅 `-DRES_CHECK` 触发 clang 前端 SIGABRT。
3. **vec binary TEPL 源 tile 读零缺陷覆盖面收窄**：原"114+42+10+10=176 全归 vec bug"过宽。`micro/scalar`（42）走标量寄存器 ALU 路径（`scalar_bench.hpp`/`bench_latency`）、`micro/memory`（10）走 tile load/store、`micro/cube`（10）走 MA/CUBE PE，均非 gfrun binary TEPL 路径。**经核实的 binary TEPL `srcTile[0]` 读零缺陷精确覆盖 `micro/vector` 114 个 tile 二元/一元算子**；scalar/memory/cube 的数值失败为并行路径，根因待查（见 §5.1）。

---

## 5. gfrun 模型问题（219 + 2 待定）

### 5.1 数值失配 rc=0 R2=1（178，全 microbench）

| 子集 | 数量 | 路径 | 根因 |
|------|------|------|------|
| `micro/vector` | 114 | gfrun binary TEPL（emulator） | **已核实（gfrun trace）**：`ExecuteTADD` 正常执行，但 `srcTile[0]` 经 `LoadFromTileRegisterSrc`→`LoadFromTileRegisterCube` 读出**全零**（trace：SrcL=0, SrcR=正确, Dst=SrcL op SrcR）。根因：`srcTile[0]->tileInfo->validRow/validCol` 在 block decode/binder 未置位 → `CubeEngine.cpp:599` 的 `if(row>=validRow\|\|col>=validCol) continue` 全跳过 → 返回初值零 matrix。`srcTile[1]` 正常。详见 `SuperScalarModel/issue_gfrun_binary_tepl_srctile0_zero.md`。 |
| `micro/scalar` | 42 | 标量寄存器 ALU（`bench_latency`/`bench_throughput`） | 并行路径，根因待查（疑似标量 ALU 结果写回/读就绪缺陷） |
| `micro/memory` | 10 | tile load/store/gather/scatter（`tload`/`tstore`/`mgather`/`mscatter`） | 内存/tile 载入路径，根因待查 |
| `micro/cube` | 10 | CUBE/MA PE（`tmatmul`/`tmatmul_acc`/`tmatmul_bias`） | 疑似 CUBE 路径源 tile 载入缺陷，根因待查 |
| `micro/fixp` | 2 | `s_qf_s4`/`v_qf_s4` | S4 4-bit `__int4x2` dtype 处理缺陷（S8 同构 PASS） |

`micro/vector` 114 覆盖的算子族：`tadd tsub tmul tdiv tmin tmax tneg tnot tshl tshr tshls tshrs txor txors tand tands tor tors tcolexpand* trowexpand* tconcat tcvt texp texpands tlog tmul tmuls tpartadd tpartmax tpartmin tpartmul trecip trelu tabs tadds tdivs tmins tmaxs tsubs tci` 等（fp16+fp32/i16+i32 变体）。

### 5.2 非法指令断言 rc=1（40）

| 断言 | 数量 | 归属 | 受影响模式 |
|------|------|------|-----------|
| `m_handlers.find(grp) != m_handlers.cend()` | 21 | microbench | shift/sqrt 未注册：`sll/sra/srl/sqrt`（i32/i64 lat+thr）+ `tshl/tshr/trsqrt/tsqrt`（fp16/fp32/i16/i32） |
| `srcTile.size()==1 && dstTile.size()==1 && ... TileCarrierW...` | 5 | one-level | `hashtable_lookup`×4 + `multi_thread/fa_MXFP4_VECBF16`×1 |
| `RawTileSourceFits(source, shape)` | 3 | one-level | `hashtable_lookup` col512×2 + `multi_thread/broadcast_PE4`×1 |
| `IsCompatibleLogicalTile(inst->srcs[1], ...)` | 3 | one-level | `aux_fi` / `get_fused_mapping` / `group_count`（deepseek 控制流） |
| `priorSources == 0 && ... IsCompatibleDataTile` | 1 | one-level | `inplace_unique_group_indices` |
| `broadcastShapeLegal` | 1 | one-level | `broadcast_vec_07` |
| `IsCompatibleOperationDataTile(source, ..., "binary TEPL source ...")` | 1 | one-level | `mask_indices_by_tp` |
| `illegal TROWSUM operand or descriptor contract` | 1 | one-level | `reduction_reducesum_row_rowsum_subview` |
| 协作 max-reduction gap（R2=?） | 4 | microbench/fixp | `shared_f16_groupmax` / `shared_rowgroup_maxabs` / `shared_rowmax_init` / `shared_s8_rowmax` |

### 5.3 gfrun SIGABRT rc=1（1）

| 模式 | 现象 |
|------|------|
| `multi_thread_fa_..._HIF8_VECFP32` | gfrun 进程 SIGABRT（exit 134），HIF8 dtype 在 fa 路径触发模型崩溃 |

### 5.4 待定（2）

| 模式 | 现象 |
|------|------|
| `concat_concat_scatter_half` | 日志截断，R2 不可定 |
| `topk` | 日志截断，R2 不可定 |

---

## 6. 与既有 issue 文档的关系

| 既有文档 | 覆盖 | 与本次关系 |
|---------|------|-----------|
| `SuperScalarModel/issue_gfrun_cross_type_dtype_check.md` | broadcast×3 + gelu×1（`source->tileInfo->dataType == block->dataType`） | 断言串不同；本次 broadcast 失败为 `RawTileSourceFits`/`broadcastShapeLegal`/`IsCompatibleOperationDataTile`，属不同/更早断言点，未重叠 |
| `SuperScalarModel/issue_gfrun_bior_stride_unit.md` | B.IOR stride | 未重叠 |
| `SuperScalarModel/issue_gfrun_fixp_tmatmul.md` | fixp TMATMUL | 未重叠（本次 fixp 仅 S4 + 4 协作 gap） |
| `SuperScalarModel/issue_gfrun_hosted_syscall_x1_selection.md` | syscall 选核 | 未重叠 |
| **新建 `SuperScalarModel/issue_gfrun_binary_tepl_srctile0_zero.md`** | micro/vector 114 binary TEPL `srcTile[0]` 读零 | 本次最高影响项，gfrun trace + emulator 代码路径，详见该文档 |

---

## 7. 建议处置

1. **模型（最高优先级）**：在 `isa/Block.cpp` 的 binder finalize 对 binary TEPL 各源正确置 `tileInfo->validRow/validCol`（现仅 `srcTile[0]` 漏置致 `LoadFromTileRegisterCube` 全 continue 返回零），可一次解决 114 个 micro/vector 数值失败。详见 focused issue 文档。
2. **模型（次优先）**：注册 shift/sqrt 的 `m_handlers`，解 21 个非法指令；补 tile carrier / broadcast / deepseek / TROWSUM 合同校验，解 15 个 one-level 断言。
3. **编译器**：向 linx-toolchain-build 报 clang-15 前端在 `-DRES_CHECK`+`-fenable-matrix`+`__half`/`std::fmod` 的 SIGABRT。
4. **算子**：CUBE API 迁移为唯一算子侧实质项，已建档，建议单独立项推进。`report_fixp.py` 的 `mxacc_cscale` 期望校准已本次修复（FAIL→PASS）。
