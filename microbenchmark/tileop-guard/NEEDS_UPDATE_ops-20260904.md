# microbench 看护 demo — 待按 ops-20260904 新 API/model 更新清单

- 生成：2026-09-04（新 tag `ops-20260904` 工具链下实跑 microbench 后整理）
- 工具链：clang++ `e427d1429c0e`（llvm `67d3ac98`）/ gfrun `0c433cd1`（SuperScalarModel `codex/consolidate-post-main-fixes-20260903` @ `49547742`）/ TileOP-API `f8fb8943`
- 本轮结果：**82 精度正确 / 13 精度失败 / 4 run-only / 9 编译失败 / 23 run-fail = 131**
- 对比重建前（旧工具链）：**101 精度正确** → 大幅下滑。**主因判断：guard demo（PR #96，针对更旧 TileOP API 编写）与 ops-20260904 的新 API/model 代际不兼容**——整族一起崩是 API/契约漂移的签名，不是零散 bug。每条仍需按 `references/golden-discipline.md` 逐个「回归 vs witness 甄别」最终定性，本清单是排查起点。

---

## A. 需按新 API/契约更新的 demo（重建前是 PASS，本轮崩）—— 优先处理

### A1. SFU 行归约 —— 编译失败（新 TileOP API 改了签名）
`trowmax` `trowmin` `trowsum` `trowprod` `trowargmax` `trowargmin`
- 观测：`compile=FAIL`（旧工具链下全 PASS）。
- 判断：新 TileOP-API 头（`f8fb8943`，较 demo 目标晚 40+ commit）改了行归约 intrinsic 签名/模板参数。
- 动作：对照新头 `include/**` 与 tileop-usage 新版文档订正调用签名；列归约 `tcol*` 编译仍过但精度崩（见 A2），一并核对。

### A2. SFU 列归约 —— 精度失败（新 model 归约行为变化）
`tcolmax` `tcolmin` `tcolsum` `tcolprod` `tcolargmax`（`tcolargmin` 仍 PASS）
- 观测：`compile=ok gfrun=ok precision=MISMATCH`（旧工具链下全 PASS）。
- 判断：编得过但新 model 归约结果与旧 golden 不符——可能新 model 归约语义/布局变化，或 golden 需随新规范更新。
- 动作：查 pto-spec 新版归约 ASL + 新 model 实现，甄别是 model 回归还是 golden 该更新。

### A3. CUBE matmul —— 全部 run-fail（新 model 收紧 cube 契约）
`tmatmul` `tmatmul_acc` `tmatmul_bias` `tmatmul_mx` `tmatmul_rowmax` `tgemv`
（`tmatmul_f16` `tmatmul_relu` 仍 PASS）
- 观测：`gfrun=FAIL`（旧工具链下全 PASS）。
- 判断：新 model（consolidation 分支，含 cube store layout / CUBE subview / SizeCode 校验 等收紧）拒绝旧 demo 的 cube 用法。
- 动作：读新 model 的 cube 断言 + `modelSpec/cube/`，按新 CUBE layout/SizeCode 契约更新 demo 的 tile 声明与 matmul 调用。

### A4. FIXP fixpipe 归约/cscale —— run-fail（同 cube 契约收紧）
`chain` `cscale` `group_max` `rowmax_acc`（`convert` `prelu` 仍 PASS）
- 观测：`gfrun=FAIL`（旧工具链下全 PASS）。
- 判断：与 A3 同源——fixpipe 走 cube 通路，新 model cube 契约变化牵连。
- 动作：随 A3 一起按新 cube 契约订正。

### A5. TLSU —— 个别从 PASS 掉落
- `range_assemble`：`gfrun=FAIL`（旧 PASS）。查新 model TLSU/range descriptor 契约。
- `mgather_cas`：`precision=MISMATCH`（旧 PASS）。查新 model gather/CAS 语义或 golden。

---

## B. 新增算子 demo（今日新写，尚未通过，需继续开发）
`tgpr2t`（compile-fail）`tpack` `tpermute` `tshuf` `tunpack`（run-fail）
- 为新 TileOP layout 算子（TGPR2T/TPACK/TPERMUTE/TSHUF/TUNPACK，随 TileOP 文档更新新增）写的 demo，本就在开发中。按 `references/add-new-guard.md` 从 pto-spec 核实语义后完善。

---

## C. 已有缺口（本次之前就在崩，非本轮回归）—— 不必混入更新清单
- 编译失败：`thistogram`（linx 后端 Match Error）、`tmatmul_bf16`（clang frontend abort）。
- run-fail（含就绪 golden）：`textract` `tgather` `timg2col` `tscatter`（就绪 golden，模型缺口）、`gmov` `range_subview` `reinterpret_tcmp`。
- precision-fail witness（模型缺口，挂 issue）：`tinsert`(gfrun-8) `tquant`(gfrun-5) `s8_scalar`/`scalar_generic`/`s8_vector`/`vquant_f16`/`lrelu`(gfrun-N4)。
- `region_tilearray`：由 precision-fail(gfrun-N3) 变 run-fail（新 model 下更早崩），仍属已知缺口。

---

## D. 本轮改善（新 model 修复）
- **`tlog`：precision-fail witness → PASS** ✅ 新 model 的 TLOG=自然对数修复（`d10e6808`）生效,原 gfrun-6 缺口已闭合。

---

## 备注
- run-only 4 项（`tload`/`tstore`/`tmov`/`tprefetch`）无可校输出,不在更新范围。
- VEC 全 35 PASS,不受影响。
- 更新 demo 后须 `make clean_all` 全量重跑再签字（改共享头/换工具链增量 make 不重编,见 SKILL 硬约束）。
- 待 demo 按新 API 更新到位前,新 tag 上的 microbench 绝对数字应理解为「demo 与新 API 的代际差」,而非工具链质量。
