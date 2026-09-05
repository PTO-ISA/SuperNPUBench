# [gfrun][ops-20260904] 三类模型缺口：B.FPATR None matmul/fixp、行列归约 header↔model skew、TINVALID layout selector

## 验证环境

| 组件 | 分支 | commit / md5 |
|---|---|---|
| SuperScalarModel (gfrun) | codex/consolidate-post-main-fixes-20260903 | `4954774` / md5 `0c433cd11c00` |
| Linx-TileOP-API | linx | `f8fb894` |
| linx clang++ | dev-llvm15_56 | md5 `e427d1429c0e` |
| SuperNPUBench | ops-20260904 | `a0ddcc3` |

三类缺口均已用 **release 自带参考源现编复现 / model 源码定位**，排除 kernel 侧 dtype/shape/layout/operand-role 写错。

---

## §1 B.FPATR None：CUBE matmul 全族 + FIXP float 输出族崩

### 现象
```
gfrun: illegal instruction: ASSERTION FAILED:
  (currentBlock->preQuantMode != 0 ||
   (currentBlock->blockAttr->rMode == FRMMode::FRM_NONE && !currentBlock->blockAttr->saturation))
  && "B.FPATR None requires B.DATR RMode=NONE and Sat=0"
  func AccumulateBlockInfo, file emulator/engine/AccumulateBlockInfo.cpp:1109
```

崩的用例（float 累加输出，PreQuantMode=0）：
- CUBE：`tmatmul` / `tmatmul_acc` / `tmatmul_bias` / `tmatmul_mx` / `tmatmul_rowmax` / `tgemv`
- FIXP：`chain` / `cscale` / `group_max` / `rowmax_acc`

正常的用例（量化 / relu 后处理，PreQuantMode≠0，走另一分支）：`tmatmul_f16` / `tmatmul_relu` / fixp `convert` / `prelu`。

### 根因
TileOP-API header 宏 `PTO_MATMUL_HEADER` / `PTO_FIXP_ATTR`（`jcore/template_asm.hpp`）对**非量化（float 累加输出）**
路径**硬编码字面量** `B.DATR %D, byte0, Zero, RNE, NOSAT`（RMode=RNE），同时 B.FPATR 的 PreQuantMode=0。
model 断言在 PreQuant=0 时要求 RMode=FRM_NONE 且 Sat=0，于是 RNE 直接触发。RNE 是 asm 字符串字面量、无模板参数，
**上层无法通过 API 改写**。

### 自证「非 demo 写法错」
release 自带、`gen_cases.py` 生成、**带 `verify()` 数值校验**的参考微基准，用当前工具链**现编**后：
```
$ make TESTCASE=tmatmul_fp32_32x32x32   # microbenchmark/cube/src/
$ gfrun -f tmatmul_fp32_32x32x32.elf
  → ASSERTION FAILED ... B.FPATR None requires B.DATR RMode=NONE and Sat=0
```
全部 11 个 cube 微基准（fp32/fp16/bf16/i8 × plain/acc/bias）+ fixp `KEEP_ACC/BIAS/ACC/ACC_CSCALE/ROWMAX/GROUPMAX`
现编全崩同一断言。**预编 ELF 之所以「PASS」是因为它们是旧 header（发 RMode=NONE）编出的陈旧二进制**。
`microbenchmark/fixp/gfrun_fixp_issue_20260830.md`（release 自带，更早快照）亦记录同类模型侧拒绝。

### 建议
header 非量化 matmul/fixp 路径改发 `B.DATR …, NONE, NOSAT`，或 model 放宽：PreQuant=0 且 float 输出时允许 RMode=RNE。

---

## §2 行/列归约 header↔model 维度 skew

### 现象
- 行归约 `trowmax/trowmin/trowsum/trowprod/trowargmax/trowargmin`：
  ```
  gfrun: illegal instruction: ASSERTION FAILED ...
    "PTO v0.58 row reduction requires final compatible source and destination descriptors"
    file isa/Block.cpp:1817
  ```
- 列归约 `tcolmax/tcolmin/tcolsum/tcolprod/tcolargmax`：不崩，但 `out.bin == 输入 row0`（逐字节），
  即 model 只归约了 1 行。`tcolargmin` 恰好 PASS（真 argmin 落 row0 巧合）。

### 根因
header `template_asm.hpp` 的 TROW*/TCOL* 对静态-valid 源走 **Branch-1**，把 **dst** 的
`ValidCol/ValidRow/Cols` 发进指令 `lb0/lb1/lb2`（objdump 实证 TROWMAX 发 `lb0=1, lb1=M, lb2=1`）。
model 侧把这三个字段解码为**源**归约几何：
- 行：`Block.cpp:1817` 要求 `source->validCol==lb0(=1)` 且 `source->col==lb2(=1)`，与 16×16 源不符 → 崩。
- 列：`TEPLEngine.cpp:2669` 取 `validRow=lb1`，`ExecuteTCOLMAX` 只 `for i in 0..lb1` 归约；header 发 `lb1=out.ValidRow=1`
  → 只归约 1 行 → 输出 == 源 row0。

header 的 Branch-2（源 DYNAMIC valid）才发 `src.GetValidCol()`（正确），但 DYNAMIC tile 的 mask 成员未初始化=0 →
TLOAD 先崩，且需戳未文档化公有成员——非 API 可达路径。

### 自证「非 demo 写法错」
release 自带预编 `reducemax_row …int32_t_tM16_tN128….elf`（由官方参考 kernel
`benchmark/one-level-arch/kernels/single_thread/reduction/reducemax_rowvec.hpp` 编出，静态源 + 真 tM×1 dst）
在当前 model 上崩同一断言，TROWMAX 同发 `lb0=1/lb1=16/lb2=1`。guard demo 已按参考 kernel 改为真 M×1 / 1×N dst。
（注：SuperNPUBench reduction test main 无 res_check，其「PASS」仅代表跑通不崩，非数值正确。）

### 建议
header Branch-1 应发**源**归约维度（`src.ValidCol/ValidRow`）到 lb0/lb1，而非 dst 维度；或 model 改按 dst 维度解码。

---

## §3 TINVALID 保留/删除选择子 —— 0.58.5 layout 算子

### 现象
`tpack / tpermute / tshuf / tunpack`：
```
gfrun: illegal instruction at 0x...: reserved/deleted tile selector
  file emulator/SoftCore.cpp:930  (ExecuteTemplate: currentBlock->tileOp == TileOp::TINVALID)
```

### 根因
model 解码器把这些 0.58.5 layout-and-rearrangement（TEPL mode 3）算子的 tile 选择子映射到
`TileOp::TINVALID`（保留/删除），即 model 侧尚未实现这些选择子。

### 自证「非 demo 写法错」
TileOP-API header（f8fb894）已暴露这些 intrinsic（`template_asm.hpp` 内有 `TPACK/TPERMUTE/TSHUF/TUNPACK`
完整模板 + static_assert），demo 按 doc 签名调用，编译器接受并下译出合法 bundle（编译通过、运行到 block），
仅 model 解码阶段判 TINVALID。header 暴露 + 编译器发射 vs model 未实现 = 纯模型缺口。

### 建议
model 解码器补齐 TPACK/TPERMUTE/TSHUF/TUNPACK 选择子与执行语义（对齐 0.58.5 layout ASL）。
