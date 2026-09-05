# [gfrun][NA] ops-20260904 模型缺口汇总（7 项）：B.FPATR None、行列归约 skew、TINVALID layout selector、pre-quant 丢 FP19 scale、TCVT 舍入、TQUANT 忽略 mult/zp、TINSERT 不保留 base

> **已提交**（2026-09-05）：`LinxISA/SuperScalarModel` **#560** — https://github.com/LinxISA/SuperScalarModel/issues/560

接口按文档正确写、编译通过，但 gfrun 侧崩溃或数值错误。每项均已用 **release 自带参考源现编复现** 或 **model 源码
定位 / 反汇编隔离 / 数值探针**自证——排除 kernel 侧 dtype/shape/layout/operand-role 写错。编译器后端类缺陷
另行提 linx issue。（本 issue 只收**已严格自证为模型侧**的缺口；另有若干 run-fail 崩点尚未证非 demo，暂不列入。）

## 组件版本清单

| 组件 | 仓库 | 分支 / tag | commit |
|---|---|---|---|
| SuperNPUBench(看护 demo，**主复现**) | ziyang-cheng/SuperNPUBench | `tileop-guard-batch1`(PR #96) | `7395fec` |
| SuperNPUBench(release 参考源，**独立自证**) | PTO-ISA/SuperNPUBench | `ops-20260904` | `a0ddcc3` |
| **SuperScalarModel(本 issue 目标)** | LinxISA/SuperScalarModel | `codex/consolidate-post-main-fixes-20260903` | `49547742` |
| Linx-TileOP-API | LinxISA/Linx-TileOP-API | `linx` | `f8fb894` |
| llvm-project(linx clang/lld) | LinxISA/llvm-project | `dev-llvm15_56` | `67d3ac9` |
| musl | LinxISA/linx-musl | `linx` | `af0dfc2` |
| jemalloc | LinxISA/jemalloc | `linx` | `4495309` |
| linux-linxisa | LinxISA/linux | `main` | `1055a74` |

> 看护 demo 代码见 **SuperNPUBench PR #96**：https://github.com/PTO-ISA/SuperNPUBench/pull/96 （分支 `tileop-guard-batch1` @ `7395fec`）
> 工具链指纹（2026-09-05）：clang++ md5 `e427d1429c0e`、gfrun md5 `0c433cd11c00`。

## 复现步骤

**主复现（看护 demo，PR #96 自包含，覆盖本 issue 全部 case）**：
```bash
# 0. checkout 看护 demo（PR #96, 分支 tileop-guard-batch1 @ 7395fec）
git fetch origin tileop-guard-batch1 && git checkout 7395fec
# 1. 指向工具链（COMPILER_DIR=linx clang / GFRUN=SuperScalarModel/bin/gfrun）
source microbenchmark/tileop-guard/env.sh
# 2. 单 case 四态验证（编译 → gfrun → host golden）
bash microbenchmark/tileop-guard/run_guard.sh <domain> <case>
```

**独立自证（§1/§2 附加）**：在 **SuperNPUBench release `ops-20260904`(`a0ddcc3`)** 检出上现编其**自带**
参考源（带 `verify()` 的 `microbenchmark/{cube,fixp}` 与 reduction 参考 kernel）——该检出才含 fixp 微基准与
reducemax 参考 kernel（PR 分支基线较旧不含）。用于佐证「即便 release 自身规范写法也同崩」。

---

## gfrun-1 · B.FPATR None：CUBE matmul 全族 + FIXP float 输出族崩

**涉及接口**：TMATMUL / TMATMUL_ACC / TMATMUL_BIAS / TMATMUL_MX / TMATMUL_ROWMAX / TGEMV / fixpipe(keep_acc/bias/acc/cscale/rowmax/groupmax)。

**问题**：float 累加输出（PreQuantMode=0）的 matmul/fixpipe 一律崩。header 宏 `PTO_MATMUL_HEADER` / `PTO_FIXP_ATTR`
（`jcore/template_asm.hpp`）对非量化路径**硬编码字面量** `B.DATR %D, byte0, Zero, RNE, NOSAT`（RMode=RNE），
model 断言在 PreQuant=0 时要求 RMode=FRM_NONE 且 Sat=0，直接触发。RNE 是 asm 字符串字面量、无模板参数，
上层无法通过 API 改写。量化/relu（PreQuant≠0）路径正常。

**复现**：
```bash
# (a) 主复现 · 看护 demo（PR #96）
bash run_guard.sh cube tmatmul        # tmatmul/_acc/_bias/_mx/_rowmax/tgemv 同崩
bash run_guard.sh fixp chain          # chain/cscale/group_max/rowmax_acc 同崩
# (b) 独立自证 · 在 SuperNPUBench release a0ddcc3 检出上现编其自带微基准（带 verify()）
#     cube 微基准在 PR 分支亦有；fixp 微基准仅 release 检出有
cd microbenchmark/cube && make TESTCASE=tmatmul_fp32_32x32x32 && \
  $GFRUN -f ../../output/microbenchmark/cube/elf/cube/tmatmul_fp32_32x32x32.elf
cd microbenchmark/fixp && make FIXP_MODE=KEEP_ACC && \
  $GFRUN -f ../../output/microbenchmark/fixp/elf/fixp/fixp_tmatmul_keep_acc*.elf   # release 检出
```

**错误信息**：
```
gfrun: illegal instruction: ASSERTION FAILED:
  (preQuantMode != 0 || (rMode == FRMMode::FRM_NONE && !saturation))
  && "B.FPATR None requires B.DATR RMode=NONE and Sat=0"
  func AccumulateBlockInfo, file emulator/engine/AccumulateBlockInfo.cpp:1109
```

**附加说明（自证非 demo）**：release `gen_cases.py` 生成、带 `verify()` 数值校验的全部 11 个 cube 微基准
（fp32/fp16/bf16/i8 × plain/acc/bias）+ fixp `KEEP_ACC/BIAS/ACC/ACC_CSCALE/ROWMAX/GROUPMAX` 现编全崩同一断言；
预编 ELF「PASS」是旧 header（发 RMode=NONE）编出的陈旧二进制。`microbenchmark/fixp/gfrun_fixp_issue_20260830.md`
（release 自带）亦记同类模型侧拒绝。**建议**：header 非量化路径改发 `RMode=NONE`，或 model 放宽 PreQuant=0 + float 输出时允许 RNE。

---

## gfrun-2 · 行/列归约 header↔model 维度 skew

**涉及接口**：TROWMAX/MIN/SUM/PROD/ARGMAX/ARGMIN、TCOLMAX/MIN/SUM/PROD/ARGMAX。

**问题**：header 对静态-valid 源走 Branch-1，把 **dst** 的 `ValidCol/ValidRow/Cols` 发进指令 `lb0/lb1/lb2`
（objdump 实测 TROWMAX 发 `lb0=1, lb1=M, lb2=1`）；model 把这三字段解码为**源**归约几何。
- 行：`Block.cpp:1817` 要求 `source->validCol==lb0(=1)` 且 `source->col==lb2(=1)`，与 16×16 源不符 → 崩。
- 列：`TEPLEngine.cpp:2669` 取 `validRow=lb1`，`ExecuteTCOLMAX` 只 `for i in 0..lb1(=1)` 归约 → 输出 == 源 row0。

**复现**：
```bash
bash run_guard.sh sfu trowmax     # 行族崩（trowmin/sum/prod/argmax/argmin 同）
bash run_guard.sh sfu tcolmax     # 列族精度失败：out.bin == 输入 row0（tcolmin/sum/prod/argmax 同）
```

**错误信息**（行）：
```
gfrun: illegal instruction: ASSERTION FAILED:
  "PTO v0.58 row reduction requires final compatible source and destination descriptors"
  file isa/Block.cpp:1817
```
（列不崩，`out.bin` 逐字节等于输入第 0 行。）

**附加说明（自证非 demo）**：在 **SuperNPUBench release `a0ddcc3` 检出**上，官方参考 kernel
`benchmark/one-level-arch/kernels/single_thread/reduction/reducemax_rowvec.hpp`（静态源 + 真 tM×1 dst）编出的
预编 ELF `output/kernel/reduction/reducemax_row/…_DTypeint32_t_tM16_tN128_GM16_GN8192.elf`，在当前 model
崩**同一断言**（已实测），同发 `lb0=1/lb1=16/lb2=1`。guard demo 已按此参考 kernel 改真 M×1 / 1×N dst。
（SuperNPUBench reduction test main 无 res_check，其「PASS」仅代表跑通不崩、非数值正确。）**建议**：header
Branch-1 应发**源**归约维度到 lb0/lb1，或 model 改按 dst 维度解码。

---

## gfrun-3 · TINVALID 保留/删除选择子 —— 0.58.5 layout 算子

**涉及接口**：TPACK / TPERMUTE / TSHUF / TUNPACK。

**问题**：model 解码器把这些 0.58.5 layout-and-rearrangement（TEPL mode 3）算子的 tile 选择子映射到
`TileOp::TINVALID`（保留/删除），即 model 侧尚未实现这些选择子。

**复现**：
```bash
bash run_guard.sh sfu tpack       # tpermute / tshuf / tunpack 同
```

**错误信息**：
```
gfrun: illegal instruction at 0x...: reserved/deleted tile selector
  file emulator/SoftCore.cpp:930  (ExecuteTemplate: currentBlock->tileOp == TileOp::TINVALID)
```

**附加说明（自证非 demo，且与 demo 内容无关）**：崩点 `Block.cpp:815-834` 是纯**解码期、仅看选择子位**的检查
——把 `(mode, function)` 位与硬编码 `function_masks[4]`（注释注明源自 `pto-spec@3b8cd26 tile-operations.json`
的 87 个 active TEPL pair）比对，不在表中即 `tileOp=TINVALID; return`，**在任何 operand 绑定 / Tile 分配之前**
（源码注释原文 "Fail closed before any operand binding or Tile allocation"）。故本崩**与我 demo 的 tile 形状 /
control 值 / 操作数完全无关**：只要指令发出 TPACK/TPERMUTE/TSHUF/TUNPACK 选择子（该 catalog 快照未含），
就在解码期被拒。另核对 demo 本身亦严格合规：tpack control=0x0202(字段宽 2/2,和≤4)、tunpack control=0x0201
(offset1/count2)=doc 示例原样、tpermute 索引 tile=32×128(U32 每元素 4 索引字节,合 doc 规则)、tshuf controls
U32 同几何 + control=0(UP/seg0/SELF)——均按 `*.md` 签名与约束写。**建议**：model 解码器 catalog 升级到含
0.58.5 layout 选择子并补执行语义。

---

## gfrun-4 · 矩阵 pre-quant 丢弃 FP19 scale（只保留 offset / LReLU slope）

**涉及接口**：TMATMUL + `fixp::s8` / 标量·向量量化后处理（PreQuantMode QF322S8Pre=24、VQF322S8Pre=23、F16 量化等）。

**问题**：矩阵 pre-quant 路径**完全忽略 FP19 scale 乘子**，输出只体现 offset（LReLU 负半轴 slope 仍生效）。
pto-spec `matrix-quantization.asl` NDF-001 明文 "Matrix PreQuant **MUST multiply by the selected FP19 scale**,
round and saturate at an assigned S5/S9/S17 intermediate, add the signed offset, and encode"。模型违反此项。

**复现**：
```bash
bash run_guard.sh fixp s8_scalar        # scalar_generic / s8_vector / vquant_f16 同（输出恒=offset）
bash run_guard.sh fixp lrelu            # 正半轴丢 scale=offset、负半轴保 slope
```

**错误信息（数值 witness，非崩溃）**：
```
[s8_scalar] MQUANT-S8 MISMATCH 993/1024  got=5(=offset) ref=clip(clip(rint(D*16),-256,255)+5,-128,127)
```

**附加说明（自证 demo+golden 绝对正确，缺口在模型）**：
- descriptor 位段：demo `make_s8_quant` 打包 `scale<<13 | offset<<37`，与 `MatrixQuantParameter`（`result[31:13]=fp19`、
  9-bit offset `result[45:37]`）逐位一致；FP19 解码（exp[17:10]/frac[9:0]）与 `fp19.asl` 一致；`fixp::s8()`
  （header `pto_tile.hpp:1661`）原样转发 64bit 描述符，反汇编确认发 `B.FPATR 24`（QF322S8Pre）。
- **scale 不变性探针**：把 FP19 scale 依次设 1.0 / 16.0 / 256.0（0x1FC00 / 0x20C00 / 0x21C00），模型输出**恒为
  uniform=offset**——scale 取值对输出零影响 ⇒ 模型彻底丢弃 scale，与 demo/golden 无关。
- **lrelu 交叉验证**：模型输出 == `where(D<0, D*slope, 0)+offset` **1024/1024** 精确 ⇒ 模型丢正半轴 scale、
  保负半轴 slope，行为自洽。
- tmatmul_f16 通过（golden=A@B→f16）证明 CUBE 矩阵乘本身正确，故非 matmul 问题、纯 pre-quant scale 丢弃。

**建议**：矩阵 pre-quant 消费 FP19 scale 乘子（`MatrixQuantParameter[31:13]`）参与 `D*scale`，与 TQUANT 同类缺口（gfrun-6）。

---

## gfrun-5 · TCVT float→int 用 RNE，违反规范默认 RTZ

**涉及接口**：TCVT（fp32→s32，默认 RMode=0）。

**问题**：`TCVT.asl` normative 明文 "RMode zero selects **RTZ for floating-to-integer conversion** and RNE for
every other conversion"，ASL 返回 `NumericRound_RTZ`；工具链 jcore `TCvt.hpp` 亦 `static_cast<int>`（RTZ）。
但模型对 float→int 做 **RNE（四舍六入五取偶）**。

**复现**：
```bash
bash run_guard.sh vec tcvt
```

**错误信息（数值 witness）**：
```
[tcvt] CVT MISMATCH 63/256 first@2 got=-126 ref=-125 src=-125.5 mode=rtz
```

**附加说明（自证 demo+golden 正确）**：demo `TCVT(d,s)` 省略 RMode → RMode=0 → 规范应 RTZ（截断），调用无误。
实测模型输出 == RNE **256/256**、== RTZ 仅 193/256；中点 −125.5 模型吐 −126(RNE) 而规范 RTZ 应 −125（63/256
中点分歧）。golden 现按规范钉 `rtz`（`np.trunc`），暴露此舍入缺口。**建议**：TCVT float→int 默认舍入改 RTZ。

---

## gfrun-6 · TQUANT 静默忽略 multiplier / zeroPoint

**涉及接口**：TQUANT。

**问题**：`TQUANT<RNE,Sat>(dst,src,float multiplier,int32 zeroPoint)` 文档语义
`q=clamp(round(src*multiplier)+zeroPoint,-128,127)`，bundle 编码 `B.IOR MultiplierFP32, ZeroPoint`。
但模型忽略这两个参数：传 `mult=0.5f, zp=1` 实测输出逐字节 == `clamp(round_RNE(src),-128,127)`（按 mult=1/zp=0）。

**复现**：`bash run_guard.sh sfu tquant`（demo 传真实 `mult=0.5f, zp=1`）。

**错误信息（数值 witness）**：
```
[tquant] QUANT MISMATCH 2020/2048  首元素 src=-256 got=-128 ref=-127
```

**附加说明（自证非 demo）**：对 `TQUANT<RNE,true>(d,s,0.5f,1)` 反汇编，`BSTART.TEPL TQUANT` 之后确有
`B.IOR`，0.5f 常量 `0x3f000000` 经 materialize 送入 bundle——**后端已正确发射 multiplier operand，是 gfrun
执行时未消费**。golden 按 pto-spec `format-conversion/TQUANT.md` 全语义钉，现为精度失败 witness。与 gfrun-4
（矩阵 pre-quant 丢 scale）同类。**建议**：TQUANT 执行消费 multiplier/zeroPoint。

---

## gfrun-7 · TINSERT 不保留 base、只写部分窗口

**涉及接口**：TINSERT。

**问题**：pto-spec `layout-and-rearrangement/layout/TINSERT.md` / ASL 规定 TINSERT 把 source patch 插入
**快照的旧 destination**（窗口外 base **保留**），即 `dst=base.copy(); dst[OR:OR+SM,OC:OC+SN]=patch`。
但模型写出全新 tile、不保留 base，且只落 patch 的部分窗口。

**复现**：`bash run_guard.sh sfu tinsert`。

**错误信息（数值 witness）**：
```
[tinsert] INSERT MISMATCH 480/512  首元素(窗口外)即与 base 不符
```

**附加说明（自证非 demo）**：demo host 拥有 base + patch，golden 按 pto-spec「快照旧 dst + 窗口覆盖」断言
（`check_insert`：`base.copy()` 后窗口赋值，语义经审计核实正确）。模型不保留 base → 精度失败 witness。
**建议**：TINSERT 按「快照旧 dst + 窗口插入」实现。

