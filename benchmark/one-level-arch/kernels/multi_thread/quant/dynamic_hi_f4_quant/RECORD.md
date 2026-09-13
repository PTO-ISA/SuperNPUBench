# DynamicHiF4Quant 问题记录

> 本文档如实记录 `dynamic_hi_f4_quant`（hifloat4 / hi_f4，3 级共享 scale，BlockSize 64）落地过程中
> 发现的**真实功能缺口**与在**不新增硬件指令**前提下的**规避方案**。缺口 = 硬件/编译器/仿真器**确实
> 缺失**的功能；规避 = 用现有 tile-op 组合出的等效手段（算子内实现技巧，非硬件补齐）。二者严格区分。
> 权威格式定义与量化算法见 `DESIGN.md`（§1 格式、§2.1 算法权威）；本文对应 DESIGN §3.1/§3.2/§4 的缺口条目。
>
> **状态**：算子处于设计+探针阶段。工具链↔仿真器 skew 下新编 ELF 无法稳定 gfrun/gfsim，故所有验证均为
> **op-review + 编译/反汇编（diss）**，无运行期数值验证。已落地探针：`test/.../src/e6m2_probe.cpp`（问题3）。

---

## 问题1：无原生分段（segmented）max —— `maxPer4` / `maxPer2`

### 结论

§2.1 三级归约要求「相邻 4 元素取 max→16 个」「相邻 2 个再取 max→8 个」「整块取 max→1 个」，即**分段归约**。
PTO-ISA 的 `TROWMAX`（`test/common/template_asm.h:83`）/ `TCOLMAX` 只支持 **tile 内整体归约**（跨整条
ValidCol/ValidRow → `[R,1]` / `[1,C]`），**无原生分段归约**指令。

### 影响场景

hi_f4 每个 64-元素 block 的三级 scale 求值第一步（Vmax16 / Vmax8 / Vmax）全部依赖分段 max，无法用单条归约完成。

### 规避方案（已定）

**零成本 TRESHAPE 化整为零**：`jcore/TReshape.hpp:13` 的 `TRESHAPE_Impl` 实体是 `tile_out.data() =
tile_in.data()`——纯句柄重指向，无数据搬运（约束：静态 shape、非 boxed）。把 `[·,64]` reshape 成
`[·×16, 4]` 后 `TROWMAX` → Vmax16；再 `[·×8, 2]`→Vmax8；再 `[·, 8]`→Vmax。参考现成实现
`kernels/reduction/reducemax_colvec_unalign_120_8_pto.hpp`（TCOLMAX→TRESHAPE→TCOLMAX 分段归约范式）。
融合变体 `jcore/TRowMaxExpand.hpp` 的 `TROWMAXEXPAND` 可一条指令「组内 max + 广播回原列」。

- **缺陷所在仓**：无（这是 ISA 能力边界，非 bug）。**解除路径**：若未来补原生 segmented-reduce 指令可省去 reshape；当前规避已足够、零开销。

---

## 问题2：无原生多对多广播乘 —— 16→64 / 8→64

### 结论

§2.1 归一化 `Vin = V·rec·2^-E1_8·2^-E1_16` 中，8 个 L2 因子要广播到 64 元素、16 个 L3 因子广播到 64 元素，
即**多对多广播乘**。PTO-ISA 的 `TEXPANDMUL` 只支持 **1→多**广播，**无原生多对多**。

### 影响场景

三级 scale 的每元素有效 scale `S[j]=E6M2·2^(E1_8[j]+E1_16[j])` 应用到 64 元素时需要分组广播。

### 规避方案（已定）

**同问题1 的 reshape 技巧降维**：`TROWEXPANDMUL(dst,src0,src1)`（`jcore/template_asm.hpp:4882`，src1 为逐行
`[R,1]` 广播算子）。把 data reshape 成 `[·×16, 4]` 后 `TROWEXPANDMUL(·, data, factor16[R,1])` → 每组 4
元素乘该组因子，即 16→64；reshape `[·×8, 8]` 同理实现 8→64。多对多在 reshape 后退化为逐行 1→C。

- **缺陷所在仓**：无（ISA 能力边界）。**解除路径**：同问题1。

---

## 问题3：one-level `TCVT(e6m2, bf16)` 不可发射（type_traits + __type_code 双缺）—— 探针已实测确认

### 结论（探针 `test/.../src/e6m2_probe.cpp` 2026-08-14 实测）

§2.1 的 `E6M2 = cast_to_E6M2(SF_BF16)`（L1 base scale）在 one-level Tile API 层**无法正确发射**。缺口**双重**：

1. **`jcore/type.hpp` 未注册 `type_traits<__fp8_e6m2>`**（`__fp8_e6m2x2` 同样缺）。自然写法
   `Tile<Location::Vec, __fp8_e6m2, ...>` 喂给 `TCVT`/`TSTORE` **直接编译失败**：
   `no member named 'TypeCode' in 'type_traits<__fp8_e6m2>'`（`template_asm.hpp:120` TCVT dst、
   `:1794` TSTORE src）。**探针 Stage 1（默认）即触发此硬停**。
2. **`__type_code` 枚举（`jcore/type.hpp:7-39）根本没有 `__type_fp8_e6m2` 条目**。字符串 `"e6m2"` 只存在于
   CVT cast 宏（`template_asm.hpp:781/813`）；而 TCVT/TSTORE 的**硬件 dst 类型字段**由
   `type_traits<>::TypeCode`（`"i"()` 立即数）驱动。**探针 Stage 2（`-DSTAGE2`）**手动注入一个占位
   `type_traits<__fp8_e6m2>`（借用 code 13=e8m0）越过 type_traits 墙——编译通过并发射真实
   `BSTART.TEPL TCVT, BF16` + `B.DATR e8m0, byte0, Null` + `BSTART.TLSU TSTORE, e8m0`，但 **dst 被标成
   e8m0 而非 e6m2**，证明缺正确 TypeCode 时无法正确打标。

> 注：底层 `linx_cvt(bf16x2→e6m2x2)` 在 `template_asm.hpp:993` 确实存在，`linx_blkc.h:182/389` 也有
> `__fp8_e6m2`/`__fp8_e6m2x2` 结构体——但都停留在低层 CVT 宏，**未被 one-level Tile 层的 type_traits/枚举
> 接通**。这修正了早期「E6M2 正向 cast 已支持、仅待验发射」的乐观判断：**one-level 端到端不可用**。

### 影响场景

L1 base scale 的 bf16→e6m2 cast——三级 scale 的第一级、后续 recip/阈值判定的锚。若走 CVT 则 dst 类型错误。

### 规避方案（已定）

**自算 e6m2 base 位（bit reconstruction），绕开 CVT dst 类型**：与问题4 recip 的 M2-LUT 同一哲学。bf16→e6m2
= 指数重偏置（bf16 bias 127 → e6m2 bias 48）+ 2bit 尾数带舍入截断，在整数/bf16 位域用
`TSHRS/TANDS/TADDS/TSELS` 拼出 8bit e6m2 位，`TSTORE` 成裸 `uint8`。只作用于块级 `[TileM,1]` 的 base，成本可忽略。

> **规范定性（2026-08-25，`pto-spec` `tile-data-types.asl`）**：规范的 Tile DataType 枚举（0..28）
> **根本不含 E6M2**（有 E5M2=8/E3M2=9/E2M3=10，无 E6M2）。E6M2 **仅**作为 HiF4 scale word 的 bits 7:0 存在
> （`hif4-scale.asl` `HiF4E6M2FiniteValue`），**规范刻意不把 E6M2 暴露为 tile 元素类型**。→ 这把「位重构」从
> 「临时权宜、等上游补 dtype」重定性为**规范设计意图下的正解**：不存在应被注册的 e6m2 tile dtype，产 e6m2 位并塞进
> U32 scale word 是唯一与规范一致的路径。下方「解除路径」中「回退原生 TCVT(e6m2,bf16)」按此**大概率不会发生**。

- **缺陷所在仓**：`linx-toolchain-build` / `Linx-TileOP-API`（头文件 `jcore/type.hpp` 缺 e6m2 枚举码 +
  type_traits 特化）。**解除路径**：上游补 `__type_fp8_e6m2` 枚举码 + `type_traits<__fp8_e6m2>` 特化，
  且仿真器 `DataFormatCvt` 支持 e6m2 dst，则可回退到原生 `TCVT(e6m2,bf16)`，省去位重构。**但**（见上「规范定性」）
  规范 tile dtype 枚举本就不含 E6M2，此回退路径**大概率不会实现**，位重构应视为长期方案。

---

## 问题4：recip(E6M2)→bf16 —— 无 FusedReciprocalCast / 无 E6M2 倒数 / 无反向 E6M2→BF16 cast

### 结论

§2.1 的 `SF_BF16_REC = recip(E6M2)` 需要 e6m2 的倒数并输出 bf16。PTO-ISA **三缺俱全，均无原生指令**：

- **无 FusedReciprocalCast** 等价（AscendC 有、PTO 只有独立 `TRECIP`，`template_asm.h:206`）。
- **无 E6M2 原生倒数**：`EleRecip`（`SuperScalarModel/isa/calculate/TileOpCommonCalc.cpp:281`）switch 只含
  FP64/FP32/FP16/BF16/INT*，**无 e6m2 分支**；`TRECIP`（`jcore/template_asm.hpp:3474`）dst/src 同 tile_shape、
  **同 dtype、不含 cast**。
- **无反向 E6M2→BF16 cast**：仿真器 funcMap 只有正向 `bfloat16_to_e6m2`，无 `e6m2_to_*`。故既不能把 e6m2 直接
  喂 `TRECIP`，也不能先反 cast 再 recip。

### 影响场景

三级 scale 全链路的核心系数 `rec`——L2/L3 阈值判定与最终归一化都乘它。

### 规避方案（已定 → (b) M2-LUT 位重构倒数，与 DESIGN §2.1:63 spec 一致、精确、前向 only）

ea 为 e6m2 uint8：`exp6=(bits>>2)&0x3F`（bias 48）、`m2=bits&3`；`1/ea = 2^-(exp6-48)·1/(1+m2/4)`。以 bf16
位算：`rec_bits = LUT[m2] + ((48-exp6)<<7)`，`LUT[m2]=bf16(1/(1+m2/4))` = {`0x3F80`,`0x3F4D`,`0x3F2B`,
`0x3F12`}（m2=0..3）。op 序：`TSHRS/TANDS` 取 exp6、`TSHLS(7)`+`6144-·` 得指数项、`TANDS(1/2)`+两次 `TSELS`
选 LUT、`TADD` 合并、HBM 字节别名 reinterpret u16→bf16。仅作用于块级 `[TileM,1]` ea，成本可忽略。

- **bring-up 快速退路 (a)**：对未 round 的 `SF_BF16` 直接 `TRECIP`——但 dequant 用的是已 round 的 e6m2，
  二者比值引入**每块 ±≤12.5% 系统 scale 偏差**（还可能翻转 L2/L3 阈值），仅供早期打通、不作最终。
- **(c) 除法不独立成立**：`TDiv` 除数仍须是 bf16 域的 ea_decoded，同样卡在反 cast，退化回 (a)。
- **缺陷所在仓**：ISA/仿真器（无 e6m2 recip、无反向 cast）。**解除路径**：补原生 e6m2 倒数或反向
  E6M2→BF16 cast 后，可省去 M2-LUT 位重构。

---

## 问题5：`≥阈值 ? 1:0` + `2^(-E1_*)` 选择 —— GE 比较（v0.58 已切原生 TCMPS<GE>）

### 结论（2026-08-14 更新：v0.58 已切原生）

§2.1 的 `E1_8=(Vmax8·rec≥4)?1:0`、`E1_16=(…≥2)?1:0` 需要 compare(≥) + select（从 {`2^0`,`2^-1`} 二选一）。
**v0.58 起 linx 头 `jcore/template_asm.hpp:5887` 提供 `template<CmpMode Mode> TCMPS`，含原生 GE**
（发射 `B.DATR Zero,ge`），`ge_threshold_e1_factor` 已从早期的 max-EQ 模拟切换为**单条原生
`TCMPS<pto::CmpMode::GE>(mask, t, K)`**。`TSEL(dst,mask,src1)` 仍是 **in-place**
`dst = mask ? src1 : dst_prev`（`:3353`）。

> ⚠️ **注意模式**：hi_f4 这里确实是 **GE**；但这不代表所有 compare 都用 GE——同工作区
> `dynamic_mx_quant` 的 cuBLAS scale 用的是 **LT/NE/GT/EQ**（见其 RECORD 问题3）。逐点按算法取模。

### 影响场景

L2/L3 两级 boost 位的判定 + 归一化用的 `2^-E1_8`、`2^-E1_16` 因子选择。

### 落地（原生 GE + 双 TSEL）

- **GE**：`TCMPS<pto::CmpMode::GE>(mask, t, hif4_bf16c(Kbits))` → `mask=(t≥K)`。K 仍以 bf16 位模式
  作编译期模板参数传入（tile-scalar 立即数，避免 float→bf16 崩后端，问题11）；TCMPS 与 TMAXS 同为
  tile-scalar 路径，同法安全。
- **factor**：`TEXPANDS(f,1.0); TSEL(f,mask,half)` → `f=(t≥K)?0.5:1.0`，供 `TROWEXPANDMUL` 归一化。
- **E1 位**：`TSUB(e,t,t)（=0）; TSEL(e,mask,one)` → `e=(t≥K)?1:0`，打包进 L2/L3 scale。K=4（L2）/2（L3）；
  t 非负有限、无 NaN。
- **发射见证**（`FRAMEWORK diss`）：8× `B.DATR Zero, GE`（L2 K=4 + L3 K=2）× bf16/half 输入 × full/tail。
- **早期规避（已弃用）**：`t≥K == TMAXS(tmp,t,K); TCMP(mask,tmp,t)`（max-EQ 模拟，`max(t,K)==t⟺t≥K`），
  v0.58 前无原生 GE 时用；现已删。

---

## 问题6：仿真器 `TQUANT` 只做无元数据纯 cast —— 带 scale tile 直接 ASSERT

### 结论

路线 B（单指令 `TQUANT` 走 §2.1 最后的元素 cast）不可用：仿真器 `TQUANT`
（`SuperScalarModel/.../TEPLEngine.cpp:1629-1678`）**只实现无元数据纯 cast**；带 scale/offset tile
（`srcTile.size()>1`）直接 `ASSERT(false)`（:1650-1656）——`(in-offset)*scale` 量化数学未实现。

### 影响场景

若想用 TQUANT 一步完成「乘 scale + cast e1m2」。且三级 scale 的**求值**本就必须算子内完成，TQUANT 至多做最后 cast。

### 规避方案

**走路线 A（手工多遍）**：三级 scale 求值 + 归一化全部用 §2.1 的 tile-op 组合（问题1/2/4/5 的规避）完成，
最后 `TCVT(Vin → e1m2)`（y≡e1m2，见问题7）。不依赖 TQUANT。

- **缺陷所在仓**：`SuperScalarModel`（TQUANT scale 路径未实现）。**解除路径**：仿真器补 `(in-offset)*scale`
  量化数学后，最后一步 cast 可选走 TQUANT；但三级 scale 求值仍须算子内做。

---

## 问题7：`HIF4` 作为 cast 目的类型在仿真器 `DataFormatCvt` 不支持 —— 复用 e1m2 通路

### 结论

§1.3：y 的**数值编码与 e1m2 完全一致**（`__fp4_e1m2x2` 每字节 2 个）。仿真器中 **`HIF4` 作为 cast dst 在
`DataFormatCvt` 未支持**；但 `OPCVT_HIF4` 复用 e1m2 的 `bfloat16_to_float4_1`
（`FloatPointUtils.cpp:794`）——与 y≡e1m2 定义一致。现有 two-level `fa_hif4` 也靠 `BF16→FP4_1(e1m2x2)` HACK 过关。

> **规范 dtype 身份（2026-08-25，`pto-spec`）**：规范 `tile-data-types.asl` 把 y 登记为**独立 dtype `HiF4X2`（码 14）**，
> 与 `E1M2X2`（码 12）**数值等价但为不同 tile 类型**；`hif4x2.asl`（`PTO-ARCH-DATA-TYPES-FORMAT-HIF4X2`）给权威
> lane 表 `{0,0.25,0.5,0.75,1.0,1.25,1.5,1.75}`（含 ±0、次正规），`TCVT.asl` legality「**HiF4X2 is TCVT-only**」+
> `matrix-functions.asl` `TileMXInputTypeSupported` 含 HiF4X2（Matrix-MX 专用，group 64、U32 scale carrier）。
> → 规范正名 dst = `HiF4X2`(14)，本 kernel 现 emit `__fp4_e1m2x2`(12) 是数值等价替身；工具链补 `HiF4X2` type_traits+枚举码后应改正名。

### 影响场景

最终量化输出 y 的元素 cast。

### 规避方案

**y 直接走 e1m2 通路**：`TCVT(Vin → __fp4_e1m2x2)`（e1m2 emit 已在 dynamic_mx_quant 侧验证可发射，见其
RECORD 问题2/6）。hi_f4 的「新」只在 scale 的三级结构，不在 y 的元素编码。

- **缺陷所在仓**：`SuperScalarModel`（HIF4 dst 名义未接，但语义等价 e1m2 已可用）。**解除路径**：仿真器
  `DataFormatCvt` 显式登记 HIF4 dst（映射到 e1m2 实现）即可正名，功能上当前已可用。

---

## 问题8：三级 scale→元素映射 —— 规范 ADR-0101 裁决 = 相邻(floor)；仿真器取模是相对规范的 bug

### 结论（规范裁决 2026-08-24，ADR-0101 / `pto-spec` 提交 `d0ce06ad`）

权威规范 `asl/arch/data-types/formats/hif4-scale.asl`（`PTO-CUBE-HIF4-SCALE-001`）的 `HiF4ScaleExponentIncrement`
用 **`q DIVRM 8`（L2/E1_8）、`q DIVRM 4`（L3/E1_16）向下取整** = **相邻分组**，与 §1.2/§2.1 的 `j/8`、`j/4` 逐位一致。
boundary demo `tests/asl/.../hif4-scale/arch-bound-hif4-scale-001.asl` 佐证（`q=0` increment=2、`q=8` increment=0）。
**这一分歧由规范一锤定音：相邻是唯一权威语义。** 仿真器 `MatrixScaleL/RHiF4`（`SuperScalarModel/.../CubeEngine.cpp:1326-1328`）
历史上若按**取模（strided）**消费（`eb=(e8>>(i%8))&1`、`ec=(e16>>(i%16))&1`，bit 跨步 8/16 共享 = 相邻的转置），
则**是相对 ADR-0101 的 bug**，须以规范为准修正。

### 影响场景

**运行期反量化**：量化算子按规范相邻 packing 输出即正确；若仿真器仍按取模消费 → 反量化错位。**encode 落地不受影响**
（编码阶段不消费该映射），但 gfrun/上板前须让仿真器对齐规范。

### 消解方案（规范已定，唯一路径）

**(a) 仿真器改 floor 分组 `j/8`/`j/4`（相邻消费），对齐 ADR-0101。** 早期备选 (b)「送 Cube 前转置重排匹配 strided」
**已废弃**——规范裁决相邻权威，不应为迁就 emulator bug 而转置。当前被 skew 阻塞，暂不影响 encode，运行期验证前落地 (a)。

- **缺陷所在仓**：`SuperScalarModel`（若 CubeEngine 取模消费，则与规范 ADR-0101 的相邻语义冲突，是 bug）。
  **解除路径**：仿真器改 floor 分组并 gfrun 对齐规范 `HiF4ScaleExponentIncrement`。

---

## 问题9：`hifloat4_scale` 无独立 dtype —— 规范钦定以单个 raw U32 word 承载（非规避，是正解）

### 结论（规范确认 2026-08-24，ADR-0101）

ADR-0101 明确：「A HiF4 Matrix scale MUST be **one raw U32 word**」（E6M2 bits 7:0、E1_8 bits 15:8、E1_16 bits 31:16），
消费侧 demo `block-exec-bstart-tmatmulmx-hif4-002.asl` 用 `TileDataType_U32` + `CUBE_M32` layout 承载 scale。
逻辑名 `hifloat4_scale`（L3 16 + L2 8 + L1 e6m2 8）**并非独立注册 dtype**——`jcore/type.hpp` 的 `__type_code` 枚举、
仿真器 `DataType` 枚举中**均无此名**（`grep` 零命中，DataType 仅登记 y 的 `HIF4`≡e1m2，`CubeCalculate.h:69`）——
但**这不是缺口**：规范本就以 raw U32 word 作 scale 载体，声明 `uint32` 即规范正解，无需独立 dtype。

### 影响场景

算子 scale 输出 tile 的 dtype 声明与 `TSTORE`。

### 落地（规范正解 → 单个 raw U32 word，每 block 1 元素）

把 L3(16)+L2(8)+L1(8) 用位运算（`TSHLS`/`TORS`）拼进一个 32bit word，scale 输出 tile 声明为 **`uint32`**
（或 `float32`，二者皆 32bit、仿真器按字节消费、等价）。**`scale` 的 shape = `[..., num_blocks]`，每块恰好
1 个 32bit 元素，与规范 per-block 语义一致**。字内布局（小端，对齐规范位段）：`word = e6m2 | (L2<<8) | (L3<<16)`
（byte0=e6m2/L1=bits 7:0、byte1=L2=bits 15:8、byte2..3=L3=bits 31:16）。L1 的 e6m2 字节由 bf16→e6m2 位重构产出（问题3）。

- **对比被否方案（uint8 每块 4 字节）**：`scale` shape 会变成 `[..., num_blocks*4]`——**4× 膨胀、与规范 shape
  不符**，故不采用。尽管 two-level `fa_hif4.hpp:48-50` 用 `unsigned char` 逐字节写、仿真器 `MatrixScaleLHiF4/RHiF4`
  （`CubeEngine.cpp:928/934`）也按字节消费——字节流内容相同，但**张量 shape 语义错误**。
- **非 dtype 规避**：规范载体本就是 raw U32 word，`uint32` 声明即正解；不存在「等 `hifloat4_scale` dtype 注册」的解除动作。
- **仿真器侧提醒**：`DataType` 无独立 scale 类型属正常（规范以 U32 承载）；消费按 `CUBE_M32` layout（A-scale `[M,G]`、
  B-scale `[N,G]` 语义 `[G,N]`），HiF4X2 仅在 Matrix-MX（`TMATMULMX`）路径，普通 `TMATMUL` 用之 `Fault_TileLegality`。

---

## 问题10：32B 列对齐 + TRESHAPE header gap —— 阻断「相邻窄列 reshape 分段 max」既定路线（探针实测）

### 结论（探针 `src/segreduce_probe.cpp` 2026-08-14 实测，编译失败为证）

问题1/2 的既定规避「零成本 TRESHAPE 化整为零 `[.,4]`/`[.,2]` + TROWMAX/TROWEXPANDMUL」实现分段 max /
分组广播，在 one-level 布局约束下**根本不可声明**。**双墙**：

1. **32B 列对齐（`common/pto_tile.hpp:649` static_assert，类型实例化即触发）**：RowMajor NoneBox tile 要求
   `Cols * sizeof(DType) % 32 == 0`；boxed（SFractal≠NoneBox）tile 要求 `Cols % InnerCols == 0` 而 bf16 的
   `InnerCols` 恒为 16（`pto_tile.hpp:562` `isInnerRowMajor ? 16 : byteSize*8/bits`）。**两条路都要求 bf16 tile
   的 Cols 是 16 的倍数**。分段 max 的相邻 4/2 分组需要 `[.,4]`/`[.,2]`（以及归约中间 `[.,1]`）——**列 4/2/1
   全部违反**，在 `m16c`/`m8c`/`m8`/`vmax` 的**类型声明处**即 static_assert 失败（与 TRESHAPE 无关）。
2. **TRESHAPE header gap**：顶层 `TRESHAPE` wrapper 在 `common/tileop_api.hpp:52`，但 `tileop_api_impl.hpp`
   只 include `aarch64/` + `cpu_sim/` 的 TReshape.hpp，**不含 `jcore/`**——故 `-D__linx`（经 `common/pto_tileop.hpp`
   走 jcore 路径）下 `TRESHAPE` 是**未声明标识符**（与 TCAST / TINTERLEAVE 同类 header gap，问题5）。即便直接调
   `TRESHAPE_Impl` 也救不了墙(1)。

> **修正**：这推翻了 DESIGN §3.1/§3.2、RECORD 问题1/2「reshape 可零成本落地相邻分段 max」的乐观判断——**该路线
> 在 one-level 不可行**。相邻分组本质上需要 width-4/2 的窄 tile，被布局规则禁止。

### 影响场景

三级 scale 求值的**第一步**（Vmax16/Vmax8）与最后归一化的**分组广播乘**——即路线 A 的核心。整个 encode
的骨架依赖它，故**路线 A 当前无法按 DESIGN §3.2 骨架落地**。

### 候选替代（**未定，待用户决策**——均须保证每个中间 tile 的 Cols 是 16 倍数(bf16)/8 倍数(fp32/uint32)）

- **(A) strided HBM load 拆列 + 逐元素 TMAX 树**：用带 stride 的 `global_tensor` 把每 block 按 stride 4/2 拆成
  **宽 ≥16 列**子 tile（如 a/b/c/d 各 `[TileM,16]`，元素 j=4g+{0,1,2,3}），`Vmax16=max(max(a,b),max(c,d))`
  逐元素、列宽合法；L2 转 fp32 域用 `[TileM,8]`（fp32 列8=32B）。保相邻语义，复杂度中等，**须验证 strided TLOAD 可发射**。
- **(B) 采纳 emulator 的 strided（转置）分组语义（§2.4）**：bit `b` 被 `{b,b+8,…}` 共享 → 列宽自然合法、与
  emulator `MatrixScaleL/RHiF4` 现状一致、免运行期转置；**代价**：偏离 spec 权威的「相邻」语义（§1.2/§2.1）。
- **(C) 暂停 kernel 落地、先固化缺口**：承认 one-level 当前不具备落地相邻分段的条件，待上游补 TRESHAPE 暴露
  或 sub-16 列 tile / deinterleave 后再落地。

- **缺陷所在仓**：`Linx-TileOP-API`（32B 列对齐是布局设计边界；TRESHAPE 头未在 -D__linx 暴露）。**解除路径**：
  上游放宽 sub-16-col tile 或暴露 deinterleave/TRESHAPE，则相邻窄列路线可行；否则走 (A)/(B)。

---

## 问题11：bf16 标量立即数的两处 LinxV5 后端缺陷（框架编译实测确认 2026-08-14）

### 结论（探针 `bf16s.cpp`/`z.cpp` 实测，编译崩溃/匹配失败为证）

路线 A 框架首次整体编译时，后端（非前端类型/shape 校验）暴露**两处独立的 bf16 标量缺陷**，均与
「把一个 bf16 标量喂给 tile-scalar op（TMULS/TMAXS/TEXPANDS…）」有关：

1. **运行期 float→bf16 标量 → 指令选择期崩溃**。任何 `static_cast<__bf16>(x)`（x 为运行期
   `float`，**甚至 constexpr float / 字面量 `4.0f`**）喂给 tile-scalar op，会发射一个 `fptrunc
   float→bf16`，其结果是 GPR 里的 i16 值；LinxV5 在 `LinxV5 DAG->DAG Pattern Instruction
   Selection` 阶段 `PromoteIntegerResult` 无法提升该 i16 `CopyFromReg`，直接
   `UNREACHABLE executed at LegalizeIntegerTypes.cpp:57` 崩溃。**整数 0/1（uint8/16/32）标量不受影响。**

2. **bf16 标量 0.0 → `B.IOR [zero],[]` 指令匹配失败**。TEXPANDS/TORS 等的头文件实现用
   `volatile sv = s` 反折叠以防常量落到硬件 `zero` 寄存器（`template_asm.hpp:4096` 注释自陈此
   fragility）。该反折叠对 uint8/16/32 的 0 有效，但**对 bf16 的 0.0 无效**——bf16 0.0 仍被分配到
   `zero` 寄存器，`B.IOR [zero],[]` 无匹配指令，编译报 `Match Instruction Error`。

### 影响场景

框架里所有 bf16 标量：`SF=Vmax*(1/7)`（TMULS 1/7）、GE 阈值 `TMAXS(t,K)`（K=4/2）、
E1/factor 的 `TEXPANDS(1/0.5/0)`、占位 recip 的 `TEXPANDS(1.0)`。

### 规避方案（已落地，框架编译干净 EXIT=0）

- **缺陷1**：所有 bf16 标量立即数以**原始位模式** materialize：`__builtin_bit_cast(__bf16, uint16_t
  bits)`（封装 `hif4_bf16c(bits)`，-O2 折叠为立即数，探针实测干净）。常量表：1.0=`0x3F80`、
  0.5=`0x3F00`、1/7=`0x3E12`、4.0=`0x4080`、2.0=`0x4000`。GE 阈值 K 改为**编译期模板参数
  `uint16_t Kbits`**（不再传运行期 `float K`）。与 `dynamic_mx_quant` 的
  `__builtin_bit_cast(__bf16, RECIP_EMAX)` 同法（common.hpp:404）。
- **缺陷2**：需要 bf16 0 tile 时用 **`TSUB(z, t, t)`**（t 非负有限、无 NaN，t−t=0 精确）产生，
  绝不用 `TEXPANDS(bf16 0.0)`。

- **缺陷所在仓**：`linx-toolchain-build`（LLVM LinxV5 后端：缺陷1=i16 提升 legalization 缺失；
  缺陷2=`B.IOR` 对 zero 寄存器无匹配 pattern + TEXPANDS volatile 反折叠对 bf16 失效）。
  **解除路径**：后端补 i16(bf16) 标量→tile 广播的合法化 + `B.IOR [zero]` pattern（或头文件对 bf16
  0 也走非零寄存器搬运），则可回退 `static_cast<__bf16>` 与 `TEXPANDS(bf16 0)` 的自然写法。

---

## 附：缺口 → 规避 速查

| # | 缺口（真实缺失） | 缺陷所在仓 | 规避方案（不改硬件） | 状态 |
|---|---|---|---|---|
| 1 | 无原生分段 max | ISA 能力边界 | ~~零成本 TRESHAPE + TROWMAX~~（被问题10推翻） | 见问题10 |
| 2 | 无原生多对多广播乘 | ISA 能力边界 | ~~TRESHAPE + TROWEXPANDMUL~~（被问题10推翻） | 见问题10 |
| 3 | one-level TCVT→e6m2 不可发射（type_traits + 枚举双缺） | Linx-TileOP-API 头文件 | bf16→e6m2 位重构，存裸 uint8 | 探针实测确认 |
| 4 | 无 FusedReciprocalCast / e6m2 倒数 / 反向 e6m2→bf16 cast | ISA + 仿真器 | M2-LUT 位重构倒数 | 已定 |
| 5 | ~~TCMP/TCMPS EQ-only 无 GE~~（v0.58 已补原生 GE） | ~~Linx-TileOP-API 头~~ 已解除 | 原生 `TCMPS<CmpMode::GE>`（旧 max-EQ 模拟已删） | **v0.58 已切原生，diss 见 8× B.DATR Zero,GE** |
| 6 | 仿真器 TQUANT 带 scale ASSERT | SuperScalarModel | 走路线 A 手工多遍 | 已定 |
| 7 | HIF4 作 cast dst 未登记 | SuperScalarModel | 复用 e1m2 通路（y≡e1m2） | 已定 |
| 8 | scale→元素映射 相邻 vs 取模 | SuperScalarModel（若取模则 bug） | 规范 ADR-0101 裁决=相邻(floor)；仿真器改整除对齐（(b)转置已废） | **规范一锤定音，运行期改 emulator** |
| 9 | `hifloat4_scale` 无独立 dtype | 非缺口（规范以 raw U32 承载） | 声明 `uint32` 单 word/块（ADR-0101 正解，非规避） | **规范确认为正解** |
| 10 | 32B 列对齐 + TRESHAPE header gap → 窄列 reshape 分段 max 不可声明 | Linx-TileOP-API | (A) strided load 宽子 tile + TMAX 树 / (B) 采纳 emulator strided 语义 / (C) 暂停待上游 | **探针实测确认，路线未定待决策** |
| 11 | bf16 标量后端双缺陷：运行期 float→bf16 崩溃 + bf16 0.0 落 zero 寄存器匹配失败 | linx-toolchain-build（LinxV5 后端） | bf16 立即数走 `__builtin_bit_cast` 位模式 + 阈值改模板参数；0 tile 用 `TSUB(t,t)` | **框架编译实测确认，已规避（EXIT=0）** |
