# DynamicHiF4Quant — 设计文档

`dynamic_hi_f4_quant` 把输入激活/权重量化为自研 **hifloat4（hi_f4）** 格式。其 scale 方案与
`dynamic_mx_quant` 的 OCP / cuBLAS **完全不同**：OCP/cuBLAS 每 block 只有**一个** scale（E8M0 或
fp32），而 hifloat4 采用**三级共享（3-level shared）**的分段 scale。

本文的**格式定义（§1）与量化算法（§2.1）均为需求方给出，权威**；§2.2–§2.5 是对其的
标注、验证与映射分析；§3 是 PTO-ISA 落地缺口；§4 是工具链/仿真器现状。

> **规范锚（normative，最高权威）—— PTO-spec ADR-0101「Matrix Scale Cell Layouts, HiF4 Scale Words, and CScale」**
> （`pto-spec` 提交 `d0ce06ad` "Define Matrix scale and CScale contracts" #146，2026-08-24，accepted，target 0.58.4）。
> 归一化 ASL 单元 `asl/arch/data-types/formats/hif4-scale.asl`（`PTO-CUBE-HIF4-SCALE-001`）逐位定义了 HiF4 scale word，
> 可执行 demo `tests/asl/.../hif4-scale/arch-bound-hif4-scale-001.asl`、`.../BSTART.TMATMULMX/...-hif4-002.asl` 佐证。
> 该规范**确认**了本文 §1.2 的相邻分组语义、§0/§2.5 的 U32 word 布局，并**裁决**了 §2.4/RECORD 问题8 的相邻-vs-取模分歧
> （规范用 `q DIVRM 8`/`q DIVRM 4` 向下取整 = 相邻，emulator 取模消费是相对规范的 bug）。规范要点已就地并入各节。

## 0. 算子接口（权威约定）

- **输入**：
  - `x` — 待量化激活/权重。**仅支持 `float16`（`__half`）与 `bfloat16`（`__bf16`）两种输入 dtype**
    （用 `if constexpr` 分派：算法在 bf16 域进行，fp16 输入先 `TCVT half→bf16` 进入统一域）。**无其它 dtype**。
  - `axis` — 量化轴，**预留参数**。**本期只实现 `axis = -1`（尾轴）量化**，落地文件
    `dynamic_hi_f4_quant_tail.h`。非尾轴（`axis ≠ -1`）暂不实现。
- **输出**：
  - `y` — 量化数据，**固定 `hifloat4` 类型**（4bit/元素，数值编码 ≡ e1m2，`__fp4_e1m2x2` 每字节 2 个）。
  - `scale` — **规范载体 = 单个 raw U32 word / block**（ADR-0101：「one raw U32 HiF4 scale word」，E6M2 bits 7:0、
    E1_8 bits 15:8、E1_16 bits 31:16）。逻辑名 `hifloat4_scale`（L3 16 + L2 8 + L1 e6m2 8）不是独立注册 dtype，
    **规范钦定即以 `uint32` raw word 承载**，每 block 1 元素（见 §0 类型支持性核查、RECORD 问题9）。
- **无 BlockSize 参数**（固定 64，见 §1.1）；**无输出 dtype 选择**（y/scale 类型均固定）。

> **`hifloat4_scale` 类型支持性 —— 规范钦定 raw U32 承载（ADR-0101 确认，2026-08-24）**：规范明确
> 「A HiF4 Matrix scale MUST be **one raw U32 word**」，消费侧 demo（`block-exec-bstart-tmatmulmx-hif4-002.asl`）
> 用 `TileDataType_U32` + `CUBE_M32` layout 承载 scale。逻辑名 `hifloat4_scale` 并非独立注册 dtype——
> `jcore/type.hpp` 的 `__type_code` 枚举、仿真器 `DataType` 枚举中**均无此名**（`grep` 零命中），DataType 仅登记
> y 的 `HIF4`（≡e1m2，`CubeCalculate.h:69`）；但**这不构成缺口**：规范本就以 raw U32 word 作 scale 载体，无需独立 dtype。
>
> **落地——单个 raw U32 word / block（规范正解，非权宜）**：把 L3(16)+L2(8)+L1(8) 用位运算（`TSHLS`/`TORS`）
> 拼进一个 32bit word，`scale` 输出 tile 声明为 **`uint32`**（或 `float32`，二者皆 32bit、仿真器按字节消费、等价）。
> **这样 `scale` 的 shape = `[..., num_blocks]`，每块恰好 1 个 32bit 元素，与规范 per-block 语义一致**。
> - **对比被否方案**：若声明为 `uint8` 每块 4 字节，`scale` shape 会变成 `[..., num_blocks*4]`——**4× 膨胀、与真实
>   shape 不符**，故不采用（尽管 two-level `fa_hif4.hpp:48-50` 用 `unsigned char` 逐字节写、仿真器
>   `MatrixScaleLHiF4/RHiF4`（`CubeEngine.cpp:928/934`）也按字节消费——字节流内容相同，但**张量 shape 语义错误**）。
> - **字内布局**（小端，对齐 §2.5 emulator 消费）：`word = e6m2 | (L2 << 8) | (L3 << 16)`（byte0=e6m2/L1、
>   byte1=L2、byte2..3=L3）。
> - L1 的 e6m2 字节仍由 bf16→e6m2 **位重构**产出（§3.1 cast 行 / RECORD 问题3，因 one-level `TCVT→e6m2` 不可发射）。
> - **仍属 dtype 规避**：真实类型是 `hifloat4_scale`，我们用 `uint32`/`float32` 作 32bit 容器承载其位模式；
>   一旦上游注册 `hifloat4_scale` dtype，可直接改声明、去掉容器代换。

## 1. 格式定义（权威）

### 1.1 分段量化 / BlockSize

- **分段（segmented / block）量化**，**BlockSize 固定为 64** 个元素。
- 每 64 元素的 block 独立产出：`64×4bit = 256bit` 的量化数据 + `32bit` 的 scale。

### 1.2 三级共享 scale（每个 64-元素 block）

| 级别 | 编码 | 每单位作用范围 | 单位数 / block | 位数 / block |
|------|------|----------------|----------------|--------------|
| **三级（L3）** | 1 bit | 相邻 **4** 个元素 | 64/4 = 16 | **16 bit** |
| **二级（L2）** | 1 bit | 相邻 **2** 个 L3 scale（= 相邻 **8** 个元素） | 64/8 = 8 | **8 bit** |
| **一级（L1）** | 8 bit，**e6m2** | 整个 L2 scale（= block 全 **64** 元素） | 1 | **8 bit** |

- **层级包含关系**：L1（1 个，管 64 元素）⊃ L2（8 个，每个管 8 元素）⊃ L3（16 个，每个管 4 元素）。
  即 1 个 L1 覆盖 8 个 L2；1 个 L2 覆盖 2 个 L3；1 个 L3 覆盖 4 个元素。
- **元素→bit 映射 = 相邻（adjacent，权威）**：L3 bit `k` 管逻辑元素 `[4k,4k+4)`、L2 bit `k` 管 `[8k,8k+8)`
  （`j/4` / `j/8` 整除分组）。**注意**与 emulator 现状取模消费不一致，见 §2.4。
- **拼接**：`L3(16bit) + L2(8bit) + L1(8bit) = 32bit`，组成 scale word **`hifloat4_scale`**（每 block 32bit）。
  规范位段（ADR-0101 / `hif4-scale.asl`）：**E6M2 = bits 7:0、E1_8(L2) = bits 15:8、E1_16(L3) = bits 31:16**。
  > `hifloat4_scale` 是**逻辑布局名，非独立注册 dtype**；规范钦定以 **raw `uint32`** 每 block 1 word 承载（见 §0）。
  > 字内排布见 §2.5。

### 1.3 量化输出 y

- dtype **`hifloat4`**，每元素 **4 bit**。
- **数值编码格式与 e1m2 完全一致**（1 符号位 + 1 指数位 + 2 尾数位的 fp4 变体，`__fp4_e1m2x2` 每字节打包 2 个）。
- 故 y 的**数值编码**可直接复用 e1m2 通路；hifloat4 的“新”只体现在 **scale 的三级结构**上，不在 y 的元素编码上。
- **规范 dtype 身份（`tile-data-types.asl` 确认）**：规范把 y 登记为**独立 dtype `HiF4X2`（枚举码 14）**，与 `E1M2X2`（码 12）
  **数值等价但为不同 tile 类型**；`TCVT.asl` legality 明写「**HiF4X2 is TCVT-only**」（仅能由 TCVT 产出、且被 Matrix-MX
  输入角色接受，见 §2.4 消费契约）。**当前工具链**无 `HiF4X2` type_traits/枚举码，故本 kernel emit 数值等价的
  `__fp4_e1m2x2`(12) 作 stand-in（问题7）；上游注册 `HiF4X2`(14) 后应改 emit 该正名 dst。

### 1.4 与 dynamic_mx_quant 的对比

| 维度 | OCP (mxfp4) | cuBLAS (fp8) | **hifloat4** |
|------|-------------|--------------|--------------|
| BlockSize | 32 的倍数 | 32 的倍数 | **固定 64** |
| scale 级数 | 1 级 | 1 级 | **3 级共享** |
| scale 编码 | E8M0（8bit/block） | fp32 | **L1 e6m2 8bit + L2 8bit + L3 16bit = 32bit/block** |
| 输出元素编码 | e2m1 | e4m3 | **e1m2** |
| scale 求法 | max_exp → 减 emax → cast | amax → recip | **三级逐层求（见 §2.1）** |

## 2. 量化算法

### 2.1 算法（权威，bf16 域逐步）

对每个 block 的 64 个输入元素 `V64`：

```text
V64      = cast_to_bfloat16(V64)          # 统一在 bf16 域处理
V64_abs  = |V64|                          # 取绝对值
Vmax16   = maxPer4(V64_abs)               # 相邻 4 元素取 max → 16 个（对应 L3 分组）
Vmax8    = maxPer2(Vmax16)                # 相邻 2 个 Vmax16 取 max → 8 个（对应 L2 分组）
Vmax     = max(Vmax8)                     # 整块 max → 1 个（对应 L1）

SF_BF16       = Vmax * (1/7)_bf16          # 乘 bf16 常量 1/7 → 一级 base scale（实数域）
E6M2          = cast_to_E6M2(SF_BF16)      # CVT 直转 → 一级 scale ea（写入 hifloat4_scale）
SF_BF16_REC   = recip(E6M2)                # 基于 2bit M2 查表四选一得 E6M2 倒数，输出 bf16

E1_8   = ( Vmax8  * SF_BF16_REC                 >= 4 ) ? 1 : 0   # 二级 scale（8 个）
E1_16  = ( Vmax16 * SF_BF16_REC * 2^(-E1_8)     >= 2 ) ? 1 : 0   # 三级 scale（16 个）

Vin_BF16 = V64 * SF_BF16_REC * 2^(-E1_8) * 2^(-E1_16)            # 逐元素归一化（保留符号）
E1M2     = cast_to_E1M2(Vin_BF16)                               # → y（含符号位）
```

- **三级归约走「相邻分组」**：`maxPer4`（相邻 4）、`maxPer2`（相邻 2）——与 §1.2 的相邻语义一致。
- **符号自然贯通**：`Vin_BF16` 用带符号的 `V64`（非 `V64_abs`），故 `cast_to_E1M2` 直接把符号写入 y 的 bit[3]。
  decode 侧只出幅值（§2.3），符号由消费方（Cube/matmul）另行处理。
- **每元素有效 scale** `S[j] = E6M2 · 2^(E1_8[j] + E1_16[j])`，`Vin[j] = V64[j] / S[j]`——即 §2.3 反量化的逆。

### 2.2 关键系数的由来（为什么 `1/7`、阈值 `4`/`2`）

hifloat4 每元素满量程 = `base(E6M2) × 最大 boost 4 × 最大 e1m2 幅值 1.75 = base × 7`。故：

- **`SF_BF16 = Vmax/7`**：让块内最大元素 `Vmax` 恰好落到满量程（`Vin=1.75`）。base 由此是块内 max
  派生的**上界锚**（而非 min 侧下界）；e6m2 的 2bit 尾数**参与**（`cast_to_E6M2` 是带舍入的直转，
  base ∈ `2^exp × {1,1.25,1.5,1.75}`）。
- **阈值 `4`（L2）/ `2`（L3）= binade 边界**，保证 boost 后 `Vin ≤ 1.75`（不越 e1m2 满幅）：
  - `Vmax8/base ≥ 4` → L2 置 1（×2）；再看 `(Vmax16/base)·2^(-E1_8) ≥ 2` → L3 置 1（×2）。
  - 验证块 max：`Vmax8=Vmax=7·base` → `7≥4` E1_8=1；`7/2=3.5≥2` E1_16=1 → `S=4·base`，`Vin=7/4=1.75` ✓。
  - 验证 `3·base`：`3<4` E1_8=0；`3≥2` E1_16=1 → `S=2·base`，`Vin=1.5≤1.75` ✓。
- **无冲突联合选择**：L2 用组内 max `Vmax8`、L3 用组内 max `Vmax16`，各自过阈值——确定性，无需仲裁。
  代价：同组内偏小元素的 `Vin` 可能 `<0.5` 而 underflow（round 到最近 e1m2 或 0），是共享 scale 的固有损失。

### 2.3 一致性锚：仿真器**反量化（decode）** + fa_hif4 的**部分 encode 参考**

- **部分 encode 参考（新发现）**：two-level `fa_hif4.hpp` 有 `tohif4`（:9）/`tohif4_bf16x2`（:68）——
  真正对 64 元素做 hif4 编码。但**是退化/简化版，不能直接照搬**：
  - **L2/L3 强制置 0**（`p_scale[mx_idx+1..3]=0`，注释「E1_8 && E1_16 set to zeros」）——只算 **E6M2 base**，无三级 boost。
  - **`scale_factor = 1/3.5`**（非 spec 的 `1/7`）——因无 boost，满量程退化为 `2×1.75=3.5`（用户全算法带 ×4 boost → 7）。
  - **E6M2 cast 本身是 TODO**：`linx_cvt_as<unsigned char>(upd_max)`（:48）只是把 bf16 位当字节 reinterpret（HACK）、
    `tohif4_bf16x2` 用 `__fp8_e4m3x2` 占位（:109 注释「TODO: cast fp16x2 to E6M2x2」）——**未做真正 E6M2 数值 cast**。
  - **是 two-level 标量 `blkv_*` 代码**（`blkv_bf16_fmax`/`blkv_bf16_fdiv`/`linx_cvt_package`），非 one-level tile-op；
    归一化走**除法**（`fdiv`）而非 recip+mul。
  - **价值**：印证结构（每 64 一组、`max×scale_factor→base`、`div→cvt hif4`、4 字节/块 scale 布局），但三级 scale 与
    E6M2 cast 都缺——仍需按 §2.1 全算法自行落地。
- **权威反向锚（normative）= ADR-0101 `hif4-scale.asl` 的 `HiF4ScaleFiniteValue`**：
  `x̂[j] = HiF4E6M2FiniteValue(word[7:0]) · 2^(E1_8[floor(j/8)] + E1_16[floor(j/4)]) · HIF4elem(y[j])`，
  其中 `HiF4E6M2FiniteValue = (1 + m2/4)·2^(exp6-48)`（`exp6=bits[7:2]` bias 48、`m2=bits[1:0]`、`FF`=quiet NaN）。
  §2.1 encode 必须是它的逆。仿真器 `CubeCalculate::GetElementValue`（`CubeCalculate.cpp:1113`）是该规范的实现，
  应与之逐位一致（`bak/` 无 mxfp 外的 hif4 AscendC；matmul `HiF4_HiF4.cpp:50` scale 缓冲未初始化；无 python golden）：

```text
x̂[j] = E6M2(ea) · 2^(eb[j] + ec[j]) · HIF4elem(y[j])
```

- **ea = L1（e6m2, 8bit）**，`E6m2ToF32Bits`（:1093）：`exp6=bits[7:2]`（bias 48）、`m2=bits[1:0]`
  → `2^(exp6-48)·(1+m2/4)`，无符号，尾数 {1.0,1.25,1.5,1.75}，范围 `2^-48 … 2^15`。
- **eb=L2, ec=L3（各 1bit）** → `2^(eb+ec)∈{1,2,4}`（相对 base 上调 0/1/2 个 binade）。
- **HIF4elem = y（4bit）—— 权威 lane 表见规范 `hif4x2.asl` `HiF4X2FiniteDecomposition`**（`PTO-ARCH-DATA-TYPES-FORMAT-HIF4X2`
  + demo `arch-bound-hif4x2-decomp-001.asl`）：`sign=bit[3]`、`exp=bit[2]`、`frac=bits[1:0]`；**`exp=0 frac≠0` 走次正规
  `frac/4`、`exp=1` 走 `1+frac/4`**，故幅值 = **`{0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75}`**（0x0/0x8 = ±0，有可表示零）。
  > ⚠️ 早前把它当作全 normal 的 `2^(e1-1)·(1+m2/4)`→`{0.5,0.625,…,1.75}` **不正确**：低端是次正规、且含零。encode 的
  > underflow/round 分析须按此权威表（§2.2 的满幅 1.75 上界不变）。emulator `Hif4ToF32Bits`（`CubeCalculate.cpp:1103`）
  > 若按 normal-only 实现则与规范不符，运行期前须核对。**符号 bit[3] decode 未乘**（取绝对值路径）。

即 encode 的 `S[j]=E6M2·2^(E1_8+E1_16)`、`y=e1m2(V/S)` 与 decode 严格互逆。

### 2.4 三级 scale → 元素映射（规范裁决 = 相邻；emulator 取模是相对规范的 bug）

- ✅ **规范权威 = 相邻（floor 整除）**。ADR-0101 / `hif4-scale.asl` 的 `HiF4ScaleExponentIncrement`（`PTO-CUBE-HIF4-SCALE-001`）：

  ```asl
  e1_8_index  = 8  + (q DIVRM 8)     // L2：lane q 用 bit[8 + floor(q/8)]  —— 8 相邻元素/bit
  e1_16_index = 16 + (q DIVRM 4)     // L3：lane q 用 bit[16 + floor(q/4)] —— 4 相邻元素/bit
  increment   = E1_8[..] + E1_16[..] ∈ {0,1,2}
  ```
  `DIVRM`（向下取整除）= **相邻分组**，与 §1.2 / §2.1 的 `j/8`（L2）、`j/4`（L3）逐位一致。boundary demo
  `arch-bound-hif4-scale-001.asl` 佐证：scale[7:0]=00 设 E1_8[0]/E1_16[0]→`q=0` increment=2→`2^-46`、`q=8` increment=0。
- ⚠️ **emulator 现状若为取模（strided）则是相对规范的 bug**：`MatrixScaleL/RHiF4`（`CubeEngine.cpp:1326-1328`）历史上按
  `(e8>>(i%8))&1`、`(e16>>(i%16))&1` 消费（bit `b` 被跨步 8/16 共享，是相邻的转置）——**与 ADR-0101 的 floor 分组冲突**，
  须以规范为准修正。
- **运行期消解 = (a) 改 emulator 对齐规范 floor 分组**（规范裁决已定，**不再走 (b) 送 Cube 前转置**）。量化算子按相邻
  packing 输出即正确；运行期被 toolchain↔emulator skew 阻塞，encode 落地不受影响，gfrun/上板前对齐 emulator 即可。

### 2.5 scale 物理打包与尾轴布局

- **量化轴**：尾轴 / 非尾轴都要，**优先实现尾轴**；尾轴**不做特殊排布**（compact 平铺，对齐 dynamic_mx_quant
  尾轴——块行天然连续、无需交织）。
- **scale 布局**：**每块 1 个 32bit 元素**（tile 声明 `uint32`/`float32`），`word = e6m2 | (L2<<8) | (L3<<16)`，
  用 `TSHLS/TORS` 拼接（见 §0 规避 / RECORD 问题9）；scale 输出 shape = `[..., num_blocks]`，与真实 per-block 语义一致
  （**非** uint8-4B 的 `[..., num_blocks*4]`）。字内小端字节序（byte0=e6m2/L1、byte1=L2、byte2..3=L3）对应 emulator
  **左** `ScaleLExtract`（`CubeEngine.cpp:1273-1287`）的 AoS 每块配对语义：`srcScale[2b]` 低字节=e6m2、次字节=L2；
  `srcScale[2b+1]`=L3——32bit 容器按字节消费时与该配对布局**逐字节等价**。
  > **右** `ScaleRExtract`（:1289-1306）是 16 块乒乓的 SoA（前 16 word 给 16 块的 (e6m2,e8)、后 16 word 给 e16），
  > 供右操作数消费——本算子尾轴优先方案暂不产出该布局。

## 3. PTO-ISA 落地缺口 + tile-op 映射

### 3.1 关键约束（§2.1 各步 → 当前 ISA 缺口）

> **缺口 vs 规避**：第 3 列「现状/缺口」记录的是硬件/编译器**真实存在的功能缺失**（无原生的分段 max、
> 1→多广播不够、无 FusedReciprocalCast、无反向 E6M2 cast 等）——这些不因下文而消失；第 4 列「规避方案」
> 是在**不新增指令**的前提下用现有 tile-op 组合出的等效手段（属算子内实现技巧，非硬件补齐）。

| 算法步骤 | 需要的能力 | PTO-ISA 现状 / 缺口 | 规避方案（不改硬件） |
|----------|-----------|---------------------|---------------------|
| `maxPer4` / `maxPer2` | **分段（segmented）max** | `TROWMAX` 只支持 tile 内**整体**归约，**无原生分段归约** | ⚠️ **早期「零成本 TRESHAPE 窄列 reshape」路线已被探针推翻**（`segreduce_probe.cpp`，RECORD 问题10）：32B 列对齐强制 bf16 tile `Cols%16`，`[.,4]`/`[.,2]`/`[.,1]` 窄列在**类型声明处**即 static_assert 失败；且 `TRESHAPE` 在 `-D__linx` 未暴露（header gap）。**当前落地方式**：中间 tile 声明**物理宽 PW=64（=BlockSize）+ ValidCol 收窄**（64/16/8/1），满足对齐；分段 max 本身**无原生指令**，框架以 `seg_max_adjacent_PLACEHOLDER` 预留（待 (A) strided 宽子 tile+TMAX 树 / (B) emulator strided 语义，问题10 候选）。 |
| `E1_16`、`Vin` 的 16×8 / 64×16 乘 | **多对多广播乘**（8→64、16→64） | `TEXPANDMUL` 只支持 **1→多**广播，**无原生多对多** | ⚠️ **同上，reshape 路线被问题10推翻**。`TROWEXPANDMUL`（`jcore/template_asm.hpp:4882`，src1 逐行 `[R,1]`）仍是 1→C，多对多需先降维——但降维依赖窄列 reshape（不可声明）。框架以 `group_bcast_mul_PLACEHOLDER` 预留（同 (A)/(B) 候选待决）。 |
| `cast_to_E6M2`（BF16→E6M2） | 正向 E6M2 cast（one-level 可发射） | ⚠️ **探针实测：one-level `TCVT(e6m2,bf16)` 不可正确发射（双缺）**（`e6m2_probe.cpp`，见 RECORD 问题3）。底层 `CVTF_E6M2`（`MInst.cpp:715/773`）、`bfloat16_to_e6m2`（`FloatPointUtils.cpp:773`）、`__fp8_e6m2` 结构体（`linx_blkc.h:182`）、`linx_cvt(bf16x2→e6m2x2)`（`template_asm.hpp:993`）都在，但 **① `jcore/type.hpp` 未注册 `type_traits<__fp8_e6m2>` → TCVT/TSTORE 编译失败（`template_asm.hpp:120/1794` no TypeCode）；② `__type_code` 枚举无 `__type_fp8_e6m2` → 即便补 type_traits 也无正确 dst 码，Stage2 占位借 e8m0 发射出 `B.DATR e8m0`（dst 误标）**。修正早期「原生可用」判断 | **bf16→e6m2 位重构**（指数重偏置 127→48 + 2bit 尾数舍入，`TSHRS/TANDS/TADDS/TSELS`，存裸 uint8），绕开 CVT dst 类型；与 recip M2-LUT 同哲学。上游补枚举码+type_traits 后可回退原生 TCVT |
| `recip(E6M2)`→bf16（`SF_BF16_REC`） | E6M2 倒数（bf16） | ⚠️ **三缺俱全，均无原生指令**：① 无 FusedReciprocalCast 等价；② 无 E6M2 原生倒数——`EleRecip`（`TileOpCommonCalc.cpp:281`）switch 只含 FP64/FP32/FP16/BF16/INT*，**无 e6m2 分支**；`TRECIP` dst/src 同 tile_shape、**同 dtype 不含 cast**（`jcore/template_asm.hpp:3474`）；③ 无反向 E6M2→BF16 cast（emulator funcMap 只有正向 `bfloat16_to_e6m2`，无 `e6m2_to_*`）。故不能把 e6m2 直接喂 `TRECIP`，也不能先反 cast 再 recip | **已定 → (b) M2-LUT 位重构倒数**（与 §2.1:63 spec 一致，精确、前向 only、绕开反 cast 缺口）。ea 为 e6m2 uint8：`exp6=(bits>>2)&0x3F`（bias 48）、`m2=bits&3`；`1/ea = 2^-(exp6-48)·1/(1+m2/4)`。以 bf16 位算：`rec_bits = LUT[m2] + ((48-exp6)<<7)`，`LUT[m2]=bf16(1/(1+m2/4))`={`0x3F80`,`0x3F4D`,`0x3F2B`,`0x3F12`}（m2=0..3）。op 序：`TSHRS/TANDS` 取 exp6、`TSHLS(7)`+`6144-·` 得指数项、`TANDS(1/2)`+两次 `TSELS` 选 LUT、`TADD` 合并、HBM 字节别名 reinterpret u16→bf16。仅作用于 `[TileM,1]` 的块级 ea，成本可忽略。<br>**bring-up 快速退路 (a)**：对未 round 的 `SF_BF16` 直接 `TRECIP`——但 dequant 用的是已 round 的 e6m2，二者比值 `E6M2(SF)/SF` 引入**每块 ±≤12.5% 系统 scale 偏差**（还可能翻转 L2/L3 阈值判定），仅供早期打通、不作最终。**(c) 除法不独立成立**：`TDiv` 除数仍须是 bf16 域的 ea_decoded，同样卡在反 cast，退化回 (a) 的 pre-round 不精确 |
| `E1_8`/`E1_16` 的 `≥阈值 ? 1:0`、`Vin` 的 `2^(-E1_*)` | compare(≥)+select（从 {`2^0`,`2^-1`} 二选一） | ✅ **v0.58 已补原生 GE**：`jcore/template_asm.hpp:5887` `template<CmpMode Mode> TCMPS`（发射 `B.DATR Zero,ge`）；`TSEL(dst,mask,src1)` 为 **in-place** `dst=mask?src1:dst_prev`（`:3353`）。早期无带 mode 比较时靠 max-EQ 模拟（已弃用） | **已落地 → 原生 `TCMPS<pto::CmpMode::GE>` + 双 TSEL**。**GE**：`TCMPS<pto::CmpMode::GE>(mask,t,hif4_bf16c(Kbits))` → `mask=(t≥K)`（单指令，替代旧 `TMAXS;TCMP`）。**factor**：`TEXPANDS(f,1.0); TSEL(f,mask,half)` → `f=(t≥K)?0.5:1.0`，供 `TROWEXPANDMUL` 归一化。**E1 位**：`e=0（TSUB(e,t,t)）; TSEL(e,mask,one)` → `e=(t≥K)?1:0`，打包进 L2/L3 scale。K=4（L2）、2（L3）；t 非负有限（无 NaN）。⚠️ hi_f4 用 **GE**，但 mx_quant cuBLAS 用 LT/NE/GT/EQ——逐点按算法取模，非一律 GE |
| **所有 bf16 标量立即数**（`1/7`、阈值 K、`1.0`/`0.5`/`0`） | tile-scalar op 喂 bf16 常量（TMULS/TMAXS/TEXPANDS） | ⚠️ **框架编译实测：LinxV5 后端双缺陷**（RECORD 问题11）：① 任何 `static_cast<__bf16>(float)`（含 constexpr/字面量）发射 `fptrunc→i16`，`PromoteIntegerResult` 无法提升 → `UNREACHABLE (LegalizeIntegerTypes.cpp:57)` 崩溃；② bf16 `0.0` 落硬件 `zero` 寄存器 → `B.IOR [zero],[]` 匹配失败。整数 0/1 标量不受影响 | **已落地（框架 EXIT=0）**：bf16 立即数走**原始位模式** `__builtin_bit_cast(__bf16, uint16_t bits)`（封装 `hif4_bf16c`，-O2 折叠为立即数）；阈值 K 改**编译期模板参数 `uint16_t Kbits`**（不传运行期 float）。位表：1.0=`0x3F80`、0.5=`0x3F00`、1/7=`0x3E12`、4.0=`0x4080`、2.0=`0x4000`。**需 bf16 0 tile**：用 `TSUB(z,t,t)`（t 非负有限），绝不用 `TEXPANDS(bf16 0.0)`。与 mx_quant `__builtin_bit_cast(__bf16, RECIP_EMAX)` 同法 |

### 3.2 tile-op 映射路线

- **路线 A（手工多遍，仿 dynamic_mx_quant）**：分段归约求 `Vmax16/Vmax8/Vmax` → 乘 `1/7` + `cast E6M2`
  → 倒数 → 阈值比较得 L2/L3 → 广播乘 `2^-(E1_8+E1_16)` → `TCVT → e1m2`。可控、可逐 op 对齐。
  **落地状态（2026-08-14）**：整体流程**框架**已落地并**编译干净（EXIT=0）**、发射真实 BSTART.TEPL 块
  （见 `test/.../src/framework_emit.cpp`，`make ... TYPE=FRAMEWORK diss`）；但**含 4 个 `_PLACEHOLDER` 缺口**
  （分段 max、分组广播乘、bf16→e6m2 base、e6m2 recip），故**数值不可信**——是 emit/diss 见证，非数值 harness。
  骨架（尾轴、每行=1 个 64-elem block，plain RowMajor Vec tile）：
  ```text
  # 中间 tile 均声明"物理宽 PW=64（=BlockSize）+ ValidCol 收窄"，满足 32B 列对齐（bf16 Cols%16）
  Vabs   = TABS(x)                              # [TileM, 64]
  Vmax16 = seg_max_adjacent_PLACEHOLDER(Vabs)   # 每 4 相邻 → ValidCol=16（⚠️无原生分段 max，问题10）
  Vmax8  = seg_max_adjacent_PLACEHOLDER(Vmax16) # 每 2 相邻 → ValidCol=8
  Vmax   = TROWMAX(Vmax8)                        # 整块 → ValidCol=1（此步原生）
  SF     = TMULS(Vmax, hif4_bf16c(0x3E12))       # ×1/7；bf16 立即数走位模式（问题11）
  ea     = bf16_to_e6m2_bits_PLACEHOLDER(SF)     # ⚠️ one-level TCVT→e6m2 不可发射，位重构预留（问题3）
  rec    = e6m2_recip_bf16_PLACEHOLDER(ea)       # ⚠️ 无 e6m2 倒数/反向 cast，M2-LUT 预留（问题4）
  # L2：t8=TROWEXPANDMUL(Vmax8,rec)（rec 1→8）；ge_threshold_e1_factor<...,K=0x4080>(t8)→E1_8[VC=8]、f8
  # L3：t16=TROWEXPANDMUL(Vmax16,rec)；group_bcast_mul_PLACEHOLDER(t16,f8)（8→16）；ge_threshold<...,0x4000>→E1_16、f16
  # 归一化（保符号，用 V 非 Vabs）：Vin=TROWEXPANDMUL(V,rec)；group_bcast_mul(Vin,f8)(8→64)；group_bcast_mul(Vin,f16)(16→64)
  y      = TCVT(Vin → __fp4_e1m2x2)              # y≡e1m2（问题7）
  scale  = pack_hif4_scale_word(ea, E1_8, E1_16) # uint32：e6m2 | (L2<<8) | (L3<<16)（问题9）
  ```
  **已固化的规避**（框架编译干净即证据）：GE 用 max-EQ 模拟 + 双 TSEL（§3.1 compare 行）；所有 bf16 标量走
  `hif4_bf16c` 位模式、阈值 K 为编译期模板参数、0 tile 用 `TSUB(t,t)`（§3.1 bf16 标量行 / 问题11）。
  **仍为占位的真实缺口**（4 个 `_PLACEHOLDER`，均待与硬件商讨适配）：分段 max（问题10）、分组广播乘（问题2/10）、
  bf16→e6m2 base 位重构（问题3）、e6m2 M2-LUT 倒数（问题4）。替换前数值不可信。
- **路线 B（单指令 `TQUANT`）**：`TQUANT` 至多做最后的元素 cast；三级 scale 的**求值**仍须算子内完成。
  且**当前仿真器 `TQUANT` 只实现无元数据纯 cast，带 scale tile 直接 ASSERT 失败**
  （`TEPLEngine.cpp:1650-1656`）→ 近期不可用。

### 3.3 目标形态伪码（后端补齐 subview/assemble 后，M=32×N=64 单块示例）

> **前提**：后端计划新增 **tile subview + assemble** 能力（对 CellReg 做 `[:,a:b]` 子视图读写、按写入位置组装）。
> 一旦补齐，§3.2 路线 A 里最丑的部分——16 条显式 `TROWMAX` + 位置式 compaction——即可压成**带 subview 的单条循环**。
> 下面是该形态下的完整、**已复核正确**的伪码蓝图（`M=32` 行、`N=64` 列 = 1 个 block，其它 M/N 按块循环平铺）。
>
> **指令描述符记法**：`<groupCols, rows, cols, dtype…>`。第 1 槽 `groupCols` 对**归约类**（TROWMAX）是每组归约的列宽、对
> **elementwise** 恒 = `cols`（冗余）；归约类的输出列数由目标 subview 隐含。`<size>` 标注：整 tile op 标**整块**字节数，
> 带 subview 循环的行标**单次写入**字节数（整块另计）。谓词 tile（TCMPS 输出）是 **1bit/elem 位图**（描述符沿用 BF16 仅表宗
> 域，实际按 `<size>` 判定；如 `T10<32B>`=32×8bit）。
>
> **立即数图例**：`X1 = 1/7`（满刻度锚，§2.2）、`X2 = 4`（L2 阈值）、`X3 = 0.5`（boost 因子 2^-1）、
> `X4 = 1.0`（无 boost 因子 2^0）、`X5 = 2`（L3 阈值）。全部走 `hif4_bf16c` 位模式立即数（问题11）。

```text
# ── 取绝对值 ──
TABS          <64,32,64,BF16>              T1 -> T2<4KB>          # A=|X|
# ── 三级相邻 max：64 → 16 → 8 → 1 ──
TROWMAX       <4,32,4,BF16>  T2[:,4g:4g+4] -> T5[:,g]<64B>       # for g in 0..15：每 4 相邻 → Vmax16
TROWMAX       <2,32,2,BF16>  T5[:,2g:2g+2] -> T7[:,g]<64B>       # for g in 0..7 ：每 2 相邻 → Vmax8
TROWMAX       <8,32,8,BF16>            T7  -> T8<64B>            # 整块 8 → 1 → Vmax
# ── 一级 base scale：SF = Vmax/7 → E6M2 → 倒数 ──
TMULS         <1,32,1,BF16>        T8,X1  -> T11<64B>            # SF = Vmax·(1/7)
TCVT          <1,32,1,BF16,E6M2,RNE>  T11 -> T12<32B>           # SF → E6M2（一级 scale，写 word[7:0]）
TCVT          <1,32,1,E6M2,BF16,RNE>  T12 -> T12D<64B>          # decode E6M2 → bf16（用量化后的值）
TRECIP        <1,32,1,BF16>          T12D -> T13<64B>            # rec = 1/decode(E6M2)
# ── L2（E1_8，8 个）：Vmax8·rec ≥ 4 ? ──
TROWEXPANDMUL <8,32,8,BF16>       T7,T13  -> T9<512B>            # V8 = Vmax8 · rec（rec 1→8 广播）
TCMPS         <8,32,8,BF16,GE>    T9,X2   -> T10<32B>           # E1_8 mask = (V8 ≥ 4)，1bit/elem
TEXPANDS      <8,32,8,BF16>          X3   -> T14<512B>          # T14 = 0.5（全 8 列）
TSELS         <8,32,8,BF16>    T10,T14,X4 -> T15<512B>          # K8_factor = mask?0.5:1.0 = 2^-E1_8
TROWEXPANDMUL <8,32,8,BF16>      T15,T13  -> T16<512B>          # K8 = rec · 2^-E1_8（每 8 元素一个）
# ── L3（E1_16，16 个）：Vmax16·K8 ≥ 2 ? ──
TROWEXPANDMUL <2,32,2,BF16> T5[:,2i:2i+2],T16[:,i] -> T17[:,2i:2i+2]<128B>  # for i in 0..7：V16=Vmax16·K8[i]（K8[i]→2列）
TCMPS         <16,32,16,BF16,GE>  T17,X5  -> T18<64B>           # E1_16 mask = (V16 ≥ 2)，16 个 1bit
TEXPANDS      <16,32,16,BF16>        X3   -> T19<1KB>           # 0.5（全 16 列）
TSELS         <16,32,16,BF16> T18,T19,X4  -> T20<1KB>           # 2^-E1_16（16 列）
# ── 组装每元素有效缩放因子 Fscale ──
TROWEXPANDMUL <2,32,2,BF16> T20[:,2i:2i+2],T16[:,i] -> T21[:,2i:2i+2]<128B>  # for i in 0..7：F16=K8[i]·2^-E1_16（K8[i]→2列）
TEXPANDS      <4,32,4,BF16>          X4   -> T22<256B>          # 1.0（4 列常量，供 4-lane 复制）
TROWEXPANDMUL <4,32,4,BF16>  T22,T21[:,i] -> T23[:,4i:4i+4]<256B>          # for i in 0..15：F16[i] 铺到相邻 4 列 → Fscale64
# ── 最终乘 + 量化 ──
TMUL          <64,32,64,BF16>      T1,T23 -> T24<4KB>           # Z = X · Fscale（用带符号原始 X=T1）
TCVT          <64,32,64,BF16,HIF4,RNE> T24 -> T25<1KB>          # Z → hifloat4（y，含符号）
# ── 打包 scale word：E6M2|E1_8<<8|E1_16<<16 ──
TPACK         <1,32,1,U32,HIF4_SCALE> T12,T10,T18 -> T26<128B>  # word[7:0]=E6M2、[15:8]=E1_8、[31:16]=E1_16
```

**复核结论**：算法语义完整、字节标注自洽、**嵌套关系正确**——关键的 E1_16 应用步（`T21`）左操作数用
`T16[:,i]`（= K8[i]，复制到相邻 2 列），使 F16[2i]、F16[2i+1] 都乘同一 K8[i]（`floor(2i/2)=floor((2i+1)/2)=i`），
与三级嵌套（L3 组 2i、2i+1 的父级 L2 = `floor(j/2)=i`）逐位一致（修正了早期把左操作数误用为 `[32,8]` 全 K8、导致
奇子组错乘 + view 越界的 bug）。最终乘用 `T1`（带符号原始 X）保号；`TCVT→HIF4` 命中 `HiF4X2`(code14) 规范目标 dtype；
`TPACK<U32,HIF4_SCALE>` 位布局与 ADR-0101 逐位一致（§2.5）。

**此形态仍需的能力（subview/assemble 之外）**——即便后端补齐子视图，下列 gap 仍独立阻塞运行期落地：
- **BF16 `TROWMAX` 被 emulator 运行期白名单拒**（仅 FP16/FP32/INT32，见 RECORD `reference_emulator_trowmax_dtype_whitelist`）
  → 三级 max 全中，**最硬阻塞**；需在 FP16/FP32 域求 max 或等 emulator 修白名单。
- **`TCVT BF16↔E6M2`**（`T12`/`T12D`）→ E6M2 非注册 tile dtype（问题3），需 scale-profile TCVT 或退位重构。
- **`TCVT BF16→HIF4`**（`T25`）→ `HiF4X2` TCVT-only，需工具链暴露 code14（问题7）。
- **`TROWEXPANDMUL`**（1→C 行广播乘）对应问题2「多对多广播无原生 op」；现 kernel 仍为 `group_bcast_mul_PLACEHOLDER`。
- **`TRECIP` / `TEXPANDS` / `TSELS` / `TCMPS<GE>` / `TPACK<HIF4_SCALE>`** 逐个确认是现有 intrinsic 还是同待后端新增
  （`TCMPS<GE>` 已在 v0.58 原生，见 §3.1 compare 行）。

> **与 §3.2 路线 A 框架的记法差异**：本节 subview 版走 **tile 域**阈值→因子（`TCMPS<GE>`+`TSELS`），与现有
> `dynamic_hi_f4_quant_tail.h` 的 `ge_threshold_e1_factor` 同路线（非早期讨论过的 GPR-predicate `TCMP.GPR`/`TSEL.GPR`）。

## 4. 工具链 / 仿真器现状（已知缺口）

- 仿真器 `TQUANT`（`TEPLEngine.cpp:1629-1678`）**只实现无元数据纯 cast**；带 scale/offset tile
  （`srcTile.size()>1`）直接 `ASSERT(false)`——`(in-offset)*scale` 量化数学未实现。
- **`HIF4` 作为 cast 目的类型在仿真器 `DataFormatCvt` 中不支持**；现有 fa_hif4 靠 `BF16→FP4_1(e1m2x2)` HACK 过关。
- `OPCVT_HIF4` 复用 e1m2 的 `bfloat16_to_float4_1`（`FloatPointUtils.cpp:794`）——与 y≡e1m2 定义一致。
- **E6M2 cast：底层在 emulator/ISA，但 one-level 层不可发射（探针已实测，RECORD 问题3）**。正向底层原语
  （`bfloat16_to_e6m2` / `CVTF_E6M2` / `linx_cvt bf16x2→e6m2x2` / `__fp8_e6m2` 结构体）都在，但 **(a) `jcore/type.hpp`
  未注册 `type_traits<__fp8_e6m2>` → TCVT/TSTORE 编译失败；(b) `__type_code` 枚举无 e6m2 码 → 无正确 dst 类型
  （占位借 e8m0 发射出 `B.DATR e8m0`，误标）；(c) 反向 E6M2→BF16 缺**（recip 路径受阻，见 §3.1）。→ L1 base 与
  recip 均改走 **bf16↔e6m2 位重构**，绕开 one-level CVT。
- **无 FusedReciprocalCast**：PTO 只有独立 `TRECIP`（`template_asm.h:206`），recip+cast 需拆两步（§3.1）。
- 工具链↔仿真器 skew 下新编 ELF 无法稳定 gfrun/gfsim，运行期验证暂不可行。
