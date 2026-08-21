# DynamicMxQuant 问题记录

## 问题1：TileSize 大小约束分析

### 结论

- **唯一强制的尺寸检查**是 `IsValidActiveSize`（`tileop-api/jcore/type.hpp`）：
  `Rows × Cols × sizeof(elem) × 4` 必须**恰好**等于 {512,1024,…,32768} 之一，即
  `Rows × Cols × sizeof(elem) ≤ 8192`（`pto_tile.hpp` 里带文字的尺寸 `static_assert` 全被
  注释掉、不生效）。
- 该检查**只作用于实际穿过 load/store 家族或 matmul 的那个 tile 自身**（`template_asm.hpp`
  逐行核对：load/store 家族 393/421/449/481/1697/1721/1759/1788/1810/1833/1852、全部
  TMATMUL 变体 2516/2690/…/2843 检查；**纯 VEC 逐元素/归约 op 一处都不检查**）。
- 因此上限是**纯粹按 tile 自身 dtype 位宽**分档的，**与量化算法无关**：

  | tile 元素位宽 | sizeof(elem) | 元素个数上限 R×C |
  |:---:|:---:|:---:|
  | 8-bit（fp8 / uint8 / fp4-packed byte） | 1 | **8192** |
  | 16-bit（bf16 / fp16 / uint16） | 2 | **4096** |
  | 32-bit（fp32 / uint32） | 4 | **2048** |

- 这是**每个 tile 各自**的上限，不是「kernel 的上限」。一个 kernel 的实际瓶颈 = 它
  **穿过 load/store 的最宽 dtype tile** 的档位。**仅参与 VEC 运算、始终留在寄存器里的中间
  tile（如 TCVT 产生、TMUL 消费的 fp32）不走 load/store，不计入此约束。**

注：算子约束输入x为fp32时，BlockSize仅支持32，故对 fp32 输入，最大 Tile 为尾轴量化：[64, 32] 或非尾轴量化：[32, 64]

### 影响场景

量化轴恒为 BlockSize，故一个穿 load/store 的 tile 上限即**非量化轴的元素数上限 = 元素个数上限 ÷ BlockSize**。
下表按 dtype × BlockSize 列出非量化轴上限（fp4 按打包 byte 计元素个数上限 8192，即 16384 个 fp4 值）：

  | BlockSize \ dtype | fp32（4B，≤2048） | fp16/bf16（2B，≤4096） | fp8（1B，≤8192） | fp4（打包，≤16384 值） |
  |:---:|:---:|:---:|:---:|:---:|
  | 32   | 64 | 128 | 256 | 512 |
  | 64   | 32 | 64  | 128 | 256 |
  | 128  | 16 | 32  | 64  | 128 |
  | 256  | 8  | 16  | 32  | 64  |
  | 512  | 4  | 8   | 16  | 32  |
  | 1024 | 2  | 4   | 8   | 16  |

本 kernel 里不同算法的绑定 tile 不同，故上限不同：

- **OCP / DynRange**：穿过 load/store 的最宽 tile 是 16-bit（bf16 输入 / uint16 scale），
  fp32 中间 tile 留在寄存器 → 上限 **4096** 元素。
- **cuBLAS**：位重解释往返（问题4）多引入一个 **32-bit** load/store tile → 上限被压到
  **2048** 元素。

### 规避方案

- **通用**：尺寸上限是硬件档位，无法绕过；只能**按预算切分 tile**——选 TileM/TileN 使每个
  穿过 load/store 的 tile 满足 `R × C × sizeof(elem) ≤ 8192`（按其 dtype 位宽取档）。
- **cuBLAS 的 2048 元素限制可解除**：它并非算法固有，而是问题4 的位重解释缺寄存器 bitcast、退而用
  HBM 往返所引入的 32b load/store tile。一旦补齐寄存器级 bitcast（见问题4 解除路径），该 32b
  tile 消失，cuBLAS 的绑定 tile 回落到 16-bit 输入 → 上限升到 4096 元素，与 OCP 一致。

---

## 问题2：32 字节对齐约束分析

### 结论

`pto_tile.hpp:649` 的三分支 static_assert（**Tile 构造时**检查——**与问题1 的关键区别**：适用于
**所有 tile**，含只做 VEC、留在寄存器的中间 tile，不像问题1 只卡穿过 load/store 的 tile）。

```cpp
static_assert(
    ((pto::BLayout)0 == BLayout::RowMajor && (pto::SLayout)0 == SLayout::NoneBox &&
     Cols * type_traits<DType>::bits % (32 * 8) == 0) ||
    ((pto::BLayout)0 == BLayout::ColMajor && (pto::SLayout)0 == SLayout::NoneBox &&
     Rows * type_traits<DType>::bits % (32 * 8) == 0) ||
    ((pto::SLayout)0 != SLayout::NoneBox) &&
     (Rows % InnerRows == 0 && Cols % InnerCols == 0),
    "BFractal_ is RowMajor and SFractal_ is NoneBox: Rows must be 32 bytes align, ..."
);
```

**它卡的是 tile「连续轴」的物理字节宽度，必须是 32 字节（256 位）的整数倍**——纯物理的一条线粒度，
与量化算法无关。连续轴 = 内存里相邻元素所在的那根轴，由布局决定（对 tile `[M, N]`）：

| 布局 | 连续轴 | 约束 |
|---|---|---|
| RowMajor | **-1 轴（N）** | `N × sizeof(DType) % 32 == 0` |
| ColMajor | **-2 轴（M）** | `M × sizeof(DType) % 32 == 0` |

（报错文字笼统写「Rows must be 32 bytes align」，但 RowMajor 分支实际卡的是 -1 轴 N。）

### 影响场景

**关键前提：对齐检查卡的是 tile 的物理 Cols（分配宽度），不是 valid/box 宽度。** kernel 里逻辑上很窄的
输出（scale 每 block 每行 1 列、尾块 M_tail 行）一律用「物理撑满 + valid-box 收窄」表达——如
`tile_sred/tile_sstore = Tile<…, TileM, BlockSize, RowMajor, TileM, 1>`：物理 `[TileM,BlockSize]`、
有效列=1，靠 box 只落 1 字节/block。valid-box **不进对齐断言**，故真正被卡的永远是每个 tile 的
**共享物理 -1 轴**——尾轴 = BlockSize，非尾轴 = TileN。

「32 字节」按元素位宽换算成连续轴的最小元素数：

| DType | 每元素字节 | 连续轴最小元素数 |
|---|---|---|
| fp4_e2m1x2（打包，2 值/字节） | 1 字节 = 2 值 | 64 |
| fp8_e4m3 | 1 | 32 |
| bf16 / uint16 | 2 | 16 |
| float | 4 | 8 |

#### 尾轴 `[TileM, BlockSize]`：物理 -1 轴 = BlockSize，除 fp4 外全部天然满足

物理 -1 轴恒为 BlockSize（MX 定义就是 32 的倍数），代入 `物理Cols × sizeof % 32B == 0`：

| tile dtype | 32 元素 × 字节 | 32B 倍数？ |
|---|---|---|
| fp8 / uint8（1B） | 32B | ✓ |
| bf16 / uint16（2B） | 64B | ✓ |
| fp32（4B） | 128B | ✓ |
| **fp4-packed（1B=2 值）** | **16 打包 = 16B** | **✗** |

- **任何 ≥1 字节的非打包类型天然满足**（`32 × 任意整数`恒为 32 倍数）。scale 逻辑上是 `[TileM,1]`，
  但物理沿用 `[TileM,BlockSize]`（uint8→32B、uint16→64B 均过），靠 valid col=1 收窄落盘 →
  **尾轴 fp8 及所有非打包类型不构成额外限制**。
- **fp4 输出是唯一例外**：打包把字节减半，单 block fp4 输出物理宽 = BlockSize/2 = 16 打包 = 16B
  < 32B → 失败。**box 只能收窄不能加宽**，救不了这个物理下限 → fp4 输出 tile 的物理 -1 轴必须撑到
  **64 个 fp4 值的倍数**（= 32 打包字节 = 2 个 block）。
- fp4 的**打包轴与 reduce 轴重合**：直接把物理宽加到 64 会把 2 个 block 并入一次 `TROWMAX`（归约错误）
  → 必须**解耦**：归约仍每 32 一段（分块 `TROWMAX`、各自算 scale/recip），落盘才 `TCONCAT` 到 64 宽
  做单次 `TCVT`；中间 `[TileM,16]` 单 block fp4 tile 根本不构造（非法）。

#### 非尾轴 `[BlockSize, TileN]`：物理 -1 轴 = TileN（自由轴，被强加下界）

物理 -1 轴 = TileN，而 TileN 是**与量化无关**的自由（Post）轴（Post 无需为 TileN 的倍数）。故此处对齐
约束成为对 TileN 的真实**下界**：连续轴 `TileN × sizeof % 32B == 0`。

**与问题1 的 TileSize 上界在同一根轴上对撞**：对齐给 TileN **下界**，问题1 给 TileN **上界**（见
问题1 二维表：非量化轴上限 = 元素个数上限 ÷ BlockSize）。某档 BlockSize 有合法 TileN ⟺ **下界 ≤ 上界**。
上界随 BlockSize 增大而收窄，一旦降到下界以下即无解。

例如 fp8：BS=256 时问题1 上限 32、对齐下界 32，恰好只剩 TileN=32；BS≥512 时上限 < 32，无解。

**fp8 编译验证**（BlockSize=1024，该档已无解，逐步印证两界）：

| tile | Cols×sizeof | 对齐（:649） | 问题1 尺寸（16b≤4096） | 结果 |
|------|:--:|:--:|:--:|:--:|
| `[1024,2]` | 4B | ✗ | — | ✗ 停在 `pto_tile.hpp:649` |
| `[1024,4]` | 8B | ✗ | — | ✗ 停在 `pto_tile.hpp:649` |
| `[1024,32]` | 64B | ✓ | 32768 > 4096 ✗ | ✗ 停在 `template_asm.hpp:1696`（`ext_vector_type(32768)`） |

即 BlockSize=1024 对齐要 TileN ≥ 32、问题1 要 TileN ≤ 4，交集为空 → 无合法 TileN，与表一致。

> **fp4 发射本身可用**（约束只落在 tile 切分，非发射能力）：探针
> `test/kernel/quant/dynamic_mx_quant/src/fp4_probe.cpp`（`TYPE=FP4_PROBE diss`）证实
> fp32→`__fp4_e2m1x2` 单步 `TCVT`+`TSTORE` 发射真实指令（`BSTART.TEPL TCVT, FP32` +
> `B.DATR e2m1x2, byte0`；`BSTART.TLSU TSTORE, e2m1x2`），无 Match-Instruction-Error、无对齐断言。

### 规避方案

通用底线：连续轴须同时满足**对齐下界**（结论表：fp32=8 / fp16=16 / fp8=32 / fp4=64 值）与**问题1
上界**（元素上限 ÷ BlockSize，见问题1 二维表）。有合法切分 ⟺ 下界 ≤ 上界。

**尾轴**（-1 轴 = BlockSize，天然 32B 对齐）：
- **除 fp4 外全部 dtype**：连续轴即量化轴，恒 32 倍数，**无需任何处理**。
- **fp4**：打包轴与 reduce 轴重合，物理宽须撑到 64 值却不能整块并入一次归约 → **解耦**：每 block
  子归约（分块 `TROWMAX` 各算 scale/recip）→ `TCONCAT` 到 64 值宽 fp32 → 单次 `TCVT` → fp4
  `TSTORE`；单 block `[TileM,16B]` fp4 tile 不构造（非法）。

**非尾轴**（-1 轴 = TileN，自由轴，被强加下界）：
- **fp8**：TileN 取满足「下界 32 ≤ 问题1 上界」的值。上界随 BlockSize 收窄——只走 16b tile（bf16 入 /
  fp8 出 / uint16 scale）→ 上界 4096/BS → **BS ≤ 128**；若引入 32b 位重解释往返 tile（问题4）→ 上界
  2048/BS → **BS ≤ 64**。大 BlockSize 无合法 TileN。
- **fp4**：打包轴 ⊥ reduce 轴（打包沿 -1=TileN，归约沿 -2=行=BlockSize），`TileN=64` 走 plain RowMajor
  NoneBox（`tile_o=[BlockSize,32]` packed），**归约零改动、已落地**（`dynamic_mx_quant_nontail_ocp_fp4.hpp`）。
  但 fp4 对齐下界 = 64 值（`(TileN/2)*8 % 256 == 0` → `TileN % 64 == 0`），问题1 上界 = 4096/BS
  （16b 输入 tile 绑定）→ 交点在 **BS > 64**：BS=32 → TileN≤128（64 可）、BS=64 → 恰 TileN=64、
  **BS≥128 上界<64 → plain 方案无合法 TileN**。大 BlockSize 的两解见下（方案 A/B）。

#### 非尾轴大 BlockSize（BS≥128）冲突的两个解

冲突根因：当前一次性载入满 `[BlockSize, TileN]`，`Rows×Cols = BlockSize×TileN` 被顶满，对齐下界（TileN）
与 TileSize 上界（乘积）压在同一根轴上对撞。两条解都从**拆开这个乘积**或**豁免下界**入手。

**方案 A（推荐，通用解）——切分归约轴 + 累积归约**：非尾轴归约轴是 -2 行轴（长 BlockSize）。把
BlockSize 行切成 `R_sub` 行子块（`R_sub | BlockSize`），载入 `[R_sub, TileN]` 子 tile，用 running-`TMAX`
把每列 max 累积到 `[1,TileN]`（每子块 `TCOLMAX` → 累积），跨 `numSub = BlockSize/R_sub` 子块。累积完算
一次 scale/recip，再第二遍重载子块做 `TCOLEXPANDMUL` 广播乘 + fp4 `TCVT` + 落盘。
- **为何总可行**：TileSize 现在约束 `R_sub×TileN ≤ 4096`，`R_sub` 是自由旋钮（可缩到 1，`1×64=64≤4096`
  恒成立）→ TileN 永远能满足 64 对齐下界，**与 BlockSize 无关**。max 结合律保证跨子块归约正确，每列
  scale 对所有子块广播一致。典型取 `R_sub = min(BlockSize, 64)`、`TileN = 64`（BS=128 → 4 子块、
  `[32,64]` 子 tile、budget 4096 ≤ 8192、对齐 64 ✓）。
- **代价**：两遍结构、重读输入（HBM 流量↑）、代码变多；结构与尾轴两遍同构。保持 plain RowMajor，
  无 fractal 落盘风险。**已落地（两个 kernel，同结构）**：
  - `dynamic_mx_quant_nontail_ocp_fp4_bigbs.hpp`（`TYPE=NONTAIL_OCP_FP4_BIGBS`）——pass1 每子块
    `TANDS`（exp 位）→`TCOLMAX`→跨子块 `TMAX` 累积到 `[R_sub,TileN]` valid=1、finalize 用
    **kernel 文件内 static 局部** helper `ocp_scale_from_maxexp_not_tail_boxed_bigbs`（应要求未放入
    common.hpp、未改既有函数）；pass2 `reinterpret`→fp32 inv_scale + 每子块 `TCOLEXPANDMUL`→fp4。
    BS=128 编译/链接/反汇编通过（4×累积链 + fp4 cast 发射，无对齐/TileSize 断言）。
  - `dynamic_mx_quant_nontail_cublas_fp8_bigbs.hpp`（`TYPE=NONTAIL_CUBLAS_FP8_BIGBS`）——pass1 在
    **uint16 abs-bit 域**累积（`TANDS` 0x7FFF→`TCOLMAX`→跨子块 `TMAX`；非负 bf16 位序与幅值单调故
    等价 bf16 amax，inf/NaN 由 `compute_cublas_core` 的 `finite` 掩码兜住）——**改用位域首要是为精确匹配
    AscendC 归约域,附带规避 bf16 `TEXPANDS` seed 崩溃 LinxV5 后端**（`getCopyToParts` illegal-type，与
    `tail_ocp_fp4` .bak 记录同因；改用 uint16 `TEXPANDS(0)` seed 合法）。**注:bf16 逐元素 `TMAX` 本身
    不崩——已探针实测编译通过——故 bf16 值域 + peeled-first-sub-chunk `TCOLMAX` seed 亦可编译,选 uint16
    位域是为对齐 AscendC,非因 bf16 `TMAX` 不可用**；累积后 **`reinterpret_tile<__bf16>` 零指令视图**→`TCVT`→fp32 amax，
    再**就地内联展开** `compute_cublas_core`（规避问题8 的 S64 栈往返，与 plain 同法：`raw`/`s32v` 两处
    `reinterpret_tile<uint32_t>` 视图 + 原生 `TCMPS<CmpMode::{LT,NE,GT,EQ}>`，无 min/max-EQ 模拟、无 scratch-HBM）；
    pass2 `reinterpret_tile<__bf16>`→fp32 inv_scale + 每子块 `TCOLEXPANDMUL`→fp32→fp8。bf16/half/fp32
    三输入 BS=128（`R_sub=32`/`numSub=4`/`TileN=32`）compile+diss 通过、发射 42 条原生 TCMPS、零 scratch-HBM。
  - **预算（cublas-bigbs 实测更正 2026-08-20）**：cuBLAS 固有 fp32 amax + uint32 位运算 + pass2 fp32 数据 cast，
    这些都是 physical `[R_sub,TileN]` 的 **32b tile**，经 8192B tile 律（`TilesizeCode` enum 无 >8192B 码位）锁死
    `R_sub*TileN≤2048`——**实测 2048 编过、4096 撞 TADDS `B.IOT unknown operand`**。故此 32b 中间量是**指数抽取固有**、
    非可去 workaround，「formal 4096」**不可达**；`static_assert` 已从 4096 收紧到 **2048**（当前 `TileN=32`/`R_sub=32`
    合法，`R_sub=64`/`TileN=32` 亦可，`R_sub=64`/`TileN=64`=4096 编不过）。**ocp-fp4-bigbs 不变**（走 16b、无 32b 中间量，
    仍 `≤4096`）。**两者 runtime 因 skew 未验**（默认 BS=32 走 plain）。**两个 bigbs 均已逐 op 对齐 AscendC**：
    - **cublas-fp8-bigbs 对齐 `ComputeScaleCuBlas`**：归约域与 AscendC 一致（uint16 abs-bit——AscendC 非尾轴
      cuBLAS bf16 分支即 `And(BF16_ABS_MASK)`+`uint16 Reg::Max` 累积，`..._not_tail_axis_optimize_high_perf_large_tail.h:426-440`）、
      守卫+recip 为**就地内联展开的 `compute_cublas_core`**（原生 CmpMode，AscendC guard 忠实移植，`TOR`≡`MaskXor`
      因 p0/p1 互斥）、pass2 同 plain `compute_cublas_scale_not_tail` → 与 plain 同结果，唯残缺口 = 问题5 parity 交织。
    - **ocp-fp4-bigbs 对齐 `ComputeScaleOcp`**（`..._not_tail_axis_optimize_high_perf_large_tail.h:663-777`）：
      pass1 拆子块 `TMAX` 累积因 max 结合律 == 单遍全行 max，与 AscendC 同为 **uint16 指数位域**（`And(0x7F80 exp
      mask)`+`uint16 Reg::Max` 累积，种子 0）；finalize 的 kernel 内局部 helper `ocp_scale_from_maxexp_not_tail_boxed_bigbs`
      与已 review 的 plain `compute_ocp_scale_not_tail_boxed` 尾段**字节一致**，而 plain 的 `clamp-up 到 emax 再减
      emax` ≡ AscendC `762-764` 的 `减 subNum 再对 <emax 置 0`（代数恒等），`finalize_scale_recip_u16` 对应
      AscendC `765-776`、常量全核对（0x7F00/0x7F81/0x0040/0x00FF）。**残留**：仍缺问题5 parity 交织（compact
      平铺）；data 路径 fp32→fp4 直转 cast 语义待确认（问题6）。
    两个 plain kernel 本身仍是各自可用范围（ocp BS≤64 / cublas 当前 BS≤64）的单遍。

**方案 B（备选）——fractal/Box 布局整体豁免对齐（尾/非尾皆可）**：改用 `SLayout::Box`/fractal（:649
分支3）**完全绕开 32B 字节下界**（只需 `Rows%InnerRows==0 && Cols%InnerCols==0`），TileN 可任意小、只剩
TileSize 上界。`fa_hif4.hpp:85-92` 发射已验证；但为寄存器侧 fractal 布局（`SFractalSize_==512/1024`，
`pto_tile.hpp:658`），落盘 plain global 的字节序需运行期核实（当前 skew 不可测）→ 作 fallback 而非首选。

（附：**方案 C 不适用于 fp4 大 BS**——上界由最宽 16b 输入 tile 决定，fp4 输入无法窄于 16b，拉不动上界；
它只对 cuBLAS 有效：补寄存器 bitcast 去掉 32b 往返 → 2048→4096，属工具链侧修复、与 fp4 对齐冲突无关。）

---

## 问题3：带 CmpMode 的 4-参 TCMP/TCMPS（工具链已原生支持；业务 kernel 待迁移）

> **状态（当前工具链）**：`template_asm.hpp` 已在 `-D__linx` 下原生提供带 CmpMode 的
> `TCMP`/`TCMPS`——`template <CmpMode Mode, ...>` 覆盖全 6 模式（EQ/NE/LT/GT/LE/GE，
> `template_asm.hpp:3341` / `:4076` 以 `if constexpr` 分派），并保留 3-参 EQ-default 重载
> （`:3461`）向后兼容。故下文所述「4-参重载仅存在于 `jcore/TCmp.hpp`、linx 未包含」的缺口
> **已闭合**。剩余工作纯在**业务 kernel 侧**：`compute_cublas_core`
> （`dynamic_mx_quant_common.hpp:432`）仍用 min/max + 默认-EQ 的规避写法，末尾保留了
> `IDEAL VERSION (blocked)`（`:522`），待切换到原生 CmpMode。
>
> **注意（实测 2026-08-19）**：CmpMode 是 **模板参数**，正确调用形如
> `TCMPS<CmpMode::LT>(dst, src, s)`——**不是** docs / IDEAL VERSION 注释里那种 4-参运行期形式
> `TCMPS(dst, src, s, CmpMode::LT)`（该形式在 `-D__linx` 下不存在、从未编译过）。签名见
> `template_asm.hpp:4076` `template <CmpMode Mode, ...> void TCMPS(out&, in&, DType s)`。
>
> **已迁移（plain 路径）**：`dynamic_mx_quant_nontail_cublas_fp8` 的 `nontail_cublas_fp8_plain`
> 已把 `compute_cublas_core` 就地展开并全部换成原生 `TCMPS<CmpMode::{LT,NE,GT,EQ}>`（无 GE），
> compile+diss 实测每条 InT 路径发射 7 条原生 TCMPS，min/max+EQ 模拟序列消失。**`nontail_cublas_fp8_bigbs`
> 亦已同步迁移（2026-08-20）**：就地展开 `compute_cublas_core` + 原生 `TCMPS<CmpMode>`，bf16/half/fp32 BS=128
> compile+diss 通过（42 条原生 TCMPS，无模拟序列）。**其余 5 个 kernel** 仍调用 `common::compute_cublas_core`
> （规避版）保持不变（注：ocp-fp4 系列从不走 cuBLAS core）。
>
> **gfrun 端到端已验证（2026-08-20）**：ELF 用 env_test 工具链（含 B.IOR 元素步长修复 `f35d3aa`）编译、
> 工作目录 gfrun（feat/pto-v058-adaptation + 5 处 emulator 反应式移植，见问题9/14/17/18）执行到底
> `R2=0`。**data 输出逐字节匹配 golden**（32 行全对，无棋盘错位）；**scale 值逐字节匹配**（仅布局差
> = 问题5 parity 交织，值本身一致）。故原生 CmpMode 路径数值正确性已坐实。

### 结论（缺口闭合前的历史记录）

`-D__linx` 构建下曾**只有 3-参 mode-less（语义固定 EQ）的 `TCMP`/`TCMPS`**，没有带 CmpMode 的
4-参重载，故 `>` / `<` / `!=` / `<=` / `>=` 无法直接发射。这是 **linx intrinsic 头文件封装的
缺口，不是硬件/仿真器能力问题**：

- 根因：`-D__linx` 分发链（`common/tileop_api_impl.hpp:4-5`）只包含 `jcore/template_asm.hpp`；
  带 CmpMode 的 4-参 `TCMP`/`TCMPS` 当时只定义在 `jcore/TCmp.hpp` / `aarch64/TCmp.hpp` /
  `cpu_sim/TCmp.hpp`——这三个在 linx 构建里都没被包含。（现已在 `template_asm.hpp` 原生补齐，见上方状态。）
- linx 当时可用的是 `template_asm.hpp` 的 3-参版，无 CMode 操作数，解码固定为 EQ（`CMode::EQ=0`，
  `SuperScalarModel/isa/ISACommon/BlockAttribute.h:25`）。`CmpMode` 枚举本身可见
  （`common/pto_tile.hpp:19`），故报错停在「4 参无匹配重载」而非「未定义标识符」。
- 底层 ISA/仿真器**支持全部 6 种模式**（`isa/calculate/CubeCalculate.cpp` `EleCmp`，
  `isa/Block.cpp:535` 从 `srcs[SRC6_IDX]` 解 cMode）——缺口纯在头文件封装。

**解除路径**：已由 `template_asm.hpp` 原生 4-参重载实现；业务 kernel 切换到下述 IDEAL 版本即可。

### 影响场景

cuBLAS 算法的 scale 计算（`compute_cublas_core`，`dynamic_mx_quant_common.hpp`）需要
`>` / `<` / `!=`，以 1:1 对照 AscendC `ComputeScaleCublas`
（`dynamic_mx_quant_tail_axis_fp8.h:741-759` 的 `Compare<LT>/<NE>/<GT>`）。文档
`docs/content/intrinsics/{tcmp,tcmps}.md` 记载的 4-参签名：

```cpp
TCMP (TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, CmpMode cmpMode);
TCMPS(TileDataDst& dst, TileDataSrc0& src0, T src1,            CmpMode cmpMode);
```

在 `-D__linx` 下直接调用即编译失败：

```
error: no matching function for call to 'TCMPS'
    TCMPS(p0a, exp32, 0, CmpMode::GT);
note: candidate template not viable: requires 3 arguments, but 4 were provided
```

### 规避方案

`compute_cublas_core` 用 min/max + 默认-EQ 的 3-参比较模拟 GT/LT/NE（语义等价）：

```
a<b  == TMINS(t,a,b-1); TCMP(m,t,a)   // t==a ⇔ a<=b-1 ⇔ a<b
a>b  == TMAXS(t,a,b+1); TCMP(m,t,a)
a!=b == TNOT(TCMPS(m,a,b))
```

理想的 1:1 CmpMode 版本以注释形式保留在 `compute_cublas_core` 函数末尾
（标记 `IDEAL VERSION (blocked)`），待 linx 补齐 4-参重载后可直接切换。

---

## 问题4：寄存器级 reinterpret（位重解释）（工具链已原生支持；业务 kernel 待迁移）

> **状态（当前工具链）**：主线已提供零指令寄存器级 bitcast `reinterpret_tile<>`
> （`common/pto_tile.hpp`），把 tile 元素按位重解释为等宽 dtype、不经 HBM、不做数值转换。
> 探针 kernel（`probe_dynamic_mx_quant_tail_ocp_fp8.hpp`）已改用它。故下文所述「没有任何寄存器级
> reinterpret/bitcast」的缺口**已闭合**。剩余工作纯在**业务 kernel 侧**：`dynamic_mx_quant_common.hpp`
> 仍用 `reinterpret_u16_to_bf16` / `reinterpret_f32_to_u32`（`:201`/`:217`）的 scratch-HBM 往返，
> 待迁移到 `reinterpret_tile<>`。迁移后 cuBLAS 非尾轴的 32b 往返消失、绑定回落 bf16 输入宽度，
> TileSize 上界从 2048 回到 4096（见下方「代价2」与问题1）。
>
> **已迁移（plain 路径，实测 2026-08-19）**：`nontail_cublas_fp8_plain` 已把 scale pass 就地展开，
> 两处位重解释换成 `reinterpret_tile<>`：（a）recip(uint16) → bf16 视图喂 TCVT；（b）clamp 前后
> 各开一次 uint32 视图（`raw` 供 finite/nonzero 的 TCMPS 直接消费；`s32v` 供指数/尾数抽取）。
> compile+diss 实测：scale 段仅 2×TLOAD + 2×TSTORE（纯数据流），**无 scratch-HBM 往返**，
> `reinterpret_f32_to_u32`/`reinterpret_u16_to_bf16` 零命中。
>
> **gfrun 端到端已验证（2026-08-20）**：见问题3 状态块——同一 ELF+gfrun 跑到底 `R2=0`，data 逐字节
> 匹配 golden、scale 值逐字节匹配。`reinterpret_tile` 视图被 compare/select 消费时暴露的 emulator
> dtype 标签断言（问题14 的 302 兄弟）已按位宽相等放松，见问题14 补记与问题17/18。
>
> **实测约束**：`reinterpret_tile<>` 返回的 view 是**同寄存器视图**，与 TCVT / TCMP 这类**双模板参**
> op 兼容（out/in 可异型，直接吃 view）；但 TSHRS/TANDS/TAND/TOR/TSEL/TADDS 等**单模板参** op 要求
> dst 与 src **同类型**，view ≠ 真实 tile → 编译失败。故指数/尾数位运算需先用一条 u32→u32 恒等 TCVT
> 把 view 物化成真实 uint32 tile（一条寄存器级指令，仍远优于 scratch-HBM 往返），之后的位运算全在真实
> tile 上做。**`nontail_cublas_fp8_bigbs` 亦已迁移（2026-08-20）**：pass1 amax、core 的 `raw`/`s32v`、pass2 recip
> 四处全换 `reinterpret_tile<>` 视图，零 scratch-HBM。**其余 5 个 kernel** 仍走 `common::reinterpret_*`（scratch-HBM）
> 保持不变。

### 结论（缺口闭合前的历史记录）

`-D__linx` 构建下曾**没有任何寄存器级 reinterpret/bitcast**——float tile 与等宽 int tile 之间
不能在寄存器内按位重解释：

- `TCAST` 在 `-D__linx` 下未声明（`common/tileop_api.hpp` 的 `#ifndef __linx` 块内）；且即便
  可用，它 lower 成单条 `TCVT`，是**数值转换**（`static_cast`，`0.001f → 0`），不保留
  IEEE-754 位型——用它抽 exponent 位在任何后端都语义错误。
- 当时没有 tile 级 `__builtin_bit_cast` / reinterpret intrinsic 能在寄存器内把 tile 元素按位
  重解释为等宽的另一 dtype。（现已由 `reinterpret_tile<>` 补齐，见上方状态。）

**解除路径**：已由主线 `reinterpret_tile<>` 实现；业务 kernel 把下述 `reinterpret_*` 的
scratch-HBM 往返替换为寄存器写法，即可省去多余 store/load 与 scratch buffer。

### 影响场景

MX / E8M0 量化必须把 float 位型当整数用（抽 exponent/mantissa 位）、把整数位型再当 float 用
（构造 `2^(-E)` recip 乘子），即**位重解释**而非数值转换——**三种 scale 算法都要做**。因此
缺寄存器级 reinterpret 直接卡住三条 scale 计算路径（OCP / DynRange / cuBLAS），必须走下述规避。

### 规避方案

经 scratch-HBM 同宽度字节别名往返实现位重解释（`dynamic_mx_quant_common.hpp` 的
`reinterpret_u16_to_bf16` / `reinterpret_f32_to_u32`）：

```
static uint8_t buf[R*C*sizeof(T)];               // scratch HBM
TSTORE(alias_global_as_srcT(buf), src_tile);     // 按源 dtype 写原始字节
TLOAD(dst_tile, alias_global_as_dstT(buf));      // 按目标 dtype 读同一字节
```

- `TLOAD`/`TSTORE` 只按**元素字节宽度**搬原始字节、不做值转换，故同一段字节用两个不同 dtype
  的 `global_tensor` 别名读写即为真·reinterpret。
- `Slot` 模板参数为每个并发存活的 reinterpret 选一块独立 buffer，避免互相踩踏。
- 代价1：每次位重解释多一次 HBM 往返（store+load）+ 一块静态 scratch buffer。

**代价2：往返的 store+load 让位重解释 tile 穿过 load/store，被问题1 的 TileSize 约束按其位宽卡住。**
cuBLAS 在 fp32 域抽 exponent（`reinterpret_f32_to_u32`，32-bit），这个 32b tile 比 kernel 里其它
tile 都宽 → 上限压到 **2048**；OCP/DynRange 在 bf16 域（`reinterpret_u16_to_bf16`，16-bit），与
输入 tile 同宽、不加宽绑定 → 上限仍 **4096**。故 cuBLAS 的 2048 是此 workaround 的产物，非算法
预算；补齐寄存器级 bitcast 后往返消失，cuBLAS 也回到 4096。

### 政策：assert 编码正式（补齐后）边界，当前缺口以注释记录

kernel 的 `static_assert` 尺寸/BlockSize 边界一律**按编译器补齐寄存器 reinterpret 后的正式模型**编写
（cuBLAS 非尾轴 = `BlockSize*TileN ≤ 4096` → BS ≤ 128），**不**按当前 fp32 往返的收紧值（2048 → BS ≤ 64）。
理由：数据路径的 scratch-HBM 往返是**临时 workaround**（本问题），一旦补齐即消失；把临时值烧进 assert 会在
修复后反而误报。当前工具链的更严实际上界（`64 < BS ≤ 128` 仍停在 `IsValidActiveSize`）作为**已记录缺口**
写在 kernel 头注释与 README「已知限制」中，而非 assert。**对不走 fp32 往返的路径（OCP 在 16b 域）正式与
当前同界，assert 即真实界**（见问题1 的 OCP/cuBLAS 上界区分）。

---

## 问题5：linx `-D__linx` 头未暴露 TINTERLEAVE/TDEINTERLEAVE（需 linx 侧解决）

### 结论

`-D__linx` 构建下**没有 zip/unzip 交织 intrinsic**——`TINTERLEAVE`/`TDEINTERLEAVE` 无法直接发射。
但这**不是「ISA 没有该指令」，而是「ISA 有、linx 头未暴露其 lowering」**——与问题3（4-参 TCMP）、
问题4（寄存器 reinterpret）**同一类头文件封装缺口**，非硬件能力问题：

- **ISA 层有定义**：`docs/scripts/data/linxisa-0.57-intrinsics.txt:101-102` 列出
  `TDEINTERLEAVE`/`TINTERLEAVE`；文档页 `docs/content/intrinsics/{tinterleave,tdeinterleave}.md`
  给出 4-参签名，语义即 parity zip：`result[2i]=even[i]`、`result[2i+1]=odd[i]`，结果按中点拆到
  两个 dst tile：

  ```cpp
  template <typename DstTile, typename SrcTile>
  PTO_INST void TINTERLEAVE(DstTile &dst0, DstTile &dst1, SrcTile &even, SrcTile &odd);
  ```

- **但 `-D__linx` 头里没有**：`-D__linx` 分发链（`common/tileop_api_impl.hpp:4-5`）只包含
  `jcore/template_asm.hpp`。对整个 `tileop-api` 目录做大小写不敏感搜索
  `interleave|intlv|deintlv` → **零匹配**。`template_asm.hpp` 的 TEPL 操作码本身有空档
  （29/30/31、48-55 缺失），无对应发射模板。
- **规范自身的限制标注**：0.57 workbook 把该 op 标为 **A5-only**，且**无独立 PTO-AS 汇编页**，
  注明「由选定 backend 提供 block-template lowering」——而 linx block backend 未提供。

**解除路径（linx 工具链侧）**：在 `-D__linx` 下暴露 `TINTERLEAVE`/`TDEINTERLEAVE`（在
`template_asm.hpp` 补 4-参发射模板，或让 `tileop_api_impl.hpp` 的 `#ifdef __linx` 分支也包含
定义它的头）。补齐后即可实现非尾轴 scale 的 parity 交织。

### 影响场景

非尾轴 scale 的 AscendC-faithful 落盘布局是 `[ceil(Axis/32/2), Post, 2]` 的 **parity 交织**
（even/odd 块行按列 zip，parity 在最内层），对应 AscendC `DataCopy<DIST_INTLV_B8>` /
`Reg::Interleave`（standalone `dynamic_mx_quant_post.h` / swiglu `axis_not_last.h`）。缺交织
intrinsic 即无法在寄存器内做这个 zip。**尾轴无需交织**（块行在行内已连续，compact 平铺即等价，见
host tiling :415-416 / swiglu `axis_last.h:585-592`），故本缺口只影响非尾轴。

### 规避方案

当前非尾轴 scale 输出退化为 **compact 平铺**（每块 1 字节，归约轴 ÷BlockSize + 偶数对齐，见
`nontail_cublas_fp8` / `nontail_ocp_fp4`）：每块 E8M0 语义逐 op 对齐 AscendC，但**块行的 parity
交织尚未施加**，故非尾轴 scale 布局尚非 AscendC-faithful。补齐 4-参交织重载后，在 compact 平铺
之上追加一次 `TINTERLEAVE`（even/odd 块行 → 交织落盘）即可对齐。

### 验证：compact-planar 的数值与列序已实测坐实，与 golden 差异纯 = 交织（2026-08-20）

问题17 记录 `nontail_cublas_fp8_bigbs` gfrun 端到端跑通后，结论「scale 与 golden 仅差 parity 交织
布局、数值/列序均正确」不能只靠**值集相同**判据——随机输入下非尾轴 scale **近常量**（每列是
`BlockSize=128` 个同分布样本，amax 相近 → E8M0 多为同一字节 `0x78`，仅个别 `0x79`），此时列错位或
数值错误都可能被同值掩盖。故该结论由两条**布局无关**证据坐实：

1. **data 逐字节匹配 = 布局无关的 scale 正确性证据**：data = `quantize(x × recip)`，`recip = 1/scale`
   取自**寄存器**、**不经 `scale_output.bin` 落盘布局**。故 4096B data 全对 ⟹ 每一列的 scale 数值都对，
   与 scale 存储布局（planar vs 交织）无关。这是「数值正确」的主证据。
2. **单调判别实验 = 列序 + planar 布局证据**：为破近常量，另构造输入令第 `c` 列 amax = `0.02·2^(c/2)`，
   得**逐列单调**的 scale `0x71→0x81`（32 列 17 个不同值）。参考两布局：
   planar = `[0x71,0x72,…,0x81, 0×32]`、interleave = `[0x71,0,0x72,0,…]`。工作目录 gfrun 跑出
   kernel `scale_output` 前 32B = `[0x71,0x72,…,0x81]` **严格单调**、**== planar 字节精确**、
   **≠ interleave**、后 32B pad 全 0 ⟹ kernel 写 **planar 布局、列序正确（单调保持 = 无列错位）**，
   与 golden 差异**纯粹 = parity 交织**（本问题），非数值/列序错误。

（判别实验为一次性验证，产物已清理、compare 目录复原为随机 golden；复现只需按上式生成单调输入重跑。）

---

## 问题6：fp4 数据路径的 cast 域 —— fp32→fp4 直转是否合法，待 ISA/编译器确认

### 背景

**两个 OCP-fp4 kernel 同构**，数据路径都在 **fp32 域**乘 inv_scale 后**直接 `TCVT`
fp32→`__fp4_e2m1x2`**：
- `dynamic_mx_quant_tail_ocp_fp4.hpp`：每块 `TCVT(xf, xq)`（bf16→fp32）+ `TROWEXPANDMUL`（fp32 域），
  两块经 scratch-HBM concat 后 `TLOAD(xcat) → TCVT(oq, xcat)`（fp32→fp4）。
- `dynamic_mx_quant_nontail_ocp_fp4.hpp`：line 92-95 `TCVT(xf, xq)`（bf16→fp32）+ `TCOLEXPANDMUL`（fp32 域）
  + `TCVT(oq, xf)`（fp32→fp4）。

对照 AscendC 基准（`dynamic_mx_quant_tail_axis.h` `ComputeDataOptimizeBf16` /
`ComputeDataFloatToFP4`，及 swiglu `swiglu_mx_quant_common.h` `ComputeDataF4Last`），
AscendC **从不做 fp32→fp4 直转**，只有两条合法路径：

- **bf16 输入（默认路径）**：在 **bf16 域**乘 halfScale，然后 **直接 bf16→fp4** cast，**不调
  网格对齐 helper**（硬件 bf16→fp4 cast 会正确 round 到 E2M1 非均匀网格 {0,0.5,1,1.5,2,3,4,6}）。
- **fp32 / fp16 输入**：先在 fp32 域调 `ComputeFP4FromFp32` / `ComputeFP4FromHalf`
  （E2M1 网格对齐取整 helper，抽指数构造 2 幂缩放 → 抬到整数网格 → `Truncate` → 缩回，
  见 swiglu `ComputeFP4FromHalf:145-179`），再 fp32→bf16→fp4。

即：**只有 bf16→fp4 硬件 cast 能自对齐 E2M1 网格；fp32→fp4 若无 helper 会 round 错**。

### 待确认

Bench 直转能否成立，取决于两个 linx ISA/编译器事实（**尚未确认**）：

1. linx `TCVT` 是否**支持 fp32→`__fp4_e2m1x2` 直转发射**（fp4_probe 已验证单步 fp32→fp4
   `TCVT`+`TSTORE` 能发射真实指令、无 Match-Instruction-Error，见问题2 —— 故发射层面看似可用）。
2. 该直转在硬件/仿真器上**是否正确 round 到 E2M1 网格**（若像 AscendC 假设的那样 fp32→fp4
   不自对齐，则数值错误，须改走 bf16 域 或补 `ComputeFP4FromFp32` 等价的网格对齐）。

### 决策

**代码暂时保持 fp32→fp4 直转**，待 ISA/编译器确认第 2 点（round 语义）后再决定：
- 若 fp32→fp4 直转 round 正确 → 保持现状即可，无需改。
- 若不正确 → 改为 Option A：concat / 乘 inv_scale / cast 全留 **bf16 域**，末尾 `TCVT` bf16→fp4
  （对齐 AscendC bf16-input 默认路径）；或补 fp32 域网格对齐 helper。

**验证阻塞**：端到端精度当前因工具链↔仿真器 skew 不可测，无法直接观测直转 round 结果，
故此项须由 ISA 规格 / 编译器 cvt 语义静态确认。

---

## 问题7：OCP 新算法（bf16 乘 + 直转 e8m0）替代移位法的两个边界

### 背景

AscendC 因**无法直转 fp8_e8m0**，OCP scale 用「clamp 到 emax → 减 emax → 右移 7 位（`>>BF16_SHR_NUM`）
→ inf-Select」的移位法（见 `bak/AscendC_dynamic_mx_quant_tail_axis_fp8.h:620-683`）。已确认 **PTO-ISA
可直转 bf16→e8m0**（`TCVT`，见问题2/type 表），故有更简的等价算法（示例见
`bak/AscendC_dynamic_mx_quant_tail_axis_fp8_ocp_new.h:79-98`）：

```
sharedExp(bf16) = xMaxExp(bf16, 即 2^E_max) × 2^(-emax)      // Reg::Mul，bf16 域
scaleByte(e8m0) = Cast<bf16→e8m0, TRUNC>(sharedExp)          // 直转，替代 clamp+减+移位+inf-Select
recip           = 0x7f00 − reinterpret_u16(sharedExp) + 三 Select（inf→nan / zero / special）
```

**主体等价性已证**（代数 + 实例）：有限 normal 区间内，新 `sharedExp` 的 bf16 位型
`(E_max−emax+127)<<7` 与旧 `shared_exp` 字段**逐位相同**，故直转产出的 E8M0 字节 == 旧右移产出的字节，
recip 的 `0x7f00−bits` 也逐位一致；bf16 乘为**精确 2 幂相乘**（两操作数尾数皆 0、有限区间不溢出），无
round 误差。**仅两处边界发散**，因 skew 无法运行期验证，记录如下。

### 边界1：inf/nan 的 scale 字节改依赖硬件 `Cast(inf/nan)==0xFF`

- **旧法**：显式 `Select(cmpResult=(maxExp==0x7F80), scaleValue, fp8NanU16=0x00FF)` 把 inf/nan block
  的 scale 字节强制置 `0xFF`。
- **新法**：**去掉这条 inf-Select**，inf/nan 的 `sharedExp` 会是 inf/nan（`0x7F80 × 有限 = inf`），
  scale 字节完全由 `Cast<bf16→e8m0, TRUNC>(inf/nan)` 的硬件语义决定。**期望**其产出 E8M0 全 1
  `0xFF`（E8M0 的 NaN/最大编码），从而与旧法一致。
- **风险**：`Cast(inf/nan)→0xFF` 是**假设**，PTO-ISA/仿真器对 bf16→e8m0 溢出/非有限输入的饱和语义
  **未经运行期确认**。若硬件 clamp 到 `0xFE` 或其它值，inf/nan block 的 scale 字节将与 AscendC 不符。
- **影响面**：仅 inf/nan block 的 **scale 字节**（recip 路径仍保留显式 inf→nan Select，不受影响）。

### 边界2：极小非零下溢 `|x| < 2^(emax−127)` 的 recip 发散

- **旧法**：`sharedExp` 先 **clamp 到 ≥emax**（`Select(maxExp<=emax, maxExp=emax)`），故 E_max<emax 时
  `sharedExp` 被顶到 emax → 减 emax 后为 0 → **recip=0**（scale 也为最小档）。
- **新法**：**去掉 clamp**，E_max<emax 时 `sharedExp = 2^(E_max−emax)` 为**真·denormal 小值**（指数域
  负），reinterpret 后 `0x7f00 − 小正值 ≈ 0x7f00`，得到一个**接近 1 的 recip**而非 0。
- **影响面**：**仅 recip 路径**，且**仅当** block 的 `E_max < emax`（即整块 |x| 都 `< 2^(emax−127)`：
  e4m3 为 `<2^-119`、e5m2 为 `<2^-112`、fp4_e2m1 为 `<2^-125`）——极端下溢区。**精确零**仍由 `zeroMask`
  单独置 0，不受影响；scale 字节两法一致（直转对 denormal 也 TRUNC 到最小档）。
- **性质**：这是 AscendC clamp 的**保守截断** vs 新法**保留真实缩放**之别。数学上新法给出的
  `recip=2^(emax−E_max)`（放大到 dtype 可表示范围）**更贴近 MX 定义本意**，旧法的 recip=0 是
  clamp 的副作用。孰为「正确」取决于 golden 参考，**待 skew 解除后按 AscendC golden 比对确认**。

### 决策

**先落地新算法**（主体等价、去掉 clamp/移位/inf-Select，代码更简、少一次 `TSHRS`）；两个边界作为
**已记录缺口**，待 skew 解除后运行期比对 AscendC golden：边界1 验 `Cast(inf/nan)==0xFF`，边界2 验
下溢 block 的 recip 是否需补回 clamp 对齐 AscendC。**验证阻塞同问题6**（skew）。

---

## 问题8：非内联 helper 的 tile 参数经 `TSTORE/TLOAD, S64` 栈传参，被 emulator `ValidateLocalTlsu` 拒绝（需 emulator / linx 后端侧解决）

### 结论

当一个收/发 tile 的 helper（如 `compute_cublas_core` / `compute_cublas_scale_tail`）**未被内联**时，
LinxV5 后端用 **`TSTORE, S64` / `TLOAD, S64`（把 tile 当 64-bit 通用字节块 memcpy）在栈上传递
tile 实参**；而 emulator 的 `ValidateLocalTlsu`（`SuperScalarModel/emulator/engine/AccumulateBlockInfo.cpp:60`）
要求 Local TSTORE 的**源 tile dtype 精确等于 store block 的 dtype**
（`IsCompatibleDataTile` :270 `source->tileInfo->dataType == dataType`），S64 块 ≠ 源 tile 的真实 dtype，
故触发断言：

```
AccumulateBlockInfo.cpp:60 ValidateLocalTlsu
ASSERT(... IsCompatibleDataTile(inst->srcs[1], block->dataType, ...)
       && "Local TSTORE requires one compatible source Tile")
```

这**不是 kernel 逻辑缺陷、也与位重解释（问题4）无关**——反汇编与诊断实测的证据链：

- **反汇编**（`dynamic_mx_quant_tail_cublas_fp8_fp16.elf.diss`）出现源码里从未写过的 `BSTART.TLSU
  TSTORE, S64`：源码只写了 `TSTORE` scale（U8）与 output（e4m3）两种，diss 却有 **5 条 `TSTORE, S64`**。
  首条 PC `0x113d8`：`max_f`（`TCVT FP16→FP32` 的 per-row amax tile，1KB）被 store 到栈 `sp+1152`，
  紧接 `C.BSTART.STD DIRECT <compute_cublas_core>`；进入 core 后 `0x1140c: BSTART.TLSU TLOAD, S64`
  把它取回。→ 这是**跨函数传 tile 参数的 caller-save spill**，后端统一用 S64 块搬运。
- **emulator 诊断**（在 :60 断言前打印）：`block.dtype=16`（`DataType::INT64`/S64，`elemB=8`）、
  维度被解成 `1/1/1`；而 `src.dtype=1`（FP32）、`[validRow=8, col=32]`、`size=1024`。block 的 S64 与
  源 tile 的 FP32 不匹配 → 断言必然失败。
- **与 dtype 无关**：cuBLAS tail/not_tail 对**所有输入类型都走** `compute_cublas_core`；bf16 只是先在
  `TABS(BF16)`（见问题9，emulator TABS 白名单缺 BF16）崩溃、根本走不到
  这条 store，所以此障碍被前置断言**掩盖**。fp16 越过 TABS 白名单后才第一个撞上它。

**缺陷所在仓（交互问题）**：
- **`SuperScalarModel`（emulator）**：`ValidateLocalTlsu` 对编译器合成的 tile-块搬运（S64）按 dtype
  精确匹配过严——tile 参数的栈往返是**等宽字节 memcpy**，按字节宽度匹配即可，不应要求 block dtype
  逐一等于源 tile dtype。
- **`linx-toolchain-build`（llvm-project / LinxV5 后端）**：传 tile 实参时统一 lower 成 S64 通用块，
  而非与 tile 真实 dtype 一致的块类型；若后端按源 tile dtype 发 store/load block，则天然满足 emulator 校验。

**解除路径**：emulator 侧放宽 `ValidateLocalTlsu` 对通用块搬运的 dtype 匹配（按字节宽度）；或 linx 后端
传 tile 参数时保持 block dtype 与 tile dtype 一致。二者任一即可。

### 影响场景

任何在 kernel 主体外**以独立（非内联）函数**收/发 tile 的调用链都会命中——本 kernel 的 cuBLAS scale
路径 `compute_cublas_scale_{tail,not_tail}` → `compute_cublas_core`。runtime 在 fp16/fp32 输入下（越过
问题1 的 TABS 白名单后）首个障碍即此断言。bf16 输入被问题1 掩盖、未暴露。

### 规避方案

**把整条调用链全内联**，消除跨函数栈传参，S64 块 store/load 随之消失：给 `compute_cublas_core` 与
`compute_cublas_scale_tail`（及对称的 `_not_tail`）加 `__attribute__((always_inline))`。实测 S64 store
计数 **5 → 3 → 0**，`ValidateLocalTlsu` 断言被完全越过。此规避**不破坏正确性**（只改调用约定/内联决策，
不改数值语义），可作为长期规避保留；根治仍需 emulator 或 linx 后端按上「解除路径」修复。

> 越过本断言后，runtime 推进到**问题3**（`compute_cublas_core` 的 3-参 mode-less `TCMP`/`TCMPS` 被
> emulator `ValidateCompareSelectTepl` 要求显式非保留 CMode 而拒绝，`AccumulateBlockInfo.cpp:301`）——
> 即 fp16 cuBLAS 路径的下一个障碍，性质同问题3（linx 缺 4-参 CmpMode）。

---

## 问题9：TABS 作用于 BF16 被 emulator 拒绝（需 emulator 侧解决）

> release_ver0812 未收录的报错。缺陷所在仓 **`SuperScalarModel`（emulator）**，复现入口在本仓
> cuBLAS scale 路径。

### 结论

cuBLAS scale（`dynamic_mx_quant_common.hpp:702` `compute_cublas_scale_tail`）对 **BF16** tile
执行 `TABS(abs_x, x_in)` 时，被 emulator 拒绝：

```
SuperScalarModel/emulator/engine/AccumulateBlockInfo.cpp:393
ASSERT(IsBasicUnaryTeplDataType(...) &&
       "TEPL opcode/data-type tuple is not defined by PTO ISA v0.2")
```

根因：白名单 `IsBasicUnaryTeplDataType`（`AccumulateBlockInfo.cpp:229-230`）把 TABS 限为
`FP16 || FP32`，不含 BF16：

```cpp
case TileOp::TABS:
    return dataType == DataType::FP16 || dataType == DataType::FP32;
```

规范 ASL 中 TABS 的 legality handler `TileOperandsLegal_ExecuteTileUnary`
（`pto-spec: asl/tile/model/legality/operand-schema.asl:20`）**不含任何 dtype 白名单**，仅要求
`TileShapeAndTypeMatch(dst,src)`；BF16 是合法 tile dtype。故 BF16-TABS 为 **spec-legal**，
emulator 白名单过窄；同函数 `TNEG`（:233-236）已允许 BF16，可见白名单本身支持 BF16 表达。

**解除路径（SuperScalarModel）**：`AccumulateBlockInfo.cpp:229` TABS 分支加入 `DataType::BF16`
（对齐 `TNEG:233-236`）。

> **前提说明**：结论以规范 ASL 为权威（`TileOperandsLegal_ExecuteTileUnary` 无 dtype 白名单，
> BF16 为合法 tile dtype）。若最终以硬件实际支持为准、且硬件确不支持 BF16-TABS，则结论回到
> 「合法但不受支持」，仍需在 emulator 或规范侧明确对齐。

### 影响场景

cuBLAS 路径（tail/not_tail）对 **bf16 输入的首个障碍**——bf16 输入在此崩溃、走不到后续，故
问题8（S64 栈传参）在 bf16 下被本断言**前置掩盖**，只有换 fp16 越过本报错后才暴露。

### 规避方案 / 验证过程链

把**输入 dtype 从 bf16 改为 fp16（`__half`）**规避：`TABS(abs_x, x_in)` 随之作用于 **FP16** tile，
落入白名单 `FP16` 分支，不再触发本报错。**这是规避而非修复**——问题9 作为 emulator 白名单缺陷仍在。

- **改动**：新增 driver `test/kernel/quant/dynamic_mx_quant/src/{tail,nontail}_cublas_fp8_fp16.cpp`
  （`InT=__half`）与 Makefile `TYPE={TAIL,NONTAIL}_CUBLAS_FP8_FP16`；kernel/common 逻辑未改。
- **越过本报错后 fp16 cuBLAS 路径的后续链条**（逐一均在 emulator/toolchain 侧，非 kernel）：
  1. → **问题8**（非内联 helper 的 tile 参数 `TSTORE/TLOAD, S64` 栈传参被 `ValidateLocalTlsu` 拒）
     ——给调用链加 `__attribute__((always_inline))` 越过（S64 store 5→3→0）。
  2. → **问题3**（3-参 mode-less `TCMP`/`TCMPS` 被 `ValidateCompareSelectTepl` 要求显式 CMode）
     ——临时把该断言缓和为 warning 探路。
  3. → 最终停在 **README「失败分类」第 2 类 Text-store 被拒绝**（`AssertNotTextStore`，
     `emulator/engine/AaccelssMemoryEngine.cpp:12`）。触发点在 libc 启动例程 `__init_libc`
     （`addtpc`+`sdi.u` 存 text 相对地址），属新鲜 ELF startup 阶段、与 kernel 无关，是已归档的
     toolchain↔emulator 版本 skew。
- **代码状态**：探路用的 emulator 断言缓和、`common.hpp` 的 `always_inline` 均已 `git checkout`
  还原；保留 fp16 driver 与 Makefile TYPE 作复现入口。

---

## 问题10：TROWMAX 作用于 UINT16 被 emulator 拒绝（与 TCOLMAX 不对称，需 emulator 侧解决）

> release_ver0812 未收录的报错。缺陷所在仓 **`SuperScalarModel`（emulator）**，复现入口在本仓
> OCP tail scale 路径。

### 结论

OCP tail scale（`dynamic_mx_quant_common.hpp:639` `compute_ocp_scale_tail_boxed_pw`）对 **UINT16**
指数位 tile 执行 `TROWMAX(max_exp, exp_bits)` 时，被 emulator 拒绝：

```
SuperScalarModel/emulator/engine/AccumulateBlockInfo.cpp:607
ASSERT(IsReduceAndExpandTeplDataType(...) &&
       "reduce/expand TEPL tuple is not defined by PTO ISA v0.58")
```

根因：白名单 `IsReduceAndExpandTeplDataType`（`AccumulateBlockInfo.cpp:525-528`）给 **TROWMAX**
的 dtype 集为 `FP16||FP32||INT32`，不含 U16/BF16；而同函数 **TCOLMAX**（:539-547）含
`INT8/UINT8/INT16/UINT16/INT32/UINT32/BF16`：

```cpp
case TileOp::TROWMAX:  return FP16 || FP32 || INT32;                        // 无 U16/BF16
case TileOp::TCOLMAX:  return FP16||FP32||INT8||UINT8||INT16||UINT16||INT32||UINT32||BF16;
```

但规范 ASL 中 TROWMAX 与 TCOLMAX **共用同一** legality handler
`TileOperandsLegal_ExecuteTileReduction`（`pto-spec: asl/tile/reduce-and-expand/row-reduction/TROWMAX.asl`
与 `.../column-reduction/TCOLMAX.asl` 的 `legality_handler` 字段一致），该 handler
（`operand-schema.asl:70`）不含 dtype 限制（axis 只影响 destination shape 检查）。故 U16-TROWMAX
为 **spec-legal**；emulator 给 TROWMAX 配了比 TCOLMAX 窄、且与共用 handler 不符的 dtype 集。

**旁证**：not-tail 变体用 TCOLMAX（`dynamic_mx_quant_common.hpp:800/832`），落在已放行集合内，故
nontail scale pass 不触发此断言——差异纯在 emulator 的 TROWMAX/TCOLMAX 不对称。

**解除路径（SuperScalarModel）**：`AccumulateBlockInfo.cpp:525` TROWMAX 分支 dtype 集扩到与
TCOLMAX（:539-547）一致（或按共用 handler 语义去掉 dtype 白名单）。前提说明同问题9。

### 影响场景

OCP tail 路径（`TAIL_OCP_FP4`）的 scale pass 首个障碍。

### 规避方案 / 验证过程链

按解除路径**临时**在 emulator `:525` TROWMAX 分支加入 `UINT16||BF16`（与 TCOLMAX 对齐）探路，重编
gfrun 后跑 `TAIL_OCP_FP4`（res_check=on）：

- **越过本报错后直达同一堵墙**：OCP tail 路径**没有**再撞新的 kernel/TEPL 障碍，直接收敛到
  **README「失败分类」第 2 类 Text-store 被拒绝**（`AssertNotTextStore`，
  `AaccelssMemoryEngine.cpp:12`）。faulting 指令与问题9 的 fp16 cuBLAS 路径**完全相同**：libc
  `__init_libc` 的 `sdi.u s1,[t#1,-1616]`（指令 bin `0x618679d9`，存 text 相对地址）——同一 startup skew。
- **比 cuBLAS 路径更干净**：问题9 的 fp16 cuBLAS 越过后还需经问题8（S64 栈传参）、问题3（CMode）两道；
  OCP tail 路径**不经这两道**（OCP 不走 `compute_cublas_core`、无 `TCMP`），越过本报错一道即达 skew 墙。
- **代码状态**：探路用的 emulator TROWMAX 白名单放行已 `git checkout` 还原。

---

## 问题11：`B.IOT ... ->u<>` unknown operand（nontail_ocp_fp4 -O1/-O2 编译失败，需 toolchain 侧解决）

> release_ver0812 未收录的报错。缺陷所在仓 **`linx-toolchain-build`（`llvm-project` LinxV5 后端）**，
> 复现入口在本仓 `NONTAIL_OCP_FP4` 编译。

### 结论

编译 `nontail_ocp_fp4.cpp`（Axis=32/Post=64/BS=32，fp4 输出 tile `[32,32]` RowMajor NoneBox）于
**`-O1`/`-O2`** 时，vendor 头内联汇编报错：

```
tileop-api/jcore/template_asm.hpp:115   (TCVT_T)        "B.IOT %3, mask=15, last, ->%0<%Z4>\n"
tileop-api/jcore/template_asm.hpp:5106  (TCOLEXPANDMUL) "B.IOT %5, %6, mask=15, last, ->%0<%Z7>\n"
→ instantiated:  B.IOT u#1, mask=15, last, ->u<>          // box 为空 <>
error: unknown operand
```

根因：`%Z` 是 LinxV5 后端自定义操作数修饰符，打印 B.IOT 的 TileSize 文本。打印器
`LinxV5AsmPrinter.cpp:176-183` 在 `%Z` 对应操作数**不是立即数**（`!MO.isImm()`）时 `return true`
→ clang 报 "unknown operand"，且提前返回、box 未写入 → 空 `<>`：

```cpp
if (ExtraCode[0]=='Z' ...) {
    if (!MO.isImm()) return true;              // 返回 true = "unknown operand"
    static const char* TileSizes[] = {"0B","128B",...,"8KB"};
    if ((unsigned)MO.getImm() < 8) OS << TileSizes[MO.getImm()];
    return false;
}
```

证据链，指向**优化 pass** 而非 kernel：
1. C++ 层该 fp4 `[32,32]` 输出 tile 的 `TilesizeCode = 4`（=1KB，合法枚举），与 fp8 输出 tile 取值
   相同（static_assert 实测 fp4→4、fp8→4）；同套 TCVT_T/TCOLEXPANDMUL 模板对 fp8 输出、tail-fp4
   输出均编译干净。
2. 最小复现（单独对 `Tile<Vec,__fp4_e2m1x2,32,32,RowMajor>` 做 TCVT）operand 保持立即数、打印
   `<1KB>`、编译干净；仅在整 kernel 上下文失败。
3. 失败随优化等级出现：同一 `nontail_ocp_fp4.cpp` — `-O0`→**0** 处、`-O1`→**6** 处、`-O2`→**10** 处。

→ `-O1/-O2` 的某个 LinxV5 优化 pass 把经 INLINEASM `"i"` 约束传入的 `%Z` 立即数降级为非立即数
（vreg），触发打印器 `!MO.isImm()` 分支。源码合法、仅 -O 变化即触发，是后端优化 pass miscompile
的签名。

**解除路径（linx-toolchain-build）**：保证经 INLINEASM 传入、`"i"` 约束的 `%Z` 操作数在优化后仍以
立即数抵达 AsmPrinter（或相关 pass 对 INLINEASM imm 操作数做保守处理）。

### 影响场景

编译 `nontail_ocp_fp4` 于 `-O1`/`-O2`（`diss` 默认 `-O2`）。

### 规避方案 / 验证过程链

尝试**降 `-O0` 规避 `%Z`** → **规避不成立**：`-O0` 不再触发 `%Z`，但在更早的编译期撞上
**问题12**（`-O0` spill/reload 寄存器类不对称）。故 `nontail_ocp_fp4` 在**任一优化等级都无法编译**
——`-O1/-O2` 撞本问题、`-O0` 撞问题12。二者是 LinxV5 后端两个独立缺陷。

---

## 问题12：`-O0` 溢出/重载寄存器类不对称，`layout_type_to_str` 崩溃（需 toolchain 侧解决）

> 由问题11 的「降 -O0 规避」尝试触发。缺陷所在仓 **`linx-toolchain-build`（`llvm-project` LinxV5
> 后端）**，与具体 kernel 无关（通用）。

### 结论

以 **`-O0`** 编译任一含 `pto::layout_type_to_str`（`two-level-arch/include/common/layout.hpp:59`，
返回字符串字面量的平凡 helper，各 kernel 都链入）的翻译单元时崩溃：

```
llvm_unreachable("Can't load this register from stack slot")
llvm-project/llvm/lib/Target/LinxV5/LinxV5InstrInfo.cpp:670
```

`tail_ocp_fp4`、`tail_cublas_fp8_fp16` 在 `-O0` 下都崩在同一处、faulting 函数均为
`layout_type_to_str`，证明**与本 kernel 无关**。

根因：物化字符串字面量地址的 `PseudoADDTPC_HI` 产出寄存器类 **`mixedgprnora`**（MIR，
`-print-before=regallocfast`，`layout_type_to_str` 的 `sw.bb`）：

```
%8:mixedgprnora = PseudoADDTPC_HI <mcsymbol>, target-flags(linx-tpcrel-hi) @.str
%9:mixedgpr    = ADDI killed %8, target-flags(linx-tpcrel-lo) ...
SDI killed %9, %stack.0.retval, 0
```

store 与 load 处理**不对称**：
- `storeRegToStackSlot`（`LinxV5InstrInfo.cpp:607-648`）**无**寄存器类白名单——非 `Tile_ABS` 一律
  `SDI` 无条件溢出，**接受** `mixedgprnora`。
- `loadRegFromStackSlot`（`:650-670`）有白名单 `{GR, LTR, LUR, Tile_ABS, SIMTCGV}`，
  `hasSubClassEq(mixedgprnora)==false` → 落到 `:670` `llvm_unreachable`。

`-O0` 用 Fast RegAlloc，激进溢出/重载短活跃期虚寄存器，故命中 load 路径；`-O2` 的 Greedy 把
`mixedgprnora` 保留在寄存器、不经栈往返，故不触发（但 `-O2` 会命中问题11）。

**解除路径（linx-toolchain-build）**：`loadRegFromStackSlot:665` 白名单加入 `mixedgprnora`
（或 `PseudoADDTPC_HI` 结果对应的正确寄存器类），与 `storeRegToStackSlot` 对齐。

### 影响场景

任一 kernel 降 `-O0` 编译（本 kernel 中由问题11 的规避尝试触发）。

### 规避方案

**无有效规避**——`-O0` 撞本问题、`-O1/-O2` 撞问题11，`nontail_ocp_fp4` 任一优化等级均无法编译。
需按解除路径在 toolchain 侧修复（问题11、问题12 各修一处）。

---

## 问题13：TCVT 不发 lb2，valid-col-1 的 TCVT 输出直接 TSTORE 被 emulator 拒绝（需 toolchain 侧解决）

> release_ver0812 未收录的报错。缺陷所在仓 **`linx-toolchain-build`（Linx-TileOP-API `-D__linx`
> intrinsic header `template_asm.hpp`）**，复现入口在本仓 probe 探针
> `probe_dynamic_mx_quant_tail_ocp_fp8.hpp` 的 OCP e8m0/bf16 直转 scale 落盘。

### 结论

一个 **boxed valid-col-1**（物理 `Cols>1`、`validCol=1`）的 tile，若其**最后一个 producer 是
`TCVT`** 并随即 `TSTORE`，gfrun 运行期挂：

```
SuperScalarModel/emulator/engine/AccumulateBlockInfo.cpp:60
ValidateLocalTlsu "Local TSTORE requires one compatible source Tile"  (EXIT=1)
```

probe 中命中点（full + tail pass 各一对，共 4 处）：
- `TSTORE(gw, max_bf)`，`max_bf = TCVT(max_h)`（half→bf16），boxed `[TileM,1] / 物理 Cols=32`。
- `TSTORE(gs, scale_e8m0)`，`scale_e8m0 = TCVT(shared_bf)`（bf16→e8m0），boxed `[TileM,1] / 物理 Cols=32`。

### 根因（工具链 header 侧不一致）

tile 形状经内联汇编 `B.DIM ..,->lb0/lb1/lb2` 传给 emulator：`lb0=validCol`、`lb1=validRow`、
**`lb2=物理列宽 tile::Cols`**。`template_asm.hpp` 里几乎所有产 tile 的 TEPL elementwise op
（`TABS`:3401 / `TEXP`:3461 / `TRECIP`:3501 / `TADDS`:3646 / `TMULS` / `TANDS` / `TMAX` /
`TROWMAX`:4586 …）都发 `B.DIM zero,%c4,->lb2`；**唯独逐元素的 `TCVT_T`（TEPL 27，:109）与
`TMOV` 不发 lb2**（TMOV 是 layout move、dst 形状另有来源，可理解；TCVT dst 与 src 逐元素同形、
无理由区别对待）。

emulator 对「缺 lb2」兜底（`isa/Block.cpp:1074`）：

```cpp
physicalCol = (lb2 <= 1) ? validCol : lb2;   // TCVT lb2=0 → 塌成 validCol
```

于是 TCVT 产出的 boxed tile 被记 `col = validCol = 1`（本应是物理的 32）。而 `TSTORE`（:1806）
固定声明 `lb2 = tile::Cols`（=32）。校验 `IsCompatibleDataTile`（:265）要求
`源tile.col == store.physicalCol` → `1 != 32` → 断言。

**关键澄清**：并非「physical≠valid 就非法」。部分列 store（valid=1, physical=32）**是合法的**
（`kernels/reduction/reducemax_rowvec_pto.hpp` 正常这么干），前提是 store 的 tile 由**发 lb2 的
op** 产出、`col` 记成物理宽即可。症结**只**在 `TCVT` 塌了 `col`。数据满宽 store
（`TSTORE(gy, oq)`，`oq=TCVT(...)` 但 `validCol==Cols==32`）**不中招**——塌陷后 `col` 仍等于
`Cols`。经 `TANDS`/`TLOAD` 中转的 store 也不中招（那些 op 发 lb2）。

**定性**：疑似工具链 header 缺陷（`TCVT_T` 与同类 TEPL elementwise op 在 lb2 上不一致），
非 LLVM codegen、非 emulator（兜底合理）。100% 坐实还需核对 PTO ISA 规范 TEPL-27(CVT) 编码
`lb2` 是否为产 tile 指令的必填/合法字段。详见 `ISSUE_tcvt_no_lb2.md`。

**解除路径（linx-toolchain-build）**：`template_asm.hpp` 的 `TCVT_T`（:109）汇编体补一行
`"B.DIM zero, %cN, ->lb2\n"`（`N` 绑定 `tile_shape_out::Cols`），与 `TABS` 等对齐。

### 影响场景

任何「boxed valid-col-1 tile 由 TCVT 直接产出并落盘」的路径。OCP scale 直转
（bf16→e8m0）尤其命中——e8m0 只能由 `TCVT` 产出（无 e8m0 域算术 op 可作替代 producer）。

### 当前提交状态：复现版（未应用任何规避）

**本仓提交的 `probe_dynamic_mx_quant_tail_ocp_fp8.hpp` 是未加盖章的复现版**——按
`ISSUE_tcvt_no_lb2.md` 的复现命令编译并 `gfrun` 执行，会**直接在 `TSTORE(gw, max_bf)` 处挂
M47 `ValidateLocalTlsu` 断言**。这样保证按 ISSUE 可原样复现缺陷。下面两条解法均已实测验证，但
**均未写入提交代码**（避免掩盖问题）。

### 解法 A（kernel 侧盖章，gfrun 实测越过 M47）

「让喂 store 的 tile 最后由发 lb2 的 op 重新盖章 `col=Cols`」，两处均经 gfrun 验证越过：
- **`max_bf`（bf16）**：TCVT 后插一元恒等发-lb2 op 盖章。**不能用 `TABS`**（问题9：emulator 拒
  BF16 TABS）；改用 `TMULS(max_bf, max_bf, __builtin_bit_cast(__bf16,(uint16_t)0x3f80))`（×1.0，
  发 lb2、bf16 合法；立即数经 `__builtin_bit_cast` bits 规避问题11 后端崩溃）。
- **`scale_e8m0`（e8m0）**：e8m0 无 elementwise 算术，但**二元 `TMAX(t,t,t)` 恒等盖章发 lb2 且被
  emulator 接受**（`TMAX(scale_e8m0, scale_e8m0, scale_e8m0)`）。此前判断的「e8m0 无 producer 死结」
  被 gfrun 推翻——TMAX 可作 in-place 重盖 op。

### 解法 B（toolchain 侧根本修复，汇编实测越过 M47）

`template_asm.hpp` 的 `TCVT_T`（:109）汇编体在 `->lb1` 后补一行 `"B.DIM zero, %c7, ->lb2\n"`，
并追加输入操作数 `"i"(tile_shape_out::Cols)`（第 7 个输入，`%c7`），与 `TABS`(:3401) 等对齐。
**已实测**：改后重编 probe（**无需 kernel 盖章**），反汇编确认每个 `BSTART.TEPL TCVT` 块后紧跟
`C.B.DIMI 32, ->lb2`；`gfrun` **越过原 M47 断言**。零回归：满宽 tile 旧兜底 `col=validCol==Cols`
与新显式 `lb2=Cols` 一致；boxed valid-col-1 旧 `col=1`（错）→ 新 `col=Cols`（对）。缺陷根仓在
Linx-TileOP-API 组件源（`src/Linx-TileOP-API/include/jcore/template_asm.hpp`），改 build artifact
会被 `make build-tileopapi` 覆盖，故根本修复须落组件源——**本次未改，保留复现**。

### 两解法共同的收敛点

A、B 任一应用后，probe 全部 tile 指令链跑通，均收敛到与问题9/10 相同的 **startup skew 墙**——
libc `__init_libc` 的 text 相对 store `sdi.u s1,t#1,-12xx`（`AssertNotTextStore`，
`AaccelssMemoryEngine.cpp:12`），非 kernel 缺陷，是三仓版本 skew 的已知阻塞。

## 问题14：零指令 `reinterpret_tile` 的位重解释运行期不可见，emulator dtype 相等断言误杀（需 emulator 侧解决）

> 详见 `ISSUE_reinterpret_dtype_tag.md`。缺陷所在仓 **`SuperScalarModel`（emulator 建模层
> `AccumulateBlockInfo.cpp:440` `ValidateScalarLogicalTepl`）**，复现入口在 probe 的 OCP 清尾数链
> `reinterpret_tile<uint16_t>(max_bf) → TANDS → 读回 bf16`。**这是问题4「无寄存器 bitcast」被 v0.58
> `reinterpret_tile` 解除后，暴露出的下一道墙**（HBM 往返时被 TLOAD 重打标签掩盖，零指令 bitcast 才显形）。

### 结论

v0.58 `reinterpret_tile<uint16_t>(max_bf)` 把 bf16 tile **零指令**重解释成 u16 喂 `TANDS`（清尾数
留指数位），随后仍按 bf16 读回 `TMULS`。编译+反汇编正确，gfrun 挂：

```
gfrun: ASSERTION FAILED: ... source->tileInfo->dataType == block->dataType ...
  "scalar logical TEPL source dtype/shape/stride is incompatible"
  func ValidateScalarLogicalTepl, AccumulateBlockInfo.cpp:440   (EXIT=1)
faulting = BSTART.TEPL TANDS UINT16 (B8, M38, TPC 0x1134e)
```

### 根因（编译期/运行期不一致）

- **反汇编证明 kernel 数据流正确**：TCVT(BF16)→TANDS(U16)→TMULS(BF16) 三块在**同一个 512B 的 `t`
  寄存器**就地读写（中间无 copy）。`-D__linx` intrinsic 经 `"=Tr"(dst.data())`/`"Tr"(src.data())`
  绑定操作数，reinterpret view 的 `.data()` 转发到源 tile 同一 `data_`（`pto_tile.hpp:1352`），掩码
  就地写进 max_bf 寄存器、被 TMULS 读到。「对象身份 SSA」抽象模型在 linx 后端不适用。
- **缺陷在 emulator per-tile 运行期 dtype 标签**：`tileInfo->dataType` 由**产出该 tile 的指令**
  设置（TCVT→BF16），reinterpret 零指令不改它。断言 440-445 要求源 tile 运行期标签 `== block->dataType`
  （U16，reinterpret 编译期设的）→ `BF16 == U16` 失败。位重解释信息只活在编译期指令 datatype 字段，
  emulator 物理 tile 运行期标签收不到。

### 关键澄清

- **424「逻辑op datatype 必须整数」是对的、必须保留**：这正是须 reinterpret 到 u16 的原因——把
  datatype 合法化成整数、同时保 bit。故**不能直接 `TANDS(max_bf, max_bf, mask)` 跳过 reinterpret**
  （会先挂 424；且 bf16 标量立即数数值转换掉掩码 bit 并崩后端，见问题11）。
- **HBM 往返「能过」纯属副作用**：`TSTORE(bf16)→TLOAD(u16)` 的 TLOAD 是真指令，会重打标签成 U16。
  零指令 reinterpret 只干位重解释，才暴露此断言没建模 bitcast。

### 定性

emulator 建模缺陷（过严校验），非 ISA/工具链/kernel。reinterpret 是 v0.58 合法特性、工具链发的 bit
正确、真实硬件按 datatype 字段当场解释 bit 即可跑。断言把「本op如何解释 bit」与「上一条产出此 tile 的
dtype」强行划等号，禁掉一切「零指令 bitcast 后被异类型op消费」。

### 解除路径（emulator 侧，本次采用）

`AccumulateBlockInfo.cpp:441` 的 `source->tileInfo->dataType == block->dataType` 放松为**位宽相等**
`BytesOf(source->tileInfo->dataType) == BytesOf(block->dataType)`，保留 424 整数约束 +
全部 shape/stride/valid 校验。同族相等断言（270/410/475）**probe 阶段暂不动**——probe 只撞 441。

> **补记（2026-08-20，cuBLAS plain 路径 gfrun 落地）**：`nontail_cublas_fp8_plain` 把
> `reinterpret_tile<uint32_t>` 视图直接喂 compare/select 后，撞上了 441 的**同族兄弟**——
> `IsCompatibleDataTile`（`AccumulateBlockInfo.cpp:~302`）里 `source->tileInfo->dataType == dataType`。
> 症结同 441：reinterpret 视图运行期仍带 producer（fp32）的 dtype 标签，与 compare/select block 的
> u32 不等。按同一思路放松为 `BytesOf(...) == BytesOf(...)`（等宽即容），commit `c022a929`。至此
> 问题14 的「解除路径」由 probe 的 441 单点，扩到 cuBLAS plain 路径的 302 兄弟；410/475 仍未撞、不动。

---

## 问题15：emulator 未实现 TCVT bf16→e8m0(SF8)，MX 共享 scale 转换缺失（需 emulator 侧解决）

### 现象

放松 441 + 补发 TCVT lb2 后，probe 越过 TANDS/TMULS，撞到 OCP scale 产出的
`TCVT bf16 → e8m0`（`B.DATR e8m0/SF8`）：

```
CubeEngine.cpp:374  DataFormatCvt lambda OpCvtType
assert(0 && "Not support such type convert yet")   // SF8 在不支持列表
```

### 根因

`OpCvtType`（CubeEngine.cpp）把 `DataType::SF8` 归到「尚未支持」assert 分支，根本没走到
`ConvertAggre`；且 `FloatPointUtils.cpp` 的 funcMap 也无 `{BF16, SF8}` 条目。即 emulator
功能模型从未实现「浮点→e8m0」这条 MX 共享 scale 转换。其**逆向**已存在：
`CubeCalculate.cpp:416-419 EleMulScale` 用 `value = 2^(E-127)` 反解 e8m0（bias 127）。

### 解除路径（emulator 侧，本次采用）

e8m0 与 bf16 同为 8-bit 指数、同 bias 127，故 bf16→e8m0 = 丢符号、把 7 位尾数 RNE 舍入进指数：

1. `CubeEngine.cpp` `OpCvtType`：`DataType::SF8 → OPConvertType::OPCVT_SF8`（移出 assert 分支）。
2. `FloatPointUtils.cpp` `InitConvertMapFp`：新增 `{OPCVT_BF16, OPCVT_SF8}` lambda——
   `exp=(bits>>7)&0xFF; mant=bits&0x7F; if(exp==0xFF)→0xFF(NaN); RNE(mant>0x40||(==0x40&&exp&1))→exp++;
   clamp[0,0xFE]`。对 OCP 的 max_exp 输入（尾数已被 TANDS 清零、恒为 2 的幂）**精确等于指数抽取**。

### 结果

probe **gfrun 跑到底**：23 blocks / 120 insts，`R2 = 0`（Success to Reach End of Benchmark）。
完整 OCP 链 `TCVT→TANDS(u16)→TMULS→TCVT bf16→e8m0→TSTORE(e8m0)` + data 路径
（TRECIP/TROWEXPANDMUL/TCVT→e4m3/TSTORE）全部执行无断言。

> probe 为**全零输入的执行 smoke test**（无 golden 对比）：x=0 → scale E=0、y=0，确定性无崩。
> 数值正确性需另接 golden harness（gen_dynamic_mx_quant_data.py），属独立更大任务。

### 三处修复汇总（打通 probe 执行）

| # | 文件 | 改动 |
|---|---|---|
| 1 | `SuperScalarModel/emulator/engine/AccumulateBlockInfo.cpp:441` | dtype 相等 → `BytesOf` 位宽相等（支持零指令 reinterpret） |
| 2 | `Linx-TileOP-API/include/jcore/template_asm.hpp` TCVT_T | 补发 `B.DIM zero, %c7, ->lb2`（+`tile_shape_out::Cols` 操作数） |
| 3 | `SuperScalarModel` `CubeEngine.cpp` + `FloatPointUtils.cpp` | 实现 TCVT bf16→e8m0(SF8) |

---

## 问题16：TCVT 形状契约（TileLogicalShapeMatch）对打包 fp4 与源无法同时满足 → 结构性必崩；当前工具链落在**编译期 static_assert**（需工具链头 + emulator 双侧解决）

> 触发 kernel `dynamic_mx_quant_tail_ocp_fp4.hpp:206`（`TCVT(oq, xf)`，`TYPE=TAIL_OCP_FP4`）。
> 同一契约（TCVT 的 src/dst 必须 physical Rows 与 Cols 全等）落在**两层**，缺陷在**工具链头
> （`Linx-TileOP-API/jcore/template_asm.hpp`）+ emulator（`SuperScalarModel/isa/Block.cpp`）双侧**。
> 专文见 `ISSUE_tcvt_fp4_shape_contract.md`。

### 编译期 static_assert（当前工具链主障碍）

**当前工具链（clang15.0.4，自报 PTO 0.58.1）的 `jcore/template_asm.hpp:115` `TCVT_T` 有硬
`static_assert`**：`tile_shape_out::Rows==tile_shape_in::Rows && Cols==Cols`（msg:"TCVT source
and destination must have identical physical Rows/Cols (PTO 0.58.1 TileLogicalShapeMatch)"）。其
**自身注释**（template_asm.hpp:110-114）写明这是把运行期的 `TileOperandsLegal_TCVT` /
`TileLogicalShapeMatch` 契约「提前到编译期拒绝」（"reject it at compile time"）。故 fp4 data 路径
`TCVT(oq[8,32], xf[8,64])` **编译期即崩、连 `.o` 都出不来**：

```
template_asm.hpp:115:3: error: static assertion failed ...
  'Tile<...__fp4_e2m1x2, 8, 32...>::Cols == Tile<...float, 8, 64...>::Cols':
  TCVT source and destination must have identical physical Rows/Cols (PTO 0.58.1 TileLogicalShapeMatch)
```

**编译期崩的维度是 `Cols`（32≠64）**——同契约在两层各读不同来源的 physical col：

| 层 | physical col 从哪读 | 崩的维度 |
|---|---|---|
| **编译期** `static_assert`（`template_asm.hpp:115`，本障碍） | **声明的模板** `Cols_`（fp4 tile_o=PW/2=32、fp32 tile_f=PW=64） | **Cols 32≠64** |
| **运行期** `ValidateOperandContract`（emulator `Block.cpp:1039`，放宽编译期后） | emit 的 lb2 / inherit（不读模板声明） | Row 8≠4 / Col 64≠32（随 lb2） |

**Rows/Cols 数值来源坐实**：`static_assert` 比的是 `Tile` 模板第 3、4 参 `Rows_`/`Cols_`
（`pto_tile.hpp:604-611` 模板头、`:638-639` `Rows=Rows_; Cols=Cols_`）。kernel（`:96-103`，M=8/N=64/BS=32→
TileM=8/PW=64）声明 `tile_f=Tile<...,TileM,PW,...>`（Rows=8,Cols=64）、`tile_o=Tile<...,TileM,PW/2,...>`
（Rows=8,Cols=32）→ Rows 8==8 ✓、**Cols 32≠64 ✗**。

**复现最小化（编译期崩的直接推论）**：既是编译期崩，复现**只需工具链编译**——
- **不需要 `SuperScalarModel`（gfrun/gfsim）**：崩在 clang++ 编译阶段，到不了链接/执行。
- **不需要携带任何本地改动**：崩点 `TCVT(oq, xf)` 在 data 路径、HEAD 即存在（本轮改动只在 scale/finalize 区）。
```bash
make TESTCASE=dynamic_mx_quant TYPE=TAIL_OCP_FP4 res_check=on   # 编译即失败
```

### 最小 TCVT 探针闭环实证（2026-08-21）

`test/kernel/quant/dynamic_mx_quant/src/fp4_shape_probe.cpp`（`TYPE=FP4_SHAPE_PROBE`，`WIDEN=on`→
`-DWIDEN`）：单条 `TLOAD(fp32)→TCVT→TSTORE`，`R=8,PW=64`。
- **变体 A（`OCOL=PW/2=32`，打包正确）**：编译**唯一实质错误**即 `template_asm.hpp:115` static_assert
  （`grep -c error: ==1`），坐实「不一致→编译崩 Cols」。
- **变体 B（`WIDEN`，`OCOL=64`，加宽骗过断言）**：编译过、gfrun `R2=0`，但 `output.bin` 数据错——
  反汇编 TCVT 发 `->lb2 64`+`B.DATR e2m1x2,byte0`→非打包；喂交替 1.0/4.0（fp4 nibble 0x2/0x6）得每行
  64B（前 32B `02 06…` 低 nibble 单装 + 后 32B 补零、共 512B），对 golden 每行 32B 打包（`0x62`×32、共
  256B）三重不符（字节翻倍/未打包/每行只落半数源值）。坐实「加宽一致→数据错」。

> **运行期同契约（放宽编译期后的第二道）**：instrumented gfrun 实测崩点是 **`row==row` conjunct
> （src.row=8 ≠ dst.row=4）**（不发 lb2→inherit col=64→row=4），非 `col==col`——见「根因」「排除法」。

### 现象

`tail_ocp_fp4` 编译+反汇编通过，gfrun 在**块解码组装阶段**（任何 Execute 之前）崩：

```
gfrun: illegal instruction: ASSERTION FAILED:
  srcTile[0]->tileInfo->row == dstTile[0]->tileInfo->row && ...
  "PTO 0.58 TCVT requires matching source/destination logical shapes"
  func ValidateOperandContract, file isa/Block.cpp:1039   (EXIT=1)
```

调用点：`SetBlockIsComplete()`（Block.cpp:1136）在**块解码完成时**调用 `ValidateOperandContract()`。
TCVT 分支逐 conjunct 断言 `src.row==dst.row && src.col==dst.col && src.validRow==dst.validRow &&
src.validCol==dst.validCol && src.layout==dst.layout`，其中 `row`/`col` 是 **physical**（含 padding/打包）。

### 根因（打包类型的结构性半宽，instrumented gfrun 实证）

instrumented gfrun 打印的实际 tileInfo（当前 fresh build，fp4 TCVT **不发 lb2**）：

| conjunct | src（上游 fp32 tile） | dst（fp4） | 结果 |
|---|---|---|---|
| physical col | 64 | 64（inherit） | ✓ 相等 |
| validCol | 32 | 32 | ✓ 相等 |
| validRow | 8 | 8 | ✓ 相等 |
| **physical row** | **8** | **4** | **✗ 崩** |

- `__fp4_e2m1x2` 是**打包类型**（`BytesOf(FP4)=HF4_DATA_WIDTH=1`，`DataType.h:203`；1 字节 = 2 个 fp4）。
- physical `row` 由 `row = size / (col × BytesOf(elem))`（Block.cpp:1380-1386）算：
  - src（fp32，elemBytes=4）：`2048 / (64 × 4) = 8`。
  - dst（fp4，BytesOf=1）：`256 / (64 × 1) = 4`。dst 继承了 src 的 col=64，但 fp4 每元素只占 1 字节 →
    同 size 下 row 减半 → **8 ≠ 4** → 断言失败。
- **col 单位（源码实证）**：emulator 的 physical `col` 对打包 fp4 = **fp4x2 打包单元（字节）数**，不是单个
  fp4 元素数。依据：(1) `BytesOf(FP4)=1`；(2) `col=physicalCol` 直接取自 lb2 或 inherit；(3) `row=size/(col×elemBytes)`
  自洽要求 col 按打包单元计。故 kernel 的 `tile_o Cols` 应保持 `PW/2=32`，**不能改成 64**。
- **两种 emit 变体都崩，只是崩不同 conjunct**——宽→打包-fp4 的 TCVT **无法同时**满足 col==col 和 row==row：
  - **不发 lb2**（当前 build）：`UpdateDstTileInfo` inherit 分支生效 → dst.col=64 → **row 崩**（8≠4）。
  - **发 lb2=32**（旧 build 另一变体）：`explicitPhysicalCol=true` → dst.col=lb2=32 → **col 崩**（64≠32）。
  选哪个 col 都有一个 physical 维度 != src。这是打包类型的**结构性**特征，与 BlockSize、具体数值、哪个 fp4
  kernel 无关。**validRow/validCol 两边恒相等**（8/8、32/32），只比 valid 维度即可放过。

### 排除法：`TCVT lb2`（问题15 修复 #2）不是元凶，反而必需

曾怀疑「问题13/15 给 `TCVT_T` 补发 lb2（`template_asm.hpp:118` `B.DIM zero,%c7,->lb2` +
`tile_shape_out::Cols` 操作数）」是否导致 fp4 崩。实证否定：

- **当前 build fp4 TCVT 已不发 lb2**（反汇编：块内只有 `B.DIM a6->lb0` / `B.DIM a7->lb1`）。无-lb2 →
  `UpdateDstTileInfo` 的 inherit 分支（`inheritTcvtShape && !explicitPhysicalCol`，Block.cpp:1260-1262）
  **确实生效**，`physicalCol = shapeSource->col = 64`，dst.col 继承 src.col。**这一点推翻了上一版「inherit
  未生效、回退 validCol=32」的推断**——inherit 生效了，只是继承来的 col=64 让 row 崩。
- 若发 lb2=32（正常 TCVT 如 bf16→fp32 需要它把 dst.col 取对），fp4 会走 `explicitPhysicalCol` → dst.col=32
  → 改崩 col==col。故问题15 的 lb2 补发是让正常 TCVT dst.col 取对的**必需件**；**fp4 崩纯属 emulator 对打包
  类型逐 physical 维度比对的缺陷，与 lb2 无关**（发不发 lb2 都崩，只换崩 row 还是 col）。

### 定性

emulator 建模缺陷（过严校验），非 ISA/工具链/kernel。TCVT 到打包类型时，dst 的 physical col/row 与 src 天然
不同（打包改变了 col↔row↔size 关系），这是**正确**的描述符；断言把 physical row/col 都强行要求相等，等于
禁掉一切「宽类型 → 打包窄类型」的 TCVT。

### 解除路径（工具链头 + emulator 双侧）

同一契约现落在两处，须**双侧**放宽（只保留 `validRow==validRow` + `validCol==validCol` + `layout`
+ physical≥valid 健全性检查；fp4 两边 valid 维度恒相等 8/8、32/32，放宽后过校验且不影响正确性）：

1. **工具链头（编译期崩点）**：`Linx-TileOP-API/jcore/template_asm.hpp:115-118` `TCVT_T`
   的 `static_assert` —— 删 `Rows==Rows && Cols==Cols` physical 强等，保留 valid 包含关系（:119-124 那条）。
   **这是让 fp4 data 路径能编译的必要条件。**
2. **emulator（运行期崩点）**：`SuperScalarModel` `ValidateOperandContract()`
   （`Block.cpp:1042-1044`）—— 删 physical `row==row`/`col==col` 两条 conjunct，保留 valid + layout。
   **这是让编译出的 elf 能跑到底的必要条件。**

- 对症提交：**`eaa3dfe7` "fix(tile): relax TCVT legality to valid shape only"**（作者 jialewang，2026-08-14），
  正是删掉 `row==row`/`col==col` 只留 valid 维度 + 在 `UpdateDstTileInfo` 给 TCVT 加 dstPhysicalCol 继承/
  dense-pack 逻辑——**仍是对症的正确修复**（它删的是 row 和 col 两个 physical 检查，同时覆盖上面两种崩法）。
- **该提交游离于当前分支之外**：它曾是 `origin/feat/pto-v058-adaptation` 的 tip，后远程被 force-push 重写
  到 `b3227fe5`（不含此提交），现仅存在于本地分支 `feat/pto-v058-adaptation` 的 tip，**不可作依据**。当前
  工作区 `local_test`（tip `0e213a2c`）的 Block.cpp 仍是严格版。已并入 origin/main 与 origin/feat 的 TCVT
  提交（PR#175 组：`d5a088f5`/`4a8c4f86`/`5a4d2774`）**没动 row==row/col==col**。
- **落地方式（二选一）**：(A) 把 `eaa3dfe7` 的 Block.cpp 放宽 patch 摘到 `local_test`（含问题15 的 e8m0
  修复）上，重编 gfrun → 直接实测 fp4 精度；(B) fp4 data 路径维持 compile-only，等 emulator 放宽 PR 并入。
  **修复方向明确在 emulator，kernel 无需改**（tile_o Cols 保持 PW/2=32 是对的）。

### 反证：kernel 侧「加宽 dst tile 骗过断言」产出错误数据（2026-08-19 实测）

为确认「不改 emulator、只在 kernel 侧规避」是否可行，做过对照实验：把 fp4 输出 tile 的 physical 列
`PW/2 → PW`、valid 列 `BlockSize/2 → BlockSize`（声明成未打包宽）。此时 dst size=`8×64×1=512B`、继承
col=64 → `row=512/64=8==src.row` → **断言过、gfrun 跑到底 R2=0**。但**输出数据错**（M=8,N=64,BS=32）：
`scale_output` 逐字节对，**`output` vs golden 不同**——golden 每字节
打包 2 个 fp4（byte0=`0x26`=nibble 2,6），加宽输出把每 fp4 摊进一整字节高半字节恒 0（byte0=`0x06`
byte1=`0x02`），32 个 fp4 被解包成 32 字节、宽度翻倍越界。**证实：加宽 tile 只是让 physical row 恰好
等于 src 而绕过断言，代价是 fp4 打包布局被破坏；kernel 侧无损规避不可行，修复只能在 emulator。** 亦见
`ISSUE_tcvt_fp4_shape_contract.md`「反证」子节。

### 与问题2 / `ISSUE_32B_align` 的分层区别

问题2 是 tile **声明**的 32B 列对齐 `static_assert`（`pto_tile.hpp:408`）——用 padded-physical col-box
方案已规避。本问题是 **TCVT 的 src/dst 形状匹配契约**（TileLogicalShapeMatch），与列对齐无关，且现分两层：
- 编译期：`template_asm.hpp:115` `TCVT_T` 的 `static_assert`（崩 Cols 32≠64）。
- 运行期：`Block.cpp:1039` 的解码校验（崩 Row 8≠4 或 Col 64≠32，放宽编译期后暴露）。

即 fp4 现有三道独立关卡：问题2（列对齐声明，已规避）→ 本问题编译期（TCVT_T 形状匹配）→
本问题运行期（Block.cpp 形状匹配）。前者与后两者无关；后两者同契约、不同层。

---

## 问题17：TCMPS 作用于 UINT32 被 emulator compare/select 白名单拒绝（需 emulator 侧解决）

> 缺陷所在仓 **`SuperScalarModel`（emulator）**，复现入口 = `nontail_cublas_fp8_plain` 就地展开后、
> 在 `reinterpret_tile<uint32_t>` 视图上做 `TCMPS<CmpMode::{LT,NE,GT,EQ}>` 抽 fp32 指数/尾数位。
> **这是问题3（原生 CmpMode）+ 问题4（reinterpret_tile）落地到 cuBLAS plain 路径后暴露的下一道 emulator 墙。**

### 结论

`compute_cublas_core` 的 IDEAL 版本在 **u32 域**用 `TCMPS<CmpMode::LT>(finite, raw, FP32_EXP_MASK)`
等直接比较 fp32 位型（`raw = reinterpret_tile<uint32_t>(max_f)`）。gfrun 挂在 compare/select 白名单：

```
SuperScalarModel/emulator/engine/AccumulateBlockInfo.cpp  IsCompareSelectTeplDataType (~:291)
ASSERT(IsCompareSelectTeplDataType(...) && "...")
```

根因：`IsCompareSelectTeplDataType` 的 **TCMPS** 分支 dtype 集为
`INT32||FP32||FP16||UINT16||INT16`，**不含 UINT32**：

```cpp
if (op == TileOp::TCMPS) {
    return dataType == DataType::INT32 || dataType == DataType::FP32 ||
           dataType == DataType::FP16 || dataType == DataType::UINT16 ||
           dataType == DataType::INT16;   // 缺 UINT32
}
```

而 kernel 在 fp32 位型上做整型比较（抽 exp/mantissa）天然要 UINT32 域；INT32 已在集内，UINT32
是等宽同类，spec-legal。

**解除路径（SuperScalarModel）**：TCMPS 分支加入 `DataType::UINT32`。已实测放行后 gfrun 推进到问题18。

### 影响场景

任何在 u32 位型域做 TCMPS 的路径——cuBLAS scale 的原生 CmpMode 版本（`nontail_cublas_fp8_plain`）首当其冲。
scratch-HBM 版（其余 5 个 kernel）走真实 uint32 tile 也同样需要，只是此前被更前置的断言掩盖。
**`nontail_cublas_fp8_bigbs` 端到端 gfrun 亦依赖此修复**（其就地展开的 `compute_cublas_core` 同在
`reinterpret_tile<uint32_t>` 视图上发 `TCMPS<CmpMode>`）：2026-08-20 实测 **env_test 的 gfrun（未打
`50afe316`，`AccumulateBlockInfo.cpp` TCMPS 白名单确无 UINT32）复现本断言崩溃；工作目录的 gfrun（已打
`50afe316`）放行、跑到底 `R2=0`**——两个 gfrun 版本对 UINT32 compare/select 的支持点即差在此 commit。

### 已落地

emulator commit `50afe316`「fix(emulator): accept UINT32 for TCMPS compare/select TEPL」。
**验证升级（2026-08-20）**：`nontail_cublas_fp8_bigbs` 走独立 harness（`TYPE=NONTAIL_CUBLAS_FP8_BIGBS`，
Axis=128/Post=32/BS=128→`R_sub=32/TileN=32` 自动路由 bigbs）+ BS 参数化 golden（`--block-size 128`），
env_test linx 编译、工作目录 gfrun 执行到底：**data 逐字节匹配 golden（4096B 全对）、scale 值逐字节匹配**
（32 个真实 E8M0 全对，仅 parity 交织布局差=问题5）。cublas-bigbs 由此从「逐 op 对齐」升级为
「gfrun 端到端逐字节验证」，成为继 plain 之后第二个带 golden 实测数值正确的业务路径。
「仅布局差、数值/列序均正确」这一判定**不靠近常量下的值集相同**，而由 data 逐字节匹配（布局无关）+
单调判别实验（planar 字节精确、无列错位）坐实——详见**问题5「验证」小节**。

---

## 问题18：linx 就地 TSEL（false-source 融进 dst）被 emulator 双侧拒绝（需 emulator 侧解决）

> 缺陷所在仓 **`SuperScalarModel`（emulator）**，validate 侧 `AccumulateBlockInfo.cpp` +
> execution 侧 `TEPLEngine.cpp`。复现入口 = `nontail_cublas_fp8_plain` 展开出的 `TSEL`（roundup 选择、
> extractExp/recip 的 TSEL 链）。

### 结论

LinxV5 后端把 `TSEL(dst, mask, trueSrc)` lower 成**单条带两个 tile 源**的 B.IOT
（`mask=srcTile[0]`、`trueSrc=srcTile[1]`），**false-source 即 dst 自身、就地写**
（`dst = mask ? trueSrc : dst_prior`，与 TSELS「dst = mask ? src : scalar」同构，只是标量换成 dst 活字节）。
emulator 两处都按「TSEL 必有 3 个 tile 源、无 dst」的旧契约建模，双侧崩：

- **validate 侧**（`ValidateCompareSelectTepl`，`priorSources==0` 分支，~:388）：
  `ASSERT(inst->srcs.size() == 3 && inst->dsts.empty() && ...)` —— 就地 TSEL 只有 2 源且带 1 个 dst，断言失败。
- **execution 侧**（`ExecuteTSEL`，`TEPLEngine.cpp` ~:1118）：无条件 `LoadFromTileRegisterSrc(..., block->srcTile[2], ...)`
  —— 就地 TSEL 无 `srcTile[2]`，越界。

### 解除路径（SuperScalarModel，本次采用）

- validate 侧放宽为「3 源显式-false」**或**「2 源+1 dst 就地」皆合法：
  `srcs.size()==3 && dsts.size()<=1 && (dsts.empty() || dsts[0]->size >= dataBytes)`。
- execution 侧 false-source 回退：`srcTile.size()>=3 ? srcTile[2] : dstTile[0]`（读 dst 活字节作 false 源）。

两侧须成对改（validate 放行、execution 才不越界读）。

### 已落地

- validate 侧 commit `ab822e7a`「accept in-place TSEL dst fused onto first select B.IOT (validate side)」；
- execution 侧 commit `1f398190`「read in-place TSEL false-source from dst tile (execution side)」。
