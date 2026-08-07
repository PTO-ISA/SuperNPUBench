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
- **cuBLAS 的 2048 可解除**：它并非算法固有，而是问题4 的位重解释缺寄存器 bitcast、退而用
  HBM 往返所引入的 32b load/store tile。一旦补齐寄存器级 bitcast（见问题4 解除路径），该 32b
  tile 消失，cuBLAS 的绑定 tile 回落到 16-bit 输入 → 上限升到 4096，与 OCP 一致。

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
- **fp4**：打包轴 ⊥ reduce 轴，`TileN=64` 走 plain RowMajor NoneBox（`tile_o=[BlockSize,32]` packed），
  **归约零改动，优先落地**；上界内可更大（BS=32 → ≤128）。

**通用备选（尾/非尾皆可）——fractal 布局整体豁免**：改用 SLayout::Box/fractal（:649 分支3）**完全绕开
32B 字节约束**（只需 `Rows%InnerRows==0 && Cols%InnerCols==0`），`fa_hif4.hpp:85-92` 发射已验证；但为
寄存器侧 fractal 布局（`SFractalSize_==512/1024`，`pto_tile.hpp:658`），落盘 plain global 的字节序需
运行期核实（当前 skew 不可测），作备选而非首选。

---

## 问题3：linx 缺失带 CmpMode 的 4-参 TCMP/TCMPS（需 linx 侧解决）

### 结论

`-D__linx` 构建下**只有 3-参 mode-less（语义固定 EQ）的 `TCMP`/`TCMPS`**，没有带 CmpMode 的
4-参重载，故 `>` / `<` / `!=` / `<=` / `>=` 无法直接发射。这是 **linx intrinsic 头文件封装的
缺口，不是硬件/仿真器能力问题**：

- 根因：`-D__linx` 分发链（`common/tileop_api_impl.hpp:4-5`）只包含 `jcore/template_asm.hpp`；
  带 CmpMode 的 4-参 `TCMP`/`TCMPS` 只定义在 `jcore/TCmp.hpp` / `aarch64/TCmp.hpp` /
  `cpu_sim/TCmp.hpp`——这三个在 linx 构建里都没被包含。
- linx 实际可用的是 `template_asm.hpp` 的 3-参版（TCMP `:3190` `TEPL 13`；TCMPS `:3819`
  `TEPL 45`），无 CMode 操作数，解码固定为 EQ（`CMode::EQ=0`，
  `SuperScalarModel/isa/ISACommon/BlockAttribute.h:25`）。`CmpMode` 枚举本身可见
  （`common/pto_tile.hpp:19`），故报错停在「4 参无匹配重载」而非「未定义标识符」。
- 底层 ISA/仿真器**支持全部 6 种模式**（`isa/calculate/CubeCalculate.cpp` `EleCmp`，
  `isa/Block.cpp:535` 从 `srcs[SRC6_IDX]` 解 cMode）——缺口纯在头文件封装。

**解除路径（linx 工具链侧）**：在 `-D__linx` 下暴露带 CMode 操作数的 4-参 `TCMP`/`TCMPS`
（在 `template_asm.hpp` 增加发 cMode 操作数的 4-参重载，或让 `tileop_api_impl.hpp` 的
`#ifdef __linx` 分支也包含 `jcore/TCmp.hpp`）。补齐后切换到下述 IDEAL 版本即可。

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

## 问题4：编译器尚未支持 reinterpret（位重解释）语法形式（需编译器侧解决）

### 结论

`-D__linx` 构建下**没有任何寄存器级 reinterpret/bitcast**——float tile 与等宽 int tile 之间
不能在寄存器内按位重解释：

- `TCAST` 在 `-D__linx` 下未声明（`common/tileop_api.hpp` 的 `#ifndef __linx` 块内）；且即便
  可用，它 lower 成单条 `TCVT`，是**数值转换**（`static_cast`，`0.001f → 0`），不保留
  IEEE-754 位型——用它抽 exponent 位在任何后端都语义错误。
- 没有 tile 级 `__builtin_bit_cast` / reinterpret intrinsic 能在寄存器内把 tile 元素按位
  重解释为等宽的另一 dtype。

这是**编译器工具链的缺口，待补齐**（同问题3 的性质）。**解除路径（编译器侧）**：在 `-D__linx`
下提供寄存器级 reinterpret（一条把 tile 元素按位重解释为等宽 dtype、不经 HBM、不做数值转换
的 tile-op，或让 `__builtin_bit_cast` 作用于 tile 类型）。补齐后可把下述 `reinterpret_*` 的
scratch-HBM 往返替换为寄存器写法，省去多余 store/load 与 scratch buffer。

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

---

## 问题6：fp4 数据路径的 cast 域 —— fp32→fp4 直转是否合法，待 ISA/编译器确认

### 背景

**两个 OCP-fp4 kernel 同构**，数据路径都在 **fp32 域**乘 inv_scale 后**直接 `TCVT`
fp32→`__fp4_e2m1x2`**：
- `dynamic_mx_quant_tail_ocp_fp4.hpp`：`ocp_fp4_block` line 57-58 `TCVT(xf_out, xq)`（bf16→fp32）+
  `TROWEXPANDMUL`（fp32 域）；`ocp_fp4_pair` line 90-92 `TLOAD(xcat) → TCVT(oq, xcat)`（fp32→fp4）。
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
