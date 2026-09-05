# DynamicMxQuant 问题记录

## 2026-09-01 状态更新（当前基线 codex/pr-0.58.4，取代下方 0828 清单中的 codex 判断）

**当前工作基线**：model `SuperScalarModel dmxq-ops-20260828`（源自 `origin/codex/pr-0.58.4-shared-model`）
+ 工具链 `Linx-TileOP-API d6a52b8`（fp4 `bits` 保持**官方 8**）+ llvm `0f878a8`。

### 端到端状态

| kernel | 状态 |
|---|---|
| `dynamic_mx_quant_tail_ocp_fp8` | ✅ **res_check PASS**（output MSE=0/MaxAE=0.0117，scale MSE=0） |
| `dynamic_mx_quant_tail_ocp_fp4` | ❌ **res_check FAIL**（output MSE=7.83；scale=pass）—— fp4 **写侧**未修 |

### 已落地（本轮）

| 仓 | 内容 | 状态 |
|---|---|---|
| model | e8m0(SF8) 转换（问题15） | 已提交 `60ce26fd` |
| model | compare-select 用 carrier-width 兼容（问题14，reinterpret 守卫） | 已提交 `3ca5744c` |
| model | `GFRUN_FORCE_DIRECTBOOT_ABI` 注释更正为 X1（问题23） | 已提交 `19e2e97a` |
| 工具链 | PR#41 两拍显式 false-source TSEL（问题18，源+已装头） | 未提交（`linx` 分支工作树） |
| Bench | `readBinary.h`/`writeBinary.h` 在 `#ifndef RES_CHECK` 静音 stdout printf（res_check writev 挂起规避） | 未提交，跟踪 `ISSUE_gfrun_res_check_writev_hang.md` |

### ⚠️ 更正：codex/pr-0.58.4 **并未**原生修复 fp4 打包（下方 0828 清单该结论作废）

下方 0828 清单多处写「上游 codex/pr-0.58.4 已原生修复 fp4 打包，应弃用 31f7a8f 移植、整体对齐 codex」
—— **实测证伪**。当前基线就是 codex/pr-0.58.4，`dynamic_mx_quant_tail_ocp_fp4` 实跑 **output=fail**。

- codex 只有 fp4 **读侧**（`TMAEngine::ExecuteTSTORE` `packed=CubeCellElementBits==4`、`CubeEngine` `srcType==FP4`
  解码），**缺写侧**（`DataFormatCvt` 的 `dstType==FP4` 值编码 + `ExecuteTCVT` nibble 打包 + `IsFourBitDataType`）。
  读侧必要非充分：TCVT 从没把编码/打包好的数据写进 tile。
- 跟踪 issue：**`ISSUE_fp4_pack_tcvt_regression.md` → SuperScalarModel issues454**（已按远端最新代码重写，去本地引用）。

### 最小复现探针（隔离模型值编码，纯 origin + 官方工具链）

`fp4_shape_probe.cpp` `WIDEN=on`，单条 `TCVT(fp32→fp4)`，输入 `[0.5,1,1.5,2,3,4,6,0]`：
```
期望 E2M1 码 [1,2,3,4,5,6,7,0]  vs  实测 [1,2,3,4,5,6,6,0]   # 6.0→码6(4.0) 应码7(6.0),MaxAE=2.0
```
坐实模型缺 E2M1 顶档编码（全 1 码被通用 softfloat 当 NaN/Inf 舍一档），与打包/工具链无关。

### 本地修复思路（分两层，反应式、每步可验）

1. **A 值编码（先做，隔离低风险）**：`CubeEngine::DataFormatCvt` 加 `dstType==FP4/FP4_1` E2M1/E1M2 值表编码
   （顶档饱和到码7）→ WIDEN 探针验 `6.0→7`。不碰打包/工具链/形状。
2. **B 打包（后做，尊重 codex 原生读侧）**：`ExecuteTCVT` 打包 + `IsFourBitDataType` + `Block.cpp` 用
   `ElementBitsOf` 派生打包行。**fc643829 的 TMAEngine `eleRSize` 配套大概率不需要**（codex 已有原生 `packed`
   读侧，套了反而打架）；工具链打包宽度是否需动 **以补完 B 跑全 kernel 的实报错为准**（不预套 bits=4）。
3. **C 全 kernel 复测** output=pass 才闭环。

### 工具链 fp4 `bits`：官方是 `ddd07b9`，不是 `bits 8→4`

远端官方对 fp4 TCVT tile size 的修复是 `ddd07b9 [tileop-api] Encode TCVT destination logical tile size`
（作者 zhuwei0003，**已在 d6a52b8**，改 `TilesizeCode` 取 `tile_shape_out::TilesizeCode`，**未动 bits=8**）。
「`bits 8→4`」无官方提交，是本地 stopgap（下方 0828 清单 A 段记的即此）。当前已把 bits 恢复官方 8。
`ddd07b9` 单独不足以让 fp4 打包（仍需模型写侧 B）——见 `ISSUE_linx_tileop_fp4_tile_size_bits`。

---

## 0828-tag 补丁清单（工具链 / model / kernel）

> ⚠️ 本节针对**旧基线 d8903938**（非当前 codex）；其中「codex 已原生修复 fp4 打包 / 应对齐 codex」
> 的判断**已被上方 2026-09-01 节证伪作废**，其余站点清单仍可参考。

> 面向 `ops-20260828` 配套（TileOP-API `f94bc12` / llvm `adcb879` / model `d8903938`+`ad288c24`）为让
> `dynamic_mx_quant_tail_ocp_fp4`（4PE + 特殊值守卫）端到端跑通所打的补丁清单。类型：**移植**=反应式重放
> 上游已有/应有修复；**配套**=0.58.4 重构后新增、需与打包对齐的站点；**临时**=待正式方案的 TEMP 规避。

### A. 工具链 installed 头（构建产物，非 git 跟踪 → 重建工具链会丢，须重打）

路径 `linx-toolchain-build/output/linx_blockisa_llvm_musl/lib/clang/15.0.4/include/tileop-api/jcore/`

| 文件 | 补丁 | 对应 | 类型 |
|---|---|---|---|
| `template_asm.hpp` | TCMP/TCMPS `B.DATR Zero, cmode0..5` → 命名 `eq..ge`（12 处） | 官方 `16dc674`（PR 已合） | 移植 |
| `template_asm.hpp` | **TSEL 单拍就地 → 两拍显式 false-source**（`B.IOT Mask,True; B.IOT Prior,last,->Dst`，`[Prior] "0"(dst)`） | **Linx-TileOP-API PR #41 `072ea70`** | 移植 |
| `type.hpp` | `__fp4_e2m1x2`/`__fp4_e1m2x2` 的 `bits` 8→4 | `ISSUE_linx_tileop_fp4_tile_size_bits`；**上游正解见下** | 本地 stopgap |

- **cmode**：头 f94bc12 发 `cmode0..5` 超前 adcb879 后端（后端只认命名 `eq..ge` @ `Inst{31-29}`）→ 编译期
  `Match Instruction Error`。移植官方修复即解（当前后端 `eq`/`EQ` 都认、编码同）。**cmode 编解码本身正确**
  （后端 `enum CmpMode{EQ=0..GE=5}` 与模型 `CMode` + `(word>>29)&0x7` 逐值同位）——曾误判为"cmode 错位"，已澄清。
- **TSEL**：就地 `TSEL(dst,mask,true)` 旧头发单拍（2 源 + dst 就地 false-source），false-source(prior-dst)
  被读成 0 → `mask=0` 的 no-op 也清零 dst → 三守卫塌成 output 全 ±0。PR #41 改发两拍显式 false-source（用 `"0"`
  匹配约束把 Prior 绑到 dst 旧值），emulator **原始**显式三源建模即正确。详见 `ISSUE_tsel_inplace_falsesrc.md`。
- **fp4 bits 8→4**：fp4 是 4bit/元素，头误声明 8bit（"x2 打包对载体"视角）。该 bits 喂 `pto_tile.hpp`
  `kBytes=Rows*Cols*bits/8`：bits=8→每元素 1 字节（未打包、2× 大），与模型 `ElementBits(FP4)=4`（打包 2/byte）
  矛盾 → 运行期 `IsLegalLocalTileDescriptor` 尺寸校验崩。bits=4 令工具链 tile 尺寸对齐模型打包布局。
  - **以上游方案为准**：本地 bits=4 仅是配合本仓「31f7a8f TCVT 侧打包」移植的 stopgap。**上游正解**是
    `codex/pr-0.58.4-shared-model` 的 `ElementBitsOf` 抽象 + **存储侧打包**（tile 寄存器保持未打包，只在落 GM 时
    按 nibble 打包）——该路线**无需改 `bits`**、不撞尺寸校验，侵入更小。采用上游方案时应弃用本 bits=4 补丁与本仓
    的 TCVT 侧打包移植，整体对齐 codex 分支。跟踪：`ISSUE_linx_tileop_fp4_tile_size_bits`。

### B. SuperScalarModel（model 仓，工作区未提交，HEAD=`ad288c24`）

| 文件 | 补丁 | 对应 | 类型 |
|---|---|---|---|
| `isa/ISACommon/DataType.h` | `ElementBits` / `IsFourBitDataType` | 移 `31f7a8f`(#314，被 `930d9981` 连坐删) | 移植·fp4打包 |
| `emulator/engine/CubeEngine.cpp` | `DataFormatCvt` fp32→fp4 有限表 + RNE 编码 | 移 `31f7a8f` | 移植·fp4打包 |
| `emulator/engine/TEPLEngine.cpp` | `ExecuteTCVT` 偶低奇高 nibble 打包 | 移 `31f7a8f` | 移植·fp4打包 |
| `isa/Block.cpp` | `UpdateDstTileInfo` 用 `ElementBits` 派生打包行 | 移 `31f7a8f` | 移植·fp4打包 |
| `emulator/engine/AccumulateBlockInfo.cpp` | `IsLegalLocalTileDescriptor` `size==r*c*BytesOf` → `size*8==r*c*ElementBits` | 0.58.4 新增站点（缺则 double-free） | 配套·fp4打包 |
| `emulator/engine/TMAEngine.cpp` | NORM TSTORE `srcRowWidth = totalCol*eleSize` → `*eleRSize`(fp4=0.5) | 0.58.4 新增站点（缺则逐行错位） | 配套·fp4打包 |
| `emulator/engine/AccumulateBlockInfo.cpp` | `IsCompatibleDataTile` `dtype==` → 按位宽 `BytesOf==BytesOf` | **问题14 同类**（reinterpret_tile 位重解释被 dtype 相等断言误杀，`SuperScalarModel issue254`；仿 `3739068c`） | 配套·守卫 |
| `isa/Block.cpp` | TCVT `ValidateOperandContract` 去物理 `row==row`/`col==col` 两条 conjunct | 问题22；**`codex/pr-0.58.4-shared-model` 已解**（改为只比 valid 形状+layout+合法物理描述符） | 临时·对齐上游 |
| `emulator/main.cpp` | `GFRUN_FORCE_DIRECTBOOT_ABI=1` env 规避 | 问题23（纯本地，**不单开 issue**） | **临时**（仅 env 生效） |
| `kernels/multi_thread/quant/dynamic_mx_quant/ISSUE_fp4_pack_tcvt_regression.md`(新，本仓) | fp4 打包回归 issue（由 model 仓迁入 Bench 仓） | — | 文档 |

- **移植 + 配套 = 同一类问题（"fp4 打包在 0.58.4 model 上跑通"），统一由 `ISSUE_fp4_pack_tcvt_regression.md`
  （本仓）跟踪**：4 条移植是重放被 `930d9981` 连坐删的 `31f7a8f` 打包算法；2 条配套（`IsLegalLocalTileDescriptor`
  位宽、TMA `srcRowWidth` eleRSize）是 0.58.4 重构后新增、需与打包对齐的站点。三者 + 工具链 bits 一体，缺一即崩。
  另与既有 `ISSUE_tcvt_fp4_shape_contract`（问题16，TCVT 打包形状契约）、`ISSUE_linx_tileop_fp4_tile_size_bits`
  （tile 位宽）同族。
- fp4 打包（`31f7a8f`，#314）与 e8m0（`52f56d5`，#253）是**同一个 `930d9981` "Revert unify" 连坐删除**的两个受害者；
  e8m0 已由 `ad288c24` 恢复（问题15），fp4 打包本轮补齐。
- **远程现状**：`origin/main` 至今仍缺 fp4 打包；**上游 `codex/pr-0.58.4-shared-model`（另一 0.58.4 子线，不含
  d8903938）已有原生修复**——`ElementBitsOf` 抽象 + **存储侧打包**（`packed=CubeCellElementBits==4`、
  `memoryRowBytes=(validCol+1)/2`、偶低奇高 nibble），且其 TCVT 契约已放宽（去物理 row/col，见上表问题22 行）。
  即本轮 fp4 打包 + 问题22 两处，远程 `codex/pr-0.58.4` 都已正式解决，本仓补丁为对齐前的本地落地。
- **已还原（不在上表）**：本地一度加的 TSEL 就地适配 `ab822e7a`(validate)+`1f398190`(execute) —— PR #41 后原始
  模型即可处理显式两拍，已回退。**判 fp4 守卫问题勿再查 cmode**（已证清白）。
- **已提交（早前 commit，不在工作区）**：`ad288c24` e8m0 恢复、`4cacc579` `ValidateScalarLogicalTepl` 位宽。

### C. SuperNPUBench（kernel 仓，工作区未提交）

| 文件 | 改动 |
|---|---|
| `kernels/multi_thread/quant/dynamic_mx_quant/dynamic_mx_quant_tail_ocp_fp4.hpp` | 重写为 4PE SPMD（get_thread_idx/run_pe/process_tile，reduce tile Cols=1）+ 三守卫（`TCMPS`+`TSEL`）+ fp4 打包输出 |
| `test/kernel/multi_thread/quant/dynamic_mx_quant/src/tail_ocp_fp4.cpp` | driver 改 PM=512/PN=256/RES_CHECK/4 线程 |
| `test/kernel/multi_thread/quant/dynamic_mx_quant/src/tsel_inplace_falsesrc_probe.cpp`(新) | 就地 TSEL false-source 最简探针（mask=0，期望保留 prior-dst） |
| `test/kernel/multi_thread/quant/dynamic_mx_quant/Makefile` | 加 `TYPE=TSEL_INPLACE_FALSESRC_PROBE` 条目 |
| `kernels/multi_thread/quant/dynamic_mx_quant/ISSUE_tsel_inplace_falsesrc.md`(新) | TSEL false-source issue（已标 PR #41 根治） |

### 验证（补丁全到位后）

```
dynamic_mx_quant_tail_ocp_fp4 (4PE, M=512 N=256 BS=32, bf16 in / fp4 out / e8m0 scale, 含三守卫):
  output = pass (MSE=0.022, MaxAE=0.5)   scale = pass (MSE=0, MaxAE=0)   gfrun 4线程 R2=0
tsel_inplace_falsesrc_probe (mask=0): 0x1234 ✓   tsel_inplace_probe (mask=1): 0x1234 ✓
```

---

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
> `test/kernel/multi_thread/quant/dynamic_mx_quant/src/fp4_probe.cpp`（`TYPE=FP4_PROBE diss`）证实
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

## 问题3：带 CmpMode 的 4-参 TCMP/TCMPS（工具链已原生支持；业务 kernel 待迁移）【已解决】

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
> **gfrun 端到端已验证（2026-08-20）**：ELF 用 env_test 工具链编译、
> 工作目录 gfrun（feat/pto-v058-adaptation + 5 处 emulator 反应式移植，见问题9/14/17/18）执行到底
> `R2=0`。**data 输出逐字节匹配 golden**（32 行全对，无棋盘错位；e4m3 输出为 1B tile，B.IOR 元素步长==字节步长故无需字节步长补丁，见问题19）；**scale 值逐字节匹配**（仅布局差
> = 问题5 parity 交织，值本身一致）。故原生 CmpMode 路径数值正确性已坐实。

> **更新（2026-09-01，供 codex 系列）**：本节下文（2026-08-19/20）中「`compute_cublas_core` 末尾保留
> `IDEAL VERSION (blocked)`（`:522`）」「其余 5 个 kernel 仍调用规避版」等**表述已过时**——(1) 该函数末尾的
> IDEAL VERSION 注释块**已删除**；(2) 3 个 cuBLAS kernel（tail / nontail plain / nontail bigbs）均已**就地内联**
> native `TCMPS<CmpMode>` 版本、`common::compute_cublas_core` **无 kernel 调用**（dead reference）；(3) 该 reference
> 函数保留 min/max 比较模拟，但其守卫掩码组合已从数据域 TAND/TOR 改为**合规嵌套 TSEL**（PTO ISA 规范，见
> README「cuBLAS 守卫掩码的 PTO ISA 合规写法」小节，属 kernel 合规、非本 CmpMode 缺口）。下文按历史记录保留。

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

## 问题4：寄存器级 reinterpret（位重解释）（工具链已原生支持；业务 kernel 待迁移）【已解决】

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

## 问题5：非尾轴 scale「parity 交织」缺口——【已解除 2026-09-03：pto-spec 规范定义无需交织，问题不存在】

### 结论（定性反转）

**PTO-ISA 规范不要求 parity 交织；非尾轴 scale 的正解就是 kernel 现有的 compact planar。**
原「必须交织、卡在缺 `TINTERLEAVE` intrinsic」的框定，是把 golden 对齐到了**错误的目标**
（AscendC/ttk 的 `DataCopy<DIST_INTLV_B8>` 打包约定，属 Ascend 硬件而非 PTO-ISA）。规范落地后
本缺口自然消解，**无需交织、无需等 intrinsic**。

**规范依据 = pto-spec `d0ce06ad`（ADR-0101「Matrix Scale Cell Layouts, HiF4 Scale Words, and
CScale」，target 0.58.4）**：matmul 消费 MX scale 分两类载体，两者**都不做 even/odd parity zip**：

- **Shared scale（普通 Tile，常规 HBM→tile 路径）**：A-scale 纯 planar `[group_M, G]`、
  B-scale 纯 planar `[G, N]`。取元素走非 CUBE 分支
  `MatrixRightScaleElement → TileStorageIndex(scale, group, column)` = 行主序 `[G,N]`
  （`asl/tile/model/execution/matrix-scale.asl`）；`TileStorageIndex` 对非 cube 布局 =
  `TileLinearIndex` = 纯 planar。转置靠 primary+scale 一起重绑定，非字节交织。
- **Local scale（CUBE_M32 CellReg 网格）**：`[M,G]`/`[N,G]`，128B CellReg 网格「K-fast /
  32-row-slow」（`PTO-CUBE-MATRIX-SCALE-CELL-001`）——CUBE 专属分块，仍非 parity zip，且通常由
  CUBE 的 scale-load 从 planar tile 加载时完成，量化算子只需存 planar。

### 实证核实（2026-09-03）

1. **真实消费方 `matmul_shared_lowp.hpp` 逐行坐实 Shared + planar**：
   `gmAScale = global_tensor<e8m0, RowMajor<gM, gK/32>>`（`[M,G]`）、
   `gmBScale = global_tensor<e8m0, RowMajor<gK/32, gN>>`（`[G,N]`），
   `SharedMatrixLeft/Right` 绑定 + 普通 `global_iterator` planar 加载，**全链零交织** ——
   与 ADR-0101 Shared 契约、与 kernel 现有 compact-planar 三方一致。
2. **golden 已修正 + 端到端复验**：`gen_dynamic_mx_quant_data.py:compact_scale_bytes` 非尾轴分支
   由 parity 交织改为纯 planar `[scaleRows, cols]`（尾轴本就 planar，无改动）。
   `nontail_cublas_fp8_4pe`（Axis=512/Post=256/BS=32，numKb=16 → 交织本会大幅重排 16 块行）
   4-PE gfrun：**output=pass（MaxAE=0.0117）、scale=pass（MaxAE=0，逐字节精确）**。
   kernel 未改一行 → 证明 compact-planar 一直就是 PTO-ISA 正解。

### 历史遗留 intrinsic 缺口（现无功能影响，仅存档）

`-D__linx` 头确未暴露 `TINTERLEAVE`/`TDEINTERLEAVE`（ISA 0.57 有定义，`template_asm.hpp` TEPL 操作码
29/30/31、48-55 空档无发射模板；0.57 workbook 标 A5-only 无独立汇编页）。但**该 intrinsic 对本量化
算子已无需求**——scale 无需交织。若未来别处（非 MX scale）确需 zip，再按「补 4-参发射模板」路径解除。

### 历史验证：compact-planar 的数值与列序早已实测坐实（2026-08-20，现回看即正解证据）

> 注：以下为定性反转前的记录。当时把 planar 与 golden 的差异归为「待补的 parity 交织」；
> ADR-0101 落地后回看，**planar 本身即 PTO-ISA 正解**，这段实验正是「kernel 输出一直正确」的旁证。

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

## 问题8：非内联 helper 的 tile 参数经 `TSTORE/TLOAD, S64` 栈传参，被 emulator `ValidateLocalTlsu` 拒绝（需 emulator / linx 后端侧解决）【已解决】

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

## 问题9：TABS 作用于 BF16 被 emulator 拒绝（需 emulator 侧解决）【已解决】

> **✅ 2026-08-24 已解决（ops-20260823 / a5dca25a 官方基线）。** 探针 `TABS_TROWMAX_PROBE`
> （`TLOAD bf16 → TABS → reinterpret_tile<u16> → TROWMAX → TSTORE`）gfrun 跑到底 R2=0、无 assert
> 实证。旧白名单 `IsBasicUnaryTeplDataType`(FP16/FP32) 已不存在；TABS 现走统一白名单
> `TileVecArithmeticDataTypeSupported`（AccumulateBlockInfo.cpp:255-280，含 BF16:265）。属官方基线
> 自带，非本地 cherry-pick。下文为历史记录。

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

- **改动**：新增 driver `test/kernel/multi_thread/quant/dynamic_mx_quant/src/{tail,nontail}_cublas_fp8_fp16.cpp`
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

## 问题10：TROWMAX 作用于 UINT16 被 emulator 拒绝（与 TCOLMAX 不对称，需 emulator 侧解决）【已解决】

> **✅ 2026-08-24 已解决（ops-20260823 / a5dca25a 官方基线）。** 同探针 `TABS_TROWMAX_PROBE`
> 的 `reinterpret_tile<u16> → TROWMAX uint16 → TSTORE` 链，gfrun 跑到底 R2=0、无 assert 实证。旧
> 白名单 `IsReduceAndExpandTeplDataType` 给 TROWMAX 的窄集(FP16/FP32/INT32) 已不存在；TROWMAX 现走
> 统一白名单 `TileVecArithmeticDataTypeSupported`（含 UINT16:274），与 TCOLMAX 对称。执行侧
> `ExecuteTROWMAX`(TEPLEngine.cpp) 有 UINT16 dstStride 分支佐证。属官方基线自带。下文为历史记录。

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

## 问题13：TCVT 不发 lb2，valid-col-1 的 TCVT 输出直接 TSTORE 被 emulator 拒绝（需 toolchain 侧解决）【已解决】

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

## 问题14：零指令 `reinterpret_tile` 的位重解释运行期不可见，emulator dtype 相等断言误杀（需 emulator 侧解决）【SuperScalarModel issue254】

> **状态更新（2026-08-24 实测 git 对比）**：位宽放宽是**本地反应式移植，官方从未采纳**——
> `origin/main`（e82817e1，08-23）/ `origin/feat/pto-v058-adaptation`（63dbb5a2）/`_rebase_main`
> 全保持**严格 dtype 相等**；本地放宽源自 `local_test`（f60ca82f），本次以工作树提交 `1fecf9e6`
> （author ziyang-cheng）落地，patch-id 扫全 origin 无命中=本地原创、非官方移植。与问题15（官方已修
> 52f56d5f）互为对照：**问题14 每次跟官方分支同步后都须重新反应式补这一处**（分支分叉/force-push 会丢，
> 见 memory `reference_v058_branch_force_pushed`）。补后 `probe_ocp_fp8_newcalc` **gfrun R2=0 + gfsim 620 cyc 双通**。

> 详见 `ISSUE_reinterpret_dtype_tag.md`。缺陷所在仓 **`SuperScalarModel`（emulator 建模层
> `AccumulateBlockInfo.cpp:440` `ValidateScalarLogicalTepl`）**，复现入口在 probe 的 OCP 清尾数链
> `reinterpret_tile<uint16_t>(max_bf) → TANDS → 读回 bf16`。**这是问题4「无寄存器 bitcast」被 v0.58
> `reinterpret_tile` 解除后，暴露出的下一道墙**（HBM 往返时被 TLOAD 重打标签掩盖，零指令 bitcast 才显形）。

> **同根 sibling（ops-20260904 / fp4 再次命中）**：compare/select 路径的 `IsCompatibleDataTile`
> （`AccumulateBlockInfo.cpp` `ValidateCompareSelectTepl`）是本问题的姊妹断言——`tail_ocp_fp4` 三守卫
> `TCMPS(eq_inf, reinterpret_tile<uint16_t>(max_bf), …)` 对 bf16 的 u16 视图比较，同样按 `dataType 相等`
> 误杀（崩 `TCMPS requires one compatible Tile source`）。与 `ValidateScalarLogicalTepl` 同根、须同样放宽为
> **carrier 位宽匹配**（`BytesOf==BytesOf`）。规范依据：`TCMP/TCMPS.asl` Legacy RowMajor 形式"the existing
> sixteen-type domain remains unchanged"（源不限定=操作 dtype）+ ADR-0112 reinterpret「observable only
> through carrier width」。每次同步官方分支后与本问题一并须重新反应式补。

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

## 问题15：emulator 未实现 TCVT bf16→e8m0(SF8)，MX 共享 scale 转换缺失（需 emulator 侧解决）【官方 52f56d5 修复 → 930d9981 误回退 → 本地恢复，SuperScalarModel issues439】

> **状态更新（2026-08-31 复核）——这是一个未恢复的回归**：官方 `52f56d5f`（#253「TCVT float→E8M0」，
> jialewang 08-21，PTO-TCVT-E8M0-PROFILE-001）**确实修复过**，但该实现被并进 "unify TCVT/ARGMAX/FPATR,
> cooperative TMATMUL, TSORT" 大包后，被 **`930d9981 Revert "unify…"` 连坐删除**。实测：
> `git log -S ConvertFloatToE8M0 52f56d5..d8903938` 唯一命中 `930d9981`；**当前 `origin/main` 与侧支基线
> `d8903938` 都已无该实现**（`ConvertFloatToE8M0` 出现 0 次、SF8 仍落 `CubeEngine.cpp` 的
> `Not support such type convert` assert）。故**不是**「基线落后、同步官方即自带」，而是官方线自身的**未恢复回归**。
>
> **本地已恢复**：model 分支 `dmxq-ops-20260828` 提交 `ad288c24`，**忠实重放 52f56d5 的 E8M0 部分**
> （不含被 revert 的 cooperative TMATMUL/TSORT/ARGMAX #271）——proper header `PtoRoundingMode` +
> `ConvertFloatToE8M0`（全 rounding mode + Sat + IEEE flag）+ `DataFormatCvt(bundleRMode,saturating)`
> 重载 + `CubeEngine.cpp` inline `if(dstType==SF8)` 块。验证：`tail_ocp_fp8` 4-PE 官方 res_check
> **`scale=pass（MaxAE=0，逐字节精确）/ output=pass（MaxAE=0.011719）`**（真随机输入）。跟踪
> **SuperScalarModel issues439**，详 `SuperScalarModel/docs/ISSUE_e8m0_tcvt_regression.md`。
>
> **过时口径（2026-08-24，已作废）**：曾记「官方已修、已在 origin/main 等 6 分支、同步官方分支即自带、
> 干净 cherry-pick `189e45f6`」——基于当时工作目录尚含 52f56d5 的组合，`930d9981` 回退后不再成立。
> 下方为修复前的原始记录。

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

## 问题16：TCVT 形状契约（TileLogicalShapeMatch）对打包 fp4 与源无法同时满足 → 结构性必崩；当前工具链落在**编译期 static_assert**（需工具链头 + emulator 双侧解决）【SuperScalarModel issue314 已解决 + Linx-TileOP-API issue30 已解决（官方 ddd07b9）】

> **状态更新（ops-20260904 / TileOP `804eb03`）：工具链侧已由官方 `ddd07b9`（Encode TCVT destination
> logical tile size：`TCVT_T` 改用 `tile_shape_out::TilesizeCode`）解决。** `tail_ocp_fp4` 在 `804eb03`
> 未打任何工具链改动即编译通过且 TCVT 运行期形状自洽（工具链发的物理容量经 `DerivedTileRows` 派生成
> 与 `validRow` 相容的物理行数，emulator 侧按 packed 落盘——见问题26）。原「工具链头 `type_traits::bits=8`
> 使 tile size 2×」的推断不再需要工具链改动。

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

`test/kernel/multi_thread/quant/dynamic_mx_quant/src/fp4_shape_probe.cpp`（`TYPE=FP4_SHAPE_PROBE`，`WIDEN=on`→
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

## 问题17：TCMPS 作用于 UINT32 被 emulator compare/select 白名单拒绝（需 emulator 侧解决）【linx-toolchain-build issue5】

> **❌ 2026-08-24 实测仍未解决（ops-20260823 / a5dca25a 官方基线）。** 探针 `TABS_TROWMAX_PROBE`
> 尾段（`TCVT bf16→fp32 → reinterpret_tile<u32> → TCMPS uint32 → TSTORE`）gfrun 崩在
> `IsCompareSelectTeplDataType`（AccumulateBlockInfo.cpp:384）。当前 HEAD 的 TCMPS 分支(:337-341)仍为
> `INT32/FP32/FP16/UINT16/INT16`、**缺 UINT32**，与下文旧记录逐字一致。下文「已落地」的 emulator
> commit `50afe316` 是旧 local_test 上的修复，**未并入官方 a5dca25a**（`git log|grep 50afe316` 查无），
> 我为 tail_ocp_fp4 打的 4 个 cherry-pick 亦不含它。与问题9/10（官方已放宽 arithmetic/reduce 白名单）
> 对比：compare/select 白名单官方仍未放宽。

> 缺陷所在仓 **`SuperScalarModel`（emulator）**，复现入口 = `nontail_cublas_fp8_plain` 就地展开后、
> 在 `reinterpret_tile<uint32_t>` 视图上做 `TCMPS<CmpMode::{LT,NE,GT,EQ}>` 抽 fp32 指数/尾数位。
> **这是问题3（原生 CmpMode）+ 问题4（reinterpret_tile）落地到 cuBLAS plain 路径后暴露的下一道 emulator 墙。**

### 结论

cuBLAS plain 路径（各 kernel 内联的 native CmpMode 版本）在 **u32 域**用 `TCMPS<CmpMode::LT>(finite, raw, FP32_EXP_MASK)`
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

## 问题18：linx 就地 TSEL（false-source 融进 dst）被 emulator 双侧拒绝（需 emulator 侧解决）【SuperScalarModel issue338+Linx-TileOP-API issue39，官方pr修复 [Linx-TileOP-API pr39](https://github.com/LinxISA/Linx-TileOP-API/pull/41)】

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

## 问题19：工具链 `TLOAD/TSTORE` 内联汇编模板把 B.IOR 的 GM 行步长按**元素数**发射（漏 ×bits/8），违反 pto-spec ADR 0074 字节步长契约 → 非 1B dtype 的 GM 访存半行错位（需 linx-toolchain / Linx-TileOP-API 侧解决）【未修·上游 issue31 挂起；本地 installed 头两行字节步长补丁可跑通 MSE=0（未提交）】【Linx-TileOP-API issue31】

> 缺陷所在 = **工具链头** `Linx-TileOP-API` `jcore/template_asm.hpp`（安装于
> `linx-toolchain-build/output/linx_blockisa_llvm_musl/lib/clang/15.0.4/include/tileop-api/jcore/`）。
> 复现入口 = `TAIL_CUBLAS_FP8`（M=8/K=32/BS=32，bf16 in → e4m3 out）data+scale 双失配。**非 gfrun bug、非 kernel bug。**

### 结论

kernel 写的 `TLOAD()`/`TSTORE()` 宏解析到 `template_asm.hpp` 里 **raw `asm volatile(...)` 模板路径**
（单 tile `TLOAD` :1755 / `TSTORE` ~:1838），**不走** `blk_tload/blk_tstore` clang builtin 路径
（`TLoadBackend.hpp`/`TStoreBackend.hpp`——那条自 `91e3d7a`(2026-06-22) 起已字节正确）。raw-asm 路径把
B.IOR 的 GM 行步长操作数 `[GmStride]` **直接绑 `GetStride(3)`（元素行数），漏了 `×element_size_bytes`**：

```cpp
// 修前（:1774 TLOAD / :1857 TSTORE）
[GmStride]"r"(src.GetStride(3))          // 元素行数，当字节喂 B.IOR
```

违反 pto-spec `e9e9934f` **ADR 0074「TLOAD/TSTORE GM Byte Row Stride」**：`B.IOR.RegSrc1 = row_stride_bytes`
（**字节**），`byte_address = base + row*row_stride_bytes + column*element_size_bytes`，「encoded stride is
a byte quantity, **not multiplied by element size a second time**」。emulator 侧已跟进字节契约
（`TMAEngine.cpp` 行推进 `addr += i*stride` 原样字节、无 ×BytesOf；model 提交 `2d467114`「interpret as bytes」
取代早期 `10e6557f`「scale by element size」），故 gfrun 完全合 ADR 0074，是**工具链未跟进**。

**为何只有非 1B dtype 中招**：1 字节类型（fp8/e4m3/uint8）下「元素数 == 字节数」，B.IOR 巧合正确；
bf16(2B) 输入 load 被发成半行 stride（32 元素当 32 字节 = 16 个 bf16），GM 行 pitch 减半 → scale 错行 +
data 每行 16B 错位（同一半 stride 的两个投影）。本 kernel 反汇编 `__blkc_bf16`：修前 `addi zero,32,->a4`
后 bf16 load 与 fp8 store **共用 a4=32** 作行 stride。

### 机理（反汇编实证）

`gm_x=RowMajor<8,32>` bf16 正确行 pitch = 32 元素 = **64B**，fp8 输出 = 32 元素 = **32B**；工具链修前对两者
都发 32（元素计数）。修后两 B.IOR 按 dtype 分化：bf16 load `B.IOR[a0,a5]` a5=`addi 64`、e4m3 store
`B.IOR[a1,a3]` a3=`addi 32`。

### 解除路径（工具链头，本次采用）

把每个 `TLOAD/TSTORE` 的 `[GmStride]` 绑定折成字节步长（抄同仓 builtin 路径 `TLoadBackend.hpp:230` 口径）：

```cpp
// 修后
[GmStride]"r"((src.GetStride(3) * type_traits<typename gm_shape::DType>::bits + 7) / 8)
```

### 本地补丁（工作目录 installed 头，未提交）

- 补丁 = 单 tile `TLOAD`(:1774) / `TSTORE`(:1857) 两处 `[GmStride]` 折成字节步长（= 本 kernel 走的路径）。
  应用后官方流水线 `gen(seed=8) → 编 res_check=on → gfrun 代 QEMU → compare` **data+scale 双 pass，
  MSE=0 MaxAE=0（256/256 + 16/16 逐字节精确）**；seed=8 保证 scale 有区分度（seed=12345 全行恒 120 会掩盖）。
- 补丁仅落 installed 头、未提交，上游 `Linx-TileOP-API` src/ 与 issue31 未动。installed 头 template_asm.hpp
  另有 ~16 处同缺陷 GmStride 绑定（`TLOAD2/4`、`TSTORE2/4`、Shared、gather/scatter：约
  284/307/330/351/376/401/426/462/487/520/548/1805/1833/1889/1923/1935），正式修复应落 Linx-TileOP-API 源覆盖全部绑定点。
- 两套工具链 raw-asm 路径默认都发元素步长；1B dtype（fp8/e4m3/uint8）元素数 == 字节数、B.IOR 巧合正确，
  故 1B 主导的测点不打补丁也逐字节对，仅 bf16/fp32 等非 1B 的 GM 载入需要此补丁。

## 问题20：`tail_cublas_fp8` 多 block（numKb>1）散落个别 block 的 `max_f` 归约算成 0 → extract=0 连锁 scale 字节 0 + recip 巨值 → data 饱和到 e4m3 上限（数据相关，独立未定位）【未修·未定位】

> 复现入口 = `TAIL_CUBLAS_FP8`（driver `tail_cublas_fp8.cpp`，bf16 in → e4m3 out）临时扩到 **128×256（numKb=8）seed=8**。
> 与问题19（B.IOR 字节步长）**无关**、是独立的**第二问题**：问题19 修好后 8×32（numKb=1）seed=8 全字节 pass，扩到 128×256 才双 FAIL。

### 结论

问题19（B.IOR 字节步长）本地补丁应用后，`dynamic_mx_quant_tail_cublas_fp8` 在 **8×32（numKb=1）seed=8 逐字节 pass**；
但扩到 **128×256（numKb=8）seed=8 data+scale 双 FAIL**。步长修复在 128×256 **确认成立**（bf16 load B.IOR
步长寄存器 s0=`addi zero,512`=512B=256 元素×2B 正确缩放；data **118/128 行逐字节全对**、scale 128 行几乎全对——
步长错会是**全行规律棋盘错位**，不会是 118 行全对 + 10 散行）。残差 = **数据相关的 `max_f` 归约→0 边界**，非步长回归。

### 机理（精确解码）

- ~10 散行的个别 block（如 row56 的 block4+block5、row73 block7、row76 block5）kernel 把该 block 的 `max_f`
  **算成 0**（真实 amax=1.75 或 3.5，数据非零）。
- 连锁（同一 `extract=0` 的两个投影，非两个独立 bug）：`nonzero=(raw!=0)` 掩码=假 → `extract=0` →
  ① scale 字节存 **0**（应 120/119）；② `recip=0x7f00-(0<<7)=0x7f00`（bf16=2^127 巨值）→ data 路径
  `TROWEXPANDMUL` 乘巨值 → **饱和到 e4m3 上限**（out 得 448/416/352/320…，golden 为 88/128/32…）。
- 另有 3 处良性近零舍入差 `0x82` vs `0x80`（±0.0039）。
- **无相关性排除**：坏块 amax 正负都有、值（1.75/3.5）与正常块无异；全局 496/1024 block 的 amax 来自负值 → 符号不相关。

### 判据

out 值 = x × 2^9（inv_scale 应为 2^7，即大 4× = 2 个指数档），对应 `recip=0x7f00` 巨值 → 坐实 `extract=0`。

### 未定位

疑 emulator 对特定归约模式的 TABS/TROWMAX 边界返 0，或 tiling 边界（M=128 → TileM=64，full_m=2）。
下一步 = 锁一个坏 block，gfrun `--dump-memory` 抓中间 tile `abs_x`/`max_r`，判 TROWMAX 归约返 0 还是 TLOAD 该 block 数据本身为 0。

### 复现

driver `tail_cublas_fp8.cpp` 数组 + 模板实参临时改 128×256（scale=128×8），`gen --M 128 --K 256 seed=8`
→ 编 res_check=on → gfrun → compare。seed=8 保证 scale 有区分度（seed=12345 全行恒 120 会掩盖）。
**注**：验证后 driver 已还原 8×32（registered CONFIG=8×32）。记忆 `reference_tail_cublas_fp8_maxf_zero_multiblock`。

## 问题21：fp32→e4m3 TCVT 溢出饱和漏「舍入进位越 exp_max」一路 → 进位溢出值吐 0x78=256 而非饱和 0x7e=448（需 emulator 侧解决）【SuperScalarModel issue364】

> 缺陷所在仓 **`SuperScalarModel`（emulator）**，softfloat 定点内核 `softfloat/fpu/softfloat-parts.c.inc`
> `partsN(uncanon_normal)` 的 `ocp_e4m3` 溢出分支（:193-205）。复现入口 = `PROBE_OCP_FP8_NEWCALC`
> （`probe_dynamic_mx_quant_tail_ocp_fp8_newcalc`，fp16 in → e4m3 out，BS=32，M=8/K=32）。**非 kernel bug、非 scale 路径 bug。**

### 前置：一阶 inf 语义已修（7d01ed6f）

commit `7d01ed6f`「fix(softfloat): implement OCP E4M3 finite exponent range」（当前 HEAD `f0c488c8` 祖先链）
已把 e4m3 从「IEEE 1-4-3、biased-exp=15 全判 inf/nan」改成 OCP 语义：`exp=15 / frac 0..6` 为有限正规数、
仅 `0x7f`=NaN、溢出饱和 0x7e=448、E5M2 保持 IEEE 不受影响。本问题是该修复**未覆盖的二阶边界**。

### 结论

`ocp_e4m3` 溢出分支的饱和判据 `(frac & 0b111) >= 0b111`**只认「移位后尾数字段==全 1(7)」**，漏了
「舍入进位使尾数溢出、exp 越过 exp_max」这一路：

```c
// :185-189  舍入进位：frac_addi 返回真 → 尾数右移、exp++（15→16），尾数被移位清 0
if (frac_addi(p, p, inc)) { frac_shr(p, 1); ...; exp++; }
...
// :196-205  ocp_e4m3 溢出
if (unlikely(exp >= exp_max)) {          // exp=16 也进这里
    exp = exp_max;                        // 钉回 15
    frac_shr(p, frac_shift);
    if ((p->frac_hi & 7) >= 7) { ... }    // 进位来的尾数=0 → (0&7)>=7 假 → 不 clamp
    goto packed;                          // → 打包 exp=15/mant=0 = 0x78 = 256（错，应 0x7e=448）
}
```

### 机理（e4m3 顶档网格）

顶档（exp 字段=15，值=1.mmm×2^8）：mant0=256=0x78，mant6=448=0x7e（最大有限），mant7=480 槽=0x7f=NaN；
512=2^9 需 exp=16 越档。相邻档距=32。

| 输入值 V | 舍入到 | 是否进位 | 现结果 | 应为 | 判 |
|---|---|---|---|---|---|
| [256,448] | 对应档 | 否 | 精确 | 同 | ✓ |
| (448,496) | 448 或 480(mant7) | 否 | mant==7→clamp 0x7e | 0x7e | ✓ |
| **[496,512)** | 512（进位）| **是** | exp→16、mant→0、漏 clamp → **0x78=256** | **0x7e=448** | ✗ bug |
| ≥512（MX 域外）| — | exp 直接≥16 | 尾数各异、多数漏 clamp → 0x78..乱 | 0x7e | ✗（同因，MX 归一后不达）|

**触发值**：喂 fp32→e4m3 TCVT 一个 `V ∈ [496, 512)`（如 500.0f、496.0f 边界）即触发。分界 496 = 480 与 512
中点，round-half-to-even 偏 512（mant0 偶）→ 进位。**为何 MX 域内恰是此带**：OCP e4m3 emax=8，
scale=2^(E_max−8) 令块最大元素 scaled∈[2^8,2^9)=[256,512)，故 mx_quant 永远够不到 512，相关带 = 交集 [496,512)。

### 实证（newcalc probe，seed 数据，8×32）

- **修复前**（HEAD gfrun）：`output.bin` 253/256，3 处失配 **r4c3 / r5c2 / r6c12 全 out=0x78 vs golden=0x7e**，
  三者块最大元素 scaled 分别 **511.75 / 506.75 / 498.25**，逐一落在 [496,512)。scale 16/16 本就对齐。
- **候选修法**（验证用，已回退未提交）：`ocp_e4m3` 分支内在钉 exp 前先取 `bool overflow = exp > exp_max;`，
  再把 `overflow ||` OR 进 clamp 条件——真溢出无条件 clamp mant=6=0x7e，不再只靠尾数==7 的 NaN 避让。
  只改 softfloat 内核，`fmt->ocp_e4m3` 门控、E5M2/fp4 不受影响。
- **修复后**（重编 gfrun、同一 ELF）：`output.bin` **256/256 逐字节**、scale 16/16 不变；r4c3/r5c2/r6c12 全转 0x7e。
  golden 含 6× 0x78(合法256) + 6× 0x7e(饱和448)，output 逐位复刻——坐实**修好越档、不误伤合法 256**
  （240→256 这类 exp14→15 进位 overflow=false，仍打 0x78）。

### 全情形复核（候选补丁）

| 输入 | exp | overflow=exp>exp_max | 结果 | 对否 |
|---|---|---|---|---|
| 256…448 可表示 | 15 | false | mant 0..6，`m>=7` 假→原样 | ✓ |
| (448,480) 舍到 m=7 | 15 | false | `7>=7` 真→clamp 6=0x7e | ✓ |
| [480,512) 进位 | 16 | **true** | 无条件 clamp 6=0x7e | ✓ 修复 |
| 240→256 进位(exp14→15) | 15 | false | mant=0，不 clamp→0x78 | ✓ 不误伤 |
| ≥512 大输入 | ≥16 | true | clamp 6=448 | ✓ |

### 状态

缺陷已定位、修法已实证（output 253→256/256 逐字节对齐 golden）。**补丁在 worktree 试打后已回退、未提交**，
正式修复挂 **SuperScalarModel issue364**（emulator 侧落地）。记忆 `reference_emulator_e4m3_clamp_256`。

## 问题22：boxed 尾块（validRow < 物理行高）reduce 输出物理列 stride 被模型反推塌成 1 → 下游 TCVT 形状契约崩溃（需 emulator/spec 侧解决）【Linx-TileOP-API issue42】

> 缺陷所在仓 **`SuperScalarModel`（emulator）**，`isa/Block.cpp` rowReduce 分支的输出 stride 反推逻辑。
> 复现入口 = `TYPE=TAIL_OCP_FP8`（正式 4-PE kernel `dynamic_mx_quant_tail_ocp_fp8.hpp`，本轮由
> `probe_dynamic_mx_quant_tail_ocp_fp8_newcalc_mt.hpp` 去 probe 正名而来；fp16 in → e4m3 out，
> BS=32）。**非 kernel bug、非工具链 codegen 缺陷。**

### 结论

当 reduce（`TROWMAX`）物理 tile 行高 `TileM` > 有效行 `validRow`（boxed 尾块，且 `validRow` 不整除
物理分配字节）时，工作目录模型 `f0c488c8` 对 reduce 输出**物理列 stride** 用启发式
`dst->size / (validRow * elemBytes)` 反推。该式在非整除时**塌成 1**，使 `TROWMAX` 输出被记为 `col=1`；
紧随的 `TCVT` 目标物理列由 B.DIM `lb2=BlockSize`(=32) 给出 → `col=32`。二者不等 →
`Block.cpp:1155 ValidateOperandContract`「PTO 0.58 TCVT requires matching source/destination
logical shapes」断言崩溃。

**关键点：非 codegen 分叉。** full-tile 与 tail 的 `TROWMAX`/`TCVT` 指令**编码逐位相同**（`24119181`/
`21b19181`），唯一差别是运行期 `validRow` 寄存器（64 vs 58）。崩溃纯在模型构造 reduce 输出描述符的
stride 反推上。

### 反推表（物理 tile 恒 64×32 half = 4096B）

| validRow | `4096 % (validRow*2)` | 反推 stride=col | TCVT dst col(lb2) | 判 |
|---|---|---|---|---|
| 64（full-tile） | 0 | `4096/128`=**32** | 32 | ✓ 通过 |
| 58（boxed 尾块） | 36≠0 | **1** | 32 | ✗ 崩 |

`validRow==物理 TileM` 时 `dst->size` 恰整除、反推=真实物理列；boxed 尾块 `validRow<TileM` 时
`dst->size` 含 padding 行字节，反推失效。**物理行步长在 bundle 里没有独立 B.DIM 字段承载**——模型只能
从 `dst->size` 反推，这是根因。

### 触发条件

`M % (kPeNum*TileM) != 0`，即某 PE `SubM % TileM != 0`（seg_tail>0）。例：M=1000/4PE → SubM=250 →
seg_full=3、seg_tail=58 boxed → 崩；M=1024 → SubM=256 → seg_tail=0 → 通过（R2=0）。此前所有注册配置
M 均被 `kPeNum×TileM` 整除，故一直未触发。**全 mx_quant 家族共有**（凡 reduce 输出直接喂 lb2 声明物理列的
elementwise 且发生 boxed 尾块）。

### 与 env_test（`d8903938`）回归的关系——两个独立问题

`grep` 核实 env_test 最新版**不含反推式**，rowReduce 分支**硬编码 `const uint64_t stride = 1;`**
（注释 "physical column count is always one"）。故：

| 模型 | reduce 输出 col 来源 | full-tile | boxed 尾块 |
|---|---|---|---|
| **WD `f0c488c8`**（本问题） | `dst->size/(validRow*elemBytes)` 反推 | ✅ 通过 | ❌ 塌成 1 |
| **ET `d8903938`**（另一问题） | **硬编码 =1** | ❌（≠32） | ❌ |

ET 版直接假设 reduce 输出物理列恒 1、期望 kernel 声明 `physical Cols=1`；而 kernel 现用
`physical=BlockSize / ValidCol=1`（行跨步，源于 `ISSUE_32B_align.md` 的 32B 列对齐 static_assert，
最新工具链已删该约束）在 ET 彻底不兼容。这是「physical=1 迁移」的动因，独立于本尾块问题。

### 同根的另一面（2026-08-31）：physical=1 迁移后，1 字节 e8m0 收窄 TCVT 目的 row 被 128B 最小档撑翻倍

对齐 `d8903938` 的 reduce-col=1 语义把 `tail_ocp_fp8` scale 链下游列向量迁到 `physical Cols=1`
（提交见 SuperNPUBench `dmxq-ops-20260828`）之后，在 `d8903938` 基线上前移暴露出**同一「物理几何从
`dst->size` 反推」根因的另一个触发面**：scale 链末端 `TCVT bf16→e8m0`（`DataType::SF8`，1 字节）的目的
tile 是 `[TileM,1]`（`TileM=64`，BS=32），逻辑仅 64B，但 PTO TSize 最小档 128B（工具链
`pto_tile.hpp:861 round_capacity` 从 128 起步），被抬到 128B 编进 `sizeClass`；模型
`Block.cpp` 按 `row = dst->size/(physicalCol×elemBytes) = 128/(1×1) = 128` 反推，而 bf16 源同形状
`128/(1×2)=64` → `TCVT` 契约物理 `row 64≠128` 崩。

- **与本问题同根、异触发**：本问题是 boxed 尾块 `validRow<TileM` 令 `dst->size` 含 padding 行字节 →
  反推塌 1；e8m0 这面是 1 字节窄 tile 逻辑字节 <128B 被**最小档下限**抬起 → 反推翻倍。二者都栽在
  「物理 row/col 无独立字段、一律 `size÷(col×bytes)` 反推」。**口径**：非某一侧单独 bug，是编译器（发
  `sizeClass=1`/128B 下限）与模型（按填充后字节数反推 row）对「sub-128B 窄 tile」处理未对齐。
- **临时规避（已落地、未提交）**：`SuperScalarModel/isa/Block.cpp` 的 TCVT `ValidateOperandContract`
  放宽为**只比 `validRow/validCol/layout`**、去掉物理 `row==row`/`col==col`（对齐「修复建议」方向）。
  配合问题15 的 e8m0 转换恢复（`ad288c24`），`tail_ocp_fp8` 4-PE 官方 res_check
  `scale=pass（MaxAE=0）/ output=pass（MaxAE=0.011719）`。
- 详 `ISSUE_tcvt_e8m0_row_padding.md`（含 ①kBytes→②round_capacity 128→③TilesizeCode→④B.IOT
  sizeClass→⑤BlockArgs.cpp size=128→⑥Block.cpp row=128→⑦契约断言 的逐行链条）。

### 修复建议

- **根本解（模型/spec）**：reduce 输出物理列 stride 应有权威来源而非从 `dst->size` 反推——① 改用物理
  行高 `TileM` 反推（`dst->size/(TileM*elemBytes)`，full/boxed 都得 BlockSize）；或 ② spec 给 reduce
  输出补物理列步长字段（类 TCVT lb2）。
- **kernel 侧规避（二选一，尚未落地）**：方案 C = `static_assert(SubM % TileM == 0)` 拦 boxed 尾块，
  牺牲任意 M；方案 D = 把 seg_tail 按 2 的幂分解成若干合法物理行高 full-tile（58=32+16+8+2）绕开反推失败。

### 状态

缺陷已定位、机理已实证（M=1000 崩 / M=1024 通过，指令编码逐位对照）。**未修**，正式修复挂
`ISSUE_reduce_output_stride_tail.md`（待提 SuperScalarModel issue）。kernel 侧规避方案 C/D **尚未选定落地**。
相关：问题13（TCVT 不发 lb2）、问题16（TCVT 形状契约）、`ISSUE_32B_align.md`。

### 补充（2026-09-02）：rowReduce（尾轴 TROWMAX）与 colReduce（非尾轴 TCOLMAX）在模型侧物理形状处理**不对称**

核对 `SuperScalarModel/isa/Block.cpp` 的 reduce 输出形状分支，两者机制不同：

- **rowReduce（`TROWMAX/…`，Block.cpp:2328）**：`stride=1` **硬编码** → `col=1`、`validCol=1`，
  `row = dst->size/(1×elemBytes)`。即**物理列被强制塌成 1**（列 stride 在 bundle 无独立 B.DIM 字段承载）。
  故尾轴 reduce 下游列向量 tile **必须声明 physical Cols=1** 来匹配（本问题正文）。
- **colReduce（`TCOLMAX/…`，Block.cpp:2349）**：`validRow=1`、`col=validCol`、
  `row = dst->size/(validCol×elemBytes)`。即**物理行不被强制为 1，而是按分配 size 反推**。

代入非尾轴现用声明 `Tile<…, BlockSize, TileN, …, ValidRow=1, ValidCol=TileN>`：`dst->size=
BlockSize·TileN·elemBytes`、`validCol=TileN` → 模型反推 `col=TileN`、`row=BlockSize`，与声明的物理行
`BlockSize` **自洽**，不崩；`nontail_cublas_fp8` 逐字节 `output=pass` 亦佐证数值正确。

- **两种写法都合法（colReduce 特有）**：因物理行由 `dst->size` 反推，声明 `physical row=BlockSize`（复用
  输入块物理行高）反推得 `BlockSize`；声明 `physical row=1` 则 `dst->size=TileN·elemBytes`、反推得 `1`，
  同样自洽。这与 rowReduce（物理列被无条件塌 1、只有 col=1 一种合法声明）**不同**。
- 与问题24（4-PE 落盘规模相关现象）**无关**：colReduce 形状自洽且数值正确，非该现象成因。

## 问题23：gfrun 对 res_check ELF 的 syscall-ABI 探测误判 → 读错寄存器、`Bad Syscall` 崩（需 emulator 侧解决）【未修·本地 env 规避】

> 缺陷所在仓 **`SuperScalarModel`（gfrun）**，`emulator/main.cpp` + `emulator/SysCall.h` 的 syscall
> ABI 选择。与 kernel / 工具链无关。**仅在跑 `res_check=on` 的 ELF 时触发**（官方精度流程本用 QEMU，但
> 本环境 `/remote/.../qemu-linx` 不存在，改用 gfrun 才暴露）。

### 现象

`gfrun -f <tail_ocp_fp8 res_check elf> -s softcore.multiThreadNum=4`（单/多线程均然）在 libc 对 stdout
做 `ioctl(1, TIOCGWINSZ)`（stdio/isatty 初始化）时崩：

```
At emulator/SysCall.h EcallAgent:
Bad Syscall Request: syscall(102e2, 1, 5413, ..., 0, 0, 0);   # a1=0x5413=TIOCGWINSZ
```

### 根因

`main.cpp` 按 `directBootSyscallAbi = !addrRet.hosted_runtime` 自动选 ABI：hosted→读 `A7`、
direct-boot→读 `X1`。res_check ELF 被 `HasHostedRuntime`（`ELF.cpp`）**误判成 hosted** → 读 `A7`，
而这个镜像其实是 **direct-boot（syscall 号在 `X1`，`A7` 是 caller-saved 残留旧地址 `0x102e2`）**，于是
把垃圾地址当 syscall 号 → `HandlerTable` 无此项 → `Bad Syscall`。**gfrun 本身支持 ioctl**
（`SysCall.h HANDLER(ioctl)`，含 TIOCGWINSZ），号读对（`X1=29`）即可正常返回；纯粹是 ABI 选错寄存器。
`EcallAgent` 构造注释已预警此坑（"A7 may still contain a prior address/immediate"）。

### 规避（已落地、未提交）

`main.cpp` 加 env 开关 `GFRUN_FORCE_DIRECTBOOT_ABI=1` 强制读 `X1`（direct-boot ABI），使全部 syscall
读对号、res_check 文件 I/O（open/read/write）走通：

```bash
GFRUN_FORCE_DIRECTBOOT_ABI=1 bin/gfrun -f <res_check elf> -s softcore.multiThreadNum=4
```

以此跑通 `tail_ocp_fp8` 4-PE 官方 res_check（`scale=pass MaxAE=0 / output=pass MaxAE=0.011719`）。

### 状态

**未修**（gfrun 侧 `HasHostedRuntime` 误判是根因）；当前用 env 开关规避，仅为在无 QEMU 环境跑精度流程。
正式修复应在 gfrun 侧修正 res_check 直接引导镜像的 hosted/direct-boot 判定，或提供 CLI 覆盖。属跑官方精度
流程（QEMU 缺席时）的前置，与问题15/问题22 的 e8m0 面组合后 `tail_ocp_fp8` 精度方能端到端验证。

## 问题24：非尾轴 4-PE gfrun res_check 落盘随问题规模出现"仅 PE0 段有数据、其余块行为零"，小尺寸复现、大尺寸不复现（**根因未定位**）

> **本条目记录一个观察到的现象，其根因尚未定位。下文所有"可能方向"均为未经证实的猜测，不作结论。**
> 现象在 `SuperScalarModel`（gfrun）多-PE（`multiThreadNum=4`）+ res_check 落盘流程下出现；是否属 gfrun、
> harness（driver/`writeBinaryFile`）、抑或其它，均未确定。**kernel 单 PE 计算结果已验证正确**（见下）。

### 现象

非尾轴 kernel 做 4-PE SPMD（按块行 kb 连续切 4 段，每 PE 写不重叠的 y 行段 + scale 行段）后，用
`GFRUN_FORCE_DIRECTBOOT_ABI=1 gfrun -f <res_check elf> -s softcore.multiThreadNum=4` 跑，落盘的
`output.bin` / `scale_output.bin`：

- **仅 PE0 段（kb 0..numKb/4−1）有正确数据，其余块行为零**，另有末块（kb=numKb−1）出现少量非零"碎片"。
- 输出文件为**满尺寸**（非截断），无 `writeBinaryFile` 的 "faild" 报错。

同一 kernel 随**问题规模**呈现确定性差异（`Axis=512, BlockSize=32` 固定，仅变 `Post`）：

| Post | numN | 落盘覆盖 | 结果 |
|---|---|---|---|
| 64 / 96 / 128 / 192 | 2 / 3 / 4 / 6 | 仅 kb 0–3（PE0 段）+ kb15 碎片 | 挂 |
| 256 | 8 | 全部 16 块行完整、逐字节匹配金标 | 过 |

临界点落在 Post=192 与 256 之间。

### 复现范围（**不限于本次改的 kernel**）

- `nontail_cublas_fp8`（此前 Post=256 记为"精度通过"的 kernel）：**Post≤192 同样挂，签名一致**；其
  Post=256 的"通过"是在阈值之上。故该 kernel 4-PE 精度结论**只在大尺寸成立、非规模无关的可靠证据**。
- `nontail_ocp_fp4`（本次新增 4-PE）：Post=64 挂，签名与 cublas 完全一致。
- `tail_ocp_fp8`（**运行期动态 shape 版**，尾轴/OCP/fp8，2026-09-02）：BS=64、M=130 的 4-PE 落盘出现
  **变体签名**——`output.bin` 满尺寸且逐字节正确，但**第二次落盘 `scale_output.bin` 写空（0 字节）**；
  `multiThreadNum=1` 时 scale 写满 260 字节、逐字节匹配金标。故本现象**不限于非尾轴/cublas/fp4**，尾轴
  fp8 亦触发；且签名可为"整个第二输出文件空"而非仅"部分块行零"，但同属"多-PE + res_check + 规模相关 +
  单 PE 正确"一族。
- 上述各 kernel `multiThreadNum=1` 单 PE 均**写满全部块行/字节、逐字节正确** → kernel 计算逻辑无误。

### 已排除的方向（各带证据）

1. **不是 kernel 代码差异**：`nontail_cublas_fp8` 的 Post=128 与 256 反汇编逐条对比——tile 指令流同序、
   tile 维度（lb0/lb1/lb2=64/32/64）相同；差异仅为内层 numN 循环次数、访存 stride 常量（正比于 Post，各自
   正确）、寄存器分配。无任何会改变计算的代码区别。
2. **不是 write 截断/超时**：输出文件满尺寸，无 "faild"；落盘内容本身就是"半成品"内存状态。
3. **不是通用内存/dump 竞争**：最小**标量**探针（每 PE 用标量写把自己 kb 段写成 `tid+1`）在任意尺寸、
   甚至加大每块计算延迟后，落盘**均完整**（4 段齐全）。
4. **不是单纯 tile 存储路径**：`TLOAD+TCVT+TSTORE` identity-copy tile 探针在 Post=64 落盘**也完整**。
5. **不是每 PE 工作量不均**：挂/过两种情况每 PE 块数相同（PE0 仅多约 3 条收尾指令），均衡。

### 确定性事实（供后续定位）

- 运行时：4 个 PE 均执行完整 `main()`（`readBinaryFile`→kernel→两次 `writeBinaryFile`）；worker 跑完
  `main` 后 park（`test/common/src/group_worker_runtime.c`）。
- `SoftCore::Step()`（`emulator/SoftCore.cpp`）为 round-robin（thread 0 先跑）；`emulator/SoftCore.cpp:379-388`
  注释与逻辑："a passing write from one PE terminates the complete SPMD program instance" —— 任一 PE 触发
  即置全部 `simEnd` 并立即返回。
- `writeBinary.h`/`readBinary.h` **未被本次改动**（RES_CHECK 静音 printf 是既有已提交规避，见问题相关的
  writev 挂起说明），非本问题引入。

### 可能方向（**均为未证实猜测，不作结论**）

- 多-PE 下 tile 存储对 host 侧 `writeBinaryFile` 读取的可见时序，与"首个 PE 到达终点即终止全组"之间的
  交互；是否存在提交/可见延迟未验证。
- 访存 stride/对齐的规模相关性（Post=256 时输入行 stride=512B，恰为常见 bank/对齐宽度）；是否影响多-PE
  写回落 bank 未验证。
- 注：曾试过"仅 PE0 落盘 + 共享 .bss 标志 barrier"，PE0 自旋等标志时**死锁**（标量 `.bss` 跨 PE 读未即时
  可见），说明标量与 tile 存储的跨-PE 可见性可能不同——但这同样只是线索，非定论。

### 状态

**未定位、未修**。kernel 逻辑已由单 PE 逐字节正确佐证；4-PE 下的落盘现象是**规模相关**的，且在
`nontail_cublas_fp8`、`nontail_ocp_fp4` 及**尾轴 fp8 动态 shape 版**上同样存在（跨 axis / scaleAlg /
dstType，非某一 kernel 特有）。当前 4-PE 精度验证仅在足够大尺寸可得到完整落盘。根因待进一步在
gfrun/harness 侧定位（tile 存储可见性、组终止时序、访存对齐等方向均待验证）。

完整可复现 issue（含逐一版本清单 + 最小复现 + 已排除方向 + 确定性事实 + 猜测方向）见
`ISSUE_gfrun_multipe_size_dependent_dump.md`。

## 问题25：gfrun 缺 `ppoll` syscall handler → hosted musl 启动即 `abort()`（全 res_check ELF 崩）（需 emulator 侧解决）【SuperScalarModel issue554·未修】

- **归属**：SuperScalarModel（gfrun / emulator）。
- **暴露组合**（ops-20260904）：model `49547742`；工具链 llvm `1ae4ee39` + Linx-TileOP-API `804eb03` + **musl `af0dfc20`**（新 musl 才引入此启动 poll）。

### 现象

任意 hosted-musl ELF（含全部 res_check harness）在 gfrun 启动即 `abort()`：
```
Bad Syscall Request: syscall(49, 808d420, 3, 808d410, 0, 8, 0);   // hex 49 = 十进制 73 = ppoll
```
发生在 `__init_libc` / `__libc_start_main`（block B66，程序最开头），远早于任何计算或 res_check I/O。

### 根因

新 musl `src/env/__libc_start_main.c:45-50` 启动时对 fd 0/1/2 发 `ppoll(pfd,3,&{0},0,_NSIG/8)` 探测 stdio
是否打开（检 `revents & POLLNVAL`，无效则 open `/dev/null` 顶上）。LinxISA 无 `SYS_poll` → 走 `#else` 发
`ppoll`（号 73）。gfrun `HandlerTable`（`emulator/SysCall.h:1254`）白名单**无 ppoll** → 命中未注册号 → 构造
函数走 `abort()`（`SysCall.h:668`）。老 musl 无此启动 poll，随 musl 升级到 `af0dfc20` 才暴露；三个既有
backup model 分支均无 ppoll handler → 非回归，是新增覆盖缺口。

### 与问题23 的关系

问题23 是**旧栈**上 `GFRUN_FORCE_DIRECTBOOT_ABI` 的 X1/A7 选号误判；本问题是**新栈**上 ppoll 未注册。
二者同属"syscall 处理"域但根因不同。关键：旧流程靠 `DIRECTBOOT_ABI=1` 走 direct-boot **整段跳过 musl
libc init**，因而也顺带躲过了 ppoll；新 model 把该 flag 在 `EcallAgent` 里 `(void)` 忽略、ELF 按 hosted
加载 → `__init_libc` 必执行 → 必发 ppoll。故该 env 不再是可行绕过（res_check 需 hosted 的 openat/read/write
落盘），正解只能补 handler。

### 解除路径（SuperScalarModel）

emulator `HandlerTable`（`SysCall.h`）注册 ppoll handler：stdio 0/1/2 在仿真里恒有效，清各 pollfd 的
`revents`（struct 偏移 6 的 short）=0、返回 0（timeout、无就绪 fd），musl 判无 POLLNVAL 正常继续；纯增量
注册，不影响既有 pass-list。是全 hosted-musl / res_check ELF 的共用前置（放行后 res_check 端到端可跑）。

复现 issue 见 `ISSUE_gfrun_ppoll_libc_startup.md`。

## 问题26：emulator NORM Local TSTORE 不支持打包 4-bit（描述符 size 与源行 stride 均按字节容器口径，应按元素位宽）（需 emulator 侧解决）【SuperScalarModel issue557·未修】

- **归属**：SuperScalarModel（emulator）。
- **复现入口**：`dynamic_mx_quant_tail_ocp_fp4` data pass 末 `TSTORE`（打包 fp4 输出，RowMajor/NORM）。

### 现象

打包 fp4 tile 的 Local TSTORE（NORM 布局）两处按字节容器口径处理，先崩描述符、放行后行错位：

- **(a) 描述符**：`AccumulateBlockInfo.cpp` `IsLegalLocalTileDescriptor` 用 `size == rows*cols*BytesOf`（`BytesOf(FP4)=1`）→ gfrun 崩 `Local TSTORE requires one legal source Tile descriptor`（`size` 与 `rows*cols*1` 不符）。
- **(b) 源行 stride**：`TMAEngine.cpp` `ExecuteTSTORE` NORM 分支 `srcRowWidth = totalCol*eleSize`（`eleSize=1` 字节容器口径）→ 每源行步长为 packed 的 2×，仅 tile 第 0 行落对、其余行全错位（fp4 512×256 4-PE `output=fail MSE=8.4`；8 个正确行 = 各 PE 各 tile 的第 0 行 [0,64,…,448]）。

### 根因（对照 pto-spec）

pto-spec 对打包 4-bit 用**元素位宽**而非字节容器：

- `DerivedTileRows`（`asl/tile/model/shape/rows-columns.asl`）：`rows = capacity_bits DIVRM (columns × TileElementBits(dt))`，`TileElementBits(E2M1X2)=4` → 描述符 size 校验须按 `(rows*cols*TileElementBits + 7)/8`，非 `BytesOf`。
- `TileMemoryElementAddress`（`asl/tile/model/memory/addressing.asl`）：four-bit `offset = element DIVRM 2`（2 元素/字节），`TileMemoryElementHighNibble = element MOD 2 == 1` → NORM 源行 stride 须为 `floor(totalCol × 0.5)` 打包字节。

CUBE TSTORE 分支已有 `packed ? (validCol+1)/2` 打包处理，**NORM 分支缺**。列内 nibble packing（`EleOffset`=idx/2 + `EleDataExtract` nibble mask）本就正确，唯描述符 size 与源行寻址按字节。

### 解除路径（SuperScalarModel）

`IsLegalLocalTileDescriptor` 的 NORM/RowMajor size 校验改按元素位宽（`(rows*cols*ElementBitsOf+7)/8`；字节类型 `ElementBitsOf==BytesOf*8` 等价不变）；`ExecuteTSTORE` NORM 分支 `srcRowWidth` 与 GM 回退 stride 改按打包元素地址（`floor(totalCol × EleRealSize)`）。放行后 `tail_ocp_fp4` 512×256 4-PE 逐字节 pass，gfrun 自检 pass-list 349/349 无回归。

### 影响

所有把打包 4-bit（FP4/FP4_1/HIF4/INT4/UINT4）经 NORM Local TSTORE 落盘的路径。

复现 issue（组件清单以 Bench PR#111 给出 + 前置依赖 + 复现步骤 + spec 依据）见 `ISSUE_gfrun_norm_tstore_packed4bit.md`。

## 问题27：TCVT fp→E2M1/E1M2 编码最近邻漏 code 0（小值抬到最小正档，偏离 RNE）；且参考模型无 E2M1 目的编码器（需 emulator + pto-spec 侧解决）【SuperScalarModel issue558·未修】

- **归属**：SuperScalarModel（emulator）+ pto-spec（参考模型缺口）。
- **复现入口**：`dynamic_mx_quant_tail_ocp_fp4` data pass 末 `TCVT(fp32→fp4)`，小商值 `|x/scale| < 0.25`。

### 现象

`fp32→fp4(e2m1)` 的小值应 RNE 到 `0.0`（如 `0.19→0.0`），emulator 却输出 `0.5`（最小正档）。fp4 512×256 4-PE `MSE=0.022`（`MaxAE=0.5`，1 LSB）；compare 判据 `MSE<0.1` 仍 pass，故非阻塞、但非 byte-exact、偏离 golden。

### 根因（对照 pto-spec）

`CubeEngine.cpp` `DataFormatCvt` 的 E2M1/E1M2 编码分支最近邻搜索从 `code 1`（0.5）起、跳过 `code 0`（0.0）→ 任意非零小值最小编成 `0.5`。而 E2M1X2 格式 `has_zero=TRUE`、值集含 `0.0`（`asl/arch/data-types/formats/e2m1x2.asl`），TCVT 默认 float→float 为 RNE（`asl/.../TCVT.asl` `InstructionContractDefaultRounding_TCVT`），故 `|v|<0.25` 应舍到 `0.0`。

### spec 缺口

参考模型 `ReferenceFP8Encoding`（`asl/arch/profile/matrix-quantization.asl`）断言 `dst==E4M3 ‖ HiF8`，**不含 E2M1/E1M2** → `ReferenceMatrixFloatingEncoding` 对 fp4 目的会 assert，pto-spec 参考模型**未定义 fp→E2M1 目的的编码/舍入**。emulator 的 E2M1 编码属实现扩展，须按 TCVT RNE 契约补齐，spec 侧须补 E2M1/E1M2 目的的参考编码器。

### 解除路径

emulator 最近邻搜索纳入 `code 0`（从 `code 0` 起，ties-to-even）；pto-spec 为 E2M1/E1M2 目的补 RNE 参考编码器。

复现 issue（组件清单以 Bench PR#111 给出 + 前置依赖 + 逐元素证据 + spec 依据/缺口）见 `ISSUE_gfrun_tcvt_e2m1_rne_code0.md`。
