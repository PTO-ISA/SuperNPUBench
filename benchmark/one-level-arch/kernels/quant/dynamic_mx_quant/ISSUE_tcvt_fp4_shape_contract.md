# [Issue] `TCVT_T` TileLogicalShapeMatch：当前工具链把「宽类型 → 打包 fp4」的 TCVT 形状契约落在**编译期 `static_assert`**，结构性必崩

## 摘要

`dynamic_mx_quant` 的 OCP fp4 输出 kernel（`TYPE=TAIL_OCP_FP4`）在 data 路径末尾用 `TCVT` 把
fp32 tile 转成打包 fp4（`__fp4_e2m1x2`）。**当前工具链（PTO 0.58.1）把这条 TCVT 形状契约放在编译期
`static_assert`**，源文件连 `.o` 都编不出来：

```
jcore/template_asm.hpp:115:3: error: static assertion failed ...
  'Tile<...,__fp4_e2m1x2, 8, 32,...>::Cols == Tile<...,float, 8, 64,...>::Cols'
  TCVT source and destination must have identical physical Rows/Cols
  (PTO 0.58.1 TileLogicalShapeMatch)
```

该 `static_assert` 的**自身注释明确说明**这是把运行期的 `TileOperandsLegal_TCVT` /
`TileLogicalShapeMatch` 契约「提前到编译期拒绝」（"reject it at compile time"，template_asm.hpp:110-114）。

根因是打包 fp4（1 字节 = 2 个 fp4）的输出 tile physical 列宽天然为 fp32 源的一半，而
TileLogicalShapeMatch 要求 TCVT 的 src/dst **physical Rows 与 Cols 全等** —— 打包窄化 TCVT
**结构性无法满足**。缺陷在 **工具链头（`Linx-TileOP-API` 的 `template_asm.hpp`）+ emulator 建模层
（`SuperScalarModel`）双侧的过严校验**，非 ISA、非 kernel 逻辑。

> **本条 issue 是编译期崩**，复现**只需工具链编译**，**不需要 `SuperScalarModel`（gfrun/gfsim）**、
> **不需要任何本地改动**（HEAD 的 data 路径 `TCVT(oq, xf)` 即触发）。

## 最小 TCVT 探针实证（编译 + gfrun 执行，2026-08-21）

为把「不一致→编译崩、一致→数据错」两条结论从推断升成实测，写了**只含一条 TCVT** 的最小探针
`src/fp4_shape_probe.cpp`（`TLOAD(fp32) -> TCVT(fp32->打包fp4) -> TSTORE`），几何对齐真实 kernel
（`R=8, PW=64`，源 `tile_f` physical `[8,64]`）。目标 fp4 tile 的 physical Cols 由 `-DWIDEN` 切换：
默认 `OCOL=PW/2=32`（打包正确声明）、`WIDEN` 时 `OCOL=PW=64`（加宽骗过断言）。Makefile 入口
`TYPE=FP4_SHAPE_PROBE`（`WIDEN=on` 追加 `-DWIDEN`）。

**变体 A（不一致，`OCOL=32`，打包正确）→ 编译失败（唯一实质错误）**

```bash
make TESTCASE=dynamic_mx_quant TYPE=FP4_SHAPE_PROBE          # grep -c "error:" == 1（唯一实质错误）
```
唯一报错即目标 static_assert（无其它实质错误）：
```
jcore/template_asm.hpp:115:3: error: static assertion failed due to requirement
 'Tile<...,__fp4_e2m1x2, 8, 32,...>::Cols == Tile<...,float, 8, 64,...>::Cols'
 TCVT source and destination must have identical physical Rows/Cols (PTO 0.58.1 TileLogicalShapeMatch)
```
`Rows` 8==8 过、`Cols` **32≠64** 崩，连 `.o` 都出不来 —— **坐实「不一致→编译期崩 Cols」**。

**变体 B（一致，`OCOL=64`，加宽）→ 编译通过、gfrun 跑到底、但数据错**

```bash
make TESTCASE=dynamic_mx_quant TYPE=FP4_SHAPE_PROBE WIDEN=on res_check=on   # 编过，出 .elf
gfrun -f <elf>                                                              # R2 = 0
```
反汇编实证加宽把 dst physical col 定成 64（`TCVT` 块内 `C.B.DIMI 64, ->lb2`、`B.DATR e2m1x2, byte0`）
→ **每个 fp4 摊进独立字节的低半字节，未打包**。喂交替 `1.0 / 4.0`（fp4 nibble `0x2 / 0x6`）逐行 64 值，
`output.bin` 逐字节实测：

| | 每行字节数 | 内容 | 总字节 |
|---|---|---|---|
| **golden（正确打包）** | 32 | 每字节 `0x62`（低 nibble=1.0→0x2、高 nibble=4.0→0x6），×32 | 256 |
| **变体 B 实测输出** | 64 | `02 06 02 06 …`（前 32 字节各装一个 fp4 低 nibble、高 nibble 恒 0）+ 后 32 字节全 `00` | 512 |

**加宽输出与 golden 三重不符**：(1) 字节数翻倍（512 vs 256）；(2) 未打包（每字节 1 个 fp4，高半字节丢失）；
(3) 每行只落 32 个源值、余 32 值丢失补零。**坐实「一致（加宽）→ 数据错」** —— kernel 侧加宽规避不可取。

> 两变体合起来给出**闭环实证**：保持打包正确声明则编译期结构性必崩；加宽到 physical 相等骗过断言则
> gfrun 能跑但输出布局被破坏。故合法 kernel 声明无解，修复必在工具链头 + emulator 双侧（见「修复建议」）。

## 环境

| 项 | 值 |
|---|---|
| 编译仓 | SuperNPUBench `benchmark/one-level-arch` |
| 工具链 | `linx_blockisa_llvm_musl`，clang-15.0.4，自报 **PTO 0.58.1**，target `linx64v5-linux-musl` |
| 触发头 | `.../lib/clang/15.0.4/include/tileop-api/jcore/template_asm.hpp:115`（`TCVT_T` 的 `static_assert`） |
| 触发 kernel | `kernels/quant/dynamic_mx_quant/dynamic_mx_quant_tail_ocp_fp4.hpp:206`（`TCVT(oq, xf)`） |
| 复现根目录 | `benchmark/one-level-arch/test/kernel/quant/dynamic_mx_quant` |

## 复现命令（编译即崩，无需 Model、无需本地改动）

```bash
export COMPILER_DIR=<toolchain>/output/linx_blockisa_llvm_musl/bin
export LINX_SYSROOT=<toolchain>/output/linx_blockisa_llvm_musl/sysroot/usr
cd benchmark/one-level-arch/test/kernel/quant/dynamic_mx_quant
make TESTCASE=dynamic_mx_quant TYPE=TAIL_OCP_FP4 res_check=on   # 编译即失败，不到链接/反汇编
```

崩点 `TCVT(oq, xf)`（:206）在 kernel 的 data 路径、HEAD 即存在，非任何本地改动。

## 报错（完整）

```
jcore/template_asm.hpp:115:3: error: static assertion failed due to requirement
 'Tile<pto::Location::Vec, __fp4_e2m1x2, 8, 32, pto::BLayout::RowMajor, 8, 16, ...>::Rows ==
  Tile<pto::Location::Vec, float,        8, 64, pto::BLayout::RowMajor, 8, 32, ...>::Rows &&
  Tile<pto::Location::Vec, __fp4_e2m1x2, 8, 32, ...>::Cols ==
  Tile<pto::Location::Vec, float,        8, 64, ...>::Cols':
  TCVT source and destination must have identical physical Rows/Cols (PTO 0.58.1 TileLogicalShapeMatch)
  static_assert(tile_shape_out::Rows == tile_shape_in::Rows &&
                tile_shape_out::Cols == tile_shape_in::Cols, ...);
note: in instantiation of function template specialization
  'TCVT_T<Tile<...__fp4_e2m1x2, 8, 32...>, Tile<...float, 8, 64...>>' requested here (template_asm.hpp:5686 TCVT_T(dst, src))
note: ... 'TCVT<...>' requested here (dynamic_mx_quant_tail_ocp_fp4.hpp:206 TCVT(oq, xf))
note: ... 'dynamic_mx_quant_tail_ocp_fp4<8, 64, 32, __fp4_e2m1x2, __bf16>' requested here (tail_ocp_fp4.cpp:30)
```

## 根因分析

### Rows / Cols 数值来源（模板常量，非运行期反推）

`static_assert`（template_asm.hpp:115）比的是两个 `Tile` 类型的 `::Rows` / `::Cols` **静态常量**，
它们直接等于 `Tile` 模板的**第 3、4 个参数** `Rows_` / `Cols_`（`pto_tile.hpp:604-611` 模板头、
`:638-639` `static constexpr int Rows = Rows_; Cols = Cols_;`）：

```cpp
template <Location Loc_, typename Element_, const int Rows_, const int Cols_,
          const BLayout BFractal_ = RowMajor,
          const int RowValid_ = Rows_, const int ColValid_ = Cols_, ...>
struct Tile { static constexpr int Rows = Rows_; static constexpr int Cols = Cols_; ... };
```

kernel（`dynamic_mx_quant_tail_ocp_fp4.hpp:96-103`）对 `M=8, N=64, BlockSize=32` 声明
（`TileM=8`，`PW=⌈BlockSize/64⌉×64=64`）：

| tile | 声明 | Rows_ | Cols_ | ValidRow | ValidCol |
|---|---|---|---|---|---|
| `tile_f`（fp32 源） | `Tile<Vec, float, TileM, PW,   RowMajor, TileM, BlockSize>` | **8** | **64**（=PW） | 8 | 32 |
| `tile_o`（fp4 目标） | `Tile<Vec, OutT,  TileM, PW/2, RowMajor, TileM, BlockSize/2>` | **8** | **32**（=PW/2） | 8 | 16 |

`static_assert` 逐 conjunct：

| conjunct | out（fp4） | in（fp32） | 结果 |
|---|---|---|---|
| `Rows == Rows` | 8 | 8 | ✓ 相等 |
| **`Cols == Cols`** | **32**（PW/2） | **64**（PW） | **✗ 崩** |

**编译期崩在 `Cols`（32≠64），Rows 两边都是 8、匹配。**

### 结构性根因：打包 fp4 的 physical 列宽天然半

kernel 的 `tile_o` **故意**声明成 physical `PW/2` 列（valid `BlockSize/2`）——因为 `__fp4_e2m1x2`
每字节打包 2 个 fp4，装 `BlockSize` 个 fp4 逻辑值只占 `BlockSize/2` 字节列。这是**正确**的打包描述。
而 TileLogicalShapeMatch 要求 TCVT 的 src/dst physical Cols **相等**，即把 TCVT 当作 physical
shape-preserving 的算子。**「宽类型 → 打包窄类型」是列宽减半的 pack，本质无法满足 Cols 相等**：
- 保持 `tile_o` 打包（Cols=PW/2=32）→ Cols 32≠64，编译期崩（本 issue）。
- 强行加宽 `tile_o` Cols→PW=64 骗过断言 → 输出 fp4 布局被破坏、data 错（见「反证」）。

无合法 kernel 声明可过，故**修复必在工具链头 + emulator 侧**，kernel 无损规避不可行。

**更锐的根因（sub-byte 打包的字节会计缺陷，源码实测）**：linx 不暴露单个 4-bit fp4，只有打包对
`__fp4_e2m1x2`（`type.hpp:62` `type_traits_base<...,8>` → **bits=8**，storage `uint8_t`）。emulator
`BytesOf(FP4)=HF4_DATA_WIDTH=1`（`DataType.h:203/215`，注释「4bits still need 1 byte」）——**没有
「每 fp4 半字节」的表示**，故 fp4 的「一个元素」被迫等于打包字节。于是 64 个 fp4 的向量 physical
col=32（字节），其 fp32 源 col=64 → 契约比 32 vs 64 必不等。这不是「契约写得太严」的表面问题，而是
**sub-byte 打包在「col 按元素计 + BytesOf≥1」的会计体系里无法自洽**：要么 col 记打包字节数（=32，与源不
等）、要么加宽记逻辑 fp4 数（=64，但每 fp4 占满一字节 → 数据非打包，见变体 B 实证）。两条路都错。

### 同一契约在编译期与 emulator 运行期各落一次

TileLogicalShapeMatch（src/dst physical Rows 与 Cols 全等）在**两层**各查一次，各从**不同来源**读
physical col：

| 层 | physical col 从哪读 | 若参与校验则崩的维度 |
|---|---|---|
| **编译期** `static_assert`（本 issue，`template_asm.hpp:115`） | **声明的模板** `Cols_`（fp4=32 / fp32=64） | **Cols 32≠64** |
| **运行期** `ValidateOperandContract`（emulator `Block.cpp:1039`） | emit 的 lb2 / inherit（不读模板声明） | Row 8≠4 或 Col 64≠32（随 lb2） |

当前工具链下编译期先崩，打包声明根本到不了 emulator。**但只放宽编译期 static_assert 并不够**——放开后
打包 TCVT 块进入 emulator，同一契约在运行期 `ValidateOperandContract()` 再崩一次。故修复须双侧（见下）。

## emulator 运行期同契约（放宽编译期后的必经第二道）

emulator 在**块解码组装阶段**（任何 Execute 之前）由 `ValidateOperandContract()`（`Block.cpp:1039`，
`SetBlockIsComplete()`@:1136 调用）查同一契约：

```
ASSERTION FAILED: ... srcTile[0]->tileInfo->row == dstTile[0]->tileInfo->row &&
                      srcTile[0]->tileInfo->col == dstTile[0]->tileInfo->col && ...
  "PTO 0.58 TCVT requires matching source/destination logical shapes"
  func ValidateOperandContract, file isa/Block.cpp:1039
```

运行期不读模板 Cols，而按 emit 决定 col，再 `row = size/(col × BytesOf(elem))`（`Block.cpp:1384-1386`）
反推 row（`BytesOf(FP4)=HF4_DATA_WIDTH=1`，`DataType.h`）。对打包 fp4 TCVT（dst size=256B）：

| fp4 TCVT emit | dst.col | dst.row = size/(col×1) | 崩的 conjunct |
|---|---|---|---|
| 不发 lb2 → inherit src | 继承 src=64 | 256/(64×1)=4 | **row（8≠4）** |
| 发 `->lb2 32` | 32 | 256/(32×1)=8=src | **col（64≠32）** |

选哪个 col 都有一个 physical 维度 ≠ src（打包改变 col↔row↔size 关系），**结构性**。而两边
`validRow`/`validCol` 恒相等（8/8、32/32），只比 valid 维度即可放过。

> 佐证：加宽变体（Cols=64，见开头探针变体 B）dst size=512B、col=64→row=8=src，**运行期校验过、gfrun
> R2=0**——即当前 emulator 只在 physical 维度不等时崩；这正是打包声明（Cols=32）放开编译期后会撞上的。

### 对照：非打包 fp8 全过

OCP fp8 探针（e4m3，1 字节/元素、非打包）`R2=0`；fp8 输出 TCVT 的 physical row/col 不减半 → 过校验。
佐证缺陷**专属打包类型**（`bits<8` 的打包 dtype）。

## 反证：kernel 侧「加宽 dst tile 骗过契约」会产出错误数据（修复必在工具链/emulator）

把 fp4 输出 tile 的 physical 列 `PW/2 → PW`、valid 列 `BlockSize/2 → BlockSize`（声明成「未打包」宽）：
- 编译期：`Cols` 变 64 == src 64 → static_assert 过。
- 运行期：dst size=8×64×1=512B，col=64 → row=512/64=8=src → 断言过、gfrun 跑到底 R2=0。

但**输出 data 错**（`M=8,N=64,BS=32` 实测）：`scale_output` 逐字节对，**`output` vs golden 不同**——
golden 每字节打包 2 个 fp4 半字节（`byte0=0x26`=nibble 2,6）；加宽输出把每个 fp4 摊进一整字节、高半字节
恒 0（`byte0=0x06`、`byte1=0x02`），32 个 fp4 解包成 32 字节、宽度翻倍越界。**故加宽不可取，kernel 必须
保持 `tile_o` 打包（physical `PW/2`、valid `BlockSize/2`）。**

> 上述 kernel 级反证已由「最小 TCVT 探针」变体 B（`FP4_SHAPE_PROBE WIDEN=on`）**独立复现坐实**
> （2026-08-21，见开头「最小 TCVT 探针实证」）：单条 TCVT、交替 1.0/4.0 输入，加宽输出 512B/每行 64B
> 未打包（`02 06 …`+补零），对 golden 256B/每行 32B 打包（`0x62`×32）三重不符。

## 修复建议（双侧放宽，删 physical row/col 强等、只留 valid + layout）

同一契约现落在两处，须**双侧**放宽，只保留 `validRow==validRow` + `validCol==validCol` + `layout`
+ physical ≥ valid 的下界健全性检查（fp4 两边 valid 维度恒相等，放宽后过校验且不影响正确性）：

1. **工具链头**（本 issue 的编译期崩点）：`Linx-TileOP-API` 的 `jcore/template_asm.hpp:115-118`
   `TCVT_T` 的 `static_assert` —— 删掉 `Rows==Rows && Cols==Cols` 的 physical 强等，改为只断言
   valid 区包含关系（:119-124 已有的 `Cols >= ValidCol` 那条保留）。**这是让 fp4 data 路径能编译的必要条件。**
2. **emulator**（运行期崩点）：`SuperScalarModel` 的
   `ValidateOperandContract()`（`Block.cpp:1042-1044`）—— 删掉 physical `row==row` / `col==col`
   两条 conjunct，保留 `validRow`/`validCol`/`layout`。**这是让放宽编译期后编译出的 elf 能跑到底的必要条件。**

> **注**：历史提交 `eaa3dfe7 "fix(tile): relax TCVT legality to valid shape only"` 做过 emulator 侧
> 放宽，但已被 force-push 从 `origin/feat/pto-v058-adaptation` 重写丢弃、**不可作为依据**；建议在
> 当前分支**重新独立落地**，勿 cherry-pick 已丢弃的提交。

**保留不动**：`Block.cpp:1107` 的 Local TMOV 分支 row/col 相等校验（非 TCVT，不在本 issue 范围）。

## 影响面

- **编译期阻断**（PTO 0.58.1）：`dynamic_mx_quant_tail_ocp_fp4` 的 fp4 data 路径连 `.o` 都
  编不出，`TAIL_OCP_FP4` 无法产出 elf。
- **运行期阻断**（放宽编译期后）：打包 TCVT 块进 emulator，`ValidateOperandContract` 在块解码组装阶段崩。
- **泛化**：任何「宽类型（fp32/bf16/half）→ 打包 fp4（或其他 `bits<8` 打包 dtype）」的 TCVT 都被
  此契约结构性挡住——这是 MX-quant fp4 输出量化的必经手法。
- 不影响非打包输出（fp8 e4m3/e5m2 等 1 字节/元素）。
