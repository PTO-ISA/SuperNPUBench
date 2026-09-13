# [Issue] emulator 把就地 `TSEL` 建模成「显式三源 / 两拍 B.IOT」，与工具链实发的「单拍 B.IOT + dst 就地 false-source」不符 → gfrun 崩

## 一句话

`TSEL(dst, mask, trueSrc)`（`dst = mask ? trueSrc : dst_prior`，false-source 就是 dst 自身）经工具链
lower 成**一条** `B.IOT`：两个 tile 源（mask、trueSrc）+ `->dst` 就地。官方发布版 emulator 却按
**显式三源**建模，于是 validate 侧要求首拍 `dsts.empty()`、execute 侧读 `srcTile[2]` → 一条 TSEL 即崩。

## 最简复现探针

`test/kernel/quant/dynamic_mx_quant/src/tsel_inplace_probe.cpp`（Makefile `TYPE=TSEL_INPLACE_PROBE`）——
整个 kernel 只有一条 TSEL，不牵涉 dynamic_mx_quant 的任何 scale 计算：

```cpp
tile_u16 x, mask, k;
TEXPANDS(x,    (uint16_t)0xABCD);   // 就地 false-source 种子
TEXPANDS(mask, (uint16_t)1);        // 全真谓词
TEXPANDS(k,    (uint16_t)0x1234);   // true-source 常量
TSEL(x, mask, k);                   // 就地: x = mask ? k : x_prior
TSTORE(gy, x);
```

## 复现基线

| 仓库 | HEAD | 说明 |
|---|---|---|
| SuperNPUBench | 3ecd9083 | pr: https://github.com/PTO-ISA/SuperNPUBench/pull/49 |
| SuperScalarModel（gfrun） | 63dbb5a2 |
| linx-toolchain-build | e6a31ef |
| Linx-TileOP-API | cdeb624a |

## 实测报错（ `bin/gfrun` @ 63dbb5a2）

```
gfrun: illegal instruction: ASSERTION FAILED:
  inst->srcs.size() == 3 && inst->dsts.empty() &&
  IsCompatibleLogicalTile(inst->srcs[1], ...) &&
  IsCompatibleDataTile(inst->srcs[2], ...) &&
  "select first B.IOT requires mask then true/source Tile"
, func ValidateCompareSelectTepl, file emulator/engine/AccumulateBlockInfo.cpp:383
```

同一条 ELF 在带修复的 emulator 上 `R2 = 0` 跑通（12 block / 61 inst）——隔离出缺陷在
**emulator 的 TSEL 建模层**，非 kernel、非 ISA 编码。

## Ground truth：工具链实发的 TSEL lowering

`.../tileop-api/jcore/template_asm.hpp` `TSEL(dst,src0,src1)` 汇编体 = **一条** `B.IOT`：

```asm
BSTART.TEPL 0, 26, %D1
B.DIM %2, 0, ->lb0            ; validCol
B.DIM %3, 0, ->lb1            ; validRow
B.DIM zero, %c4, ->lb2        ; physicalCol = Cols
B.IOT %5, %6, mask=1111, last, ->%0<%Z7>   ; %5=mask, %6=trueSrc, ->%0=dst 兼就地 false-source
```

**两个源 tile（mask、trueSrc）+ dst 就地兼隐式 false-source；无第二拍、无独立第三源。** 与 `TSELS`
（scalar false-source）同族形态，属 ISA 固有 lowering。

## emulator 现状（63dbb5a2，两侧都假设显式三源）

- **validate** `AccumulateBlockInfo.cpp:383` `ValidateCompareSelectTepl` 首拍分支强制
  `inst->srcs.size()==3 && inst->dsts.empty()`（首拍不许带 dst、要独立第三源）→ 工具链单拍带 dst(`->%0`) 撞断言。
- **execute** `TEPLEngine.cpp:1118` `ExecuteTSEL` 在 `:1126` **无条件**读 `block->srcTile[2]`（假设三源）→
  工具链只 2 源 → 越界（validate 放宽后才会暴露）。

## 修法（emulator validate + execute 双侧接受单拍就地形态）

1. **validate**：首拍 `inst->dsts.empty()` 放宽为 `inst->dsts.size() <= 1 &&
   (inst->dsts.empty() || inst->dsts[0]->size >= dataBytes)`，接受 dst 融合到首拍 B.IOT。
2. **execute**：`ExecuteTSEL` 的 `block->srcTile[2]` 改为
   `block->srcTile.size() >= 3 ? srcTile[2] : dstTile[0]`，false-source 缺省取 dst 就地。
