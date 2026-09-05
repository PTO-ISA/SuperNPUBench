# TCVT float→FP4 写侧(值编码 + nibble 打包)缺失 —— `31f7a8f`(#314)被 `930d9981` 连坐回退，官方未恢复【SuperScalarModel issues454·已修复】

> **已修复**：官方 PR #510（`eededa48`，Refs #454）把 FP4 写侧四件套（E2M1/E1M2 有限值编码 + nibble 打包 + bit-based row 派生 + `IsFourBitDataType`）重放回 `codex/consolidate-post-main-fixes-20260903`（当前 model `49547742` 已含）。


- 仓库：`github.com/LinxISA/SuperScalarModel`
- 复现基线：`origin/main`、`origin/codex/pr-0.58.4-shared-model`（两者一致，均缺失）

## 摘要

对打包 FP4（`DataType::FP4` / `FP4_1`，E2M1X2/E1M2X2，每字节 2 个 4bit 元素）目的的 `TCVT`，在当前
远端各分支上**缺失写侧**两块：

1. **值编码**：fp→FP4 走通用 softfloat 编码，把 E2M1/E1M2 的**全 1 码**当 NaN/Inf 保留，导致顶档
   最大有限值被向下舍一档（`6.0` → 码 6=`4.0`，应为码 7=`6.0`）。
2. **nibble 打包**：`ExecuteTCVT` 无「两个 4bit 元素打进一字节」的逻辑，每元素独占一字节低 nibble、
   高 nibble 恒 0。

这两块曾由 `31f7a8f`（#314）完整实现，但被 `930d9981` 一次宽范围 revert 连坐删除，之后官方任何分支
都未恢复。

## 最小复现探针（纯 origin 代码 + 官方工具链即可编译运行）

`SuperNPUBench .../dynamic_mx_quant/src/fp4_shape_probe.cpp` —— 单条计算 `TLOAD → TCVT(fp32→
__fp4_e2m1x2) → TSTORE`。用 `WIDEN=on` 变体（输出 tile 与源同 `ValidCol`，专门隔离**模型值编码**，
不涉及打包宽度）：

输入 row0 前 8 个 fp32：`[0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0]`
期望 E2M1 码：`[1, 2, 3, 4, 5, 6, 7, 0]`

```
$ make TESTCASE=dynamic_mx_quant TYPE=FP4_SHAPE_PROBE WIDEN=on res_check=on
$ gfrun -f dynamic_mx_quant_fp4_shape_probe.elf
# output.bin row0 前 8 字节：
实际 = [1, 2, 3, 4, 5, 6, 6, 0]          # ← 6.0 编成码 6(=4.0)，应为码 7(=6.0)❌ MaxAE=2.0
高 nibble = [0, 0, 0, 0, 0, 0, 0, 0]     # 未打包（此处由输出宽度决定，见下）
```

**`6.0` 被编成码 6（4.0）而非码 7（6.0）**：顶档全 1 码被通用 softfloat 视作 NaN/Inf 保留、向下舍一
档，其余值正确。这直接、无歧义地暴露**模型缺 E2M1/E1M2 有限值编码**，且与工具链无关。

端到端 `dynamic_mx_quant_tail_ocp_fp4`（bf16 in / fp4 out / e8m0 scale，SPMD 4-PE）上叠加打包缺失：

```
dynamic_mx_quant_tail_ocp_fp4:
  scale  = pass  (MSE=0.000000, MaxAE=0.000000)
  output = fail  (MSE=7.831163, MaxAE=10.000000)   # 未打包 + 顶档编码错 + 相邻 block 覆盖
```

## 当前远端缺什么

实测 `IsFourBitDataType` 计数（fp4 写侧标记）均为 0：

```
$ git show origin/main:emulator/engine/TEPLEngine.cpp | grep -c IsFourBitDataType                   # 0
$ git show origin/codex/pr-0.58.4-shared-model:emulator/engine/TEPLEngine.cpp | grep -c IsFourBitDataType  # 0
$ git show origin/feat/cube-mx-scale-dequant:emulator/engine/TEPLEngine.cpp | grep -c IsFourBitDataType    # 0
$ git show origin/main-group-insts-branch:emulator/engine/TEPLEngine.cpp | grep -c IsFourBitDataType       # 0
```

远端**有 fp4 读侧**（`TMAEngine::ExecuteTSTORE` `packed = CubeCellElementBits(dataType)==4`、
`CubeEngine::DataFormatCvt` 的 `srcType==FP4` 解码），但**缺写侧**：

| 文件 | 缺失的写侧 |
|---|---|
| `emulator/engine/CubeEngine.cpp` `DataFormatCvt` | 无 `dstType==FP4/FP4_1` 的 E2M1/E1M2 有限值编码分支 |
| `emulator/engine/TEPLEngine.cpp` `ExecuteTCVT` | 无 `IsFourBitDataType(dstType)` 两 nibble 打包分支 |
| `isa/ISACommon/DataType.h` | 无 `IsFourBitDataType`（有等价的 `ElementBitsOf`） |

读侧存储打包是**必要非充分**：TCVT 从未把正确编码 / 打包的数据写进 tile 寄存器，读侧再怎么 `packed`
也读不出正确结果。

## 回归根因（已核实）

| 提交 | 作者 / 日期 | 事件 |
|---|---|---|
| `31f7a8f19165a949e1c6899ec7c4c29217d56b30` | jialewang-316server / 2026-08-21 | `fix(gfrun): derive packed four-bit TCVT rows and pack FP4 output`（**Refs #314**）：加 `ElementBits`+`IsFourBitDataType` + `DataFormatCvt` E2M1/E1M2 编码 + `ExecuteTCVT` 两 nibble 打包 + `Block.cpp` 行派生 |
| merge PR **#322** | — | fp4 写侧被并入 "unify TCVT/ARGMAX/FPATR, cooperative TMATMUL FP16/BF16, and TSORT" 大包（注：#314=31f7a8f 本体，#322=承载它的 unify merge PR） |
| `930d998163a6e520be743c57f7340fe770b216d0` | jiale-wangOwO / 2026-08-22 | `Revert "…unify…"`：整包回退。**其 diff 删除了 fp4 写侧**（实测删除行含 `IsFourBitDataType`×2、`dstType == DataType::FP4`×2、`E2M1_VALUES`×2） |

回退之后官方任何分支都没把这块单独恢复。**这与 [[ISSUE_e8m0_tcvt_regression]] 是同一次 `930d9981`
revert 的两个连坐受害者**（同一 revert 也删了 `#253` e8m0）；真正该回退的是 cooperative TMATMUL /
TSORT / FPATR，独立且已测的 `#314` / `#253` 是被误连坐删除。

## 修复：把 `31f7a8f` 的 FP4 写侧作为独立提交重放到官方

只重放 fp4 相关 hunk，不含被 revert 的 cooperative TMATMUL / TSORT：

| 文件 | 恢复内容 |
|---|---|
| `isa/ISACommon/DataType.h` | `IsFourBitDataType`（FP4/FP4_1/HIF4/INT4/UINT4）；行派生所需的 4bit 位宽已有 `ElementBitsOf` 提供 |
| `emulator/engine/CubeEngine.cpp` `DataFormatCvt` | `dstType==FP4/FP4_1` 编码分支：E2M1/E1M2 有限值表 + RNE 就近选择 + **顶档饱和到码 7（6.0 / 1.75），不当 NaN/Inf** |
| `emulator/engine/TEPLEngine.cpp` `ExecuteTCVT` | `IsFourBitDataType(dstType)` 打包块：偶元素低 nibble、奇元素高 nibble，每字节存 1 |
| `isa/Block.cpp` `UpdateDstTileInfo` | 打包目的的行派生按元素位宽（`ElementBitsOf`）而非 `BytesOf` |

**打包宽度**（工具链侧 fp4 tile size 与模型 4bit 元素对齐）不在本 issue 范围，见
[[ISSUE_linx_tileop_fp4_tile_size_bits]]；上面的最小探针用 `WIDEN=on` 正是为了绕开该宽度问题、单独
坐实模型值编码回归。

## 建议

1. **上游恢复**：把上表 FP4 写侧 hunk 作为独立提交重放到 `origin/main`（引 `#314` / `31f7a8f`），
   修复 main 现存的 FP4 写侧回归。
2. **回退纪律**：宽范围 `Revert "unify …"` 应拆分，避免把独立且已测的修复（如 `#314` FP4 写侧、
   `#253` e8m0）连坐删除。参见 [[ISSUE_e8m0_tcvt_regression]]。
