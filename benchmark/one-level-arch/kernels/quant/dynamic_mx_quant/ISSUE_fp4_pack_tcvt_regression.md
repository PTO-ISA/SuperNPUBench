# TCVT float→FP4 打包缺失 —— `31f7a8f` 的实现被 `930d9981` 连坐回退，且 main 从未恢复

## 摘要

对打包 FP4（`DataType::FP4` / `FP4_1`，E2M1X2/E1M2X2，每字节 2 个 4bit 元素）目的的 `TCVT`
在当前 `origin/main` 与侧支基线 `d8903938`（含本地 `ad288c24`）上**不再打包**：TCVT 逐元素
按 1 字节落盘（偶元素落低 nibble、高 nibble 恒 0），下游 `TSTORE` 按 `BlockSize/2` 步长写盘
→ 相邻 MX block 的写互相覆盖 → 数据整体错乱。

这**不是从未实现**：commit `31f7a8f`（2026-08-21，"derive packed four-bit TCVT rows and pack
FP4 output"，Refs #314/#322）已完整实现——新增 `ElementBits(FP4)=4` 驱动打包行派生、并在
TCVT 里按「偶低奇高 nibble」打包。但该实现被并进一个 "unify TCVT/ARGMAX/FPATR, cooperative
TMATMUL FP16/BF16, and TSORT" 的大包，随后 `930d9981 Revert "unify …"` 把整包回退——**FP4
打包被连坐删除**。回退之后 main 一直没把这块单独恢复。

**这与 [[ISSUE_e8m0_tcvt_regression]]（e8m0 转换缺失）是同一个 `930d9981` revert 的两个连坐
受害者**：同一次宽范围回退，既删了 `52f56d5` 的 e8m0，也删了 `31f7a8f` 的 FP4 打包。e8m0 已
由 `ad288c24` 单独恢复；FP4 打包尚未恢复。

## 现象 / 复现

`dynamic_mx_quant_tail_ocp_fp4`（SPMD 4-PE，M=512 N=256 BS=32，bf16 in / fp4 out / e8m0 scale）
过官方 res_check 精度流程：

```
dynamic_mx_quant_tail_ocp_fp4:
  scale  = pass  (MSE=0.000000, MaxAE=0.000000)   # e8m0 逐字节精确（归约/指数/recip 全对）
  output = fail  (MSE=7.831163, MaxAE=10.000000)   # fp4 数据未打包
```

逐字节剖析（golden 前 16 元素 vs output 前 16 字节）：

```
golden 前16元素:      [6, 2, 11, 3, 7, 15, 5, 14, ...]   # 打包 2/字节
output 字节低nibble:  [6, 2, 11, 3, 6, 14, 5, 14, ...]   # element k 落 byte k 低 nibble
output 字节高nibble:  [0, 0,  0, 0, 0,  0, 0,  0, ...]   # 高 nibble 恒 0（没打包）
```

即 TCVT 没把两个 4bit 元素打进一个字节；每元素独占一字节的低 nibble。block kb 的 32 元素
写 32 字节、而 block 步长只有 `BlockSize/2 = 16` 字节 → 相邻 block 覆盖 → MaxAE=10。只有
block 0 未被覆盖的前 16 字节基本对（仍有二阶 round 差，见下）。

## Regression 链条（commit 时间线 + 实测）

| 时间 | commit | 事件 |
|---|---|---|
| 2026-08-21 12:44 | `31f7a8f` | `fix(gfrun): derive packed four-bit TCVT rows and pack FP4 output`（#314/#322）。加 `ElementBits(FP4)=4` + `IsFourBitDataType` + TCVT 打包（偶低奇高 nibble）+ Block.cpp 行派生 |
| （之后） | "unify …" 大包 | FP4 打包被并进 "unify TCVT/ARGMAX/FPATR, cooperative TMATMUL FP16/BF16, and TSORT" |
| （之后） | `930d9981` | `Revert "unify …"`。反向 diff 把 `ElementBits`/`IsFourBitDataType` 一并删除（实测 DataType.h −2 处） |
| 2026-08-27 | `d8903938`（侧支基线） | 在 revert 之后 → FP4 打包缺失 |
| 2026-08-31 | `ad288c24`（本地） | 只恢复了 e8m0（[[ISSUE_e8m0_tcvt_regression]]），FP4 打包仍缺 |

实测：

```
# 31f7a8f 引入、930d9981 删除、当前 HEAD 与 origin/main 均无
$ git show 31f7a8f -- isa/ISACommon/DataType.h | grep -c '^+.*ElementBits'      # 2（引入）
$ git show 930d9981 -- isa/ISACommon/DataType.h | grep -c '^-.*ElementBits'     # 2（删除）
$ grep -c ElementBits isa/ISACommon/DataType.h                                  # 0（当前 HEAD）
$ git show origin/main:isa/ISACommon/DataType.h | grep -c ElementBits           # 0（main 也无）
```

## 远程现状：main 未恢复，但打包代码活在侧支上

- `origin/main`：**无** `ElementBits` / 打包 → 未修复（与 e8m0 相同）。
- `feat/gfrun-cooperative-tmatmul-fp16-bf16-rerun`（tip `a5dca25a`，**0823 验证分支**）：由
  `9d6adca1 "Reapply PR #322 after main revert"` 在 revert 之后**整包重新应用**，含 FP4 打包。
- `fix/issues-292-293-contracts-20260821`（`215a55f3`，revert 之前血统）：原生带 `31f7a8f`。
- 但上述两分支都**不含**当前模型的 `d8903938`（0.58.4）血统，无法直接切换。

### ✅ 上游正解：`codex/pr-0.58.4-shared-model`（另一 0.58.4 子线）已原生修复

`origin/codex/pr-0.58.4-shared-model`（tip 8-31）用**不同于 31f7a8f 的原生 0.58.4 方案**解决 fp4 打包：
`ElementBitsOf` 位宽抽象 + **存储侧打包**（`ExecuteTSTORE`：`packed = CubeCellElementBits(dataType)==4`、
`memoryRowBytes = (validCol+1)/2`、偶低奇高 nibble），tile 寄存器保持未打包 → **无需改工具链 `type.hpp` bits、
不撞 `IsLegalLocalTileDescriptor` 尺寸校验**（比本文档「TCVT 侧打包 + 到处补位宽感知」侵入更小）。且其 TCVT
`ValidateOperandContract` 已放宽（去物理 row/col，只比 valid 形状+layout，对应本仓 RECORD 问题22）。
**采纳方向**：整体对齐 codex 分支（弃用本地 bits=4 与 31f7a8f 移植），而非把本文档补丁并进 main。

## 根因

不是实现缺失，而是**一次宽范围 revert 的连坐删除**：`930d9981` 回退 "unify" 大包时，把其中本应
保留的 `31f7a8f`（FP4 打包，独立且已测）一并抹掉；回退后无人单独恢复。真正该回退的是 cooperative
TMATMUL / TSORT / FPATR 那些。

### 两侧尺寸约定的错位（配套背景）

FP4 打包依赖「模型 `ElementBits(FP4)=4`（4bit/元素，打包 2/字节）」与「工具链 tile 尺寸按 4bit
算」一致。当前错位：

| 侧 | 文件 | 现状 | 设计意图 |
|---|---|---|---|
| 模型 | `isa/ISACommon/DataType.h` | `BytesOf(FP4)=1`（8bit 存储视图）；`ElementBits(FP4)=4` **被 930d9981 删除** | 打包路径须用 `ElementBits=4` |
| 工具链 | `Linx-TileOP-API include/jcore/type.hpp:62` | `type_traits<__fp4_e2m1x2>::bits=8` | 应为 4（与 `ElementBits=4` 对齐），见 [[ISSUE_linx_tileop_fp4_tile_size_bits]] |

- 当前 `bits=8` + 模型无打包 → 每元素 1 字节、块覆盖（本 issue 现象）。
- 单独把工具链 `bits=8→4`、而模型仍无 `ElementBits` → 工具链 tile 减半（Rows×Cols×4/8）而模型
  `BytesOf=1` 仍按 8bit 记账 → 尺寸不一致 → **gfrun double-free / core dump**（实测）。

故两者必须**同时**修：恢复模型打包（`ElementBits` + TCVT 打包）**且**工具链 `bits=8→4`。

## 修复：恢复 `31f7a8f` 的 FP4 打包部分（不含被 revert 的 cooperative TMATMUL / TSORT）

只重放 `31f7a8f` 中 FP4 打包相关的 hunk，忠实对齐上游：

**A. 直接重放 `31f7a8f` 的 FP4 打包 hunk：**

| 文件 | 恢复内容 |
|---|---|
| `isa/ISACommon/DataType.h` | `inline uint64_t ElementBits(DataType t)`（FP4/FP4_1/HIF4/INT4/UINT4 → 4，其余 `BytesOf*8`）+ `IsFourBitDataType` |
| `emulator/engine/CubeEngine.cpp` | `DataFormatCvt` 中 `dstType==FP4/FP4_1` 编码分支：E2M1/E1M2 有限值表 + RNE 就近选择 + 顶码饱和到 6.0/1.75（不当 NaN/Inf） |
| `emulator/engine/TEPLEngine.cpp` | `ExecuteTCVT` 中 `IsFourBitDataType(dstType)` 打包块：偶低奇高 nibble、每字节 1 存储 |
| `isa/Block.cpp` | `UpdateDstTileInfo` 用 `ElementBits` 派生打包行数（`dst->size*8/(col*elementBits)`） |

**B. 0.58.4 特有的配套修复（`31f7a8f` 时不存在这两处，是 0.58.4 重构后新增、按 `BytesOf`
记账、对 4bit 打包不感知）——实测缺这两处会分别 double-free / 逐行错位：**

| 文件 | 修复 |
|---|---|
| `emulator/engine/AccumulateBlockInfo.cpp` `IsLegalLocalTileDescriptor` | TSTORE 源合法性 `size == rows*cols*BytesOf` → 改按位宽 `size*8 == rows*cols*ElementBits`（非 4bit 等价不变；否则 bits=4 时尺寸不符 → double-free） |
| `emulator/engine/TMAEngine.cpp` `ExecuteTSTORE`（NORM 分支） | 源 tile 行宽 `srcRowWidth = totalCol*eleSize`（eleSize 向上取整=1→32B/行）→ 改用实尺寸 `totalCol*eleRSize`（fp4=0.5→16B/行）；否则只有每 tile 首行落对、其余行读错位 |

**配套（工具链）**：`Linx-TileOP-API include/jcore/type.hpp:62` 把 `__fp4_e2m1x2` / `__fp4_e1m2x2`
的 `bits` 由 8 改 4（installed 头反应式补丁），与恢复后的 `ElementBits=4` 对齐。**A/B/工具链三者
必须同时到位**：单独 bits=8→4 而模型无打包 → double-free；单独恢复打包而工具链 bits=8 → tile 尺寸
按 8bit 记账、TSTORE 校验不符崩。

**不恢复**：`31f7a8f` 附带的 diff_test/isa_test 语料与 cooperative TMATMUL / TSORT，非 FP4 打包必需。

## 验证

恢复后 `dynamic_mx_quant_tail_ocp_fp4`（SPMD 4-PE，M=512 N=256 BS=32，bf16 in / fp4 out /
e8m0 scale）过 res_check（gfrun 4 线程 + `GFRUN_FORCE_DIRECTBOOT_ABI=1`）：

```
dynamic_mx_quant_tail_ocp_fp4:
  output = pass  (MSE=0.022015, MaxAE=0.500000)   # fp4 量化容差内
  scale  = pass  (MSE=0.000000, MaxAE=0.000000)   # e8m0 逐字节精确
```

## 二阶残留（在容差内，不阻塞）

`output` 的 MaxAE=0.5（一个 fp4 最小档）来自 `31f7a8f` 编码就近选择**从 code 1（0.5）起步、非零值
不回落 code 0（0.0）**（`best=1` 初值），而 golden `f32_to_fp4_e2m1` 会考虑 code 0；对落在 (0, 0.25)
的少量值二者差一档 0.5。此差**在 fp4 容差内、res_check 判 pass**，且忠实于上游 `31f7a8f` 行为，本
issue 不改动（如需逐字节相等，另议是否让 golden 或模型对齐 code-0 处理）。

## 建议

1. **上游恢复**：把本文件「修复」表中的 FP4 打包 hunk 作为独立提交重放到 `origin/main`（引 `#314`/
   `31f7a8f`），修复 main 现存的 FP4 打包回归。
2. **回退纪律**：宽范围 `Revert "unify …"` 应拆分，避免把独立且已测的修复（如 `#314` FP4 打包、
   `#253` e8m0）连坐删除。参见 [[ISSUE_e8m0_tcvt_regression]]。
