# [gfrun][NA] SuperScalarModel 模型缺口（tileop-guard 第二轮 · 2026-09-03 09:29）

本 issue 汇总 TileOP-API v0.58 文档看护**第二轮**（放宽纯文档驱动、读工具链头订正签名后）
新发现的、**报错发生在仿真器 gfrun 且无法靠补文档解决**的模型缺口。接口按头文件正确写、编译通过，
但 gfrun 侧运行期拒绝。与首轮已提 issue 不重复；以问题为单位。

## 组件版本清单

| 组件 | 仓库 | 分支 | commit |
|---|---|---|---|
| SuperNPUBench(看护 demo) | PTO-ISA/SuperNPUBench | **PR #96** | https://github.com/PTO-ISA/SuperNPUBench/pull/96 |
| **SuperScalarModel(gfrun,本 issue 目标)** | LinxISA/SuperScalarModel | `codex/pr-0.58.4-shared-model` | `762a72c` (762a72c34305f7f1df6964e7dfe202bd3e63a951) |
| Linx-TileOP-API(头) | LinxISA/Linx-TileOP-API | `linx` | `6f230c5` (6f230c598dd674d0007f6a0b6634ab4183c0ce48) |
| llvm-project(clang/lld) | LinxISA/llvm-project | `dev-llvm15_56` | `25677bb` (25677bb1a6c758d8867cc4e1b42acaf6626f9316) |
| musl | LinxISA/linx-musl | `linx` | `af0dfc2` |
| jemalloc | LinxISA/jemalloc | `linx` | `4495309` |
| linux-linxisa | LinxISA/linux | `main` | `1055a74` |

> 工具链指纹：clang++ md5 `1ee479a3f4678006953db5b0af0f50a2`、gfrun md5 `04ca39ece7533eb35c805a4741996ebb`。

## 通用复现步骤

```bash
source microbenchmark/tileop-guard/env.sh   # COMPILER_DIR=linx clang / GFRUN=SuperScalarModel/bin/gfrun
bash microbenchmark/tileop-guard/run_guard.sh <domain> <case>
```

---

## gfrun-N1 · TCMP 处理器拒绝 reinterpret_tile 视图源（官方 reinterpret 修复在模型层遗漏 TCMP）

**涉及接口**：TCMP + reinterpret_tile。
**组件**：SuperScalarModel（TCMP 运行期校验）。

**问题**：`reinterpret_tile<int32_t>(fp32Tile)` 返回的视图（`ReinterpretedTileView`）作为 **TCMP 的源操作数**时，
**编译通过**（`TCMP<tile_out, tile_in>` 分离模板参，允许普通 predicate 输出 + 视图源），但 gfrun 在 TCMP
处理器崩溃。对照实验证明**差异就是视图源**：
- 同样两个 int32 视图喂 **TANDS** → 跑通 + 精度PASS（`misc/reinterpret_tile`，golden=abs）；
- 普通 int32 tile 喂 **TCMP** → 跑通 + 精度PASS（`vec/tcmp`）；
- int32 视图喂 **TCMP** → gfrun 崩。

即 TANDS 等 elementwise 处理器已接受 reinterpret 视图源，**唯独 TCMP 处理器的 `IsCompatibleDataTile`
不认视图源描述符**——官方 reinterpret 修复遗漏了 TCMP 这一路。

**复现**：`bash run_guard.sh misc reinterpret_tcmp`。

**错误信息**：
```
gfrun: illegal instruction: ASSERTION FAILED: priorSources == 0 && inst->srcs.size() == 3 &&
  inst->dsts.size() == 1 && inst->dsts[0]->size >= predicateBytes &&
  IsCompatibleDataTile(inst->srcs[1], block->dataType, validRow, validCol, physicalCol, dataBytes) &&
  IsCompatibleDataTile(inst->srcs[2], block->dataType, validRow, validCol, physicalCol, dataBytes) &&
  "TCMP requires two compatible Tile sources"
```

**附加说明**：崩在 TCMP 本身（早于任何 TSTORE）。最小复现即 `reinterpret_tcmp.cpp`：两 fp32 tile →
`reinterpret_tile<int32_t>` → `TCMP<GT>(mask, vA, vB)`。修好后该 case 转精度PASS（golden 已就绪：
`where(int_bits(a)>int_bits(b), tru, prior)`）。建议让 TCMP 处理器像 TANDS 一样接受 reinterpret 视图源描述符。

---

## gfrun-N2 · TEXTRACT 任何真实子抽取都被判非法（block 维取自源 valid 维）

**涉及接口**：TEXTRACT。
**组件**：SuperScalarModel（`ValidateV058SpecialTepl` 的 TEXTRACT 分支）＋工具链头 emit（见附加说明）。

**问题**：`TEXTRACT(dst, src, indexRow, indexCol)` 照头签名写、编译通过，但只要是**真实的子抽取**
（indexRow/indexCol > 0，或 dst 小于 src）就被 gfrun 判非法；仅 `offset=0 且 dst 与 src 同为全尺寸`的
**退化恒等拷贝**能过。

根因（模型 `emulator/engine/AccumulateBlockInfo.cpp` TEXTRACT 校验）：
```cpp
rowOffset + validRow > source->tileInfo->validRow ||   // -> bIsIllegal
colOffset + validCol > source->tileInfo->validCol ||
inst->dsts[0]->size < validRow * block->lb2 * BytesOf(block->dataType)
```
其中 `validRow=lb1`、`validCol=lb0` 是 block 维，而工具链头 `TEXTRACT` emit 把 **源 tile 的 ValidRow/ValidCol**
发成 lb1/lb0。于是 `validRow == source->tileInfo->validRow`，`rowOffset + validRow > validRow` ⟺ `rowOffset > 0`
恒成立 → 任何非零 offset 必非法；且 dst 尺寸校验也按源全尺寸算，dst（抽取区域）小于源即失败。

**复现**：`bash run_guard.sh sfu textract`（demo 用 src 16×32、dst 8×16、offset(4,8)）。

**错误信息**：
```
gfrun: illegal instruction at 0x0: illegal TEXTRACT operand or descriptor contract
```

**附加说明**：本质是 **block 维应表示「抽取区域(dst)维」而非「源维」**的契约不一致。修复方向二选一：
①工具链头 TEXTRACT emit 改发抽取区域(dst)的 valid 维作 lb0/lb1；②模型校验改用区域维（并允许区域 < 源）。
同族的 TINSERT 用相同 emit 却能运行（其模型校验无此 `offset+validRow>source` 分支），可作对照。
