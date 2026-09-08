# [gfrun] region TileArray / TASSEMBLY 多 writer（parent>writer）B.ASSEMBLE 被 PrepareLocalAssemble 拒；发射侧已自证 spec 合规  【草稿·🆕未提交·待官方裁决】

> **状态**：本地草稿（未提交线上）。**owner 未定**——按 2026-09-08 归属政策，组件归属只由**官方对已提交
> issue 的裁决**产生，本地分析（含本文）是**提交 issue 用的问题分析**，不是权威归属。故本文**不标 owner**，
> 提交后由官方裁决（6 类 `API/model/llvm/spec/demo/退休`）。目标仓待定（本地分析指向 SuperScalarModel，但
> 需官方确认——本 case 历史上曾被本地判 model、亦有"实为 demo"之说，正因本地判断不可靠才走官方裁决）。
>
> **本地分析结论（供 issue 提交，非权威）**：发射侧（API carrier 分配 + B.ASSEMBLE offset/coverage + 规范
> INIT=parent）已逐项自证合规，证据**指向模型侧** `PrepareLocalAssemble` 的多 writer 路径；确切 return-false
> 行号未定（`bin/gfrun` 无行号调试信息，需 `-g` 重编）。**此前"maxSize_ 门"的猜测已证伪**（见下）。

## 组件版本清单

| 组件 | 仓库 | 分支 / tag | commit |
|---|---|---|---|
| **SuperScalarModel（本 issue 目标）** | LinxISA/SuperScalarModel | `codex/consolidate-post-main-fixes-20260903` | `49547742` |
| Linx-TileOP-API | LinxISA/Linx-TileOP-API | `linx` | `0566283` |
| pto-spec（ISA 规范） | PTO-ISA/pto-spec | `main` | `dea0b75e`（0.58.6.0） |
| SuperNPUBench(看护 demo) | [SuperNPUBench PR #96](https://github.com/PTO-ISA/SuperNPUBench/pull/96)（分支 `tileop-guard-batch1`） | — | `ecbb281` |
| llvm-project | LinxISA/llvm-project | `dev-llvm15_56` | `67d3ac9` |

> 工具链指纹：clang++ md5 `e427d1429c0e`、gfrun md5 `0c433cd11c00`。

## 复现

```bash
# 1. 从 PR #96 拉取最新看护 demo 代码
gh pr checkout 96                    # https://github.com/PTO-ISA/SuperNPUBench/pull/96
# 2. 指向工具链并复现
source microbenchmark/tileop-guard/env.sh
bash microbenchmark/tileop-guard/run_guard.sh tlsu region_tilearray
# gfrun: illegal B.ASSEMBLE generation or descriptor contract   (SoftCore.cpp:947)
```

## 现象

`region::TileArray<Fragment,1,NF>` + 逐 slot `TCVT` 生产 + `TASSEMBLY<Parent>` 组装（parent 32×64 = 4 个
32×16 fragment），gfrun 执行期崩 `illegal B.ASSEMBLE generation or descriptor contract`
（`SoftCore::PrepareLocalAssemble` 返回 false → `SoftCore.cpp:945-947`）。

**隔离**（探针）：**NF=1（parent==writer）PASS；NF≥2（parent>writer）FAIL**。即失败**仅出现在 parent 大于单
fragment 的多 writer 组装**。

## 发射侧自证 spec 合规（→ 第一偏离点在模型）

1. **API carrier 按 parent 分配（不是 fragment）**：`pto_tile_region.hpp` `TileArray::ParentBytes =
   SubTile::LogicalTileBytes * (Rows*Cols)`（=8192B），`ParentCarrier = linx_tile_carrier<ParentBytes>`。
   → API 没有"只分配 fragment 容量"的问题。
2. **B.ASSEMBLE bundle 逐条 spec-legal，且 offset/coverage 精确正确**（objdump `region_tilearray.elf`）：
   - `d0 B.ASSEMBLE 1,0,zero,0,7`（INIT，ParentSizeCode=7=8192B，offset=0）
   - `d1 …,a1,…`（a1=**16**）、`d2 …,a0,…`（a0=**32**）、`d3 …,1,a1,…`（a1 重赋=**48**，LAST）
   - writerCells=16，四 writer 覆盖 cells **[0,16)[16,32)[32,48)[48,64)** = 恰好全覆盖 64 cells、无重叠无空洞。
   - 每条 B.IOT 目的 `->n<2KB>`=fragment 大小（符合 writer/parent "两套几何"设计）。
3. **parent>writer 是 spec 明确意图**：pto-spec `asl/block/model/operands/local-generation.asl::
   BundleLocalDestinationAllocationBytes`——**INIT 时目的端分配 = `TileSizeCodeBytes(ParentSizeCode)` = parent 大小**。
   即规范本就要求 parent 容量 > 单 writer，非 subview 那种 fail-closed 拒绝。
4. **模型接收到的是一个完全 spec 合规的多 writer 组装**（carrier=parent 尺寸、bundle 合法、offset 全对、覆盖完整、
   几何为规范意图），却在 `PrepareLocalAssemble` 拒绝 → **第一偏离点 = 模型**。

## 证伪的旧假设 + 已排除项

- **旧猜测"`parentBytes > destinationEntry->maxSize_` 门触发"→ 证伪**：`configs/SoftCore.toml maxTileSize=262144`
  (256KB)，每 tile `maxSize_=256KB ≫ parentBytes(8192)`，该门恒 false，`ResizeTile(…,8192)` 亦不受限。
- **offset/coverage 已排除**：objdump 实证 offset=0/16/32/48、全覆盖无重叠（见上）。
- **gdb 确认**：函数断点命中 `SoftCore::PrepareLocalAssemble` 6 次、其一返回 false 触发 abort；行号无法取
  （二进制无 line table）→ 确切内部 return-false 行待 `-g` 重编定位。

## 本地分析结论与建议（待官方裁决）

- **证据指向模型**（非权威结论）：`PrepareLocalAssemble` 的**多 writer（parent>writer）本地组装路径**拒绝了一个
  发射侧已自证 spec 合规的输入；owner 以官方裁决为准。
- 建议：model 团队用 `-g` 重编 gfrun，在 `emulator/SoftCore.cpp` `PrepareLocalAssemble` 的 13 个 return-false
  处定位失败点（重点疑区：非-INIT writer 的 `generation.open/sameDescriptor` 状态持续性、或 coverage/geometry
  的多 writer 分支），修复后 NF≥2 组装应通过。
- 看护落点：`tlsu/region_tilearray` = **run-fail witness**（无 golden，producer 无可校输出）。

## 备注（方法论）

此前对本 case 曾以未证实的"maxSize_ 门"下结论——本轮按组件归属方法论逐项证伪并重新定位：先证发射侧合规
（API carrier + bundle offset/coverage + 规范 INIT=parent 分配），再用 gdb 确认崩在 `PrepareLocalAssemble`，
故责任钉在模型；仅确切行号受限于二进制无调试信息而暂缺。
