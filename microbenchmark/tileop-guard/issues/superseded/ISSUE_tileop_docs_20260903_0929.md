# [docs] Linx-TileOP-API `docs/tileop-usage` 文档缺陷（tileop-guard 第二轮 · 2026-09-03 09:29）

本 issue 汇总 TileOP-API v0.58 文档看护**第二轮**新发现的**文档类缺陷**（补文档即可解决）。与首轮
docs issue 不重复；以问题为单位。

## 组件版本清单

| 组件 | 仓库 | 分支 | commit |
|---|---|---|---|
| SuperNPUBench(看护 demo) | PTO-ISA/SuperNPUBench | **PR #96** | https://github.com/PTO-ISA/SuperNPUBench/pull/96 |
| SuperScalarModel(gfrun) | LinxISA/SuperScalarModel | `codex/pr-0.58.4-shared-model` | `762a72c` (762a72c34305f7f1df6964e7dfe202bd3e63a951) |
| **Linx-TileOP-API(本 issue 目标)** | LinxISA/Linx-TileOP-API | `linx` | `6f230c5` (6f230c598dd674d0007f6a0b6634ab4183c0ce48) |
| llvm-project(clang/lld) | LinxISA/llvm-project | `dev-llvm15_56` | `25677bb` (25677bb1a6c758d8867cc4e1b42acaf6626f9316) |
| musl | LinxISA/linx-musl | `linx` | `af0dfc2` |
| jemalloc | LinxISA/jemalloc | `linx` | `4495309` |
| linux-linxisa | LinxISA/linux | `main` | `1055a74` |

## 通用复现步骤

```bash
source microbenchmark/tileop-guard/env.sh
bash microbenchmark/tileop-guard/run_guard.sh <domain> <case>
```

---

## docs-N1 · range::Subview 的 parent 必须是 CUBE 布局 tile（文档未说，示例用 Vec 误导）

**涉及接口**：`range::Subview`（range-modifiers.md）。

**问题**：`range-modifiers.md` 的 Subview 示例用普通 Vec/RowMajor tile 作 parent，且**未说明 parent 必须是
CUBE 布局 tile**。实测：Vec parent 编译通过（Subview 头模板不限制 parent），但 gfrun 运行期拒
`illegal TSTORE operand or descriptor contract`——模型 `Block.cpp HandleBSubview` 断言
`IsCubeLayout(parent->layout)` 并走 `CubeCellDescribeSubview`（cube-cell 子视图运算）。即 Subview 是
**cube tile 的子区域抽取器**，parent 须由 `TLOAD_CUBE`/matmul 产出的 cube-layout tile（`CubeM16/M32/N8`）。

**复现**：`bash run_guard.sh tlsu range_subview`（demo 已改为 Vec witness，注释标注 cube-parent 要求）。

**错误信息**：
```
gfrun: illegal instruction at 0x0: illegal TSTORE operand or descriptor contract
```

**建议**：在 range-modifiers.md 明确 Subview 的 parent 必须是 CUBE 布局 tile，并把示例改为
`CubeTileM32 + TLOAD_CUBE + Subview` 的完整可落地形态；补 SizeCode 与 parent 容量的对应（如
`CubeTileM32<half,32,32>`=2048B→SizeCode≤5）。**注**：构造完整 cube-subview demo 时还会撞到头层的
`TSTORE_CUBE` 与 Subview 载体不兼容（另见同批 [linx] issue），文档补齐后需与该修复配套才能给出可运行示例。

---

## docs-N2 · reinterpret_tile 使用规则未文档化（双操作数视图 + 落盘重置标签）

**涉及接口**：reinterpret_tile（reinterpret-tile.md）。

**问题**：`reinterpret-tile.md` 给出 bitcast 语义，但**未说明两条关键使用规则**，导致按直觉写会失败：
1. **两个操作数必须同为视图**：tileop 模板把两参绑同一 `tile_shape`，所以**不能把视图与普通 tile 混用**
   在一次调用里（如 `TABS(普通int32tile, 视图)` 报 `no matching function`）。须两参都 `reinterpret_tile`
   成同一视图类型。**（此条同时修正首轮 [linx] issue 里「reinterpret 视图不被 TABS 接受」的判断：根因是
   操作数类型混用，不是 TABS 拒视图；TABS/TANDS 已接受视图。）**
2. **落盘须再做一次原生类型 op 重置 dtype 标签**：整数视图 op（如 `TANDS`）后，backing 的运行期 block
   dtype 标签变成视图 dtype，而 TSTORE 按 tile 的 C++ 声明类型定 block dtype 且**不接受视图**；须在存之前
   插一个原生类型 op（如 `TMULS(t, t, 1.0f)`）把标签重置回原类型。此模式即模型自带 cross_model 测试
   `bf16_backing_tands_u16_then_tmuls_bf16` 的做法。

**复现**：`bash run_guard.sh misc reinterpret_tile`（现精度PASS，golden=abs：fp32→int32 视图 TANDS 清符号位 →
TMULS 重置 → 存 |x|）。

**建议**：在 reinterpret-tile.md 补这两条规则，并给一个「整数视图 op → 原生类型 op 重置 → 存」的端到端示例。
