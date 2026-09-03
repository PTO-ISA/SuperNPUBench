# [linx] 编译器/头模板缺陷（tileop-guard 第二轮 · 2026-09-03 09:29）

本 issue 汇总 TileOP-API v0.58 文档看护**第二轮**新发现的、**报错发生在工具链头/编译器**且无法靠补文档
解决的问题。与首轮 [linx] issue 不重复；以问题为单位。

## 组件版本清单

| 组件 | 仓库 | 分支 | commit |
|---|---|---|---|
| SuperNPUBench(看护 demo) | PTO-ISA/SuperNPUBench | **PR #96** | https://github.com/PTO-ISA/SuperNPUBench/pull/96 |
| SuperScalarModel(gfrun) | LinxISA/SuperScalarModel | `codex/pr-0.58.4-shared-model` | `762a72c` (762a72c34305f7f1df6964e7dfe202bd3e63a951) |
| **Linx-TileOP-API(头模板,目标之一)** | LinxISA/Linx-TileOP-API | `linx` | `6f230c5` (6f230c598dd674d0007f6a0b6634ab4183c0ce48) |
| **llvm-project(clang/lld,目标之一)** | LinxISA/llvm-project | `dev-llvm15_56` | `25677bb` (25677bb1a6c758d8867cc4e1b42acaf6626f9316) |
| musl | LinxISA/linx-musl | `linx` | `af0dfc2` |
| jemalloc | LinxISA/jemalloc | `linx` | `4495309` |
| linux-linxisa | LinxISA/linux | `main` | `1055a74` |

## 通用复现步骤

```bash
source microbenchmark/tileop-guard/env.sh
bash microbenchmark/tileop-guard/run_guard.sh <domain> <case>
```

---

## linx-N1 · TSTORE_CUBE 无法接收 range::Subview 载体（const& 形参 vs 非 const `data()`）

**涉及接口**：`range::Subview` + `TSTORE_CUBE`。
**组件**：Linx-TileOP-API 头（`jcore/template_asm.hpp` TSTORE_CUBE + `common/pto_tile.hpp` Subview）。

**问题**：`range::Subview` 要求 parent 为 CUBE 布局 tile（模型契约，另见同批 [docs] issue）。据此构造正确的
cube-subview 落盘 demo——`CubeTileM32 s; TLOAD_CUBE(s, gm); auto sv = range::Subview<decltype(s),SC>(s,0);
TSTORE_CUBE(gmOut, sv);`——在**头层**即编译失败，暴露两处不兼容：

1. **const-ness 不兼容**：`TSTORE_CUBE` 以 `const&` 接收源，但对其调用 `data()`，而 `Subview::data()`
   **非 const** → 无法在 const 载体上调用。
2. **dtype 强绑**：`TSTORE_CUBE` `static_assert` 要求 GM 与 CUBE tile 的 dtype 完全一致
   （`std::is_same_v<...>`），fp16 cube → fp32 GM 被拒。

结果：range::Subview 虽有 cube parent，却**没有可编译的落盘路径**——`TSTORE_CUBE` 不接受 Subview 载体，
而普通 `TSTORE` 又走模型 descriptor 拒绝（Vec 层）。

**复现**：把 `tlsu/src/range_subview.cpp` 的 parent 换成 `CubeTileM32<__half,32,32>` + `TLOAD_CUBE` +
`TSTORE_CUBE(gmC, sv)` 后 `make`（当前提交里保留的是 Vec witness 版；cube 版仅用于本条复现）。

**错误信息**：
```
template_asm.hpp: static assertion failed ... "TSTORE_CUBE requires matching GM and CUBE dtypes"
template_asm.hpp: 'this' argument to member function 'data' has type 'const pto::range::Subview<...>',
                  but function is not marked const
```

**附加说明**：要让 range::Subview 端到端可用，头层需：①给 `Subview::data()` 提供 const 重载（或让
`TSTORE_CUBE` 以非 const& 接收 range 载体，与普通 `TSTORE` 一致——`TSTORE(gm, sv)` 能接受 Subview）；
②明确 cube-subview 的落盘接口（是否允许 dtype 转换 / 是否有专门的 cube-subview store）。当前 range::Subview
无任何可运行的完整示例，建议头层修复后配套文档（见同批 [docs] issue docs-N1）。
