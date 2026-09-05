# [linx] 编译器/后端无法处理文档接口(clang frontend abort / Match Instruction Error / 模板不兼容)

本 issue 汇总 TileOP-API v0.58 文档看护中,**报错发生在编译器/后端(linx clang / lld / TileOP-API 头)**
且**无法靠补文档解决**的问题:接口按文档完整签名/示例写,却在编译期崩溃、后端无法汇编、或头模板拒绝
接受合法视图。文档类缺陷、仿真器类缺陷另行分别提 issue。

以问题为单位,每条含复现与实测错误信息。

## 组件版本清单

| 组件 | 仓库 | 分支 | commit |
|---|---|---|---|
| SuperNPUBench(看护 demo) | PTO-ISA/SuperNPUBench | `ops-20260828` | `d1604cf` |
| SuperScalarModel(gfrun) | LinxISA/SuperScalarModel | `codex/pr-0.58.4-shared-model` | `762a72c` |
| **Linx-TileOP-API(头模板,目标之一)** | LinxISA/Linx-TileOP-API | `linx` | `d6a52b8` |
| **llvm-project(linx clang/lld,目标之一)** | LinxISA/llvm-project | `dev-llvm15_56` | `0f878a8` |
| musl | LinxISA/linx-musl | `linx` | `af0dfc2` |
| jemalloc | LinxISA/jemalloc | `linx` | `4495309` |
| linux-linxisa | LinxISA/linux | `main` | `1055a74` |

> 看护 demo 代码见 **SuperNPUBench PR #96**:https://github.com/PTO-ISA/SuperNPUBench/pull/96

## 通用复现步骤

```bash
# 1. checkout 看护 demo(SuperNPUBench PR #96)
# 2. 指向 env_test 工具链
source microbenchmark/tileop-guard/env.sh   # COMPILER_DIR=linx clang / GFRUN=SuperScalarModel/bin/gfrun
# 3. 单 case 三态验证;以下 case 均在「编译期」失败(clang/后端),跑不到 gfrun
bash microbenchmark/tileop-guard/run_guard.sh <domain> <case>
```

---

## linx-1 · MGATHER_MASK / MSCATTER_MASK — 后端无法汇编 `*.MASK` 双 B.IOT bundle

**组件**:llvm-project(后端指令匹配)。
**涉及接口**:MGATHER_MASK、MSCATTER_MASK。

**问题**:`MGATHER_MASK.md` / `MSCATTER_MASK.md` 给出完整签名 + dtype 表 + 带 `TmaPadValue::Zero` 的示例
(`MGATHER_MASK<out,off,mask,gm,Pad>(dst, base_gm, offset, mask)`,mask=uint8)。照签名写,static_assert
通过,但后端发射的 `BSTART.TLSU MGATHER.MASK` 双 `B.IOT`(IndexTile + MaskTile)bundle 形式无法被后端
匹配/汇编。

**复现**:`bash run_guard.sh tlsu mgather_mask` / `bash run_guard.sh tlsu mscatter_mask`。

**错误信息**(`template_asm.hpp:566` 附近发射时):
```
Match Instruction Error!
```

**附加说明**:offset 用 U32(dtype 表)或 uint16(示例原样)**都崩在同一行同一错误**(已过 static_assert,
崩在指令匹配),排除操作数类型因素;同族无 mask 的 MGATHER/MSCATTER 照文档写一次通过 + host golden PASS。
属后端未实现该指令编码。host golden 已按 `where(mask==1, base[off//4], 0)` 就绪,后端补齐即可转精度PASS。

---

## linx-2 · GMOV — 后端无法汇编

**组件**:llvm-project(后端指令匹配)。
**涉及接口**:GMOV。

**问题**:`tlsu.md` 给示例 `GMOV<15>(dst, peer_tid, src)`,照抄后后端无法汇编。

**复现**:`bash run_guard.sh tlsu gmov`。

**错误信息**:
```
Match Instruction Error!
```

**附加说明**:文档示例与后端不一致;与 linx-1 同类(后端未实现该指令编码)。

---

## linx-3 · TMATMUL bf16 options — clang frontend abort

**组件**:llvm-project(clang 前端)。
**涉及接口**:TMATMUL + `fixp::bf16()`。

**问题**:照 `matrix-postprocess.md` 写 `TMATMUL(dst_bf16, a, b, fixp::bf16())`(`__bf16` 累加器),clang
前端崩溃退出。

**复现**:`bash run_guard.sh cube tmatmul_bf16`。

**错误信息**:
```
clang frontend command failed (exit 134 / abort)
```

**附加说明**:同一 demo 骨架换 `fixp::f16()` 正常编译 + gfrun 跑通(精度PASS),定位到 bf16 options 路径触发
前端崩溃。

---

## linx-4 · reinterpret_tile 返回视图不被下游 tileop 模板接受

**组件**:Linx-TileOP-API(头模板)/ 视图类型与 tileop 入参兼容性。
**涉及接口**:reinterpret_tile → 下游 TABS\TCMPS 等 tileop。

**问题**:`reinterpret-tile.md` 签名 + 示例完整(同位宽、Local only、视图非拥有)。demo 照写
`fp32 tile → reinterpret_tile<int32_t>` 得视图,随后用 `TABS`、`TCMPS` 消费该视图,编译报模板匹配失败。视图类型不被
后续 tileop 的模板接受。

**复现**:`bash run_guard.sh misc reinterpret_tile`。

**错误信息**:
```
no matching function for call to 'TABS'
```

**附加说明**:问题在 reinterpret_tile 视图类型与 tileop 入参模板的兼容性——要么 reinterpret_tile 返回类型
应满足 tileop 模板约束,要么 tileop 模板应接受该视图类型。文档层无缺口。
