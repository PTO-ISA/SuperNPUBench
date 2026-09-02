# [gfrun][NA] SuperScalarModel 模型未实现 / 模型侧契约拒绝

本 issue 汇总 TileOP-API v0.58 文档看护中,**报错发生在仿真器(gfrun)** 且**无法靠补文档解决**的问题:
接口按文档正确写、编译通过,但 gfrun 侧要么是**未实现桩**,要么在运行期拒绝按文档示例生成的
descriptor。文档类缺陷、编译器/后端类缺陷另行分别提 issue。

以问题为单位,每条含复现与实测错误信息。

## 组件版本清单

| 组件 | 仓库 | 分支 | commit |
|---|---|---|---|
| SuperNPUBench(看护 demo) | PTO-ISA/SuperNPUBench | `ops-20260828` | `d1604cf` |
| **SuperScalarModel(本 issue 目标)** | LinxISA/SuperScalarModel | `codex/pr-0.58.4-shared-model` | `762a72c` |
| Linx-TileOP-API | LinxISA/Linx-TileOP-API | `linx` | `d6a52b8` |
| llvm-project(linx clang/lld) | LinxISA/llvm-project | `dev-llvm15_56` | `0f878a8` |
| musl | LinxISA/linx-musl | `linx` | `af0dfc2` |
| jemalloc | LinxISA/jemalloc | `linx` | `4495309` |
| linux-linxisa | LinxISA/linux | `main` | `1055a74` |

> 看护 demo 代码见 **SuperNPUBench PR #96**:https://github.com/PTO-ISA/SuperNPUBench/pull/96

## 通用复现步骤

```bash
# 1. checkout 看护 demo(SuperNPUBench PR #96)
# 2. 指向 env_test 工具链
source microbenchmark/tileop-guard/env.sh   # COMPILER_DIR=linx clang / GFRUN=SuperScalarModel/bin/gfrun
# 3. 单 case 三态验证(编译 → gfrun → host golden);以下 case 均「编译通过、gfrun 崩」
bash microbenchmark/tileop-guard/run_guard.sh <domain> <case>
```

---

## gfrun-1 · TIMG2COL 未实现桩(卷积 im2col)

**涉及接口**:TIMG2COL。

**问题**:照 `TIMG2COL.md` 示例(普通 `Tile<Vec,float,8,256>` + TLOAD + `TIMG2COL(dst,src,3,5)`)写,
编译通过,gfrun 直接命中未实现断言——模型侧卷积窗口/repeat/padding 契约未实现。

**复现**:`bash run_guard.sh sfu timg2col`。

**错误信息**:
```
TIMG2COL not yet fully implemented
```

**附加说明**:文档示例本身不用持久 Matrix-location feature-map 描述符源,故 demo 亦忠实按普通 Vec tile 写;
运行受阻纯属模型缺口,非 demo 写法问题。

---

## gfrun-2 · TMRGSORT 未实现桩(归并排序)

**涉及接口**:TMRGSORT。

**问题**:`sort.md` 原示例 shape 不可编译(见 docs issue Doc-8);修正为两源各 1×128、dst 1×256 后编译通过,
gfrun 命中未实现断言。

**复现**:`bash run_guard.sh sfu tmrgsort`(demo 已用修正后的合法 shape)。

**错误信息**:
```
TMRGSORT not yet fully implemented
```

**附加说明**:排序语义 golden 已可实现,待模型补齐后即可转精度校验。

---

## gfrun-3 · TileArray region producer 路径未实现

**涉及接口**:region::TileArray + TASSEMBLY + TPARTVIEW(TCVT slot producer 路径)。

**问题**:绑定问题修好(见 docs issue Doc-15)后,用 TPARTVIEW 父 strided 子视图作源、或独立紧凑 Fragment
作源,gfrun 均命中同一断言。结合 `range-modifiers-developer-guide.md` 文档**自身告警**
「until the … path is implemented and validated」,判定 region producer 路径在当前模型上尚不可运行。

**复现**:`bash run_guard.sh tlsu region_tilearray`。

**错误信息**:
```
raw tile spill source does not fit the carrier shape
```

**附加说明**:文档已自述该路径「尚未实现并验证」,属已知模型缺口,登记待实现。

---

## gfrun-4 · range::Subview 运行期拒绝文档示例 descriptor

**涉及接口**:`range::Subview`。

**问题**:照 `range-modifiers.md` 示例参数编译通过(签名/汇编层无缺口),gfrun 在运行期拒绝按文档示例参数
生成的 descriptor。

**复现**:`bash run_guard.sh tlsu range_subview`。

**错误信息**:
```
illegal TSTORE operand or descriptor contract
```

**附加说明**:签名/汇编层无问题、编译通过,受阻在运行期 descriptor 契约校验。若模型判定为合法用法,应放行
该 subview descriptor;若属文档示例参数问题,请回推正确参数域(此时应转 docs issue)。
