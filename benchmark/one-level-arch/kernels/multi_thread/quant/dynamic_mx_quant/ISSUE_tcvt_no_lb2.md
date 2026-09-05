# [Issue] linx `-D__linx` header：`TCVT` 不发 lb2，boxed valid-col-1 的 TCVT 输出无法 TSTORE【已解决】

## 摘要

在 `dynamic_mx_quant` OCP 探针 kernel 的 gfrun 执行中，命中一处不在 release_ver0812 已知清单内的
运行期断言：**一个 `validCol=1`、物理 `Cols>1` 的 tile，若最后一个 producer 是 `TCVT`，随即
`TSTORE` 时被 emulator 拒绝**。根因是 Linx-TileOP-API 的 `-D__linx` intrinsic header
（`template_asm.hpp`）里 `TCVT_T` **漏发 `lb2`（物理列宽）描述符**，与所有同类 TEPL elementwise
op（`TABS`/`TEXP`/`TRECIP`/`TADDS`/`TMULS`/`TANDS`/`TMAX`/`TROWMAX` …）不一致。缺陷所在仓
**`linx-toolchain-build`（Linx-TileOP-API header）**，非 LLVM codegen、非 emulator。

## 环境

| 项 | 值 |
|---|---|
| 编译仓 | SuperNPUBench `benchmark/one-level-arch` |
| 执行仓 | SuperScalarModel（`bin/gfrun` 执行 elf） |
| 工具链 | `linx_blockisa_llvm_musl`，clang-15，target `linx64v5-linux-musl` |
| 复现根目录 | `benchmark/one-level-arch/test/kernel/multi_thread/quant/dynamic_mx_quant` |
| 触发 kernel | `kernels/multi_thread/quant/dynamic_mx_quant/probe_dynamic_mx_quant_tail_ocp_fp8.hpp`（`TYPE=PROBE_OCP_FP8`） |

### 各仓代码分支（复现基线）

| 仓库 | 分支 / HEAD | commit |
|---|---|---|
| SuperNPUBench | `feat/dmxq-rel0812` | `1bd945a` |
| SuperScalarModel | detached HEAD（基于 `feat/pto-v058-adaptation`） | `319294ff` |
| linx-toolchain-build | `main` | `e6a31ef` |
| └ Linx-TileOP-API（组件源，缺陷根仓） | `temp/shared-tload-integration-20260811` | `abe8411` |

> 提交的 probe kernel 为**未应用任何规避的复现版**：按下方复现命令 `gfrun` 执行即触发 M47
> 断言。缺陷本体在 `linx-toolchain-build` 的 Linx-TileOP-API 组件头（`template_asm.hpp` 的
> `TCVT_T`），SuperNPUBench 侧只是复现入口。

## 复现命令

```bash
# 编译（成功）
export COMPILER_DIR=.../linx_blockisa_llvm_musl/bin
export LINX_SYSROOT=.../linx_blockisa_llvm_musl/sysroot/usr
cd benchmark/one-level-arch/test/kernel/multi_thread/quant/dynamic_mx_quant
make TESTCASE=dynamic_mx_quant TYPE=PROBE_OCP_FP8 res_check=on diss   # 编译+反汇编，clean

# gfrun 执行（触发断言）
SuperScalarModel/bin/gfrun -t 1 -f "$PWD/output/.../dynamic_mx_quant_probe_ocp_fp8.elf"
```

## 报错

```
SuperScalarModel/emulator/engine/AccumulateBlockInfo.cpp:60
ValidateLocalTlsu "Local TSTORE requires one compatible source Tile"   (EXIT=1)
```

faulting 指令 = probe 中 `TSTORE(gw, max_bf)`（`BSTART.TMA TSTORE BF16`），其中
`max_bf = TCVT(max_h)`（half→bf16），tile 为 boxed `[TileM,1]`、物理 `Cols=32`。

## 根因分析

### tile 形状描述符 lb0/lb1/lb2

产 tile 的指令用内联汇编 `B.DIM ..,->lbN` 把形状告诉 emulator：

| 字段 | 含义 |
|---|---|
| `lb0` | validCol |
| `lb1` | validRow |
| `lb2` | **物理列宽 `tile::Cols`（每行存储步长）** |

### TCVT 是唯一漏发 lb2 的 elementwise op

核对 `template_asm.hpp`：

| op | 发 lb2？ |
|---|---|
| `TABS`(:3401) / `TEXP`(:3461) / `TRECIP`(:3501) / `TADDS`(:3646) / `TMULS` / `TANDS` / `TMAX` / `TROWMAX`(:4586) | ✅ `B.DIM zero,%c4,->lb2` |
| **`TCVT_T`（TEPL 27，:109）** | ❌ 仅 `->lb0`/`->lb1` |
| `TMOV`(:181) | ❌（layout move，dst 形状另有来源，可理解） |

`TCVT` dst 与 src 逐元素同形，没有理由与 `TABS`/`TEXP` 区别对待——这个孤立的不一致指向
**手写内联汇编 intrinsic 的疏漏**。

### emulator 兜底把缺失的 lb2 解释成 validCol

`SuperScalarModel/isa/Block.cpp:1074`：

```cpp
physicalCol = (lb2 <= 1) ? validCol : lb2;   // TCVT lb2=0 → physicalCol = validCol
```

于是 TCVT 产出的 boxed valid-col-1 tile 被记 `col = 1`（本应是物理的 32）。

### TSTORE 声明 Cols，校验失败

`TSTORE`（:1806）固定发 `lb2 = tile::Cols`（=32）。校验
`IsCompatibleDataTile`（`AccumulateBlockInfo.cpp:265`）要求
`源tile.tileInfo->col == store.physicalCol` → `1 != 32` → 断言。

### 关键澄清：不是「部分列 store 非法」

部分列 store（`validCol=1`、`physical=32`）**是合法且被支持的**——
`kernels/reduction/reducemax_rowvec_pto.hpp` 正常这么做（store 的 tile 由 `TMAX` 产出、
`col=物理宽 8`、`validCol=1`，写入 `RowMajor<gIM,1>`）。症结**只**在 `TCVT` 塌了 `col`。
佐证：探针里数据满宽 store `TSTORE(gy, oq)`（`oq=TCVT(...)` 但 `validCol==Cols==32`）**不中招**，
因塌陷后 `col` 仍等于 `Cols`。

## 定性与待确认

- 疑似 **工具链 header 缺陷**（`TCVT_T` 与同类 TEPL elementwise op 在 lb2 上不一致）。
- emulator 的兜底逻辑本身合理，非 emulator 缺陷；LLVM codegen 未参与（intrinsic 是手写汇编）。
- **100% 坐实还差一步**：核对 PTO ISA 规范 TEPL-27(CVT) 指令编码，`lb2` 是否为产 tile 指令的
  必填/合法字段。若规范要求所有产 tile 指令带物理宽 → 确证 header 漏发；若允许 CVT 省略 → 属
  header(TSTORE 声明 Cols) 与 emulator(缺省 lb2 解释成 validCol) 的接口约定不一致。

## 修复建议（根本解，toolchain 侧）

**解除路径（linx-toolchain-build → Linx-TileOP-API 组件源）**：
`src/Linx-TileOP-API/include/jcore/template_asm.hpp` 的 `TCVT_T`（:109）汇编体在 `->lb1` 之后补一行
`"B.DIM zero, %c7, ->lb2\n"`，并在输入操作数末尾追加 `"i"(tile_shape_out::Cols)`（第 7 个输入
`%c7`），与 `TABS`(:3401) 等所有 TEPL elementwise op 对齐。

**已实测验证（汇编补丁）**：在 build artifact
（`output/.../lib/clang/15.0.4/include/tileop-api/jcore/template_asm.hpp`）打上该补丁后重编
probe（**无需 kernel 侧盖章**），反汇编确认每个 `BSTART.TEPL TCVT` 块后紧跟 `C.B.DIMI 32, ->lb2`；
`gfrun` **越过原 M47 `ValidateLocalTlsu` 断言**，收敛到 startup skew 墙。零回归：满宽 tile 旧兜底
`col=validCol==Cols` 与新显式 `lb2=Cols` 一致；boxed valid-col-1 旧 `col=1`（错）→ 新 `col=Cols`（对）。

> 注意：build artifact 会被 `make build-tileopapi` 覆盖，根本修复须落**组件源**
> `src/Linx-TileOP-API/...`。本次为保留可复现，**未提交任何 toolchain 侧改动**。

> **提交代码为复现版**：probe kernel 未应用上述 toolchain 补丁，按上方复现命令 `gfrun` 即触发
> M47 断言。toolchain 侧落组件源补发 lb2 后即可让 probe 干净跑通（收敛到已知 startup skew 墙）。
