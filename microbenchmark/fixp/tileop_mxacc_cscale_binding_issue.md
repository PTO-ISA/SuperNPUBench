# [TileOP][TMATMUL_MX_ACC] 可选 MX Scale 路径漏发 CScale source binding

## 问题摘要

`TMATMUL_MX_ACC` 在通过 `fixp::Options::cscale()` 启用 `CScaleEn`，且 A/B 类型不需要同时提供 MX ScaleA/ScaleB 时，TileOP API 会正确编码：

```text
B.FPATR CScaleEn=1
```

但生成的 CUBE source stream 中没有对应的 CScale `B.IOT` binding。gfrun 按 ASL 检查 source 数量时因此正确拒绝该指令块：

```text
CUBE local source stream does not match the current ASL contract
```

该问题最初表现为 [SuperNPUBench issue #97](https://github.com/PTO-ISA/SuperNPUBench/issues/97) 修复模型侧问题后的最后 1 个失败。进一步检查确认：用例已经通过公开 API 显式传入 CScale；binding 丢失发生在 TileOP API 的可选 MX Scale 发射路径中，不是 SuperNPUBench 用例漏传操作数。

## 影响版本

| 组件 | 分支 | commit |
| --- | --- | --- |
| SuperNPUBench | `main` | `58d436c01f9a` |
| linx-toolchain-build | `main` | `e6a31efb4cfb` |
| llvm-project | `dev-llvm15_56` | `adcb879481d8` |
| Linx-TileOP-API | `linx` | `955030977c25` |
| SuperScalarModel / gfrun | `codex/pr-0.58.4-shared-model` | `762a72c34305` |

编译器：

```text
clang version 15.0.4
Target: linx64v5-unknown-linux-musl
```

工具链安装目录中的 `template_asm.hpp` 与 Linx-TileOP-API 源码一致，SHA-1 均为：

```text
6ff45e009aa1b8045bb8f3df052772b93a160e72
```

## 复现用例

文件：

```text
microbenchmark/fixp/src/fixp_tmatmul.cpp
```

`MXACC_CSCALE` 使用 FP16 A/B、FP32 C/D 和 U8 CScale：

```cpp
#elif defined(MXACC_CSCALE)
  buf_t<__half, float> buf;
  run_matmul<__half, float>(buf.a, buf.b, buf.d,
      [&](auto &tD, auto &tA, auto &tB) {
        acc_tile_t<TM, TN> cacc;
        load_aux(cacc, buf.aux);
        cscale_tile_t<TM, TN> cscale;
        load_aux(cscale, buf.aux);
        TMATMUL_MX_ACC(tD, cacc, tA, tB,
                       fixp::keep_acc().cscale(cscale));
      });
```

CScale Tile 满足接口静态约束：

```text
dtype: U8
layout: CUBE_M32
valid shape: [M, 1]
storage: Local
```

Linx-TileOP-API 自带测试 `test/tileop_api/src/TMatmulAccCScale.cpp` 也使用相同的公开调用形式：

```cpp
TMATMUL_MX_ACC(d, c, a, b, fixp::keep_acc().cscale(scale));
```

因此调用方没有其他 positional CScale 参数或手工发布 `B.IOT` 的责任。

## 复现命令

```bash
export COMPILER_DIR=/path/to/linx-toolchain-build/output/linx_blockisa_llvm_musl/bin

make -C microbenchmark/fixp \
  TESTCASE=fixp_tmatmul \
  FIXP_MODE=MXACC_CSCALE \
  COMPILER_DIR="$COMPILER_DIR" \
  OBJ_ROOT=/tmp/fixp_issue97_repro \
  diss
```

执行：

```bash
gfrun -t 1 -f \
  /tmp/fixp_issue97_repro/microbenchmark/fixp/elf/fixp/\
fixp_tmatmul_mxacc_cscale_M32_N32_K32_tM32_tN32_tK32.elf
```

## 实际结果

CScale 已由前置 TLOAD 加载到 `N#1`：

```text
BSTART.TMA TLOAD UINT8
B.DATR CUBE_M32.normal
B.IOT -> N<1KB>
```

但随后的 CUBE block 只有 C、A、B 三个 source：

```text
BSTART.CUBE TMATMULMX.ACC, FP16
B.FPATR ..., CScaleEn=1
B.IOT m#1, mask=1111       # C
B.IOT t#1, mask=1111       # A
B.IOT u#1, mask=1111       # B
B.IOT mask=1111, last, ->n<4KB>  # D
```

缺少：

```text
B.IOT n#1, mask=1111       # CScale
```

gfrun 结果：

```text
gfrun: illegal instruction: ASSERTION FAILED:
block.srcTile.size() == expectedSources &&
"CUBE local source stream does not match the current ASL contract"
```

## 预期结果

启用 `CScaleEn` 后，TileOP API 应在 C、A、可选 ScaleA、B、可选 ScaleB 之后发布 final Local CScale source：

```text
B.IOT m#1, mask=1111       # C
B.IOT t#1, mask=1111       # A
B.IOT u#1, mask=1111       # B
B.IOT n#1, mask=1111       # CScale
B.IOT mask=1111, last, ->t<4KB>  # D
```

模型应能够完成执行并返回：

```text
Reach the End of Benchmark
R2 = 0
```

## 根因分析

`TMATMUL_MX_ACC(..., options)` 已正确取出 `options.CScale`：

```cpp
auto &cscale = pto_matmul_detail::select_fixp_operand<Attr.CScaleEn>(
    options.CScale, c);
```

完整 MX Scale 路径（`ScaleMask == 3`）使用 `PTO_FIXP_MX_ACC_EMIT_*`，其中已经：

1. 把 `[CScaleOperand] "Tr"(cscale.data())` 放入 inline-asm 输入约束；
2. 通过 `PTO_FIXP_ACC_CSCALE` 条件发射 CScale `B.IOT`。

但 `ScaleMask == 0/1/2` 时改走 `PTO_MX_DISPATCH_OPTIONAL(PTO_MX_OPT_ACC_*)`。当前可选路径存在两处遗漏：

```cpp
#define PTO_MX_EXTRA_ACC [C] "Tr"(c.data()),
```

这里只声明 C，没有声明 `CScaleOperand`；同时：

```cpp
#define PTO_MX_OPT_ACC_L(...) \
  PTO_MX_OPTIONAL_EMIT("TMATMULMX.ACC", L, \
                       "B.IOT %[C], mask=1111\n", \
                       "", ACC, ...)
```

`SUFFIX` 为空，没有追加 `PTO_FIXP_ACC_CSCALE`。

本用例使用 FP16 A/B。按照 MX 类型契约，FP16/BF16 不需要 E8M0 Scale，因此它合法地选择 `ScaleMask == 0`，恰好触发上述遗漏。

## 建议修复

为矩阵 `TMATMULMX.ACC` 的可选-scale路径增加独立的 CScale-aware extra operand，并在矩阵源之后追加已有的 `PTO_FIXP_ACC_CSCALE`：

```diff
 #define PTO_MX_EXTRA_ACC [C] "Tr"(c.data()),
+#define PTO_MX_EXTRA_ACC_CSCALE \
+  [C] "Tr"(c.data()), [CScaleOperand] "Tr"(cscale.data()),

-#define PTO_MX_OPT_ACC_L(M,S,O,I) \
-  PTO_MX_OPTIONAL_EMIT("TMATMULMX.ACC",L,\
-    "B.IOT %[C], mask=1111\n","",ACC,M,S,O,I)
+#define PTO_MX_OPT_ACC_L(M,S,O,I) \
+  PTO_MX_OPTIONAL_EMIT("TMATMULMX.ACC",L,\
+    "B.IOT %[C], mask=1111\n",PTO_FIXP_ACC_CSCALE,\
+    ACC_CSCALE,M,S,O,I)
```

同样修改 Local、SharedA、SharedB、SharedAB 四种 storage 路径。

需要保留原 `PTO_MX_EXTRA_ACC`，因为 TGEMVMX.ACC 的现有可选-scale宏也使用该名称，但当前 TGEMV emitter 没有 `cscale` 变量；直接全局修改 `PTO_MX_EXTRA_ACC` 会破坏 TGEMV 编译。

## 最小补丁验证

使用上述补丁覆盖头文件进行验证，未修改 SuperNPUBench 用例源码。

补丁后反汇编出现预期的第四个 source：

```text
B.IOT m#1, mask=1111
B.IOT t#1, mask=1111
B.IOT u#1, mask=1111
B.IOT n#1, mask=1111
B.IOT mask=1111, last, ->t<4KB>
```

同一 gfrun 模型执行结果：

```text
gfrun_rc=0
Reach the End of Benchmark
R2 = 0
```

该 A/B 对照说明模型对原始不完整 source stream 的拒绝是正确的，问题位于 TileOP API 发射模板，而不是模型或 SuperNPUBench 算子。

## 建议补充测试

建议将现有 `TMatmulAccCScale.cpp` 从“只验证编译通过”扩展为反汇编/编码检查，并至少覆盖：

| ScaleMask | 输入类型示例 | 预期 CScale binding |
| ---: | --- | --- |
| 0 | FP16 × BF16 | 存在 |
| 1 | FP8 × FP16 | 存在 |
| 2 | BF16 × FP8 | 存在 |
| 3 | FP8 × FP8 | 存在 |

同时覆盖 Local、SharedA、SharedB 和 SharedAB，避免完整-scale路径通过、可选-scale路径漏操作数的问题再次发生。

