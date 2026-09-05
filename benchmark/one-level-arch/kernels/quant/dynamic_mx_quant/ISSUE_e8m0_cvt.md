# [Issue] emulator 未实现 `TCVT` 浮点→e8m0(SF8) 转换，MX 共享 scale 产出命中「不支持类型转换」断言【已解决】

## 摘要

`dynamic_mx_quant` OCP 探针 kernel 在计算出每 block 的 bf16 指数域 scale 后，用一条
`TCVT bf16 → e8m0`（`B.DATR e8m0`，SF8）把它落成 MX 标准的 8-bit 共享 scale。编译与反汇编
均正确，但 `gfrun` 执行该 `TCVT` 时命中 emulator 运行期断言：

```
assert(0 && "Not support such type convert yet")
CubeEngine.cpp:374  DataFormatCvt::OpCvtType
```

根因是 emulator 功能模型**从未实现「浮点→e8m0」这一方向的类型转换**：`OpCvtType`
（`CubeEngine.cpp`）把 `DataType::SF8` 直接归入「尚未支持」的 `assert` 分支，根本不会走到
转换实现 `ConvertAggre`；且转换表 `FloatPointUtils.cpp::InitConvertMapFp` 的 funcMap 里也没有
任何以 SF8 为目标的条目。缺陷所在仓 **`SuperScalarModel`（emulator 建模层）**，非 ISA 编码、
非工具链、非 kernel 逻辑。

## 环境

| 项 | 值 |
|---|---|
| 编译仓 | SuperNPUBench `benchmark/one-level-arch` |
| 执行仓 | SuperScalarModel（`bin/gfrun` 执行 elf） |
| 工具链 | `linx_blockisa_llvm_musl`，clang-15，target `linx64v5-linux-musl` |
| 复现根目录 | `benchmark/one-level-arch/test/kernel/quant/dynamic_mx_quant` |
| 触发 kernel | `kernels/quant/dynamic_mx_quant/probe_dynamic_mx_quant_tail_ocp_fp8.hpp`（`TYPE=PROBE_OCP_FP8`） |

### 各仓代码分支（复现基线）

| 仓库 | 分支 / HEAD | commit |
|---|---|---|
| SuperNPUBench（我的 fork，含探针） | `feat/dmxq-rel0812`  (`github.com:ziyang-cheng/SuperNPUBench.git`)| `79d9140` |
| SuperScalarModel（README 要求的分支，取**最新** tip；另需下方「复现前置」一处补丁） | `feat/pto-v058-adaptation` | `c3051e3a` |
| linx-toolchain-build | `main` | `e6a31ef` |
| └ Linx-TileOP-API | `feat/v058-reinterpret-cmpmode-backfill`（`github.com/ziyang-cheng/Linx-TileOP-API`） | `cb47f6d` |

> **基线说明**：
> - **SuperScalarModel**：`feat/pto-v058-adaptation @ c3051e3a`（tip，含 ADDTPC page-offset 系列
>   `430e7bbb`/`e2c8ad23`/`5adc5d8a`/`8baf731e`，musl crt startup 正常通过，执行流稳定推进到 kernel）。
>   本 e8m0 断言在 OCP 链后段、属 kernel 深处，越过 startup 后还需下方「复现前置」一处补丁越过更早的
>   440（问题14）检查才抵达——它与本 e8m0 缺陷本身无关。**缺陷在此基线仍存在**：
>   `c3051e3a` 的 `CubeEngine.cpp:374` 仍是 `assert(0 && "Not support such type convert yet")`（SF8
>   分支）、`FloatPointUtils.cpp` 无任何 OPCVT_SF8 目标条目。
> - **Linx-TileOP-API**：`https://github.com/ziyang-cheng/Linx-TileOP-API.git` 的
>   `feat/v058-reinterpret-cmpmode-backfill @ cb47f6d`，单个 checkout 即得完整复现基线（含 OCP 链零指令
>   位重解释所用的 `reinterpret_tile`）。

### 复现前置（一处必要补丁）

本断言位于 kernel OCP 链的后段，要执行到它，需先让 gfrun 越过一处更早的运行期检查。以下这项
补丁只是**让执行流抵达本断言的前置条件**，与本 e8m0 缺陷本身无关：

1. **emulator 逻辑 TEPL 源 dtype 校验放宽为位宽相等**：`AccumulateBlockInfo.cpp` 的
   `ValidateScalarLogicalTepl` 把 `source->tileInfo->dataType == block->dataType` 改为
   `BytesOf(source->tileInfo->dataType) == BytesOf(block->dataType)`（否则 OCP 链里
   零指令位重解释产出的 u16 掩码源会先被误杀）。

打上上述一处补丁后，执行流稳定推进到 OCP scale 的 `TCVT bf16→e8m0`，即触发本 issue 的断言。

## 复现命令

```bash
export COMPILER_DIR=.../linx_blockisa_llvm_musl/bin
export LINX_SYSROOT=.../linx_blockisa_llvm_musl/sysroot/usr
cd benchmark/one-level-arch/test/kernel/quant/dynamic_mx_quant
make TESTCASE=dynamic_mx_quant TYPE=PROBE_OCP_FP8 diss   # 编译+反汇编，clean

SuperScalarModel/bin/gfrun -t 1 -f "$PWD/output/.../dynamic_mx_quant_probe_ocp_fp8.elf"
```

## 报错

命中断言的指令（gfrun 追踪）：

```
B11  T0 BPC 0x11386 [TEPL] [FALL] [TCVT]
M54  TPC:0x11386  BSTART.TEPL TCVT BF16
M55  TPC:0x1138a  B.DATR DR, NORM.normal, SF8, none      # dst = e8m0(SF8)
M56  TPC:0x1138e  B.IOT ... T#1 -> U<256B>
...
gfrun: CubeEngine.cpp:374:
  DataFormatCvt(...)::<lambda(DataType)>:
  Assertion `0 && "Not support such type convert yet"' failed.   (EXIT=1)
```

反汇编对应块（`->u<256B>` 即 e8m0 scale 输出）：

```
11384: BSTART.TEPL	TCVT, BF16
11388: B.DATR	e8m0, byte0, Null
1138c: B.IOT	t#1, mask=1111, last, ->u<256B>
```

## 根因分析

### 转换分派：SF8 落在「尚未支持」assert 分支

TEPL `TCVT` 在 emulator 里经 `CubeEngine.cpp::DataFormatCvt` 处理，其内部 lambda `OpCvtType`
把 `DataType` 映射成转换用的 `OPConvertType`。`DataType::SF8` 与 `TF32/HF32/HIF8/FP6/…`
一起被列进「尚未支持」分支：

```cpp
case DataType::SF8:
case DataType::INT4:
case DataType::UINT4:
    assert(0 && "Not support such type convert yet");   // <-- 命中
    return OPConvertType::OPCVT_NOT;
```

即执行流在**映射阶段**就断言退出，根本不会进入实际的转换实现 `ConvertAggre`。

### 转换表也缺 SF8 目标条目

即便放开上面的映射，`FloatPointUtils.cpp::ConvertAggre` 用一张
`funcMap[{from,to}] -> lambda` 表做实际数值转换（`InitConvertMapFp` 填充）。核对该表：**没有
任何一条以 SF8 为目标**（`{OPCVT_BF16, OPCVT_SF8}` / `{OPCVT_FP32, OPCVT_SF8}` 等均不存在）。
命中未注册对时 `ConvertAggre` 会二次断言 `"convert inst error type!!"`。故 emulator 对
「浮点→e8m0」这一整方向从未建模。

### 逆向（e8m0→浮点）已存在，佐证编码语义

同一功能模型里，e8m0 的**反解**是实现了的——`CubeCalculate.cpp:416-419`（MX 矩阵乘 dequant）：

```cpp
uint8_t scaleU8 = static_cast<uint8_t>(scaleValue);   // e8m0 byte E
int k = static_cast<int>(scaleU8) - 127;              // bias 127
float scaleF = std::ldexp(1.0F, k);                   // value = 2^(E-127)
```

即 e8m0 是**无符号、无尾数的 8-bit 指数，bias 127**——与 bf16 的指数字段（8-bit、同 bias
127）编码一致。缺的只是正向（浮点→e8m0）这一半。

## 定性

- **属 emulator 建模缺失（未实现的合法转换），非 ISA/工具链/kernel 缺陷。**
- `TCVT` 到 e8m0 是 MX 量化产出共享 scale 的标准手法，ISA 合法、工具链发射的指令正确
  （`B.DATR e8m0` + `TCVT`）；真实硬件按 `B.DATR` 的目标格式执行转换即可。
- emulator 只是没把这条转换实现进功能模型，于是在类型映射处直接 assert 兜底。

## 修复建议（emulator 侧）

e8m0 与 bf16 共享 8-bit 指数、同 bias 127，故 bf16→e8m0 = 丢符号、把 7 位尾数按 RNE 舍入进
指数字段。两处改动：

1. `CubeEngine.cpp` `OpCvtType`：把 `DataType::SF8` 从 assert 分支移出，映射为
   `OPConvertType::OPCVT_SF8`（`OPCVT_SF8` 已在枚举中，`GetTypeWidth` 已按 8-bit 处理）。

2. `FloatPointUtils.cpp` `InitConvertMapFp`：新增 `{OPCVT_BF16, OPCVT_SF8}` 转换 lambda：

```cpp
uint32_t exp  = (data >> 7) & 0xFF;   // bf16 指数字段
uint32_t mant =  data       & 0x7F;   // bf16 尾数
if (exp == 0xFF) return 0xFF;         // inf/nan -> e8m0 NaN 码
if (mant > 0x40 || (mant == 0x40 && (exp & 1))) exp++;  // RNE 舍入进指数
if (exp > 0xFE) exp = 0xFE;           // 饱和，避开 0xFF(NaN)
return exp & 0xFF;
```

对 OCP 的输入（scale 尾数已被清零、恒为 2 的幂）该转换**精确等于指数字段抽取**，无舍入误差；
对一般 bf16 输入亦给出 RNE 到最近 2 的幂的 e8m0 结果。

> 若后续需要，`{OPCVT_FP32, OPCVT_SF8}`、`{OPCVT_FP16, OPCVT_SF8}` 可按同样「取指数 + RNE
> 尾数 + 校正 bias」思路补齐（fp32 指数 bias 127、fp16 bias 15，注意各自 bias 与尾数位宽）。

### 实测验证

打上以上两处改动、重建 gfrun（仅 `CubeEngine.cpp` / `FloatPointUtils.cpp` 重编）后，探针
`gfrun` **跑到底**：23 blocks / 120 insts、`R2 = 0`（Success to Reach End of Benchmark），
完整 OCP 链 `TCVT→TANDS(u16)→TMULS→TCVT bf16→e8m0→TSTORE(e8m0)` 及后续 data 路径全部执行
无断言。（该 probe 为全零输入的执行 smoke test，尚无 golden 数值对比。）

## 影响面

- 阻断任何用 `TCVT` 落 e8m0 共享 scale 的 kernel 在 gfrun 上的执行——这是 MX 系量化
  （OCP/dynrange 等）产出标准 8-bit block scale 的通用最后一步。
- 逆向 dequant（e8m0→fp）已可用，故 TMATMUL_MX 消费端不受影响；缺口只在**产出端**的正向转换。
