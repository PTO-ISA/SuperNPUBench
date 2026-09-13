# [Issue] emulator 的 fp32→e4m3(`TCVT`) 用 IEEE 语义把 ≥2⁸ 的值溢出成 +inf(`0x78`)，OCP E4M3 应为正规数（上限 448），量化输出 block-max 恒错【已解决】

## 摘要

`dynamic_mx_quant` OCP 探针 kernel 的数据路径最后一步 `TCVT fp32 → e4m3`
（`probe_dynamic_mx_quant_tail_ocp_fp8.hpp:154`、newcalc 探针 `:133`）把归一化后的
`x/scale` 落成 e4m3 输出。编译、反汇编、`gfrun` 执行全部通过（`R2=0`，无断言），但**数值不对**：
凡 `x/scale ≥ 2⁸=256` 的元素一律输出 `0x78`，而按 OCP E4M3（无 inf、最大正规数 448）应给出
对应正规数（如 320→`0x7a`、448→`0x7e`）。

根因在 **emulator 的 softfloat 把 e4m3 当成「IEEE 风格、exp 全 1 保留给 inf/NaN」的
1-4-3 浮点**：`float8_params` 的 `exp_max=15` 被当作 inf/NaN 指数，任何舍入后 biased-exp≥15
（即数值 ≥2⁸）的输入触发 IEEE **overflow → +inf**，inf 编码 `0_1111_000 = 0x78`。
它在位型上恰好与「把 256 当正规数」重合，故此前被误读成「饱和到 256」——实际是**溢出成
无穷**。OCP E4M3 规范里 exp=1111 是正规数区间（256…448），只有 `S.1111.111` 才是 NaN，
无 inf。缺陷所在仓 **`SuperScalarModel`（emulator softfloat 建模）**，非 ISA 编码、非工具链、
非 kernel 逻辑。

## 环境

| 项 | 值 |
|---|---|
| 编译仓 | SuperNPUBench `benchmark/one-level-arch` |
| 执行仓 | SuperScalarModel（`bin/gfrun` 执行 elf） |
| 工具链 | `linx_blockisa_llvm_musl`，clang-15，target `linx64v5-linux-musl` |
| 复现根目录 | `benchmark/one-level-arch/test/kernel/quant/dynamic_mx_quant` |
| 触发 kernel | `kernels/quant/dynamic_mx_quant/probe_dynamic_mx_quant_tail_ocp_fp8.hpp`（`TYPE=PROBE_OCP_FP8`）；newcalc 变体同理 |

### 各仓代码分支（复现基线）

| 仓库 | 分支 | commit |
|---|---|---|
| SuperNPUBench（我的 fork，含探针，`github.com:ziyang-cheng/SuperNPUBench.git`） | `feat/dmxq-rel0812` | `c3401fd` |
| SuperScalarModel（无个人 fork，上游 + 下述两前置 issue） | `feat/pto-v058-adaptation` + #253 + #254 | 见下 |
| linx-toolchain-build | `main` | `e6a31ef` |
| └ Linx-TileOP-API（`github.com/ziyang-cheng/Linx-TileOP-API`） | `feat/v058-reinterpret-cmpmode-backfill` | `cb47f6d` |

**SuperScalarModel 复现前置（无个人 fork，仓库 `github.com/LinxISA/SuperScalarModel`）**：本探针执行流须先应用以下两个已提交的上游 issue 修复，否则执行流在抵达末端 `TCVT fp32→e4m3` 之前即被拦截，无法复现本缺陷：

- **#253** `fix(emulator): implement bf16->e8m0 (SF8) TCVT conversion`：scale pass 的 `TCVT bf16→e8m0` 直转，未修前命中 “Not support such type convert yet”。
- **#254** `fix(emulator): accept zero-instruction reinterpret_tile in logical TEPL`：newcalc 探针 data pass 的 `reinterpret_tile` + scalar-logical（`TXORS`）零指令位重解释，未修前逻辑 TEPL 校验按严格 dtype 相等误杀。

> 复现基线即「上游 `feat/pto-v058-adaptation` 之上 cherry-pick #253、#254 两补丁」。本缺陷本身
> 位于纯 softfloat 数值转换层，与上述两前置补丁无关（它们只是让执行流能抵达数据路径末端）；
> 只要执行流跑到末端的 `TCVT fp32→e4m3` 即触发。

## 复现命令与观测

默认探针 `x=4.0`(0x4400) **无法暴露**本缺陷：4.0 是 block max，`E_max=2`、`scale=2⁻⁶`，
`4.0×64=256` 恰落在 inf 边界，OCP-256 与 inf 位型都是 `0x78`，二者巧合重合。

**用 `x=5.0`(0x4500) 即可暴露**（`320` 应为 `0x7a`）：

```bash
export COMPILER_DIR=.../linx_blockisa_llvm_musl/bin
export LINX_SYSROOT=.../linx_blockisa_llvm_musl/sysroot/usr
cd benchmark/one-level-arch/test/kernel/quant/dynamic_mx_quant
# 把 src/probe_ocp_fp8_newcalc.cpp 的填充值改成 0x4500 (5.0)
make TESTCASE=dynamic_mx_quant TYPE=PROBE_OCP_FP8_NEWCALC
SuperScalarModel/bin/gfrun -f output/.../dynamic_mx_quant_probe_ocp_fp8_newcalc.elf \
    --dump-memory 0x15000:256:/tmp/y.bin        # y@0x15000
```

观测：`scale=0x79`（`2⁻⁶`，**正确**），`y=0x78`（应为 `0x7a=320`，**错误**）。

### 决定性证据（非均匀块，一次喂多值，同 block `recip=64`）

| 输入 | `x/scale` 算术值 | biased e4m3 exp | OCP 应得 | 实测 | 判定 |
|---|---|---|---|---|---|
| 5.0 | 320 | 15 (=exp_max) | `0x7a` | **`0x78`** | 溢出→inf |
| 4.5 | 288 | 15 | `0x79` | **`0x78`** | 溢出→inf |
| 3.0 | 192 | 14 | `0x74` | **`0x74`** | 正确 ✓ |
| 1.0 | 64  | 13 | `0x68` | **`0x68`** | 正确 ✓ |

biased-exp ≤14（<256）精确转换，biased-exp=15（≥256）一律 `0x78`。**证明是输出端
`TCVT fp32→e4m3` 的溢出语义，与 scale 计算、倒数算法（`TRECIP` 或位补）均无关**——原探针
与 newcalc 探针逐字节一致。

### 为何每个 OCP kernel 的 block-max 都中招

MX 构造下 `scale=2^(E_max−8)`，block max ∈ `[2^E_max, 2^(E_max+1))` ⇒
`max/scale ∈ [2⁸, 2⁹)=[256, 512)` —— **永远踩在 biased-exp=15 的溢出区**。故 block-max 元素
恒输出 `0x78`，与输入具体值无关。验证 mx_quant 正确性时应看 **sub-max 元素**。

## 根因分析

### 转换链路

`TCVT fp32→e4m3` 在 emulator 里经 funcMap 分派到 softfloat：

```
FloatPointUtils.cpp:335   funcMap[{OPCVT_FP32, OPCVT_FP8}] -> float32_to_float8(...)
softfloat.c:3226          float32_to_float8 -> float8_round_pack_canonical
softfloat.c:1673-1676     float8_round_pack_canonical -> parts_uncanon(p, s, &float8_params)
```

### 格式定义：exp=15 被保留给 inf/NaN

```c
// softfloat.c:479-481
static const FloatFmt float8_params = { FLOAT_PARAMS(4, 3) };   // E=4, F=3
// 展开 (softfloat.c:460-463):
//   exp_size=4, exp_bias=((1<<4)-1)>>1 = 7, exp_max=(1<<4)-1 = 15
```

`exp_max=15` 在**通用 IEEE 路径**里是「inf/NaN 专用指数」，故最大**有限**正规数只能到
biased-exp=14（`1.111₂×2⁷ = 240 = 0x77`），任何要落到 biased-exp≥15 的输入按 IEEE 溢出成 inf。

### 溢出分支：≥exp_max → +inf(`0x78`)

```c
// softfloat-parts.c.inc:199-209 (parts64_uncanon_normal, 非 arm_althp 分支)
} else if (unlikely(exp >= exp_max)) {           // exp==15 即触发
    flags |= float_flag_overflow | float_flag_inexact;
    if (overflow_norm) {                          // 仅 RTZ 等方向为真
        exp = exp_max - 1; frac_allones(p);       // -> 0x77 (240)
    } else {                                       // RNE(默认) 走这里
        p->cls = float_class_inf;
        exp = exp_max;    frac_clear(p);           // -> 0_1111_000 = 0x78 (+inf)
    }
}
```

RNE 默认下 `320.0`（biased-exp=15）命中 `else`，pack 成 `0x78`——**这是 +inf 编码，不是数值 256**。

### 正解模板已在同文件里：`arm_althp`

softfloat 对 ARM Alt-HP fp16 已有「无 inf/nan、指数区间加宽」的特例，正是 OCP E4M3 需要的语义：

```c
// softfloat-parts.c.inc:190-198
if (fmt->arm_althp) {
    /* ARM Alt HP eschews Inf and NaN for a wider exponent.  */
    if (unlikely(exp > exp_max)) {                 // 注意是 > 而非 >=
        flags = float_flag_invalid;
        exp = exp_max; frac_allones(p);            // 溢出 -> 最大正规数
    }
}
```

## 定性

- **属 emulator softfloat 建模缺陷（e4m3 用了错误的 IEEE-with-inf 格式语义），非 ISA/工具链/kernel 缺陷。**
- `TCVT` 到 e4m3 是 MX 量化产出的标准末步，ISA 合法、工具链发射的指令正确（`B.DATR FP8` + `TCVT`）；
  真实硬件按 OCP E4M3 语义转换即可。
- emulator 只是把 e4m3 建成了「1-4-3 IEEE + inf/nan」，与 OCP「1-4-3、无 inf、max=448、仅 `1111.111`=NaN」
  不符，于是 ≥256 溢出成 inf。

## 修复建议（emulator 侧）

给 e4m3 一个 OCP 专用格式语义（类比 `arm_althp`，但保留 `1111.111` 为 NaN）。方案二选一：

1. **新增 OCP 标志**：为 `float8_params` 加 `ocp_e4m3=true`，在 `parts64_uncanon_normal`
   增一分支：`exp > exp_max` 时溢出→最大正规数（`exp=exp_max`，`frac=110`=`0x7e`=448），
   NaN 单独走 `1111.111`。同时 `parts_canonicalize`（`softfloat.c:1629` 一线）里把
   `exp==exp_max && frac!=0x7` 当正规数而非 inf/nan。

2. **参数化 emax_normal**：给 `FloatFmt` 加「最大正规 biased-exp」字段，e4m3 设为 15、
   inf 判定改用 `frac==0x7` 组合，避免复用通用 IEEE 溢出路径。

改后期望：`320→0x7a`、`288→0x79`、`448→0x7e`、`≥448→0x7e`(饱和) 或按舍入到 NaN 的规范处理；
`<256` 分支不受影响（本就正确）。

### 建议的回归

- 探针 `x=5.0`：`scale=0x79`、`y=0x7a`（当前 `0x78`）。
- 非均匀块 `{5.0,4.5,3.0,1.0}`：`{0x7a,0x79,0x74,0x68}`（当前 `{0x78,0x78,0x74,0x68}`）。

## 影响面

- 影响**任何输出 e4m3 的 kernel**（OCP-FP8 量化、cuBLAS-FP8 等）：凡量化后落在 `[256,512)`
  的元素被写成 `0x78`(inf) 而非对应正规数。由 MX 构造，**每个 block 的最大元素必然中招**。
- 逆向（e4m3→fp）解码不受本 issue 影响；缺口只在**产出端** fp→e4m3 的溢出语义。
- 同样的 IEEE-with-inf 格式定义也用于 e5m2（`float8_1_params`，`FLOAT_PARAMS(5,2)`）；若目标
  ISA 的 e5m2 遵循 OCP（e5m2 确有 inf/nan，语义与 IEEE 一致），则 e5m2 无需改，仅 e4m3 需修。
