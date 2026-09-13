# ISSUE: TCVT fp→E2M1/E1M2 编码最近邻搜索漏 code 0（小值抬到最小正档，偏离 RNE）；且 pto-spec 参考模型无 E2M1 目的编码器

- **归属仓库**: SuperScalarModel（gfrun / emulator）+ pto-spec（参考模型缺口）【SuperScalarModel issue558】
- **状态**: 已定位根因（对照 pto-spec 权威 ASL），未修
- **对应 RECORD**: 问题27
- **严重度**: 非阻塞（compare 判据 `MSE<0.1`，本偏差 MSE=0.022 仍 pass），但**非 byte-exact、偏离 RNE/golden**

## 组件清单（暴露此问题的版本组合）

| 组件 | 版本 | 备注 |
|---|---|---|
| SuperScalarModel（gfrun） | 分支 `codex/consolidate-post-main-fixes-20260903`，tip `49547742` | E2M1 编码器（PR#510 `eededa48` 引入）最近邻漏 code 0 |
| 工具链 llvm-project | `dev-llvm15_56` @ `1ae4ee39` | clang 15.0.4 |
| 工具链 Linx-TileOP-API | `linx` @ `804eb03` | — |
| 工具链 musl | `af0dfc20` | — |
| Bench 算子代码 | PR: https://github.com/PTO-ISA/SuperNPUBench/pull/111 （fork 分支 `ziyang-cheng/SuperNPUBench:dmxq-ops-20260904`） | 触发 kernel/驱动/金标脚本 |
| pto-spec（规范核对） | `6e3e056f` | `TCVT.asl` / `e2m1x2.asl` / `matrix-quantization.asl` |

## 前置依赖（到达本问题所需的其它 issue）

在干净 `49547742` 上，`tail_ocp_fp4` res_check 须先解开以下才能到达本问题（否则更前置的墙先崩）：
1. `ppoll` handler 缺失（**SuperScalarModel issue554**）。
2. compare/select 源 dtype 相等误杀 reinterpret 视图（**SuperScalarModel issue254** 的 compare/select sibling）。
3. 打包 4-bit NORM TSTORE 不支持（**SuperScalarModel issue557** / RECORD 问题26）。

解开后执行流跑到底、可落盘比对，本问题以**小商值精度偏差**显形。

## 一、现象

`dynamic_mx_quant_tail_ocp_fp4`（512×256，4-PE）data pass 末 `TCVT(oq, xf)`（fp32→打包 fp4/e2m1）对**小商值** `|x/scale| < 0.25` 应 RNE 到 `0.0`，emulator 却编成最小正档 `0.5`：
```
dynamic_mx_quant_tail_ocp_fp4: output=fail-ish (MSE=0.022015, MaxAE=0.500000), scale=pass
```
逐元素核对（row0 block1，scale=0.5，`x/scale = x*2`）：

| 列 | x | x/scale | golden(RNE) | emulator |
|---|---|---|---|---|
| 32 | 0.0947 | 0.1895 | **0.0** | **0.5** ❌ |
| 39 | 0.0344 | 0.0688 | **0.0** | **0.5** ❌ |
| 40 | −0.0859 | −0.1719 | **−0.0** | **−0.5** ❌ |
| 34 | −0.1689 | −0.3379 | −0.5 | −0.5 ✓（>0.25 档，两侧一致） |

即凡 `|v| < 0.25`（第一档中点 `table[1]/2 = 0.25`）的非零小值，被错误抬到 `±0.5` 而非舍到 `0.0`。

## 二、复现步骤

```bash
export COMPILER_DIR=/path/to/.../bin
export LINX_SYSROOT=/path/to/.../sysroot/usr
cd SuperNPUBench/benchmark/one-level-arch/test/kernel/multi_thread/quant/dynamic_mx_quant

python3 src/gen_dynamic_mx_quant_data.py --M 512 --K 256 --block-size 32 \
    --algo OCP --kernel tail --dtype FP4 --in-dtype bf16 --scale-layout compact --seed 42 \
    -o ../../../../compare/dynamic_mx_quant_tail_ocp_fp4
make TESTCASE=dynamic_mx_quant TYPE=TAIL_OCP_FP4 res_check=on diss
bin/gfrun -f .../dynamic_mx_quant_tail_ocp_fp4.elf -s softcore.multiThreadNum=4 -t 1   # 前置 3 issue 解开后
python3 src/dynamic_mx_quant_data_compare.py -d .../dynamic_mx_quant_tail_ocp_fp4.elf \
    --dtype FP4 --scale-layout compact --cmp-root ../../../../compare
# output MSE=0.022（MaxAE=0.5，1 LSB）：小商值 0.0→0.5
```

## 三、根因（对照 pto-spec 权威 ASL）

**(1) emulator 编码器漏 code 0。** `emulator/engine/CubeEngine.cpp` `DataFormatCvt` 的 E2M1/E1M2 分支最近邻搜索从 `code 1`（`table[1]=0.5`）起、循环 `candidate=2..7`，**从不考虑 `code 0`（`0.0`）**：
```cpp
uint64_t best = 1;
double bestDistance = std::fabs(magnitude - table[1]);   // 起点=0.5，跳过 0.0
for (uint64_t candidate = 2; candidate < 8; ++candidate) { ... }
```
故任意非零 magnitude 最小只能编成 `0.5`（E1M2 则 `0.25`）。

**(2) 规范要求 RNE，含 0.0。** `TCVT.asl` `InstructionContractDefaultRounding_TCVT`：float→float 默认 **RNE**。E2M1X2 格式（`asl/arch/data-types/formats/e2m1x2.asl`）`has_zero=TRUE`、值集含 `0.0`。RNE 下 `|v| < 0.25` 的最近可表示值是 `0.0`（距 0.19 为 0.19，距 0.5 为 0.31）→ 应舍到 `0.0`。golden（AscendC 忠实移植）正是 `0.19→0.0`。

**(3) pto-spec 参考模型缺 E2M1 编码器（spec gap）。** `ReferenceMatrixFloatingEncoding`（`asl/arch/profile/matrix-quantization.asl`）对非 FP64/FP32/FP16/BF16 目的落到 `ReferenceFP8Encoding`，而后者 `assert destination_type == E4M3 || HiF8` —— **不含 E2M1/E1M2**。即 pto-spec **未定义** fp→E2M1/E1M2 目的的编码/舍入参考实现；emulator 的 E2M1 nearest-search 属实现扩展，其"漏 code 0"未被任何参考 ASL 约束住。

## 四、修复建议

1. **emulator**：最近邻搜索纳入 `code 0`（`best=0`，`bestDistance=|mag-table[0]|`，`candidate=1..7`；ties-to-even 保留）。使 `|v|<0.25` 按 RNE 舍到 `0.0`。
2. **pto-spec**：为 E2M1/E1M2（及 HiF4X2）目的补 RNE 参考编码器，使 `ReferenceMatrixFloatingEncoding` 不再对 4-bit 打包目的 assert，明确定义 fp→打包 4-bit 的舍入契约（含 0.0 / 最小正档边界）。

## 五、影响范围

所有 `TCVT(fp→E2M1/E1M2)` 且存在 `|v|<0.25`（E1M2 `<0.125`）小值的路径。`dynamic_mx_quant` tail/nontail OCP-FP4 的 data pass。因偏差 = 1 LSB（`MaxAE=0.5`）且 `MSE<0.1`，当前 compare 判据下仍报 pass，故为**精度偏差**而非崩溃；但非 byte-exact、偏离 RNE 与 golden。
