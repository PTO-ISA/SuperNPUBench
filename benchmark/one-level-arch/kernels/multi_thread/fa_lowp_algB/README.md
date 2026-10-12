# FA AlgoD（optD）调度开关

AlgoD 内核在 [`fa_lowp_algD.hpp`](fa_lowp_algD.hpp)，编译、校验和性能入口在
[`test/kernel/multi_thread/fa_lowp_algB`](../../../test/kernel/multi_thread/fa_lowp_algB)。
下文命令均从该测试目录执行。

## 计算与布局

- 四个 PE 协作处理一个 128 行 Q tile，每个 PE 处理 32 行。
- Q/K/V 为 packed E2M1 FP4，每个字节存两个值；scale 为 E8M0，每 32 个逻辑元素一组。
- QK 输出 BF16 score，同时融合 RowMax 与每 32 列的 GroupMax（GM2）。
- 每组以 `L = GroupMax - ln(4)` 为参考，计算 `exp(score-L)` 并转换成 FP4；
  `exp(L-R)` 向下取整为 E8M0 scale，`R` 为在线行参考值。
- 分母来自 FP4 转换前的 BF16 指数值：每组独立更新指数和及 log reference，
  最后一个 KV 块才把各组换算到 `R` 并合并。
- PV 使用 FP32 累加器，通过 CScale 合入在线缩放；中间块请求 InternalAcc/L0C
  驻留，末块发布至 CellReg，归一化后输出 BF16。`R` 的 ln2 格点计算使用 BF16，
  包含舍入误差。

默认 head dimension 为 `FA_QD=FA_VD=128`。Q/K scale 的 GM 布局分别为
`[Sq, QD/32]`、`[Skv, QD/32]`；V scale 的物理布局为 `[VD, Skv/32]`。
Q 的 `1/sqrt(QD)` 缩放由输入量化阶段处理，内核没有额外的 attention scale 乘法。

## 调度变体与开关映射

使用 `run.py --algorithm D` 选择 AlgoD；脚本自动设置
`ALGO_C=0 ALGO_D=1 GM_FUSED=2`。脚本的默认算法是 C，使用 D 时必须显式选择。

| 调度 | `run.py` 参数 | `D_PRELOAD` | `D_PRELOAD_N` | `D_DBLQ` | `D_CONST_KV_ADDR` |
| --- | --- | ---: | ---: | ---: | ---: |
| base | `--d-variant base` | 0 | 0 | 0 | 0 |
| preloadN | `--d-variant preload --preload-count N` | 1 | N | 0 | 0 |
| dblQ | `--d-variant dblQ` | 0 | 0 | 1 | 0 |
| dblQ + preloadN | `--d-variant dblQ_preload --preload-count N` | 1 | N | 1 | 0 |
| dblQ + constKV | `--d-variant dblQ --d-const-kv-address` | 0 | 0 | 1 | 1 |
| dblQ + preloadN + constKV | `--d-variant dblQ_preload --preload-count N --d-const-kv-address` | 1 | N | 1 | 1 |

Makefile 变量对应内核宏：

| Makefile 变量 | 内核宏 | 默认值及含义 |
| --- | --- | --- |
| `ALGO_D` | `FA_ALGB_ALGO_D` | 0；置 1 时同时设置 `ALGO_C=0` |
| `GM_FUSED` | `FA_ALGB_GM_FUSED` | C/D 必须为 2 |
| `D_PRELOAD` | `FA_ALGB_D_PRELOAD` | 0；置 1 开启 K/KScale 持续预取 |
| `D_PRELOAD_N` | `FA_ALGB_D_PRELOAD_N` | 0；预取开启时必须为 2、3、4 或 5 |
| `D_DBLQ` | `FA_ALGB_D_DBLQ` | 0；置 1 开启两个不同 Q tile 共享 KV |
| `D_CONST_KV_ADDR` | `FA_ALGB_D_CONST_KV_ADDR` | 0；置 1 固定读取 KV 第 0 块，仅计时实验 |

### base

一个 Q tile 遍历全部 KV 块。Q/QScale 在进入 KV 循环前加载；每块先加载
K/KScale，完成 QK 和 softmax 后加载 V/VScale，再做 PV。

### preloadN

只预取 K/KScale，V/VScale 仍在本块中加载。N 表示持续前瞻的 KV 块数：
先加载前 N 块，每次迭代在计算当前块前提交 `kb+N` 的加载。
使用 N+1 个环形槽，复用上一迭代已消费的槽，避免覆盖当前 K。
实际 N 截断为 `min(N, Skv/Tk)`；尾部没有未来块时不再提交加载。

这是持续前瞻预取，包含 prologue、环形槽复用及尾部处理。
`--preload-count` 默认 2，只有 `preload` 和 `dblQ_preload` 使用它；
其余变体由脚本设置 `D_PRELOAD_N=0`。增大 N 会增加 tile 驻留需求，
是否提速需在相同 shape 和模型配置下测量。

### dblQ

两个不同的 128 行 Q tile 共享每块 K/KScale、V/VScale。Q0/Q1 分别保存
行参考值、各组分母及 FP32 PV 累加链，最后各自归一化、各自存储。
普通 dblQ 的源码提交顺序为：

```text
load KV -> QK0 -> softmax0 -> PV0 -> QK1 -> softmax1 -> PV1
```

`Sq` 必须为 256 的正整数倍。它减少成对 Q 的重复 KV 加载，但增加了
Q 和在线状态的驻留需求；性能收益取决于模型配置。

### dblQ_preload

同时启用上述双 Q 共享和 K/KScale 环形预取，两者可以组合。
每个预取 K 块供 Q0/Q1 使用；每次迭代只加载一次 V/VScale。

### constKV（仅计时实验）

必须搭配 `dblQ` 或 `dblQ_preload`。所有 KV 迭代都读取第 0 块的
K/KScale/V/VScale，循环次数、Q0/Q1 和在线状态仍保留。
这会改变输入语义，并影响缓存复用；结果仅用于分析固定地址条件下的性能，
不能当作正常 attention 的功能结果或吞吐。
`run.py` 允许 `--mode build/perf`，拒绝 constKV 的 `--mode check`。

## 环境与约束

```bash
export COMPILER_DIR=/path/to/linx_blockisa_llvm_musl/bin
export MODEL=/path/to/SuperScalarModel
```

`COMPILER_DIR` 指向包含 `clang++` 的目录；`MODEL` 指向包含
`bin/gfrun`、`bin/gfsim` 和模型配置的目录。当前脚本即使在 build 模式也读取这两个变量。
工具链的 TileOP API 需支持 BF16 fused RowMax/GroupMax、packed group 的
`TROWEXPANDEXPDIF` 广播、MX matmul 的 CScale/raw_acc/acc_hint 等当前接口。
如果工具链默认头文件不匹配，可在以下命令中增加
`--clang-resource /path/to/compatible/clang/resource`；该目录应包含 `include/`。
具体地，region 版 `TROWEXPANDEXPDIF` 必须允许一个与输出同 shape 的矩阵
subview，加一个 BF16 packed CELL 广播源。如果头文件把两个源都要求为
`ValidCol <= CubeCellCols` 并要求列数相等，会在编译时触发 static_assert；
应使用已修正该接口约束的 API resource。

- 当前测试入口使用 `Tm=128`、`Tk=128`，四 PE 执行。
- `Sq` 必须为 128 的正整数倍；dblQ 系列必须为 256 的正整数倍。
- `Skv` 必须为 128 的正整数倍；不支持 KV 尾块。
- QD 与 Tk 必须是 32 的倍数；当前 Tk=256 路径未开放。
- `--run-id` 必须是新的目录名，已存在时脚本直接报错，避免覆盖记录。

## 使用示例

先用小 shape 编译各变体；以下均为 kernel-only 编译（`res_check=off`）：

```bash
python3 run.py --mode build --algorithm D --d-variant base \
  --sq 256 --skv 768 --tk 128 --run-id optD_base_build

python3 run.py --mode build --algorithm D --d-variant preload --preload-count 3 \
  --sq 256 --skv 768 --tk 128 --run-id optD_preload3_build

python3 run.py --mode build --algorithm D --d-variant dblQ \
  --sq 256 --skv 768 --tk 128 --run-id optD_dblQ_build

python3 run.py --mode build --algorithm D --d-variant dblQ_preload --preload-count 3 \
  --sq 256 --skv 768 --tk 128 --run-id optD_dblQ_preload3_build

python3 run.py --mode build --algorithm D --d-variant dblQ --d-const-kv-address \
  --sq 256 --skv 768 --tk 128 --run-id optD_dblQ_constKV_build

python3 run.py --mode build --algorithm D --d-variant dblQ_preload --preload-count 3 \
  --d-const-kv-address --sq 256 --skv 768 --tk 128 \
  --run-id optD_dblQ_constKV_preload3_build
```

功能校验会生成输入、运行四线程 gfrun，校验六个输入 buffer 的回读并与
AlgoD golden 比较；失败返回非零退出码。

```bash
python3 run.py --mode check --algorithm D --d-variant dblQ \
  --sq 256 --skv 256 --tk 128 --seed 1 --run-id optD_dblQ_check
```

增加 `--debug` 可导出 AlgoD 中间 tile 到 `check_data/debug.bin`；
该选项只支持 D 的 check 模式，产生额外 TSTORE 和较大文件，不应用于计时。

性能运行示例（同样可以替换为 base/preload/dblQ_preload）：

```bash
python3 run.py --mode perf --algorithm D --d-variant dblQ \
  --sq 256 --skv 8192 --tk 128 --vecq 64 \
  --l0a-resident 1 --sharetreg-bandwidth 256 --run-id optD_dblQ_perf
```

脚本固定使用 fourpe、`vec_cell_sched_enable=false`、`speROBDepth=128`、
`enable_internal_acc=true`。其余性能开关：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--vecq` | 32 | Vector issue queue 深度，可选 32/64/96 |
| `--l0a-resident` | 0 | L0A shared tile resident，可选 0/1 |
| `--sharetreg-bandwidth` | 256 | shared tile 带宽，可选 128/256 bytes |
| `--soc-random` | 未指定 | 指定时设置 `core.soc_random=true`；未指定时沿用模型配置 |

这些参数只影响 perf 的模型 wrapper，不改变调度宏。
性能对照需固定模型版本、配置、输入 shape、constKV 状态和编译器/API。
本脚本没有暴露历史实验中的所有模型开关，默认运行不保证复现历史最高 MFU。

输出保存在测试目录的 `perf_runs/<run-id>/`：ELF、`command.txt`、`build.log`；
check 另有 `gfrun.log`、`check.log`、`check_data/`；perf 通过
`test/common/run_swimlane.py` 保存 `<prefix>_swim.log`、完整 `<prefix>.json`
及用于 Perfetto 的 `<prefix>_nocounters.json`。总周期以日志的 `Total Cycles` 为准。

也可直接调用 Makefile，例如 dblQ+preload3：

```bash
make -j4 clean
make -j4 all ALGO_C=0 ALGO_D=1 GM_FUSED=2 Sq=256 Skv=768 Tk=128 \
  D_DBLQ=1 D_PRELOAD=1 D_PRELOAD_N=3 D_CONST_KV_ADDR=0 res_check=off
```

直接 make 时必须显式设置整组开关，内核宏是编译期选择；修改头文件或切换
开关后先 clean，以免复用旧 `.o`。`run.py` 会自动删除算子 `.o` 并用 `make -j4`
构建，再把 ELF 保存到运行目录。

## 当前验证状态

AlgoD 作为实验实现保存。已有小 shape 记录中 base/dblQ 输出逐字节一致，
但未通过 AlgoD golden；不能把调度之间一致视为功能正确。
大 shape 的历史性能使用 checkoff ELF，完整数值验证尚未闭环。
构建成功、gfsim 退出成功与 golden 通过应分别记录。
预取和 constKV 的功能/性能结论也需按实际变体分别验证。

2026-10-12 提交前检查：使用修正上述 region expansion 约束的 API resource，
`Sq256/Skv768/Tk128` 的六种表列组合及 preload5/dblQ_preload5 共八组
kernel-only 构建通过；非法 Sq、预取距离、constKV 搭配与 constKV check
四项参数检查通过。`Sq256/Skv256/Tk128`、seed=1 的 dblQ check 构建通过，
四线程 gfrun 的六个输入回读一致，输出字节数为 `65536/65536`，但 AlgoD
golden 比较仍 FAIL：`bad=24361/32768`、`max_abs=0.02747`、
`mean_abs=0.003736`，脚本返回非零退出码。
