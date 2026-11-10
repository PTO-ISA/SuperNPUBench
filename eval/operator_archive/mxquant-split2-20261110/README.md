# MXQuant：独立算子归档（2026-11-10）

**尚未仔细、充分分析此算子是否 TLSU bound。当前仿真观察到 TLSU 等待引入 Vector 断流；这只是局部流水现象，不是整体 TLSU bound 的定论。**

Git 目录保存源码/索引；完整可执行包（含 ELF/diss 与输入）在 [SuperNPUBench Release](https://github.com/PTO-ISA/SuperNPUBench/releases/tag/work-archive-mxquant-20261110)。下载解压后使用下面的 runner。模型与历史 trace 说明在 [Model 归档](https://github.com/LinxISA/SuperScalarModel/tree/archive/workshop-20261109/archives/ndf-mxquant-20261110-r1)。

本包固定历史源码与同一份非零输入 ELF；不替换库上默认算子。shape BF16 `[512,256]`，group 32，4 PE；输出 E4M3 payload 131072 B、E8M0 scale 4096 B。输入 262144 B，以 seed 12345 的固定 LCG 生成，完整算法见 `golden.py`。输入已嵌入 ELF，无运行时文件读取依赖；随包输入和独立 host scalar golden 用于比较。

| case | 原模型周期 | 范围 |
|---|---:|---|
| `baseline` | 13349 | 原逐 group 方案；保留用于历史对照 |
| `split2` | 5464 | 本次主例，输入 ping-pong、payload/scale assemble，每 `[32,256]` 拆两个 `[32,128]` 输入请求 |

这是指定历史 RMS runtime/config 的调度诊断，不能推广为当前主线模型的性能。上轮 packed-pingpong 5545 → split2 5464，改善 1.4608%；完整初始 baseline → split2 还包含前序不同优化，不能把全部提升归因于单一修改。

**数值验收要分开：** 原记录中两份同 ELF 均在 gfrun 全量 input/payload/scale exact 且负控拒绝通过；原 TimingSim baseline 在 `validation_fixed.conf` 下输入 exact，但 payload 129998/131072、scale 4092/4096 字节不符，未解决。split2 没有同 ELF TimingSim 数值通过结论。性能复跑成功也只表示周期重现，不等于数值正确。因此总体 `incomplete`。

## 复跑已有 ELF：不需要编译器

将本包解压，另准备指定模型根目录（含 `bin/gfsim` 或 `bin/gfrun`，模型身份、配置与恢复材料由对应 Model 归档提供）。可以从任意 cwd 执行，输出必须使用新的目录：

```bash
python3 /path/to/mxquant/replay.py split2 --phase performance \
  --model-root /path/to/timing-model --outdir /path/to/new-output/split2
python3 /path/to/mxquant/replay.py baseline --phase performance \
  --model-root /path/to/timing-model --outdir /path/to/new-output/baseline
python3 /path/to/mxquant/replay.py split2 --phase gfrun \
  --model-root /path/to/functional-model --outdir /path/to/new-output/split2-functional
```

入口严格检查模型/ELF SHA256，并验证输入和 golden 与独立生成器完全一致。Timing 模型 hash `bebcf1197d78cacdf65109ad1ad76d42604d7996b37c298d92063f82435b7224`，gfrun hash `9b0cf76eaa3ac62902f9511929a961cb064c8930fa9f9af7f8d967aa57eb1fb9`。3 GiB 地址空间、nice 15、无 core dump；`RMS_TAIL_BYPASS=1`，配置为 `configs/rms.conf`，runner 清除两个 ideal bank 消融变量。模型路径由调用者提供，未隐式使用作者机器环境。

性能输出保留 `mxquant_swim.log`、完整 `mxquant.json`、移除且仅移除 `ph:"C"` 的 `mxquant_nocounters.json`，以及 `result.json`。周期不符、模型失败或缺少 trace 均失败。gfrun 保留内存读回和全字节比较，篡改一个 golden 字节必须拒绝。本轮另做异目录 gfrun 功能入口检查，结果见 `functional_replay_review.json`；未运行新的 gfsim 性能仿真。历史 `run.json` 不代表此次已复跑性能。

## 源码、构建与证据

`cases/{split2,baseline}/source/` 是经原 identity hash 核对的完整测量时源文件集合。ELF 从历史测量文件逐字节复制，从未修改二进制；匹配 diss 仅把开头 ELF 文件标题的作者路径替换为 basename，其余字节完全相同，原始/派生 hash 见 `operator_identity.json`。

可选 `build.py CASE --toolchain-root ... --benchmark-root ... --outdir ...` 使用显式的历史编译器和 benchmark 根目录；原命令参数结构在 `build_recipe.json`，只把本机根路径换成调用者参数。旧编译器 hash、模型/API/ISA/benchmark commits 在 `historical_manifest.json`；对应源码变更闭合程度仍以 Model 交接说明为准，不声称从当前库 HEAD 能字节一致重建。没有打包整个工具链或 API 依赖；ELF replay 无需下载工具链。重建脚本未在本轮重新编译，历史重建证据单独标识为 historical，不冒充新验证。

宽输入候选 `width_candidate_outcomes.json` 仅保留结果摘要，不是本包的可运行 case：同骨架 control256 为 5459，512 整块/分半 5709/5750，1024 整块 6599，没有显示宽 tiling 收益。5459 与冻结 split2 的 5464 相差 5 拍来自代码生成，不能宣称宽 tile 提升。open-parent assembly 候选未通过合同检查、未跑性能；没有确定 ISA 合法性和最小复现前，不归咎具体 API/compiler/model，也未声明已有 issue 修复。

大 trace 不在算子 Git 内容中；原始性能日志/trace 归 Model 证据或可选 Release 资产。本包不包含新的模型修改，不宣称 TLSU 性能问题或 TimingSim 数值故障已解决。所有 historical JSON 是历史索引/观察，只有本包 `cases/`、`inputs/`、`configs/` 及 runner 是运行入口。
