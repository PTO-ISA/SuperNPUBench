# AlgoD Tk256 性能与 resident/preload 实验交接（2026-10-13）

这是 `Sq=256, Skv=8192, Tm=128, qD=vD=128, dblQ, GM2` 的同源配对实验。

| Tk | Total Cycles | Matmul theory cycles | Matmul-based MFU |
|---|---:|---:|---:|
|128|23995|16384|68.28%|
|256|23096|16384|70.94%|

Tk256 少899拍（3.7466%），加速1.038924倍，MFU增加2.6578个百分点。
逻辑MAC=536870912，四PE合计峰值32768 MAC/cycle；分母为完整kernel周期，包含前言、Vector、访存和同步。
共同配置：SharedTReg BW256、vecIssueQDepth256、ROB128、bctrl_bandwidth8、internalACC/resident/descriptor fusion开启，preload/constKV关闭，soc_random=false。
原归档Tk128=24269，ELF与当前重建版不同；Tk-only比较必须用23995，不混用旧基线。

## 原因与边界

- KV轮数64→32，global matmul数256→128；每次工作变大，总MAC不变。
- CUBE0 internalACC reads 4032→1984、L0A tag record misses 354→192；L0 in-flight归因2679→1294拍，符合块间开销与输入等待减少的方向。
- CUBE0 store in-flight归因1170→1398拍，CellReg STID0平均占用30.51%→48.91%；大tile增加写回及活跃数据压力。
- 全线程CMD issued slots 60616→58007，仅下降4.30%，因为Tk变大也增加每轮group-wise softmax工作；不能按KV轮数减半估计前端收益。
- CUBE0 CellReg写字节均为2129920；不是减少了所有结果写回。当前QK score为BF16，PV中间累加驻留InternalACC，不能直接套用“每次QK写FP32、每次PV都读写CellReg C”的带宽模型。

目前判断为输入/控制依赖、计算和写回共同限制。计数器归因不是可相加的独立关键路径收益；尚无带宽敏感性对照，不能断言已经纯写回bound或纯frontend-bound。

## 完成、尝试与计划的状态

| 项目 | 状态 |
|---|---|
|Tk128/256配对性能|已完成，模拟器exit0，有完整/去计数器SwimLane|
|Tk256编译限制|仅实验副本把static_assert放宽为128或256；补丁随本文归档，生产头文件不改|
|数值正确性|性能ELF为checkoff；Tk256尚未独立golden验证，exit0不是数值通过|
|soc_random|用户叫停前有一次失败尝试（exit -6）；不纳入比较，后续不跑|
|tag4→8、preload新矩阵|仅理论分析及实验设计，无新增性能数据|
|Kimi审查|已完成；审查中的强定量预测经主报告限定，不能当实测|
|DeepSeek审查|任务书已写，未找到接口，未提交|

## 下一步实验

先两种Tk×tag4/8、preload关，隔离目录容量收益；再K-only N=1/2；随后比较当前V优先、V-only、KV错开提交。
当前Shared Q每份预计两个数据window，双Q数据key共4，thread0还共用两个scale key，6 key争4槽；tag8有动机但不保证总周期收益。
scale record未设置底层resident位，PV Pscale也竞争leftScaleBuffer，要把目录容量与scale保持分别验证。
现有preload只提前K/Kscale，并把未来K放在当前V前面；关键目标是减少急需输入晚提交/晚就绪。
N=1当前被2..5断言拒绝，需实验副本与正确性门槛；更深预取必须预算padding、旧generation和物理分配峰值。

完整设计、矩阵、提交/issue/TLSU/consumer观测指标及停止条件见 [THEORY_AND_PLAN.md](THEORY_AND_PLAN.md) 和 [experiment_matrix.csv](experiment_matrix.csv)。

## 复现材料与归档

- [性能原报告](TK256_PERFORMANCE.md)、[配对数据](comparison.json)、[来源及哈希](provenance.json)、[历史命令](RECORDED_COMMANDS.md)。
- [Tk256实验补丁](tk256_guard_experiment.patch)：只允许在隔离副本使用，不表示正式支持已验证。
- [Kimi审查](KIMI_REVIEW.md)、[Kimi任务](KIMI_TASK.md)、[DeepSeek任务](DEEPSEEK_TASK.md)。主报告的审查合并说明优先于审查中的未证实推测。

原始ELF/diss、build log、run_swim.log、run.json、run_nocounters.json保留在工作区归档中，不复制大二进制进入Git；仓库内报告/数据/补丁可直接阅读。来源路径只用于定位本地证据，不意味着其他clone已包含原始运行产物。

归档根（相对superscalar_v5工作区）：
`performance_archive/2026-10-11_dblQ_final_47p48MFU/cube_tlsu_combined/tmov_chain_fix_20261012/tk256_compare_20261013/`

Tk128 run目录 `tk128_rebuilt_baseline/`，Tk256 run目录 `baseline_no_soc_random/`。
ELF分别在 `build_tk128/kernel/multi_thread/fa_lowp_algB/elf/`、`build_tk256/kernel/multi_thread/fa_lowp_algB/elf/`，完整名称及SHA256见provenance.json的builds字段。
原manifest顶层elf/variants继承自更早实验，配对身份以builds、paired_runs和comparison.json为准。

本次入库只整理现有结果和实验设计，未新增编译或模拟器执行。未来编译-j4，perf必须使用run_swimlane.py并归档三件套；不跑soc_random。
