# L0A resident-tag 与 preload 寻优设计（2026-10-13）

状态：仅静态代码分析、读取已有归档及外部模型审查；本轮不编译、不运行模拟器、不改算子/模型源码。后续实验也只跑 no_soc_random。本文中的预期不是测量结果。

## 结论与优先级

1. tag=8 值得先做单因素消融：当前双 Q 的目录工作集预计为 4 个数据窗口 + 2 个 scale key，4 槽有容量冲突，8 槽可容纳。但是目录容量、底层缓存保持和关键路径收益是三件事，不能直接保证提速。
2. preload 先从距离 N=1 做起。现有代码只提前 K/Kscale；V/Vscale 仍在当前迭代提交，而且未来 K 排在当前 V 前面。这可能把瓶颈从 K 晚到转成 V 晚到。
3. 核心方向是把关键 TLOAD 的地址准备、描述符提交及数据就绪提前，同时减少不必要的 Q 重读；不是靠大量预取把队列堆满。保留 descriptor fusion，先优化数据生命周期和指令位置，再判断是否值得拓宽 CMD dispatch。

## 固定基线与已有证据

Sq=256，Skv=8192，Tm=128，qD=vD=128，dblQ；SharedTReg BW=256，internalACC/resident/descriptor fusion 开，constKV 关。模型源使用 performance_archive/combined_main_20261012/model，与已归档模拟器来源一致，不拿另一份旧模型替代。

| Tk | 已有 Total Cycles | Matmul MFU | KV轮数 |
|---|---:|---:|---:|
|128|23995|68.28%|64|
|256|23096|70.94%|32|

配对比较来自 TK256_PERFORMANCE.md：Tk256 少899拍（3.7466%）。CMD issued slots 总数只下降4.30%，不能因为KV轮数减半就推断前端指令减半。CUBE0 tag record miss 为354/192，L0 in-flight归因2679/1294，store in-flight1170/1398。这些归因不可相加成独立收益，也不能证明所有剩余等待均来自前端。两组性能ELF未开 res_check；Tk256只是实验副本放宽static_assert，正确性需后续独立确认。

## resident-tag=8 实际改变什么

来源：CubeCore.h:966-997、CubeCore.cpp:3714-3895。每个 cubeThread 的目录槽数当前硬编码为4，不是一个已有命令行参数；后续应在隔离模型副本参数化，默认4，并记录二进制hash。

数据key=(physical Shared slot, isScale=false, windowOffset)；scale key=(physical Shared slot, isScale=true)。数据和scale共享同一目录（scale使用thread0目录）。命中减少 SharedTReg→L0 的请求；它不直接让全局内存→SharedTReg 的 TLOAD 更快，也不直接提高 frontend dispatch宽度。

AUsesPackedM32 在 src0IsShared 时返回false（CubeCore.cpp:3429），所以不能把共享Q误算成packed-A单窗口。当前M32 cellRows=32、m_width=16，windowOffset有两种取值；Q0/Q1为不同physical slot，预计每个数据目录有4个数据key；scale固定进入thread0目录，该目录再加两个Qscale共6个。其余目录不能照搬“6个key”这一推断。因此“只有两个Q，4槽肯定够”的推论不成立。最终应以实际key流确认；这只是按代码几何推得的目录工作集，不能替代完整地址覆盖证明。

底层还有两项约束：

- 数据tag会设置L0A entry.resident，淘汰/flush解除固定。一般情况下8槽可能留住更多entry；对本双Q若数据key确实只有4个，稳定固定数据entry未必比4槽更多，但存活更久；PV的P来自CellReg，不享受该resident命中，却仍使用L0A，须看其分配是否被挤压。有效运行配置打印L0A_size=32，不把源码默认值或旧“4×4KB”注释当成本轮物理容量。
- scale record只写目录，当前Record路径没有设置leftScaleBuffer.resident；FreeLeftScaleBuffer仍可回收它。HitScale遇到free entry会作废tag。因此8槽能容纳scale key，并不保证scale数据跨块永久保留。先测纯4→8；若scale频繁失效，再把“scale保持策略”列为独立变量，不能偷偷合入8槽结果。

额外正确性审查：key不显式含K-fractal或buffer generation；8槽延长entry保留时间，要核对physical-slot/window是否足够区分所有真实读地址，以及scale entry非free重用后的身份检查。此处提出验证项，不据此宣称存在数据错误。

可证伪预测：8槽若解决容量冲突，Q data tag eviction和重复A读应降低；若服务等待在关键路径，consumer ready等待和总周期也降低。若miss降而周期不变，说明被其他工作覆盖或最终受写回/计算/控制依赖限制。若PV L0A alloc等待增加，则保持成本抵消收益。

## preload 的理论收益与风险

来源：fa_lowp_algD.hpp:91-99、208-284。现有K/Kscale环距离N、深度N+1；prologue加载0..N-1，迭代kb提前加载kb+N，正常地址模式不改变总有效K流量。它主要改变发起时间和在飞量，可能增加前言、寄存器压力、spill、活跃Shared physical generation和队列占用，不应称作减少TLOAD总数。

当前顺序：未来K/Kscale → 当前V/Vscale → QK0/softmax0/PV0 → QK1/softmax1/PV1。现有K预取能让下一轮QK更早ready，却可能让当前PV等待V；多预取几轮不能修复前端晚提交，还可能把不紧急的TLOAD插在当前关键请求之前。

优先候选顺序：

- K-only N=1：复用现有环逻辑，建立最低改动对照。注意当前static_assert只允许N=2..5（头文件:22-24），N=1需后续在隔离实验副本放宽下界并完成复用正确性检查；本轮不修改。原代码可直接测试的是N=2。
- K-only N=1，当前V/Vscale先于未来K/Kscale：只改提交优先级，分离“预取距离”与“顺序”效应。当前K来自上一轮预取；prologue单独处理。
- V-only N=1：提前下一轮V，K照常，验证剩余等待是否主要来自PV输入。
- KV staggered N=1：当前关键数据优先；K/V各自按消费期限发起，未来K尽可能放在本轮QK/softmax/PV可覆盖的区间，未来V在下轮PV前就绪。提交位置做编译期调度，不能在热循环增加无意义动态分支。

K,V各被Q0/Q1消费一次。环槽最后一次使用是第二个Q的QK/PV，不能只看Q0已经发起，也不能把源码作用域结束视作硬件读完。需依靠合法rename、reader tracking和physical generation隔离；容量不足走in-place覆写时尤其要排空旧读者。改位置不绕过RAW/WAR/collective ready规则。

概念伪代码（不是可编译实现）：

    prologue: prepare/load initial K/V lookahead
    for kb:
        bind K[kb], V[kb]
        submit due critical loads before distant loads
        QK(Q0,K[kb]); softmax0; PV(P0,V[kb])
        at a profiled safe scheduling point: submit K[kb+Nk]/V[kb+Nv]
        QK(Q1,K[kb]); softmax1; PV(P1,V[kb])
        retain old generations until both consumers finish

发起不一定要等旧消费者执行完：若rename能分配新generation，可提前定义逻辑环槽；但实际资源预算必须包含尚未退役的旧generation，不能只统计C++数组长度。

## 容量与距离选择

由当前配对ELF反汇编可见：Q每份8KiB、Qscale每份4KiB；K(Tk128)=8KiB、Kscale=4KiB、V=8KiB、Vscale=4KiB；Tk256分别16/8/16/4KiB（Vscale ValidCol=8仍padding至32列）。这比只按有效scale元素计字节大得多。

| Tk | 双Q+scale | K环每槽 | V环每槽 | K/V均N=1时的静态对象下限 |
|---|---:|---:|---:|---:|
|128|24KiB|12KiB|12KiB|72KiB|
|256|24KiB|24KiB|20KiB|112KiB|

最后一列=24+2*K槽+2*V槽，尚未包含额外在飞generation、输出/其他Shared对象与碎片。全局Shared池配置256KiB、64个4KiB SSB，不能当成每PE独享256KiB，也不把广播的四份响应误算成四份对象。后续必须按实际physical allocation/SSB peak确认安全余量，不能仅凭此表决定N上限。

距离条件：N*T_iter 应覆盖 load发起到消费者ready的延迟加frontend提交抖动，采用稳态p95并区分K与V；全kernel平均23995/64≈375拍、23096/32≈722拍只给量级，不是稳态deadline。先N=1，再N=2，只有消费者仍频繁miss deadline且资源有余量才试N=3。Tk256单次load更大，但消费间距也更长，不能直接断言需要更深预取。

## 分阶段最小实验矩阵（尚未执行）

阶段0：静态审查与观测补齐。逐项冻结编译器/header/model hashes、所有开关、dtype和最终ELF；检查有效TLOAD数量、地址/stride、B.IOS、TMOV/spill。重点对齐每个TLOAD的frontend→TLSU→Shared ready→consumer链。禁止只根据SwimLane的外观或汇总TopDown给归因。

阶段1：Tk∈{128,256} × tag∈{4,8} × preload=off，共4条件（两个现有基线可参考；若模型新增观测/参数，则重新建立同版本配对基线）。相同Tk必须使用同一个ELF，只有模型tag数变化。阶段目标是判断目录容量冲突是否真实影响关键路径。

阶段2：在阶段1选定的tag下，两种Tk各测现有K-only N=1、N=2，共4新增条件；加各自off基线对照。若tag4与8在阶段1有明显差异，给候选N补一次另一tag配对以检测交互。不得一开始暴力扫全部高N组合。

阶段3：选较优Tk/距离，固定模型容量和带宽，对比K-only原顺序、K-only当前V优先、V-only、KV staggered。先N=1；只有deadline证据支持才扩深度。若Tk128/256接近或表现不同，给第二Tk补最终候选。主矩阵不同时改变CMD dispatch、L0A容量和BW。

阶段4（仅按失败原因启用）：

- tagmiss还高且为scale回收：分开A/scale目录，或独立scale保持策略，对照统一8槽；保持总资源预算可解释。
- tag8拖慢PV：诊断residentOff，或限制可固定的数据entry，明确这是策略变量。
- load地址已ready但dispatch晚、CMD IQ空：独立测试CMD dispatch宽度1→2，issue宽度保持不变；若IQ已积压或FIFO队头等依赖则先改善issue可达性，不能承诺多dispatch有效。
- TLOAD已经进入TLSU却等端口/在飞限制：处理TLSU准入与竞争，不能归为frontend fetch压力。

## 必须收集的指标和归因规则

每个逻辑kb、Q0/Q1、K/V/scale对应真实BID+完整代际/physical slot；不能用旧trace中可能复用的producerBid依赖线直接证明RAW。

| 链路 | 观测 | 解释 |
|---|---|---|
|前端地址准备|地址最后producer ready、描述符完整时刻|地址ready不等于block可提交|
|CMD提交|eligible、dispatch、IQ enqueue、issue及队头阻塞原因|区分全局dispatch供给不足与每线程FIFO/依赖|
|TLSU|admission、首末beat、Shared写入完成/collective ready|区分晚提交与服务带宽/在飞限制|
|消费|QK/PV operand ready、consumer首issue、最后使用|测deadline与可覆盖的load延迟|
|resident|data/scale分别hit/record/evict/free-invalid/flush、key流、resident entry峰值|不能把混合hit率当每次matmul命中率|
|资源|Shared physical bytes/SSB/generation峰值、allocation fallback、L0A alloc等待|排除预取和pin造成反向压力|
|流量|全局有效K/V bytes；Shared物理read/write服务beats分别计|广播response字节不能当物理总线流量|

普通CMD路径已有dispatchCycle未完整写入的问题，需先补真实时间戳或debug表，不能把缺失S1的trace间隙全算issue等待。按每线程和真实TLSU任务流分析，不能用四线程aggregate issue slots直接推出单线程吞吐。

frontend/TLSU等待可能同源于一个关键load，数据总线又可能同时服务别的请求；所以按关键路径解释，不能把各组件等待拍数相加。带宽利用率应同时给全kernel及稳态active窗口；利用率上升不必然让总周期下降。

## 选择标准、正确性和归档

目标函数为完整kernel Total Cycles最小，MFU=16384/TotalCycles；附带报告启动/稳态/排空分解。若收益小于1%，标为小收益并结合机制证据，不能只报一个更漂亮的利用率。小收益门槛是实验决策阈值，不代表已经测出噪声。

正确性门槛：独立golden/res_check（包括Tk256）、不同kb有不同K/V的输入、两Q内容不同、环绕与尾部非完整lookahead覆盖。覆盖至少N=1/2的复用，检查旧generation仍有读者时不会覆写；constKV只能诊断地址准备成本，不能作为最终正确性或优化成绩。

未来每次perf使用run_swimlane.py，保留log/full JSON/nocounters JSON、ELF+diss、完整命令和hash，全部进入performance_archive，与本设计相邻并写manifest。编译仅-j4。每项必须报告Total Cycles、ELF、归档目录；不跑soc_random。现在不执行这些步骤。

停止扩展条件：增加N未减少deadline miss、Shared分配开始等待、CMD/TMOV spill上升、总周期回退，或关键路径已经迁移到计算/写回。随后在已胜出的容量/顺序上做成对复验，不继续盲扫。

## DeepSeek / Kimi 分工

KIMI_TASK.md：独立审查resident目录、scale生命周期、PV容量竞争、最小消融和正确性风险；通过本机Kimi CLI提交，原始响应与CLI日志保存在本地performance_archive设计目录。

DEEPSEEK_TASK.md：独立审查preload距离、K/V提交顺序、环复用、Shared预算和前端归因。当前未找到DeepSeek可执行程序或配置接口，任务书已准备但没有实际提交；不得把其他模型结果写成DeepSeek结论。

审查建议应区分“源码事实”“待测假设”“后续实现”，最后由主分析合并，不能未经正确性和配对实验把理论候选当成优化成果。

## Kimi 审查合并说明

Kimi CLI已完成（exit0），独立报告KIMI_REVIEW.md，原始响应位于本地performance_archive设计目录的kimi_review.txt。认可窗口key、thread0目录共用与scale未设置resident这三项源码事实，补入了PV本地Pscale对leftScaleBuffer的竞争这一待测机制。

不直接采纳其所有定量预测：miss<40属于强假设，不是已经证明的必然结果；在scale不固定、读取代际与实际窗口序列尚未完整证明前，仅要求容量相关eviction下降，并拆分data/scale定位。其“所有配置Tk256/Tk128 miss比都约为轮数比”的预测不适用于真正消除重复读、只剩冷启动的情形，冷启动可能与轮数无关。对本双Q也不预先声称pin高水位一定不变，需确认同key重记录与L0A buffer复用关系。

优先执行顺序保持：纯tag4/8配对 → K-only小距离 → 调整V/K优先级 → 必要时scale保持/分池；scale策略不能与tag8同时合入首个实验，否则无法归因。DeepSeek任务书已完成，但当前没有可用接口，未提交。
