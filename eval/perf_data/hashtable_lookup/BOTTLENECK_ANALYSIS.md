# 6144-key hash 时间瓶颈分析

本分析只针对本目录数值通过的6144-key ELF。当前gfsim总周期421564，原2026-10-31为424487；模型二进制不同，不混用两次时间轴。算子源码相同。

## 结论

当前证据更支持 **probe轮次及lane分歧、gather依赖等待、谓词/选择指令链** 共同主导kernel，而非标量ALU吞吐主导。末尾逐项标量校验及同步占完整程序约30%，必须与kernel分开。没有通过消融A/B精确分解各因素，不能给出“某因素独占X%”或预测确定加速比。

## 时间边界

|事件|新运行周期|
|---|---:|
|第一条输入key TLOAD轨迹开始|388|
|PE0最后结果TSTORE轨迹结束|256526|
|PE1最后结果TSTORE轨迹结束|257803|
|PE2最后结果TSTORE轨迹结束|295619|
|PE3最后结果TSTORE轨迹结束|277589|
|程序结束|421564|

输入加载到最后输出存储的轨迹跨度为 **295231拍**。末次store后到结束 **125945拍，29.88%**，包含PE0逐项结果/guard校验、其他PE等待checked以及退出；不是hash查找计算。第一条load前只有388拍，不存在139-key测例的长时间运行时造表：本ELF的表、query、golden都由.incbin预置。

这些是SwimLane可见事件边界，不是严格函数入口/出口；也不与历史log里的dispatch/scb_observed事件混算。

## 每PE的批次分段

对每PE按执行顺序配对48次TLOAD和48次TSTORE，每批以第一条MGATHER开始分段。下表为各批次间隔之和，跨批可能轻微重叠，不能与其他PE相加当wall cycles；不是对每条指令纯执行周期求和。

|PE|加载→首条gather（含hash/模运算/寻址）|首条gather→结果store（probe及完成）|gather在途区间并集|
|---|---:|---:|---:|
|0|33120|222662|107262|
|1|33838|223232|106716|
|2|32744|262049|118727|
|3|33391|243305|111602|

最慢PE2约 **88.9%** 的批次内间隔位于probe区间。gather在途并集占其输入到最后输出跨度约40%；它同时包含排队/返回等生命周期，且能与其他执行重叠，不能解释为纯DRAM stall比例。

## 为什么probe贵

实际key探测总次数13039，平均2.1222，最大63。192批的最大探测次数之和2421，平均每批执行12.6094轮。由于每批32lane一起迭代，平均有效key探测lane比例为：

`13039 / (2421 × 32) = 16.83%`

这衡量key探测的逻辑活跃率，不等于物理ALU利用率，也不表示inactive lane发了访存。mask可以抑制无效访问，但剩余活跃lane仍使整轮的TCI、寻址、比较、TSEL和循环控制继续执行。

实际轨迹：2421次TCMP、4842次MGATHER、4842次U64 TMUL、29052次TSEL。即每轮两次gather指令、两次U64乘法和12次TSEL。部分value gather掩码可能全零，因此指令条数不等于有效内存请求数。

每PE累计批次最大probe轮数分别562、564、667、628。PE2轮数最多，且最后完成；静态均分key数量并没有均分实际探测工作。

## 访存与运算链

串行依赖为：slot/offset → key MGATHER → empty/equality比较 → active/offset选择与更新 → 下一轮；命中还需value MGATHER。下一次冲突探测依赖前一次读回结果，随机访问延迟难以由同一批自身隐藏。

MGATHER共4842个block事件，生命周期median31拍、p95399拍、max780拍。统计包含key/value、不同mask和cache状态，不能直接当成固定访存延迟或据此反推cache命中率。TREM384个事件（每批两次）median211拍，但这也是含依赖等待的block生命周期，不是TREM执行器延迟211拍。

事件记录按VECTOR SLOT排除INT/FMA/LNEXP执行轨迹重复，详见analysis.json。并发block持续时间不可直接求和占比；gather的全核wall并集240197拍也不等于所有PE都被内存阻塞。

## 是否标量瓶颈

整程序Unified Top-Down显示：Scalar ALU4.45%、Vector13.94%、CMD operand-not-ready15.98%、Machine Clear29.16%。这些是模型的整程序归因，受校验/同步污染，不能视作kernel互斥时间分解，也不能把Machine Clear等同于普通分支预测失败。

特别是旧PMU的MGATHER resource-cycles显示0，但轨迹有4842个MGATHER事件，说明该分类不能用于否认gather开销。当前证据不支持“标量ALU是主要瓶颈”，更支持Tile/访存/谓词依赖与批内尾部等待。仍需kernel范围PMU或消融对照才能量化独立贡献。

## 后续优化方向（尚未实施）

1. 先建立不含golden校验的kernel计时区间，继续保留独立数值回归。
2. 多批交错或更细任务分配，尝试隐藏gather延迟并缓解PE工作不均；需评估寄存器压力。
3. 尝试记录每个key最终命中slot，查找全部结束后统一gather value，以减少每probe轮的value gather和寻址指令；必须验证miss及命中语义，并实际测量收益。
4. 检查每轮12次TSEL是否可由更简洁的状态表达或compiler lowering减少。不能以简单删除谓词选择替代语义验证。
5. 大Tile不保证更快：它可能摊薄调度开销，也可能增加批内最大probe轮数和失活lane；API/PredicateCell支持亦未闭合。

本地模型的full64计算与访存可见性时序仍有未校准假设，完成端重入执行延迟问题是否在本例触发也未采集。因此这是当前模型上有数值验证支撑的性能诊断，不宣称已得到硬件准确周期。
