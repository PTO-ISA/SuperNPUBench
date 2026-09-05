# SuperNPUBench 算子工作进度总览

| 算子类别 | 代表算子 | 算子编程 | 功能验证 | 精度校验 | 多线程迁移| 性能评估优化 |
|---|---|:---:|:---:|:---:|:---:|:---:|
| 基础计算类 | MatMul、GELU、Reduction、RMSNorm | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 |
| 数据搬运类 | Transpose、Broadcast、Concat、Gather | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 |
| 控制排序类 | Hashtable Lookup、TopK | 🟢 | 🟢 | 🟢 | 🟢 | 🟡 |
| 复杂融合算子 | FlashAttention、FlashMLA、DeepSeek、MoE、SFA| 🟢 | 🟢 |  🟢 |🟢 | 🟡 |

## 状态说明

- 🟢：已完成。
- 🟡：进行中。
