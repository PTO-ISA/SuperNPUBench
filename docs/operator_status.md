# SuperNPUBench 算子状态总览

| 算子类别 | 代表算子 | 算子编程 | 功能验证 | 性能评估 |
|---|---|:---:|:---:|:---:|
| Cube 计算 | MatMul、量化 MatMul、多线程 MatMul | 🟡✓ | 🟢✓ | ○ |
| Vector 计算 | GELU、Broadcast、Reduction、Norm | 🟡✓ | 🟢✓ | ○ |
| Memory 搬运 | Gather、Concat、Transpose | 🟢✓ | 🟢✓ | ○ |
| Attention | Flash Attention、FlashMLA、SFA | 🟢✓ | 🟢✓ | ○ |
| 控制与排序 | Hashtable Lookup、TopK | 🟢✓ | 🟢✓ | ○ |
| 融合算子 | DeepSeek、MoE、SwiGLU、融合量化 | 🟢✓ | 🟢✓ | ○ |

## 状态说明

- 🟢✓：已完成或已通过
- 🟡✓：部分完成
- ○：未完成或未开展

## 口径说明

- **算子编程**：算子代码及对应实现已完成。
- **功能验证**：算子已完成正确性验证。
- **性能评估**：算子已完成性能数据采集和评估；当前所有类别均未开展。
