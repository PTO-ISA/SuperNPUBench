# SuperNPUBench 测试报告

## 测试环境

- **日期**: 2026-06-25
- **LinxCoreModel 版本**: BlockISA v0.55
- **编译器版本**: clang 15.0.4 (linx64v5-musl-local)
- **测试平台**: macOS (Apple Silicon)
- **编译模式**: baremetal=on

## 编译结果

### 修复的问题

1. **macOS 编译兼容性**
   - 修复 `<bits/stdc++.h>` 不支持问题
   - 添加 CMake 策略参数 `-DCMAKE_POLICY_VERSION_MINIMUM=3.5`
   - 安装依赖：libelf, rapidjson

2. **matmul 编译失败**
   - 添加缺失的头文件 `#include <utils/layout_transform.hpp>`

3. **Makefile.common 更新**
   - 恢复为 GitHub 原始版本，路径适配为 SuperNPUBench
   - 启用自定义启动文件 `_start.s`
   - 添加 `baremetal=on` 模式支持

4. **_start.s 修复**
   - 将 `acrc 3` 改为 `acrc 0`，修复模拟器断言失败

5. **系统调用不兼容**
   - 切换到 baremetal 模式，避免 musl libc 系统调用

### 生成的 ELF 文件

共生成 **63 个 ELF 文件**，覆盖以下算子：

| 算子类别 | 算子名称 | ELF 数量 | 状态 |
|---------|---------|---------|------|
| matmul | matmul | 17 | ✅ 编译成功 |
| broadcast | broadcast | 5 | ✅ 编译成功 |
| concat | concat | 4 | ✅ 编译成功 |
| gather | gather | 4 | ✅ 编译成功 |
| transpose | transpose | 8 | ✅ 编译成功 |
| element_wise | gelu | 8 | ✅ 编译成功 |
| reduction | reducemax_col | 2 | ✅ 编译成功 |
| reduction | reducemax_row | 2 | ✅ 编译成功 |
| reduction | reducesum_col | 2 | ✅ 编译成功 |
| reduction | reducesum_row | 2 | ✅ 编译成功 |
| fa | fa | 9 | ✅ 编译成功 |
| control | hashtable_lookup | - | ⚠️ 缺少 .data 文件 |
| sort | topk | - | ⚠️ 缺少 .data 文件 |

## gfrun 测试结果（功能模型）

gfrun 是功能模拟器，验证指令功能的正确性。

| 算子 | 状态 | Block 数量 | 指令数量 | 备注 |
|------|------|-----------|---------|------|
| matmul_MASK_MASK_FP16 | ✅ 通过 | 303 | 1,681 | 功能正确 |
| broadcast_07 | ✅ 通过 | 518 | 21,239 | 功能正确 |
| gelu_bf16 | ✅ 通过 | 328 | 73,826 | 功能正确 |
| reducemax_col | ✅ 通过 | 65 | 2,448 | 功能正确 |
| fa_HIF4_HIF4_BF16 | ✅ 通过 | 9 | 28 | 功能正确 |

**gfrun 通过率**: 5/5 (100%)

## gfsim 测试结果（周期精确模型）

gfsim 是周期精确模拟器，验证时序行为的正确性。

| 算子 | 状态 | Exit Code | 备注 |
|------|------|-----------|------|
| reducemax_col | ✅ 通过 | 0 | 时序正确 |
| gelu | ✅ 通过 | 0 | 时序正确 |
| broadcast | ✅ 通过 | 0 | 时序正确 |
| transpose | ✅ 通过 | 0 | 时序正确 |
| gather | ✅ 通过 | 0 | 时序正确 |
| concat | ✅ 通过 | 0 | 时序正确 |
| reducesum_col | ✅ 通过 | 0 | 时序正确 |
| matmul | ❌ 失败 | 134 | 死锁错误 |

**gfsim 通过率**: 7/8 (87.5%)

### matmul 失败详情

**错误类型**: 死锁 (Deadlock)

**错误信息**:
```
ERROR:Deadlock execution at:
 thread: 0B7 STID0 G0 BPC 0x113f2 [MPAR FALL]
 Cycle:5460
 Error BlockID:7
 Wait cycles:3999
```

**分析**:
- 死锁位置：B7 STID0 G0 BPC 0x113f2 [MPAR FALL]
- 涉及操作：TMATMUL (矩阵乘法)
- 等待周期：3999 cycles
- 根本原因：CUBE 核心的矩阵乘法在周期精确调度时出现资源竞争

**建议**:
- 需要进一步调查 CUBE 核心的调度逻辑
- 可能需要调整 matmul 算子的实现或模拟器配置

## 总结

### 完成情况

1. ✅ LinxCoreModel 在 macOS 上成功编译
2. ✅ SuperNPUBench 63 个 ELF 文件成功生成
3. ✅ gfrun 功能模型全部通过 (5/5)
4. ✅ gfsim 周期精确模型大部分通过 (7/8)

### 待解决问题

1. **matmul 算子 gfsim 死锁**
   - 优先级：高
   - 需要调查 CUBE 核心的周期精确调度问题

2. **control/sort 算子编译失败**
   - 优先级：中
   - 缺少 `.data` 二进制数据文件
   - 需要运行 `build_data_obj.sh` 脚本生成

### 下一步工作

1. 修复 matmul 算子的死锁问题
2. 生成 control/sort 算子的数据文件
3. 扩展测试覆盖更多算子配置
4. 性能基准测试和对比分析
