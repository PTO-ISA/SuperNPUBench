# SuperNPUBench 编译报告

**生成时间**: 2026-06-24  
**编译器**: linx_blockisa_llvm_musl (clang-15, linx64v5-musl)  
**编译目标**: Linx64 V5

## 编译统计

**总计生成 ELF 文件**: 59 个

| 算子类别 | ELF 数量 | 状态 |
|---------|---------|------|
| matmul | 13 | ✓ 成功 |
| fa | 9 | ✓ 成功 |
| transpose | 8 | ✓ 成功 |
| reduction | 8 | ✓ 成功 |
| gelu | 8 | ✓ 成功 |
| broadcast | 5 | ✓ 成功 |
| gather | 4 | ✓ 成功 |
| concat | 4 | ✓ 成功 |

## 算子详情

### 1. Matmul (13 个 ELF)

**Kernel 实现**: `kernels/matmul/`
- `matmul.hpp` - 通用 matmul 实现（mask、dynamic、vec 等变体）
- `matmul_mx.hpp` - MX 量化 matmul 实现

**测试文件**: `test/kernel/matmul/src/`
- `HiF4_HiF4.cpp` - FP4 x FP4 量化 matmul
- `A16W4.cpp` - BF16 x FP4 混合精度 matmul
- `matmul.cpp` - 通用 matmul（支持多种数据类型和变体）

**支持的类型**:
- `TYPE=HIF4_HIF4`: FP4 x FP4 量化 matmul (4 configs)
- `TYPE=A16W4`: BF16 x FP4 混合精度 matmul (3 configs)
- `TYPE=MASK`: 通用 matmul，支持多种模式 (6 configs):
  - `MASK_FP32`, `MASK_FP32_REUSEA`, `MASK_FP32_DYNAMIC`
  - `MASK_FP16`, `MASK_FP16_REUSEA`
  - `MASK_FP8`, `MASK_FP8_REUSEA`, `MX_FP8`

**编译示例**:
```bash
# FP4 x FP4
make TESTCASE=matmul TYPE=HIF4_HIF4 M=256 N=2048 K=2048 tM=128 tN=128 tK=128

# BF16 x FP4
make TESTCASE=matmul TYPE=A16W4 M=256 N=2048 K=2048 tM=128 tN=128 tK=128

# FP32 mask matmul
make TESTCASE=matmul TYPE=MASK MODE=MASK_FP32 M=256 N=256 K=256 tM=64 tN=64 tK=64
```

### 2. Flash Attention (9 个 ELF)

**Kernel 实现**: `kernels/fa/`
- `fa_2d_unroll.hpp` - 基础 2D 展开版本
- `fa_2d_unroll_pto.hpp` - PTO 风格 2D 展开
- `fa_unalign_2d_unroll.hpp` - 非对齐边界处理
- `fa_hif4.hpp` - HIF4 量化版本
- `fa_dcore.hpp` - DCore 优化版本
- `fa_fp4_utils.h` - FP4 工具函数
- `fa_utils.h` - 通用工具函数

**测试文件**: `test/kernel/fa/src/`
- `fa_2d_unroll.cpp` - 2D 展开测试 (8 configs)
- `fa_hif4.cpp` - HIF4 量化测试 (1 config)

**支持的配置**:
- `X=1, Y=2/4`
- `X=2, Y=2/4`
- 序列长度: 256, 512

**已知问题**:
- `X=1, Y=1` 和 `X=2, Y=1` 会导致编译器崩溃（见 Issue #6）

### 3. Transpose (8 个 ELF)

**Kernel 实现**: `kernels/transpose/`
- `transpose.hpp` - 基础转置
- `transpose_vector_007.hpp`, `transpose_vector_050.hpp` - 向量化版本

**测试文件**: `test/kernel/transpose/src/transpose.cpp`

**支持的配置**:
- 3D transpose (2 configs)
- 4D transpose (2 configs)
- 5D transpose (1 config)
- 6D transpose (2 configs)
- 不同数据类型 (__half, int32_t)

### 4. Reduction (8 个 ELF)

**Kernel 实现**: `kernels/reduction/`
- `reducemax_colvec.hpp` - 列方向最大值
- `reducemax_rowvec.hpp` - 行方向最大值
- `reducesum_colvec.hpp` - 列方向求和
- `reducesum_rowvec.hpp` - 行方向求和

**测试配置**:
- reducemax_col: 2 configs (int32_t, __half)
- reducemax_row: 2 configs (int32_t, __half)
- reducesum_col: 2 configs (int32_t, __half)
- reducesum_row: 2 configs (int32_t, __half)

### 5. GELU (8 个 ELF)

**Kernel 实现**: `kernels/element_wise/`
- `gelu.hpp` - 新版多项式拟合
- `gelu_origin.hpp` - 旧版多项式拟合 & tanh 近似

**测试文件**: `test/kernel/element_wise/gelu/src/gelu.cpp`

**支持的配置**:
- 数据类型: __bf16, __half
- 形状: 24_8_1024 (3D), 128_1024 (2D)
- 近似模式: false (exact), true (tanh approximation)

### 6. Broadcast (5 个 ELF)

**Kernel 实现**: `kernels/broadcast/`
- `broadcast.hpp` - 基础广播
- `broadcast_07.hpp`, `broadcast_019.hpp`, `broadcast_039.hpp` - 不同维度配置
- `broadcast_Hunyuan.hpp` - Hunyuan 模型优化
- `broadcast_vec_*.hpp` - 向量化版本

**测试配置**:
- 2D broadcast (1 config)
- 3D broadcast (1 config)
- 4D broadcast (2 configs)
- 5D broadcast (1 config)

### 7. Gather (4 个 ELF)

**Kernel 实现**: `kernels/gather/gather.hpp`

**测试配置**:
- 大规模输入 (200000, 875000)
- 中等规模输入 (754)
- 2 的幂次维度 (131072)

### 8. Concat (4 个 ELF)

**Kernel 实现**: `kernels/concat/`
- `concat_gather.hpp` - 拼接-收集
- `concat_scatter.hpp` - 拼接-散射

**测试配置**:
- concat_gather: 2 configs (int32_t, __half)
- concat_scatter: 2 configs (int32_t, __half)

## 已知问题

### 1. 编译器崩溃 (Issue #6)
- **问题**: `fa_2d_unroll` 在 `X=1, Y=1` 和 `X=2, Y=1` 配置下触发编译器断言失败
- **错误**: `Assertion failed: (Reg != 0 && "LinxV5 CallingConv Fail!")`
- **临时方案**: 避免使用 `Ydim=1` 的配置

### 2. Control 算子
- **状态**: 编译失败
- **原因**: 缺少数据文件（`.data` 文件）
- **需要**: 生成 hashtable lookup 测试数据

### 3. Sort 算子
- **状态**: 编译失败
- **原因**: 缺少数据文件（`.data` 文件）
- **需要**: 生成 topk 测试数据

## 工具链信息

- **编译器路径**: `/Users/liyi/Documents/SuperNPU编译器构建/output/linx_blockisa_llvm_musl/bin/clang++`
- **编译标志**: `-mlxbc -fenable-matrix -O2 -mllvm -enable-all-vector-as-tilereg=true -std=c++20`
- **目标架构**: `linx64v5-unknown-linux-musl`

## 文件结构

```
SuperNPUBench/
├── kernels/                    # Kernel 实现
│   ├── broadcast/             # 广播算子
│   ├── concat/                # 拼接算子
│   ├── element_wise/          # 逐元素算子 (gelu)
│   ├── fa/                    # Flash Attention
│   ├── gather/                # 收集算子
│   ├── matmul/                # 矩阵乘法
│   ├── reduction/             # 归约算子
│   ├── transpose/             # 转置算子
│   └── utils/                 # 工具函数
├── test/kernel/               # 测试代码
│   ├── broadcast/
│   ├── concat/
│   ├── element_wise/
│   ├── fa/
│   ├── gather/
│   ├── matmul/
│   ├── reduction/
│   └── transpose/
└── output/                    # 编译产物（不入库）
```

## 编译脚本

全量编译脚本: `compile_all.sh`

```bash
./compile_all.sh
```

## 最近更改

### 2026-06-24
- 删除冗余文件：`gemm.cpp`、`matmul_dynamic_reuse.hpp`、`FA.py`
- 重命名文件：`fa_fp4.h` → `fa_fp4_utils.h`，统一 `.hpp` 后缀
- 增强 matmul 支持：添加 `TYPE=MASK` 支持多种数据类型
- 修复工具链问题：创建 `fa_utils.h` 提取共享函数
- 清理 compile.all：注释掉冗余编译配置
- 修复 gelu 编译：移除对已删除 `fa_fusion.h` 的依赖
- 扩展各算子典型场景：transpose (8), reduction (8), gelu (8)
- 精简 matmul 编译目标：从 125 个减少到 13 个典型配置
