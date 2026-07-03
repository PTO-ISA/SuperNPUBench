# LinxISA Microbenchmarks

针对 LinxISA 张量指令集的指令集粒度基准测试。

## 目录结构

```
microbench/
├── cube/       # 矩阵计算指令 benchmark
├── vector/     # 向量计算指令 benchmark
└── memory/     # 内存访问指令 benchmark
```

## 分类说明

### 1. Cube (矩阵计算)

测试 CUBE 块相关的矩阵计算指令：

- **MATMUL** - 矩阵乘法 (C = A × B)
- **MATMUL_ACC** - 矩阵乘加 (C += A × B)
- **TCVT** - 类型转换 (累加器到向量)
- **TCOPYOUT** - 从累加器拷贝到全局内存

测试维度：
- 数据类型：FP32, FP16, BF16, INT8
- Tile 尺寸：16×16, 32×32, 64×64, 128×128
- 矩阵形状：方阵、非方阵、尾块处理

### 2. Vector (向量计算)

测试 VPAR/VSEQ 块相关的向量计算指令：

**逐元素操作：**
- **TADD** - 逐元素加法
- **TSUB** - 逐元素减法
- **TMUL** - 逐元素乘法
- **TDIV** - 逐元素除法
- **TMAX** - 逐元素最大值
- **TMIN** - 逐元素最小值
- **TCMP** - 逐元素比较

**归约操作：**
- **TROWMAX** - 行最大值归约
- **TCOLMAX** - 列最大值归约
- **TROWMIN** - 行最小值归约
- **TCOLMIN** - 列最小值归约
- **TROWEXPAND** - 行扩展
- **TCOLEXPAND** - 列扩展

**特殊函数：**
- **TRECIP** - 倒数
- **TSQRT** - 平方根
- **TABS** - 绝对值
- **TNEG** - 取负
- **TEXP** - 指数函数

**类型转换：**
- **TCVT** - 数据类型转换

测试维度：
- 数据类型：FP32, FP16, BF16, INT32, INT16, INT8
- Tile 尺寸：1×16, 1×32, 1×64, 16×16, 32×32
- 操作数：标量-向量、向量-向量

### 3. Memory (内存访问)

测试内存访问相关指令：

**加载指令：**
- **TLOAD** - 从全局内存加载到 Tile
- **TCOPYIN** - 拷贝到 Tile (带布局转换)

**存储指令：**
- **TSTORE** - 从 Tile 存储到全局内存
- **TCOPYOUT** - 从 Tile 拷贝 (带布局转换)

**索引访问：**
- **TCI** - 生成索引序列
- **MGATHER** - 掩码 Gather
- **MSCATTER** - 掩码 Scatter

测试维度：
- 数据类型：FP32, FP16, BF16, INT32, INT16, INT8
- 内存布局：RowMajor, ColMajor
- 访问模式：连续访问、跨步访问、随机访问
- Tile 尺寸：各种尺寸组合

## 运行方式

```bash
# 编译单个 benchmark
cd microbench/cube
make TESTCASE=matmul_fp32_16x16

# 批量编译
bash compile.all

# 运行测试
gfrun -f output/microbench/cube/matmul_fp32_16x16.elf

# 性能分析
gfsim -f output/microbench/cube/matmul_fp32_16x16.elf
```

## 输出格式

每个 benchmark 输出：
- 功能验证结果
- 执行周期数
- 吞吐量 (元素/周期)
- 带宽利用率 (内存访问类)

## 注意事项

1. 所有 benchmark 使用 `baremetal=off` 模式编译
2. 性能数据仅供参考，实际性能取决于硬件实现
3. 部分指令可能需要特定的 Tile 类型 (Vec, Left, Right, Acc)
4. 内存访问类 benchmark 需要考虑对齐要求
