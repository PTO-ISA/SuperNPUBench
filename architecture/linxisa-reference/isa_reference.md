# LinxISA 张量指令集参考

## 1. 指令集概述

LinxISA 张量指令集主要包括：
- **向量指令集**：在向量数据块（VPAR/VSEQ）中执行
- **矩阵指令集**：在矩阵数据块（CUBE）中执行
- **数据搬运指令**：在数据搬运块（TMA）中执行

## 2. 向量指令集

### 2.1 算术运算

#### V.ADD - 向量加法
```
语法: V.ADD vd, vs1, vs2
功能: vd[i] = vs1[i] + vs2[i]
类型: 支持 FP64/32/16, INT64/32/16/8
```

#### V.ADDI - 向量加立即数
```
语法: V.ADDI vd, vs1, imm
功能: vd[i] = vs1[i] + imm
类型: 支持 FP64/32/16, INT64/32/16/8
```

#### V.SUB - 向量减法
```
语法: V.SUB vd, vs1, vs2
功能: vd[i] = vs1[i] - vs2[i]
类型: 支持 FP64/32/16, INT64/32/16/8
```

#### V.MUL - 向量乘法
```
语法: V.MUL vd, vs1, vs2
功能: vd[i] = vs1[i] * vs2[i]
类型: 支持 FP64/32/16, INT64/32/16/8
```

#### V.MADD - 向量乘加
```
语法: V.MADD vd, vs1, vs2, vs3
功能: vd[i] = vs1[i] * vs2[i] + vs3[i]
类型: 支持 FP64/32/16
```

#### V.DIV - 向量除法
```
语法: V.DIV vd, vs1, vs2
功能: vd[i] = vs1[i] / vs2[i]
类型: 支持 FP64/32/16
```

#### V.REM - 向量取余
```
语法: V.REM vd, vs1, vs2
功能: vd[i] = vs1[i] % vs2[i]
类型: 支持 FP64/32/16, INT64/32/16/8
```

### 2.2 比较操作

#### V.CMP.EQ - 向量相等比较
```
语法: V.CMP.EQ vd, vs1, vs2
功能: vd[i] = (vs1[i] == vs2[i]) ? 1 : 0
类型: 支持 FP64/32/16, INT64/32/16/8
```

#### V.CMP.NE - 向量不等比较
```
语法: V.CMP.NE vd, vs1, vs2
功能: vd[i] = (vs1[i] != vs2[i]) ? 1 : 0
类型: 支持 FP64/32/16, INT64/32/16/8
```

#### V.CMP.LT - 向量小于比较
```
语法: V.CMP.LT vd, vs1, vs2
功能: vd[i] = (vs1[i] < vs2[i]) ? 1 : 0
类型: 支持 FP64/32/16, INT64/32/16/8
```

#### V.CMP.GE - 向量大于等于比较
```
语法: V.CMP.GE vd, vs1, vs2
功能: vd[i] = (vs1[i] >= vs2[i]) ? 1 : 0
类型: 支持 FP64/32/16, INT64/32/16/8
```

### 2.3 条件选择

#### V.CSEL - 向量条件选择
```
语法: V.CSEL vd, vs1, vs2, vm
功能: vd[i] = vm[i] ? vs1[i] : vs2[i]
类型: 支持 FP64/32/16, INT64/32/16/8
```

#### V.PSEL - 谓词选择
```
语法: V.PSEL vd, vs1, vs2, pm
功能: 根据谓词掩码 pm 选择 vs1 或 vs2
```

### 2.4 位操作

#### V.AND - 向量按位与
```
语法: V.AND vd, vs1, vs2
功能: vd[i] = vs1[i] & vs2[i]
类型: INT64/32/16/8
```

#### V.OR - 向量按位或
```
语法: V.OR vd, vs1, vs2
功能: vd[i] = vs1[i] | vs2[i]
类型: INT64/32/16/8
```

#### V.XOR - 向量按位异或
```
语法: V.XOR vd, vs1, vs2
功能: vd[i] = vs1[i] ^ vs2[i]
类型: INT64/32/16/8
```

#### V.SLL - 向量左移
```
语法: V.SLL vd, vs1, vs2
功能: vd[i] = vs1[i] << vs2[i]
类型: INT64/32/16/8
```

#### V.SRL - 向量逻辑右移
```
语法: V.SRL vd, vs1, vs2
功能: vd[i] = vs1[i] >> vs2[i] (逻辑右移)
类型: INT64/32/16/8
```

#### V.SRA - 向量算术右移
```
语法: V.SRA vd, vs1, vs2
功能: vd[i] = vs1[i] >> vs2[i] (算术右移)
类型: INT64/32/16/8
```

### 2.5 访存指令

#### V.LB/LH/LW/LD - 向量加载
```
语法: V.LB vd, offset(base)  // 加载字节
      V.LH vd, offset(base)  // 加载半字
      V.LW vd, offset(base)  // 加载字
      V.LD vd, offset(base)  // 加载双字
功能: 从内存加载数据到向量寄存器
```

#### V.SB/SH/SW/SD - 向量存储
```
语法: V.SB vd, offset(base)  // 存储字节
      V.SH vd, offset(base)  // 存储半字
      V.SW vd, offset(base)  // 存储字
      V.SD vd, offset(base)  // 存储双字
功能: 从向量寄存器存储数据到内存
```

#### V.LW.BRG - 桥接加载
```
语法: V.LW.BRG vd, offset(base)
功能: 跨 lane 桥接加载
```

#### V.SW.BRG - 桥接存储
```
语法: V.SW.BRG vd, offset(base)
功能: 跨 lane 桥接存储
```

### 2.6 浮点特殊运算

#### V.FABS - 向量绝对值
```
语法: V.FABS vd, vs
功能: vd[i] = |vs[i]|
类型: FP64/32/16
```

#### V.FSQRT - 向量平方根
```
语法: V.FSQRT vd, vs
功能: vd[i] = sqrt(vs[i])
类型: FP64/32/16
```

#### V.FEXP - 向量指数
```
语法: V.FEXP vd, vs
功能: vd[i] = exp(vs[i])
类型: FP64/32/16
```

#### V.FRECIP - 向量倒数
```
语法: V.FRECIP vd, vs
功能: vd[i] = 1.0 / vs[i]
类型: FP64/32/16
```

### 2.7 归约操作

#### V.CADD - 向量通道归约加法
```
语法: V.CADD vd, vs
功能: vd = sum(vs[i]) 对所有 lane 求和
类型: FP64/32/16, INT64/32/16/8
```

#### V.CMAX - 向量通道归约最大值
```
语法: V.CMAX vd, vs
功能: vd = max(vs[i]) 对所有 lane 求最大值
类型: FP64/32/16, INT64/32/16/8
```

#### V.CMIN - 向量通道归约最小值
```
语法: V.CMIN vd, vs
功能: vd = min(vs[i]) 对所有 lane 求最小值
类型: FP64/32/16, INT64/32/16/8
```

## 3. 矩阵指令集（CUBE 块）

### 3.1 矩阵乘法

CUBE 块执行矩阵乘法操作：

```
C[M,N] = A[M,K] × B[K,N] + C[M,N] (可选累加)
```

#### 约束条件

- **形状约束**:
  - A.GetValidRow() = C.GetValidRow() = M
  - A.GetValidCol() = B.GetValidRow() = K
  - B.GetValidCol() = C.GetValidCol() = N
  - M, K, N ∈ [1, 4095]

- **数据类型约束**:

| 累加器类型 | A 类型 | B 类型 |
|-----------|--------|--------|
| INT32 | INT8 | INT8 |
| FP32 | FP16 | FP16 |
| FP32 | BF16 | BF16 |
| FP32 | FP32 | FP32 |

### 3.2 CUBE 块结构

```
bstart.cube
  // 1. 配置矩阵参数
  // 2. 加载 A, B 到 L0 buffer
  // 3. 执行矩阵乘法
  // 4. 存储结果 C
bstop
```

### 3.3 L0 Buffer

CUBE 块使用 L0 buffer 进行矩阵运算：

- **L0A**: 矩阵 A 输入 buffer
- **L0B**: 矩阵 B 输入 buffer
- **L0C**: 累加器 buffer（结果 C）
- **ACC**: 累加寄存器

## 4. 数据搬运指令（TMA 块）

### 4.1 DMA 传输

TMA 块支持高效的数据搬运：

```asm
BSTART.TLOAD INT32
  // B.ARG / B.DIM / B.IOR / B.IOT descriptors
```

### 4.2 搬运模式

- **连续搬运**: 连续内存区域传输
- **分块搬运**: 支持 stride 和 tiling
- **Gather/Scatter**: 非连续内存访问

## 5. 数据类型转换

### 5.1 浮点转换

| 源类型 | 目标类型 | 指令 |
|--------|---------|------|
| FP64 | FP32 | V.FCVT |
| FP32 | FP16 | V.FCVT |
| FP32 | BF16 | V.FCVT |
| FP16 | FP32 | V.FCVT |
| BF16 | FP32 | V.FCVT |

### 5.2 整数转换

| 源类型 | 目标类型 | 指令 |
|--------|---------|------|
| INT64 | INT32 | V.ICVT |
| INT32 | INT16 | V.ICVT |
| INT16 | INT8 | V.ICVT |
| INT32 | FP32 | V.ICVTF |
| FP32 | INT32 | V.FCVTI |

## 6. 指令编码

### 6.1 块头编码

块头包含以下信息：
- Block Type (块类型)
- Branch Type (分支类型)
- Loop Bounds (循环边界)
- Scheduling Hints (调度提示)

### 6.2 指令格式

向量指令采用固定长度编码：
- 16-bit 压缩指令
- 32-bit 标准指令
- 48/64-bit 扩展指令

## 7. 性能考虑

### 7.1 向量化

- 优先使用 VPAR 块（并行执行）
- 避免 VSEQ 块（串行执行）
- 充分利用 64 lanes

### 7.2 数据局部性

- 合理使用 Tile Register
- 减少全局内存访问
- 使用 TMA 块进行数据预取

### 7.3 流水线

- 计算与搬运重叠
- 使用多块指令流水线
- 避免数据依赖停顿

## 8. 错误处理

### 8.1 常见错误

1. **类型不匹配**: 操作数类型不一致
2. **对齐错误**: 数据未满足对齐要求
3. **形状不匹配**: 矩阵维度不兼容
4. **资源溢出**: Tile Register 空间不足

### 8.2 调试建议

- 使用 gfrun 进行功能验证
- 使用 gfsim 进行性能分析
- 检查指令编码和约束条件

---

**文档版本**: 1.0
**最后更新**: 2026-01-08
