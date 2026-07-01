# PTO ISA (Tile ISA) vs JCore Compiler 最终对比文档

> **目的**：汇总 PTO ISA 与 JCore 编译器 C++ API 的差异，确定哪些 Tile ISA 操作在 JCore 中缺失或实现不完整。
>
> **参考资料**：
> - PTO ISA Reference (Tile ISA Reference)
> - JCore Linx-TileOP-API (`~/NPU_bench_project/compiler/linx-toolchain-build/src/Linx-TileOP-API/include/jcore`)
> - 实际算子实现：
>   - Tile ISA 算子：`element_wise/gelu_pto.hpp`, `reduction/*`, `transpose/*`, `concat/*`
>   - JCore 算子：`~/NPU_bench_git/SuperNPUBench/benchmark-linxisa/kernels/*`

---

## 1. 汇总表格

### 1.1 按类别统计

| 类别 | Tile ISA 总数 | JCore 完全缺失 | JCore 部分支持 | JCore 完整支持 |
|------|--------------|---------------|---------------|---------------|
| **归约操作** | 28 | 18 | 4 | 6 |
| **归约扩展** | 16 | 10 | 6 | 0 |
| **Tile-Tile 逐元素** | 29 | 14 | 0 | 15 |
| **Tile-标量** | 21 | 9 | 0 | 12 |
| **内存搬运** | 6 | 0 | 3 | 3 |
| **布局重排** | 11 | 6 | 2 | 3 |
| **矩阵运算** | 8 | 6 | 2 | 0 |
| **特殊操作** | 16 | 14 | 1 | 1 |
| **同步与配置** | 11 | 2 | 3 | 6 |
| **总计** | **~146** | **~79** | **~21** | **~46** |

### 1.2 按状态分类的详细清单

#### 状态 1：完全缺失（JCore 无任何对应实现）

| 操作 | 类别 | 说明 | 使用场景 |
|------|------|------|----------|
| `TCOLSUM` | 归约 | 按列求和（含 isBinary 形式） | reducesum_colvec |
| `TCOLMAX` | 归约 | 按列取最大值 | reducemax_colvec |
| `TCOLMIN` | 归约 | 按列取最小值 | - |
| `TCOLPROD` | 归约 | 按列求积 | reduceprod_colvec |
| `TCOLARGMAX` | 归约 | 按列取最大值索引 | - |
| `TCOLARGMIN` | 归约 | 按列取最小值索引 | - |
| `TROWPROD` | 归约 | 按行求积 | reduceprod_rowvec |
| `TROWARGMAX` | 归约 | 按行取最大值索引 | - |
| `TROWARGMIN` | 归约 | 按行取最小值索引 | - |
| `TCOLEXPAND` | 扩展 | 列扩展广播 | - |
| `TCOLEXPANDADD` | 扩展 | 列扩展后加 | - |
| `TCOLEXPANDMUL` | 扩展 | 列扩展后乘 | - |
| `TCOLEXPANDSUB` | 扩展 | 列扩展后减 | - |
| `TCOLEXPANDDIV` | 扩展 | 列扩展后除 | - |
| `TCOLEXPANDMAX` | 扩展 | 列扩展后取最大值 | - |
| `TCOLEXPANDMIN` | 扩展 | 列扩展后取最小值 | - |
| `TROWEXPANDMUL` | 扩展 | 行扩展后乘 | - |
| `TROWEXPANDSUB` | 扩展 | 行扩展后减 | - |
| `TROWEXPANDDIV` | 扩展 | 行扩展后除 | - |
| `TNEG` | 逐元素 | 取负 | - |
| `TPOW` | 逐元素 | 幂运算 | - |
| `TREL` | 逐元素 | ReLU 激活 | - |
| `TPREL` | 逐元素 | PReLU 激活 | - |
| `TLEAKYREL` | 逐元素 | Leaky ReLU | - |
| `TADDC` | 逐元素 | 三元加法 | - |
| `TSUBC` | 逐元素 | 三元减法 | - |
| `TANDS` | Tile-标量 | 标量按位与 | - |
| `TORS` | Tile-标量 | 标量按位或 | - |
| `TXORS` | Tile-标量 | 标量按位异或 | - |
| `TSHLS` | Tile-标量 | 标量左移 | - |
| `TSHRS` | Tile-标量 | 标量右移 | - |
| `TEXPANDS` | Tile-标量 | 标量广播到 tile | 所有算子的初始化 |
| `TREMS` | Tile-标量 | 标量取余 | broadcast_pto |
| `TQUANT` | 特殊 | 量化 | - |
| `TDEQUANT` | 特殊 | 反量化 | - |
| `TPRINT` | 特殊 | 调试打印 | - |
| `TSORT32` | 特殊 | 排序 | - |
| `TMRGSORT` | 特殊 | 归并排序 | - |
| `TRANDOM` | 特殊 | 随机数生成 | - |
| `THISTOGRAM` | 特殊 | 直方图 | - |
| `TTRI` | 特殊 | 生成三角掩码 | - |
| `TMATMUL_BIAS` | 矩阵 | 矩阵乘加偏置 | - |
| `TGEMV` | 矩阵 | 矩阵向量乘法 | - |
| `TGEMV_ACC` | 矩阵 | 矩阵向量乘加 | - |
| `TGEMV_BIAS` | 矩阵 | 矩阵向量乘加偏置 | - |
| `TINSERT` | 布局 | 将子 tile 插入到目标 tile 的指定位置 | broadcast_vec_019_pto, broadcast_vec_039_pto |

#### 状态 2：部分支持（PTO ISA 有多种形式，JCore 仅支持部分形式）

**说明**：PTO ISA 中某些操作有多种重载形式（不同参数数量或模板参数），JCore 仅实现了基础形式，缺少高级形式。

| 操作 | Tile ISA 形式 | JCore 支持形式 | JCore 缺失形式 | 差异要点 |
|------|--------------|---------------|---------------|---------|
| `TTRANS` | `TTRANS(dst, src)` ✅<br>`TTRANS(dst, src, tmp)` ✅ | `TTRANS_Impl(dst, src)` ✅ | `TTRANS(dst, src, tmp)` ❌ | JCore 支持 2 操作数形式，缺 3 操作数形式（带 tmp） |
| `TROWSUM` | `TROWSUM(dst, src)` ✅<br>`TROWSUM(dst, src, tmp)` ✅ | `TROWSUM_Impl(dst, src)` ✅ | `TROWSUM(dst, src, tmp)` ❌ | JCore 支持 2 操作数形式，缺 3 操作数形式（无法用二叉树归约） |
| `TROWMAX` | `TROWMAX(dst, src)` ✅<br>`TROWMAX(dst, src, tmp)` ✅ | `TROWMAX_Impl(dst, src)` ✅ | `TROWMAX(dst, src, tmp)` ❌ | JCore 支持 2 操作数形式，缺 3 操作数形式（无法用二叉树归约） |
| `TROWEXPAND` | `TROWEXPAND(dst, src)` ✅ | `TEXPANDROW(dst, src)` ✅ | 无 | 名称不同；模板约束阻止 src=(N,1) dst=(N,C) |
| `TROWEXPANDADD` | `TROWEXPANDADD(dst, src0, src1)` ✅ | `TROWEXPANDSUM(dst, src0, src1)` ✅ | 无 | 名称不同 |
| `TROWEXPANDMAX` | `TROWEXPANDMAX(dst, src0, src1)` ✅ | `TROWMAXEXPAND(dst, src0, src1)` ✅ | 无 | 名称不同 |
| `TCVT` | `TCVT(dst, src)` ✅<br>`TCVT(dst, src, tmp)` ✅<br>`TCVT(dst, src, RoundMode)` ✅<br>`TCVT(dst, src, tmp, RoundMode, SaturationMode)` ✅ | `TCVT(dst, src)` ✅ | 带 tmp/RoundMode/SaturationMode 的所有形式 ❌ | JCore 仅支持最简 2 操作数形式，缺所有控制参数 |
| `MGATHER` | `MGATHER(dst, src, idx)` ✅<br>`MGATHER<Coalesce, GatherOOB>(dst, src, idx)` ✅ | `MGATHER(dst, src, idx)` ✅ | 模板参数形式 ❌ | JCore 支持基础形式，缺 Coalesce/GatherOOB 模板参数；索引语义不同（元素索引 vs 字节偏移） |
| `TLOAD` | `TLOAD(dst, src)` ✅ | `TCOPYIN(dst, src)` ✅ | 无 | 名称不同 |
| `TSTORE` | `TSTORE(dst, src)` ✅<br>`TSTORE(dst, src, AtomicType)` ✅<br>`TSTORE(dst, src, preQuantScalar)` ✅ | `TCOPYOUT(dst, src)` ✅ | 带 AtomicType/preQuantScalar 的形式 ❌ | JCore 支持基础形式，缺原子写和量化写回 |
| `TDIVS` | `TDIVS(dst, src, scalar)` ✅<br>`TDIVS(dst, scalar, src)` ✅ | `TDIVS(dst, src, scalar)` ✅ | 反向形式 `TDIVS(dst, scalar, src)` ❌ | JCore 支持 tile÷scalar，缺 scalar÷tile；缺 PrecisionType |
| `TEXP` | `TEXP(dst, src)` ✅<br>`TEXP<PrecisionType>(dst, src)` ✅ | `TEXP(dst, src)` ✅ | 带 PrecisionType 形式 ❌ | JCore 支持基础形式，缺精度模式选择 |
| `TRECIP` | `TRECIP(dst, src)` ✅<br>`TRECIP<PrecisionType>(dst, src)` ✅ | `TRECIP(dst, src)` ✅ | 带 PrecisionType 形式 ❌ | JCore 支持基础形式，缺精度模式选择 |

#### 状态 3：完整支持（JCore 有对应实现，功能基本一致）

| 操作 | 类别 | JCore 实现方式 | 使用场景 |
|------|------|---------------|----------|
| `TCI` | 特殊 | `__vec__` kernel | broadcast_pto, 所有 offset 计算 |
| `TADD` | 逐元素 | `__vec__` kernel | broadcast_pto, gelu_pto |
| `TMUL` | 逐元素 | `__vec__` kernel | gelu_pto |
| `TMULS` | Tile-标量 | `__vec__` kernel | broadcast_pto, gelu_pto |
| `TADDS` | Tile-标量 | `__vec__` kernel | gelu_pto |
| `TMAXS` | Tile-标量 | `__vec__` kernel | gelu_pto |
| `TMINS` | Tile-标量 | `__vec__` kernel | gelu_pto |
| `TSUB` | 逐元素 | `__vec__` kernel | - |
| `TDIV` | 逐元素 | `__vec__` kernel | - |
| `TABS` | 逐元素 | `__vec__` kernel | - |
| `TSQRT` | 逐元素 | `__vec__` kernel | - |
| `TRSQRT` | 逐元素 | `__vec__` kernel | - |
| `TLOG` | 逐元素 | `__vec__` kernel | - |
| `TCMP` | 逐元素 | `__vec__` kernel | - |
| `TSEL` | 逐元素 | `__vec__` kernel | - |
| `TREM` | 逐元素 | `__vec__` kernel | - |
| `TFMOD` | 逐元素 | `__vec__` kernel | - |
| `TMATMUL` | 矩阵 | `MatMul` | matmul |
| `TMATMUL_ACC` | 矩阵 | `MatMacc` | matmul |
| `TROWSUM` | 归约 | `__vec__` kernel (简单形式) | reducesum_rowvec |
| `TROWMAX` | 归约 | `__vec__` kernel (简单形式) | reducemax_rowvec |
| `TROWEXPAND` | 扩展 | `__vec__` kernel | broadcast_vec_07 |
| `TROWEXPANDADD` | 扩展 | `__vec__` kernel (融合操作) | - |
| `TROWEXPANDMAX` | 扩展 | `__vec__` kernel (融合操作) | - |
| `TPREFETCH` | 内存 | `__vec__` kernel | - |
| `TPREFETCH_ASYNC` | 内存 | `__vec__` kernel | - |
| `TMOV` | 布局 | `__vec__` kernel | - |
| `TEXTRACT` | 布局 | `__vec__` kernel | - |
| `TRESHAPE` | 布局 | `__vec__` kernel | - |
| `TASSIGN` | 同步 | `__vec__` kernel | - |
| `TSYNC` | 同步 | `__vec__` kernel | - |

---

## 2. 系统性差异

以下差异影响所有操作，是 API 设计层面的系统性缺口：

### 2.1 返回值

| 维度 | PTO ISA | JCore |
|------|---------|-------|
| 返回类型 | `PTO_INST RecordEvent` | `void` |

PTO ISA 每条指令返回异步事件令牌，支持流水线依赖图。JCore 无法表达异步依赖。

### 2.2 WaitEvents 可变参数

| 维度 | PTO ISA | JCore |
|------|---------|-------|
| 签名末尾 | `typename... WaitEvents` + `WaitEvents &... events` | 无 |

PTO ISA 允许传入前置事件实现"等待前一条指令完成后再发射"。JCore 无此机制。

### 2.3 模板类型约束

| 维度 | PTO ISA | JCore |
|------|---------|-------|
| 模板参数 | `typename TileDataDst`, `typename TileDataSrc`（各自独立） | `is_tile_data_v tile_shape`（强制全部操作数同一类型） |

PTO ISA 允许 dst 和 src 是不同的 Tile 类型。JCore 强制 dst/src0/src1 必须是完全相同的 Tile 类型，导致以下场景无法表达：
- dst 和 src 的 ValidCol 不同（如 `TROWEXPAND` src=(N,1) dst=(N,C)）
- dst 和 src 的 dtype 不同（如 `TCVT` float→half）
- index tile 和 data tile 的 dtype 不同（如 `MGATHER` dst=half, indexes=uint32）

---

## 3. 类别详细分析

### 3.1 归约操作（最关键的差异）

#### 按列归约：JCore 完全缺失

| Tile ISA 操作 | 说明 | JCore 状态 | 实际使用 |
|--------------|------|----------|----------|
| `TCOLSUM(dst, src)` | 按列求和 | ❌ 完全缺失 | reducesum_colvec |
| `TCOLSUM(dst, src, tmp, isBinary)` | 按列求和（二叉树归约） | ❌ 完全缺失 | reducesum_colvec（优化版） |
| `TCOLMAX(dst, src)` | 按列取最大值 | ❌ 完全缺失 | reducemax_colvec |
| `TCOLPROD(dst, src)` | 按列求积 | ❌ 完全缺失 | reduceprod_colvec |

**影响**：所有列方向归约操作都必须用 `__vec__` kernel 手写，无法使用 Tile ISA 的高级抽象。

#### 按行归约：JCore 支持基础形式，缺 tmp 形式

| Tile ISA 操作 | 说明 | JCore 支持 | JCore 缺失 |
|--------------|------|----------|----------|
| `TROWSUM(dst, src)` | 按行求和（2 操作数） | ✅ `TROWSUM_Impl(dst, src)` | - |
| `TROWSUM(dst, src, tmp)` | 按行求和（3 操作数，二叉树归约） | - | ❌ 缺失 |
| `TROWMAX(dst, src)` | 按行取最大值（2 操作数） | ✅ `TROWMAX_Impl(dst, src)` | - |
| `TROWMAX(dst, src, tmp)` | 按行取最大值（3 操作数，二叉树归约） | - | ❌ 缺失 |
| `TROWPROD(dst, src)` | 按行求积 | - | ❌ 完全缺失 |

**影响**：行方向归约虽然可以实现（使用 2 操作数形式），但无法使用二叉树归约优化大 tile 的性能。

### 3.2 归约扩展操作

#### 列扩展：JCore 完全缺失

所有 `TCOLEXPAND*` 系列操作在 JCore 中都不存在。

#### 行扩展：部分支持

JCore 有以下融合操作：
- `TROWEXPANDSUM` = `TROWEXPANDADD`
- `TROWMAXEXPAND` = `TROWEXPANDMAX`

但缺少：
- `TROWEXPANDMUL`（行扩展后乘）
- `TROWEXPANDSUB`（行扩展后减）
- `TROWEXPANDDIV`（行扩展后除）

### 3.3 Tile-标量操作

#### 完全缺失

| 操作 | 说明 | 重要性 |
|------|------|--------|
| `TEXPANDS` | 标量广播到 tile | **高** — 所有算子初始化都需要 |
| `TREMS` | 标量取余 | 中 — broadcast_pto 使用 |
| `TANDS`, `TORS`, `TXORS` | 位运算 | 低 |
| `TSHLS`, `TSHRS` | 移位运算 | 低 |

#### 参数不完整

| 操作 | 缺失参数 | 影响 |
|------|---------|------|
| `TDIVS` | `PrecisionType`, 反向重载 | 无法选择精度模式 |
| `TADDS`, `TSUBS`, `TMULS`, `TMAXS`, `TMINS` | 无缺失 | 功能完整 |

### 3.4 逐元素 Tile-Tile 操作

JCore 支持大部分基础算术运算（`TADD`, `TMUL`, `TSUB`, `TDIV`, `TABS`, `TSQRT`, `TRSQRT`, `TLOG`, `TCMP`, `TSEL`, `TREM`, `TFMOD`），但缺少：

| 缺失操作 | 说明 | 影响 |
|---------|------|------|
| `TNEG` | 取负 | 低 — 可用 `TMULS(x, -1)` 替代 |
| `TPOW` | 幂运算 | 中 |
| `TREL`, `TPREL`, `TLEAKYREL` | 激活函数 | 中 — 需手动用 `TMAXS`+`TMUL`+`TADDS` 实现 |
| `TADDC`, `TSUBC` | 三元运算 | 低 |

### 3.5 内存搬运操作

| Tile ISA | JCore | 差异 |
|----------|-------|------|
| `TLOAD` | `TCOPYIN` | 名称不同，功能一致 |
| `TSTORE` | `TCOPYOUT` | 名称不同；缺 `AtomicType`/`preQuantScalar` |
| `MGATHER` | `MGATHER` | 缺 `Coalesce`/`GatherOOB` 模板参数；索引语义不同 |
| `MSCATTER` | `MSCATTER` | 功能基本一致 |
| `TPREFETCH` | `TPREFETCH` | 功能一致 |
| `TPREFETCH_ASYNC` | `TPREFETCH_ASYNC` | 功能一致 |

**关键差异**：`MGATHER` 的索引语义不同：
- Tile ISA：`Coalesce::Elem` 模式使用元素索引（`dst[i,j] = src[idx[i,j]]`）
- JCore：使用字节偏移（调用方需手动乘 `sizeof(dtype)`）

### 3.6 布局重排操作

| Tile ISA | JCore | 差异 |
|----------|-------|------|
| `TTRANS(dst, src, tmp)` | `TTRANS_Impl(dst, src)` | 缺 tmp，无法做多阶段转置 |
| `TMOV` | `TCOPY` | 名称不同 |
| `TEXTRACT` | `TEXTRACT` | 功能一致 |
| `TINSERT` | 无 | 完全缺失 |
| `TRESHAPE` | `TRESHAPE` | 功能一致 |
| `TCONCAT` | 无 | 完全缺失 |
| `TPACK` | 无 | 完全缺失 |
| `TFILLPAD` | 无 | 完全缺失 |
| `TIMG2COL` | 无 | 完全缺失 |

### 3.7 矩阵运算

| Tile ISA | JCore | 差异 |
|----------|-------|------|
| `TMATMUL` | `MatMul` | 功能一致 |
| `TMATMUL_ACC` | `MatMacc` | 功能一致 |
| `TMATMUL_BIAS` | 无 | 完全缺失 |
| `TGEMV`, `TGEMV_ACC`, `TGEMV_BIAS` | 无 | 完全缺失 |

### 3.8 特殊操作

| Tile ISA | JCore | 差异 |
|----------|-------|------|
| `TCI` | `TCI` | 功能一致 |
| `TTRI` | 无 | 完全缺失 |
| `TQUANT`, `TDEQUANT` | 无 | 完全缺失 |
| `TSORT32`, `TMRGSORT` | 无 | 完全缺失 |
| `TRANDOM` | 无 | 完全缺失 |
| `THISTOGRAM` | `THISTOGRAM` | JCore 有内联汇编实现 |
| `TPRINT` | 无 | 完全缺失 |

---

## 4. 实际算子中的使用情况

### 4.1 gelu_pto（element_wise）

使用的 Tile ISA 操作：
```
TLOAD, TCVT, TMAXS, TMINS, TMUL, TMULS, TADDS, TEXP, TRECIP, TSTORE
```

| 操作 | JCore 状态 | 实现方式 |
|------|-----------|---------|
| `TLOAD` | ⚠️ 名称不同 (`TCOPYIN`) | `__vec__` kernel |
| `TCVT` | ⚠️ 签名不完整 | `__vec__` kernel（无 RoundMode/SaturationMode） |
| `TMAXS` | ✅ 支持 | `__vec__` kernel |
| `TMINS` | ✅ 支持 | `__vec__` kernel |
| `TMUL` | ✅ 支持 | `__vec__` kernel |
| `TMULS` | ✅ 支持 | `__vec__` kernel |
| `TADDS` | ✅ 支持 | `__vec__` kernel |
| `TEXP` | ⚠️ 缺 PrecisionType | `__vec__` kernel |
| `TRECIP` | ⚠️ 缺 PrecisionType | `__vec__` kernel |
| `TSTORE` | ⚠️ 名称不同 (`TCOPYOUT`) | `__vec__` kernel |

**结论**：GELU 可以在 JCore 中实现，但 `TCVT` 缺少精度控制参数，`TEXP`/`TRECIP` 缺少精度模式选择。

### 4.2 reduction 算子

#### reducesum_colvec

```
TCOLSUM(dst, src, tmp, isBinary), TEXPANDS, TADD, TLOAD, TSTORE
```

| 操作 | JCore 状态 | 影响 |
|------|-----------|------|
| `TCOLSUM(dst, src, tmp, isBinary)` | ❌ JCore 无任何 TCOLSUM 形式 | **阻塞** — 必须用 `__vec__` 手写 |
| `TEXPANDS` | ❌ 完全缺失 | **阻塞** — 必须用 `__vec__` 手写 |
| `TADD` | ✅ 支持 | - |
| `TLOAD` | ⚠️ 名称不同 (`TCOPYIN`) | - |
| `TSTORE` | ⚠️ 名称不同 (`TCOPYOUT`) | - |

#### reducesum_rowvec

```
TROWSUM(dst, src, tmp), TEXPANDS, TADD, TLOAD, TSTORE
```

| 操作 | JCore 状态 | 影响 |
|------|-----------|------|
| `TROWSUM(dst, src, tmp)` | ⚠️ 仅支持 2 操作数形式 `TROWSUM(dst, src)`，缺 3 操作数形式 | 可使用 2 操作数形式，但无法用二叉树归约优化 |
| `TEXPANDS` | ❌ 完全缺失 | **阻塞** — 必须用 `__vec__` 手写 |

#### reduceprod_colvec / reduceprod_rowvec

```
TCOLPROD / TROWPROD, TEXPANDS, TMUL, TLOAD, TSTORE
```

| 操作 | JCore 状态 | 影响 |
|------|-----------|------|
| `TCOLPROD(dst, src)` | ❌ JCore 无任何 TCOLPROD 形式 | **阻塞** — 必须用 `__vec__` 手写 |
| `TROWPROD(dst, src, tmp)` | ❌ JCore 无任何 TROWPROD 形式 | **阻塞** — 必须用 `__vec__` 手写 |
| `TEXPANDS` | ❌ 完全缺失 | **阻塞** — 必须用 `__vec__` 手写 |

### 4.3 concat 算子

#### concat_gather

```
TCI, TDIVS, TMULS, TSUB, TADD, TMOV, TEXPANDS, MGATHER, TSTORE
```

| 操作 | JCore 状态 | 影响 |
|------|-----------|------|
| `TCI` | ✅ 支持 | - |
| `TDIVS` | ⚠️ 缺 PrecisionType | 无法选择精度模式 |
| `TMULS` | ✅ 支持 | - |
| `TSUB` | ✅ 支持 | - |
| `TADD` | ✅ 支持 | - |
| `TMOV` | ⚠️ 名称不同 (`TCOPY`) | - |
| `TEXPANDS` | ❌ 完全缺失 | **阻塞** — 必须用 `__vec__` 手写 |
| `MGATHER` | ⚠️ 索引语义不同 | 需要调整 offset 计算 |
| `TSTORE` | ⚠️ 名称不同 | - |

#### concat_scatter

```
TCI, TDIVS, TMULS, TSUB, TADD, TMOV, TEXPANDS, TLOAD, MSCATTER
```

| 操作 | JCore 状态 | 影响 |
|------|-----------|------|
| `TEXPANDS` | ❌ 完全缺失 | **阻塞** — 必须用 `__vec__` 手写 |
| 其余 | 同上 | - |

### 4.4 transpose 算子

#### transpose_tile_isa（N 维通用转置）

```
TCI, TDIVS, TMULS, TSUB, TADD, TMOV, TEXPANDS, MGATHER, TSTORE
```

| 操作 | JCore 状态 | 影响 |
|------|-----------|------|
| `TEXPANDS` | ❌ 完全缺失 | **阻塞** — 必须用 `__vec__` 手写 |
| 其余 | 同 concat | - |

#### tile_transpose_2d（2D 硬件转置）

```
TTRANS(dst, src), TLOAD, TSTORE
```

| 操作 | JCore 状态 | 影响 |
|------|-----------|------|
| `TTRANS(dst, src)` | ⚠️ 缺 tmp 形式 | 无法做多阶段转置优化 |
| `TLOAD` | ⚠️ 名称不同 | - |
| `TSTORE` | ⚠️ 名称不同 | - |

---

## 5. 优先级建议

### P0：阻塞性缺失（无法用 Tile ISA 写出算子）

| 操作 | 阻塞的算子 | 建议 |
|------|-----------|------|
| `TEXPANDS` | 几乎所有算子的初始化 | **必须新增** — 这是最常用的操作之一 |
| `TCOLSUM` | reducesum_colvec | **必须新增** — 列归约的核心操作 |
| `TCOLMAX` | reducemax_colvec | **必须新增** |
| `TCOLPROD` | reduceprod_colvec | **必须新增** |
| `TROWPROD` | reduceprod_rowvec | **必须新增** |
| `TREMS` | broadcast_pto | **必须新增** — 标量取余 |
| `TCVT` 完整签名 | gelu_pto | **必须修改** — 补充 RoundMode/SaturationMode/tmp |

### P1：功能不完整（可写但表达力受限）

| 操作 | JCore 已支持形式 | JCore 缺失形式 | 建议 |
|------|----------------|---------------|------|
| `TROWSUM` | `TROWSUM(dst, src)` ✅ | `TROWSUM(dst, src, tmp)` ❌ | **建议修改** — 补充 3 操作数形式以支持二叉树归约 |
| `TROWMAX` | `TROWMAX(dst, src)` ✅ | `TROWMAX(dst, src, tmp)` ❌ | **建议修改** — 补充 3 操作数形式以支持二叉树归约 |
| `TTRANS` | `TTRANS(dst, src)` ✅ | `TTRANS(dst, src, tmp)` ❌ | **建议修改** — 补充 3 操作数形式以支持多阶段转置 |
| `TCVT` | `TCVT(dst, src)` ✅ | `TCVT(dst, src, tmp, RoundMode, SaturationMode)` ❌ | **建议修改** — 补充带控制参数的形式 |
| `MGATHER` | `MGATHER(dst, src, idx)` ✅ | `MGATHER<Coalesce, GatherOOB>(...)` ❌ | **建议修改** — 补充模板参数，明确 Coalesce/OOB 语义 |
| `TEXP` | `TEXP(dst, src)` ✅ | `TEXP<PrecisionType>(dst, src)` ❌ | **建议修改** — 补充精度模式选择 |
| `TRECIP` | `TRECIP(dst, src)` ✅ | `TRECIP<PrecisionType>(dst, src)` ❌ | **建议修改** — 补充精度模式选择 |
| `TDIVS` | `TDIVS(dst, src, scalar)` ✅ | `TDIVS(dst, scalar, src)` ❌ | **建议修改** — 补充反向形式和 PrecisionType |

### P2：命名和系统性差异

| 差异 | 影响范围 | 建议 |
|------|---------|------|
| `TLOAD` vs `TCOPYIN` | 所有内存加载 | 建议统一命名或提供别名 |
| `TSTORE` vs `TCOPYOUT` | 所有内存写回 | 建议统一命名或提供别名 |
| `TEXPANDS` vs `TEXPANDSCALAR` | 标量广播 | 建议统一命名 |
| `TMOV` vs `TCOPY` | tile 拷贝 | 建议统一命名 |
| `RecordEvent` 返回值 | 所有操作 | 建议补充异步支持 |
| `WaitEvents` 参数 | 所有操作 | 建议补充依赖等待 |
| 模板类型约束 | 所有操作 | 建议拆分 dst/src 类型约束 |

### P3：可选增强（当前不阻塞）

| 操作 | 说明 | 优先级 |
|------|------|--------|
| `TNEG`, `TPOW` | 逐元素运算 | 低 |
| `TREL`, `TPREL`, `TLEAKYREL` | 激活函数 | 低 — 可用基础运算组合 |
| `TANDS`, `TORS`, `TXORS`, `TSHLS`, `TSHRS` | 位运算 | 低 |
| `TCOLEXPAND*`, `TROWEXPAND*` 系列 | 扩展操作 | 中 — 部分场景需要 |
| `TQUANT`, `TDEQUANT` | 量化操作 | 中 — 特定场景需要 |
| `TSORT32`, `TMRGSORT`, `TRANDOM`, `THISTOGRAM` | 特殊操作 | 低 — 使用频率低 |

---

## 6. 总结

### 6.1 关键发现

1. **归约操作是最关键的差异**：所有按列归约（`TCOLSUM`/`TCOLMAX`/`TCOLPROD`）在 JCore 中完全缺失，必须用 `__vec__` kernel 手写。

2. **`TEXPANDS` 是高频缺失操作**：几乎所有算子都需要标量广播来初始化 tile，但 JCore 没有对应 API。

3. **tmp 参数缺失影响性能优化**：`TROWSUM`/`TROWMAX`/`TTRANS` 的 tmp 形式支持二叉树归约和多阶段处理，JCore 缺少这些形式会导致大 tile 性能下降。

4. **精度控制参数缺失**：`TCVT`/`TEXP`/`TRECIP`/`TDIVS` 缺少 `PrecisionType` 和 `RoundMode` 参数，无法控制数值精度和舍入行为。

5. **MGATHER 语义差异**：Tile ISA 使用元素索引，JCore 使用字节偏移，需要调用方手动转换。

### 6.2 实现策略建议

| 场景 | 推荐方案 | 原因 |
|------|---------|------|
| 需要按列归约 | **必须用 `__vec__`** | Tile ISA 的 `TCOLSUM`/`TCOLMAX` 在 JCore 中不存在 |
| 需要标量广播初始化 | **必须用 `__vec__`** | `TEXPANDS` 在 JCore 中不存在 |
| 需要高精度类型转换 | **必须用 `__vec__`** | JCore 的 `TCVT` 缺少 RoundMode 参数 |
| 需要二叉树归约优化 | **必须用 `__vec__`** | JCore 的归约操作缺少 tmp 参数 |
| 简单逐元素运算 | **可用 Tile ISA** | `TADD`/`TMUL`/`TMULS` 等基础操作 JCore 支持 |
| 内存搬运 | **可用 Tile ISA** | `TLOAD`/`TSTORE` 功能一致（注意名称差异） |

### 6.3 最终结论

**PTO ISA 的优势**：
- 完整的归约操作支持（按行/按列，带 tmp 优化形式）
- 完整的扩展操作支持
- 精度控制参数（RoundMode, SaturationMode, PrecisionType）
- 硬件加速的 `TTRANS` 转置指令
- 统一的元素索引语义（`MGATHER`）

**JCore 的现状**：
- 基础逐元素运算支持良好
- 归约操作严重缺失（特别是按列归约）
- 缺少性能优化形式（tmp 参数）
- 缺少精度控制参数
- 部分操作命名不一致

**建议**：在 JCore 编译器中优先补充 P0 级缺失操作（`TEXPANDS`、`TCOLSUM`、`TCOLMAX`、`TCOLPROD`、`TROWPROD`、`TREMS`、`TCVT` 完整签名），这些是实际算子开发中最常用的操作。
