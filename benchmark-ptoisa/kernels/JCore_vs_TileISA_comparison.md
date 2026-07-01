# Tile ISA vs JCore 对比分析

本文档对比 Tile ISA Reference 与 Linx-TileOP-API（JCore）的差异，聚焦于 **Tile ISA 有但 JCore 没有的操作** 以及 **两者都有但实现不同的操作**。

---

## 1. 两者都有但实现不同的操作

### 1.1 TTRANS：2D 转置

| 形式 | JCore | Tile ISA |
|---|---|---|
| **基础转置** | ✅ `TTRANS_Impl(dst, src)` | ✅ `TTRANS(dst, src)` |
| **带临时 tile** | ❌ 不支持 | ✅ `TTRANS(dst, src, tmp)` |

**实现差异**：
- **JCore**：通过 `__vec__` kernel 逐元素复制
  ```cpp
  void __vec__ TTrans_RowMajor(dst, src) {
      size_t i = blkv_get_index_x();
      size_t j = blkv_get_index_y();
      size_t idx_src = j * RowStride + i;
      size_t idx_dst = i * RowStride + j;
      blkv_get_tile_ptr(dst)[idx_dst] = blkv_get_tile_ptr(src)[idx_src];
  }
  ```
- **Tile ISA**：硬件原生转置指令，性能更优

### 1.2 TROWSUM：按行求和

| 形式 | JCore | Tile ISA |
|---|---|---|
| **简单形式（无 tmp）** | ✅ `TROWSUM_Impl(dst, src)` | ✅ `TROWSUM(dst, src)` |
| **带临时 tile（多阶段）** | ❌ 不支持 | ✅ `TROWSUM(dst, src, tmp)` |

**JCore 实现方式**：通过 `__vec__` kernel 实现
```cpp
void __vec__ TRowSum_NoFractal_Impl(dst, src) {
    size_t i = blkv_get_index_x();  // 行索引
    DType sum = src[i * RowStride];
    for (size_t j = 1; j < ValidCol; ++j) {
        sum += src[i * RowStride + j * ColStride];
    }
    dst[i * RowStride] = sum;
}
```

**支持特性**：
- 支持 NoFractal（普通布局）和 NzLayout（Nz 分形布局）
- 支持动态和静态 shape
- 不支持多阶段归约（无 tmp 参数）

### 1.3 TROWMAX：按行取最大值

| 形式 | JCore | Tile ISA |
|---|---|---|
| **简单形式** | ✅ `TROWMAX_Impl(dst, src)` | ✅ `TROWMAX(dst, src)` |

**JCore 实现方式**：通过 `__vec__` kernel 实现（类似 TRowSum，使用 `blkv_max`）

### 1.4 TCI：连续索引生成

| 形式 | JCore | Tile ISA |
|---|---|---|
| **升序/降序** | ✅ `TCI_Impl(dst, start)` | ✅ `TCI(dst, start)` |

**实现差异**：
- **JCore**：通过 `__vec__` kernel 实现
  ```cpp
  blkv_get_tile_ptr(dst)[index] = s + static_cast<DType>(index);  // 升序
  ```
- **Tile ISA**：可能是硬件指令，性能更优

**限制**：仅支持整型（`int32_t`, `uint32_t`, `int16_t`, `uint16_t`）

---

## 2. Tile ISA 有但 JCore 没有的操作

### 2.1 归约操作（JCore 缺失部分）

| 操作 | 说明 | JCore 状态 |
|---|---|---|
| `TROWSUM(dst, src, tmp)` | 按行求和（多阶段归约） | ❌ 不支持带 tmp 形式 |
| `TCOLSUM(dst, src)` | 按列求和 | ❌ 完全缺失 |
| `TCOLSUM(dst, src, tmp, isBinary)` | 按列求和（二叉树归约） | ❌ 完全缺失 |
| `TROWPROD(dst, src, tmp)` | 按行求积（需要 tmp） | ❌ 完全缺失 |
| `TCOLPROD(dst, src)` | 按列求积（不需要 tmp） | ❌ 完全缺失 |
| `TROWMIN`, `TCOLMIN` | 按行/列取最小值 | ⚠️ 仅有 TRowMin（类似 TRowMax） |
| `TCOLMAX` | 按列取最大值 | ❌ 完全缺失 |
| `TROWARGMAX`, `TROWARGMIN` | 按行取最大/最小值索引 | ❌ 完全缺失 |
| `TCOLARGMAX`, `TCOLARGMIN` | 按列取最大/最小值索引 | ❌ 完全缺失 |

### 2.2 归约扩展操作（JCore 缺失部分）

| 操作 | 说明 | JCore 状态 |
|---|---|---|
| `TROWEXPAND(dst, src)` | 行扩展广播 | ✅ `TROWEXPAND_Impl` 存在 |
| `TROWEXPANDADD(dst, src0, src1)` | 行扩展后加 | ⚠️ 有 `TROWEXPANDSUM` 融合操作 |
| `TROWEXPANDMUL(dst, src0, src1)` | 行扩展后乘 | ❌ 缺失 |
| `TROWEXPANDSUB`, `TROWEXPANDDIV` | 行扩展后减/除 | ❌ 缺失 |
| `TROWEXPANDMAX`, `TROWEXPANDMIN` | 行扩展后取最大/最小 | ⚠️ 有 `TROWMAXEXPAND` |
| `TCOLEXPAND`, `TCOLEXPANDADD`, ... | 列扩展系列 | ❌ 完全缺失 |
| `TROWEXPANDEXPDIF`, `TCOLEXPANDEXPDIF` | 指数差运算 | ❌ 缺失 |

**JCore 已有操作**：
- `TROWEXPAND_Impl` - 基础行扩展
- `TROWEXPANDSUM` - 行求和+扩展（融合操作）
- `TROWMAXEXPAND` - 行最大值+扩展（融合操作）

### 2.3 Tile-标量操作

| 操作 | 说明 | JCore 状态 |
|---|---|---|
| `TEXPANDS(dst, scalar)` | 广播标量到 tile | ❌ 缺失 |
| `TADDS(dst, src, scalar)` | Tile + 标量 | ✅ `TADDS_Impl` 存在 |
| `TSUBS(dst, src, scalar)` | Tile - 标量 | ✅ `TSUBS_Impl` 存在 |
| `TMULS(dst, src, scalar)` | Tile × 标量 | ✅ `TMULS_Impl` 存在 |
| `TDIVS(dst, src, scalar)` | Tile / 标量 | ✅ `TDIVS_Impl` 存在 |
| `TMAXS(dst, src, scalar)` | Tile 与标量取最大值 | ✅ `TMAXS_Impl` 存在 |
| `TMINS(dst, src, scalar)` | Tile 与标量取最小值 | ✅ `TMINS_Impl` 存在 |
| `TANDS(dst, src, scalar)` | Tile 与标量按位与 | ❌ 缺失 |
| `TORS(dst, src, scalar)` | Tile 与标量按位或 | ❌ 缺失 |
| `TXORS(dst, src, scalar)` | Tile 与标量按位异或 | ❌ 缺失 |
| `TSHLS(dst, src, scalar)` | Tile 与标量左移 | ❌ 缺失 |
| `TSHRS(dst, src, scalar)` | Tile 与标量右移 | ❌ 缺失 |

### 2.4 逐元素 Tile-Tile 操作

| 操作 | 说明 | JCore 状态 |
|---|---|---|
| `TADD(dst, src0, src1)` | Tile + Tile | ✅ `TADD_Impl` 存在 |
| `TSUB(dst, src0, src1)` | Tile - Tile | ✅ `TSUB_Impl` 存在 |
| `TMUL(dst, src0, src1)` | Tile × Tile | ✅ `TMUL_Impl` 存在 |
| `TDIV(dst, src0, src1)` | Tile / Tile | ✅ `TDIV_Impl` 存在 |
| `TABS(dst, src)` | 绝对值 | ✅ `TABS_Impl` 存在 |
| `TNEG(dst, src)` | 取负 | ❌ 缺失 |
| `TEXP(dst, src)` | 指数 | ✅ `TEXP_Impl` 存在 |
| `TLOG(dst, src)` | 对数 | ✅ `TLOG_Impl` 存在 |
| `TSQRT(dst, src)` | 平方根 | ✅ `TSQRT_Impl` 存在 |
| `TRSQRT(dst, src)` | 平方根倒数 | ✅ `TRSQRT_Impl` 存在 |
| `TPOW(dst, src0, src1)` | 幂运算 | ❌ 缺失 |
| `TREL(dst, src)` | ReLU 激活 | ❌ 缺失（需手动实现） |
| `TPREL(dst, src, slope)` | PReLU 激活 | ❌ 缺失 |
| `TLEAKYREL(dst, src, slope)` | Leaky ReLU | ❌ 缺失 |
| `TCMP(dst, src0, src1)` | 比较生成掩码 | ✅ `TCMP_Impl` 存在 |
| `TSEL(dst, mask, src0, src1)` | 条件选择 | ✅ `TSEL_Impl` 存在 |
| `TADD C(dst, src0, src1, src2)` | 三元加法 | ❌ 缺失 |
| `TSUBC(dst, src0, src1, src2)` | 三元减法 | ❌ 缺失 |
| `TREM(dst, src0, src1)` | 余数 | ✅ `TREM_Impl` 存在 |
| `TFMOD(dst, src0, src1)` | 浮点模 | ✅ `TFMOD_Impl` 存在 |

### 2.5 特殊操作

| 操作 | 说明 | JCore 状态 |
|---|---|---|
| `TTRI(dst, ...)` | 生成三角掩码 | ❌ 缺失 |
| `TQUANT(dst, src, scale, zp)` | 量化 | ❌ 缺失 |
| `TDEQUANT(dst, src, scale, zp)` | 反量化 | ❌ 缺失 |
| `TPRINT` | 调试打印 | ❌ 缺失 |
| `TSORT32`, `TMRGSORT` | 排序 | ❌ 缺失 |
| `TRANDOM` | 随机数生成 | ❌ 缺失 |
| `THISTOGRAM` | 直方图 | ⚠️ JCore 有 `THISTOGRAM`（内联汇编实现） |

### 2.6 矩阵运算

| 操作 | 说明 | JCore 状态 |
|---|---|---|
| `TMATMUL(dst, src0, src1)` | 矩阵乘法 | ⚠️ JCore 有 `MatMul` |
| `TMATMUL_ACC(dst, src0, src1)` | 矩阵乘加 | ⚠️ JCore 有 `MatMacc` |
| `TMATMUL_BIAS(dst, src0, src1, bias)` | 矩阵乘加偏置 | ❌ 缺失 |
| `TGEMV`, `TGEMV_ACC`, `TGEMV_BIAS` | 矩阵向量乘法系列 | ❌ 缺失 |

---

## 3. 总结

### 3.1 JCore 已有的归约/扩展操作

| 类别 | JCore 已支持 | Tile ISA 有但 JCore 缺失 |
|---|---|---|
| **按行归约** | `TROWSUM`, `TROWMAX`, `TROWMIN` | `TROWSUM(dst, src, tmp)`（多阶段形式） |
| **按列归约** | ❌ 全部缺失 | `TCOLSUM`, `TCOLMAX`, `TCOLMIN` 等 |
| **按行扩展** | `TROWEXPAND`, `TROWEXPANDSUM`, `TROWMAXEXPAND` | `TROWEXPANDMUL`, `TROWEXPANDDIV` 等 |
| **按列扩展** | ❌ 全部缺失 | `TCOLEXPAND` 系列 |
| **ArgMax/ArgMin** | ❌ 全部缺失 | `TROWARGMAX`, `TCOLARGMAX` 等 |

### 3.2 操作覆盖统计（修正版）

| 类别 | Tile ISA 操作数 | JCore 已支持 | JCore 缺失 |
|---|---|---|---|
| 数据搬运 | 6 | 6 | 0 |
| 转置 | 2 种形式 | 1 种形式 | 1 种形式 |
| 归约操作 | 28 | ~10 | ~18 |
| 归约扩展 | 16 | ~6 | ~10 |
| Tile-Tile 逐元素 | 29 | ~15 | ~14 |
| Tile-标量 | 21 | ~12 | ~9 |
| 特殊操作 | 16 | ~2 | ~14 |
| 矩阵运算 | 8 | ~2 | ~6 |
| **总计** | **~126** | **~54** | **~72** |

### 3.3 关键差异

| 维度 | Tile ISA | JCore |
|---|---|---|
| **按列归约** | 完整支持 | 完全缺失 |
| **多阶段归约** | 支持（带 tmp） | 不支持 |
| **融合操作** | 较少 | 有 `TROWEXPANDSUM` 等融合操作 |
| **Tile-标量位运算** | 完整支持 | 缺失（ANDS/ORS/XORS/SHLS/SHRS） |
| **三元操作** | 支持（TADDC/TSUBC） | 缺失 |
| **硬件加速** | TTRANS/归约等可能是硬件指令 | 纯 `__vec__` 软件实现 |

### 3.4 选择建议

| 场景 | 推荐 | 原因 |
|---|---|---|
| 需要按列归约 | **Tile ISA** | JCore 完全缺失 |
| 需要多阶段归约（大 tile） | **Tile ISA** | JCore 不支持带 tmp 形式 |
| 需要融合操作（如 sum+expand） | **JCore** | 有专门的融合操作 |
| 需要 Tile-标量位运算 | **Tile ISA** | JCore 缺失 |
| 追求极致性能 | **JCore** | 可手动优化 `__vec__` kernel |
| 追求开发效率 | **Tile ISA** | 高级抽象，无需编写 kernel |

---

## 4. 实际编写算子中用到的 Tile ISA 独有操作

以下操作在实际编写的 reduction / cumsum / transpose 算子中被使用，且满足 **Tile ISA 有但 JCore 没有（或形式不完整）**。未在算子中使用的 Tile ISA 独有操作不在此列。

### 4.1 按列归约操作（JCore 完全缺失）

| 操作 | 使用场景 | Tile ISA 接口 | JCore 状态 |
|---|---|---|---|
| `TCOLSUM(dst, src)` | reducesum_colvec | 简单形式 | ❌ 完全缺失 |
| `TCOLSUM(dst, src, tmp, isBinary)` | reducesum_colvec（二叉归约优化版） | 带 tmp 形式 | ❌ 完全缺失 |
| `TCOLMAX(dst, src)` | reducemax_colvec, reducemax_colvec_unalign_120_8 | 唯一形式 | ❌ 完全缺失 |
| `TCOLPROD(dst, src)` | reduceprod_colvec | 唯一形式 | ❌ 完全缺失 |

### 4.2 按行归约操作（JCore 缺失 tmp 形式）

| 操作 | 使用场景 | Tile ISA 接口 | JCore 状态 |
|---|---|---|---|
| `TROWSUM(dst, src, tmp)` | reducesum_rowvec, reducesum_rowvec_single_tree | 带 tmp 形式 | ⚠️ 仅有简单形式 `TROWSUM_Impl(dst, src)`，无 tmp |
| `TROWMAX(dst, src, tmp)` | reducemax_rowvec, reducemax_rowvec_single_tree | 带 tmp 形式 | ⚠️ 仅有简单形式 `TROWMAX_Impl(dst, src)`，无 tmp |
| `TROWPROD(dst, src, tmp)` | reduceprod_rowvec | 带 tmp 形式（唯一形式） | ❌ 完全缺失 |

### 4.3 Tile-标量操作

| 操作 | 使用场景 | Tile ISA 接口 | JCore 状态 |
|---|---|---|---|
| `TEXPANDS(dst, scalar)` | 所有算子的初始化（`TEXPANDS(oldSumTile, 0)` 等） | 广播标量到 tile | ❌ 完全缺失 |

### 4.4 使用统计

| 算子类别 | 用到的 Tile ISA 独有操作 |
|---|---|
| **reducesum_colvec** 系列 | `TCOLSUM`（含 tmp/isBinary）、`TEXPANDS` |
| **reducesum_rowvec** 系列 | `TROWSUM(dst, src, tmp)`、`TEXPANDS` |
| **reducemax_colvec** 系列 | `TCOLMAX`、`TEXPANDS` |
| **reducemax_rowvec** 系列 | `TROWMAX(dst, src, tmp)`、`TEXPANDS` |
| **reduceprod_colvec** | `TCOLPROD`、`TEXPANDS` |
| **reduceprod_rowvec** | `TROWPROD(dst, src, tmp)`、`TEXPANDS` |
| **cumsum_colvec/rowvec** | `TEXPANDS`（其余 TADD/TLOAD/TSTORE 两者都有） |

### 4.5 小结

在实际编写的算子中，**最关键的 Tile ISA 独有优势**体现在：

1. **所有按列归约操作**（`TCOLSUM`/`TCOLMAX`/`TCOLPROD`）—— JCore 完全没有对应实现，必须用 `__vec__` kernel 手写
2. **带 tmp 的按行归约**（`TROWSUM`/`TROWMAX`/`TROWPROD` 的 tmp 形式）—— JCore 只有顺序归约的简单形式，无法使用二叉树归约优化
3. **`TEXPANDS`** —— JCore 缺失，需要 `__vec__` kernel 手动实现标量广播
