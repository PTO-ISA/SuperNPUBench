# PTO ISA 与当前编译器 C++ API 表达完备性对比

> **目的**：以 `broadcast_pto.hpp`、`broadcast_vec_07_pto.hpp`、`gelu_pto.hpp` 三个 PTO 一层编程算子用到的 17 条 PTO ISA 指令为基准，逐条检查当前编译器（Linx-TileOP-API `tileop_api.hpp`）的 C++ 接口能否完整表达 PTO ISA 文档定义的语义。**只关注 API 签名层面的表达力差异（参数、名称、行为语义），不涉及后端实现方式。**

> **排序规则**：完全缺失 → 需要修改 → 名称和核心参数一致。

---

## 汇总列表

| # | 类别 | PTO ISA 指令 | 使用算子 | 当前编译器 | 差异要点 |
|---|---|---|---|---|---|
| 1 | **完全缺失** | `TREMS` | broadcast_pto | 无 | 无标量取余 API；`TREM` 仅 tile-tile 且不支持 uint32 |
| 2 | **需要修改** | `TCVT` | gelu_pto | `TCVT` (签名不完整) | 缺 `RoundMode`/`SaturationMode`/`tmp` 参数 |
| 3 | | `MGATHER` | broadcast_pto | `MGATHER` (template_asm.h) | 缺 `Coalesce`/`GatherOOB` 模板参数；索引语义不同（元素索引 vs 字节偏移） |
| 4 | | `TLOAD` | vec_07, gelu_pto | `TCOPYIN` (名不同) | 名称不同；核心参数和行为一致 |
| 5 | | `TSTORE` | 全部三个 | `TCOPYOUT` (名不同) | 名称不同；缺 `AtomicType`/`preQuantScalar` |
| 6 | | `TEXPANDS` | broadcast_pto | `TEXPANDSCALAR` (名不同) | 名称不同；核心参数和行为一致 |
| 7 | | `TROWEXPAND` | vec_07 | `TEXPANDROW` (名不同) | 名称不同；模板约束阻止 src=(N,1) dst=(N,C) |
| 8 | | `TDIVS` | broadcast_pto | `TDIVS` (缺参数) | 缺 `PrecisionType`；缺反向重载 `scalar÷tile` |
| 9 | | `TEXP` | gelu_pto | `TEXP` (缺参数) | 缺 `PrecisionType`（`DEFAULT`/`HIGH_PRECISION`） |
| 10 | | `TRECIP` | gelu_pto | `TRECIP` (缺参数) | 缺 `PrecisionType`（`DEFAULT`/`HIGH_PRECISION`） |
| 11 | **一致** | `TCI` | broadcast_pto | `TCI` | 仅系统性差异（§0） |
| 12 | | `TMULS` | broadcast_pto, gelu_pto | `TMULS` | 仅系统性差异（§0） |
| 13 | | `TADD` | broadcast_pto, gelu_pto | `TADD` | 仅系统性差异（§0） |
| 14 | | `TMAXS` | gelu_pto | `TMAXS` | 仅系统性差异（§0） |
| 15 | | `TMINS` | gelu_pto | `TMINS` | 仅系统性差异（§0） |
| 16 | | `TMUL` | gelu_pto | `TMUL` | 仅系统性差异（§0） |
| 17 | | `TADDS` | gelu_pto | `TADDS` | 仅系统性差异（§0） |

> **统计**：完全缺失 1 条 · 需要修改 9 条 · 一致 7 条。另有 3 类系统性差异（`RecordEvent` 返回值、`WaitEvents` 参数、模板类型约束）影响全部 17 条，见 §0。

---

## 0. 系统性差异（影响全部 17 条指令）

以下 3 类差异不是单条指令的问题，而是当前编译器 API 设计层面的系统性缺口，**所有指令都受影响**，在各条指令中不再重复展开：

### 0.1 返回值

| | PTO ISA | 当前编译器 |
|---|---|---|
| 返回类型 | `PTO_INST RecordEvent` | `void` |

PTO ISA 每条 Tile 指令返回 `RecordEvent` 异步事件令牌，调用方可用于构建流水线依赖图。当前编译器全部返回 `void`，无法表达异步依赖。

### 0.2 `WaitEvents` 可变参数

| | PTO ISA | 当前编译器 |
|---|---|---|
| 签名末尾 | `typename... WaitEvents` + `WaitEvents &... events` | 无 |

PTO ISA 允许在调用时传入前置 `RecordEvent`，实现"等待前一条指令完成后再发射本指令"。当前编译器无此机制。

### 0.3 模板类型约束

| | PTO ISA | 当前编译器 |
|---|---|---|
| 模板参数 | `typename TileDataDst`, `typename TileDataSrc` …（各自独立） | `is_tile_data_v tile_shape`（C++ concept，强制全部操作数同一类型） |

PTO ISA 允许 dst 和 src 是不同的 Tile 类型（不同 ValidRow/ValidCol/甚至不同 dtype），在 impl 层做约束检查。当前编译器在 API 签名层强制 dst/src0/src1 必须是**完全相同的 Tile 类型**，导致以下场景无法表达：
- dst 和 src 的 ValidCol 不同（如 `TROWEXPAND` src=(N,1) dst=(N,C)）
- dst 和 src 的 dtype 不同（如 `TCVT` float→half）
- index tile 和 data tile 的 dtype 不同（如 `MGATHER` dst=half, indexes=uint32）

> **下文各条指令的差异对比不再重复以上 3 点，只列出该指令特有的差异。**

---

## 1. 完全缺失（当前编译器无对应 API）

### 1.1 TREMS — 标量取余

**使用算子**：`broadcast_pto.hpp`（`coord = idx % out_dim`）

**PTO ISA 签名**：
```cpp
template <auto PrecisionType = RemSAlgorithm::DEFAULT,
          typename TileDataDst, typename TileDataSrc,
          typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TREMS(
    TileDataDst &dst,
    TileDataSrc &src,
    typename TileDataSrc::DType scalar,
    TileDataTmp &tmp,
    WaitEvents &... events);
```

**当前编译器**：无 `TREMS` API。最接近的是 `TREM`（tile-tile 取余），但：

| 维度 | PTO ISA `TREMS` | 当前编译器 `TREM` |
|---|---|---|
| 操作形式 | tile ÷ **标量** | tile ÷ **tile** |
| `tmp` 参数 | 有（`TileDataTmp &tmp`） | 无 |
| `PrecisionType` | 有（`DEFAULT` / `HIGH_PRECISION`） | 无 |
| 支持类型 | A5: float, int32, **uint32**, half, int16, uint16 | 仅 **int32, int16**（不支持 uint32） |
| 标量形式 | ✅ 直接传标量 | ❌ 需先 `TEXPANDS` 把标量广播成 tile 再调用 |

**无法表达的场景**：broadcast_pto 中 offset tile 是 `uint32_t`，需要 `TREMS(coord, idx, out_dim, tmp)` 但当前编译器既没有标量取余 API，`TREM` 也不支持 `uint32_t`。

**需补充**：
1. 新增 `TREMS` API，签名按 PTO ISA 文档
2. 包含 `tmp`、`PrecisionType` 参数
3. 支持 `uint32_t`

---

## 2. 需要修改（API 存在但名称或参数不完整）

### 2.1 TCVT — 类型转换

**使用算子**：`gelu_pto.hpp`（fp16↔fp32）

| 维度 | PTO ISA | 当前编译器 |
|---|---|---|
| **签名** | `TCVT(dst, src, tmp, RoundMode, SaturationMode, events...)` （4 个重载） | `TCVT(dst, src)` |
| **`tmp` 参数** | 有（可选）：某些转换路径需要临时 tile | 无 |
| **`RoundMode`** | `CAST_RINT` / `CAST_ROUND` / `CAST_FLOOR` / `CAST_CEIL` / `CAST_TRUNC` | 无（无法控制舍入方式） |
| **`SaturationMode`** | `ON` / `OFF`（可选） | 无（无法控制饱和行为） |

**PTO ISA 4 个重载**：
```cpp
TCVT(dst, src, tmp, RoundMode, SaturationMode, events...);  // 全量
TCVT(dst, src, tmp, RoundMode, events...);                  // 无 SaturationMode
TCVT(dst, src, RoundMode, SaturationMode, events...);       // 无 tmp
TCVT(dst, src, RoundMode, events...);                       // 最简
```

**无法表达的场景**：GELU 中 fp32→fp16 转换需要指定舍入模式（`RoundMode::Nearest`）和饱和模式，当前 API 无法控制。

**需修改**：增加 `RoundMode`（必选）、`SaturationMode`（可选）、`tmp`（可选重载）参数。

---

### 2.2 MGATHER — 按索引从 global memory 取数

**使用算子**：`broadcast_pto.hpp`

| 维度 | PTO ISA | 当前编译器 |
|---|---|---|
| **模板参数** | `<Coalesce Mode, GatherOOB Oob, ...>` | 无模式参数 |
| **`Coalesce` 模式** | `Row`（按行 gather）/ `Elem`（逐元素 gather） | 无选择，隐含一种模式 |
| **`GatherOOB`** | `Undefined` / `Clamp` / `Wrap` / `Zero` | 无越界处理策略 |
| **索引语义** | `Coalesce::Elem`：`dst[i,j] = src[idx[i,j]]`，按**元素索引** | 按**字节偏移**取数（调用方需手动乘 `sizeof(dtype)`） |
| **索引类型** | `int32_t` 或 `uint32_t` | `uint32_t`（由 tile 类型决定） |

**无法表达的场景**：
1. broadcast_pto 使用 `MGATHER<Coalesce::Elem>` 按元素索引取数，当前编译器无法表达 `Coalesce` 模式选择。
2. PTO ISA 的元素索引语义意味着 offset tile 存的是元素下标，当前编译器存的是字节偏移——语义不同，算子代码中是否乘 `sizeof(dtype)` 的行为不一致。

**需修改**：
1. 增加 `Coalesce` 和 `GatherOOB` 模板参数
2. 增加 `Coalesce::Elem` 模式（逐元素 gather，元素索引语义）
3. 统一或明确索引语义（元素索引 vs 字节偏移）

---

### 2.3 TLOAD — 从 global memory 加载到 tile

**使用算子**：`broadcast_vec_07_pto.hpp`、`gelu_pto.hpp`

| 维度 | PTO ISA | 当前编译器 |
|---|---|---|
| **函数名** | `TLOAD` | `TCOPYIN` |
| **签名** | `TLOAD(TileData &dst, GlobalData &src, events...)` | `TCOPYIN(tile_shape &dst, gm_shape &src)` |
| **核心参数** | `(dst, src)` | `(dst, src)` — 一致 |
| **行为** | GM → UB | GM → UB — 一致 |

**差异**：仅函数名不同（`TLOAD` vs `TCOPYIN`），核心参数和行为完全一致。

**需修改**：重命名为 `TLOAD`（保留 `TCOPYIN` 为 deprecated 别名）。

---

### 2.4 TSTORE — tile 写回 global memory

**使用算子**：全部三个算子

| 维度 | PTO ISA | 当前编译器 |
|---|---|---|
| **函数名** | `TSTORE` | `TCOPYOUT` |
| **签名** | `TSTORE(GlobalData &dst, TileData &src, events...)` | `TCOPYOUT(gm_shape &dst, tile_shape &src)` |
| **`AtomicType`** | `AtomicNone`（默认）/ 原子写模式 | 无 |
| **`preQuantScalar` 重载** | 有：`TSTORE(dst, src, uint64_t preQuantScalar, events...)` — 写回前做量化 | 无 |
| **核心参数** | `(dst, src)` | `(dst, src)` — 一致 |
| **行为** | UB → GM | UB → GM — 一致 |

**无法表达的场景**：原子写和写回前量化（preQuantScalar）。

**需修改**：
1. 重命名为 `TSTORE`
2. 增加 `AtomicType` 模板参数
3. 增加 `preQuantScalar` 重载

---

### 2.5 TEXPANDS — 标量广播填充 tile

**使用算子**：`broadcast_pto.hpp`（初始化 offset tile 为 0）

| 维度 | PTO ISA | 当前编译器 |
|---|---|---|
| **函数名** | `TEXPANDS` | `TEXPANDSCALAR` |
| **签名** | `TEXPANDS(TileData &dst, DType scalar, events...)` | `TEXPANDSCALAR(tile_shape &dst, DType s)` |
| **核心参数** | `(dst, scalar)` | `(dst, s)` — 一致 |
| **行为** | `dst[i,j] = scalar` | 同 — 一致 |

**差异**：仅函数名不同。

**需修改**：重命名为 `TEXPANDS`（保留旧名别名）。

---

### 2.6 TROWEXPAND — 行广播（首元素广播到整行）

**使用算子**：`broadcast_vec_07_pto.hpp`（(N,1)→(N,C)）

| 维度 | PTO ISA | 当前编译器 |
|---|---|---|
| **函数名** | `TROWEXPAND` | `TEXPANDROW` |
| **签名** | `TROWEXPAND(TileDataDst &dst, TileDataSrc &src, events...)` | `TEXPANDROW(tile_shape_out &dst, tile_shape_in &src)` |
| **核心参数** | `(dst, src)` | `(dst, src)` — 一致 |
| **行为** | `dst[i,j] = src[i,0]` | 同 — 一致 |
| **shape 约束** | `src.ValidRow == dst.ValidRow`；`src.ValidCol` 可 < `dst.ValidCol`（如 src=(N,1) dst=(N,C)） | 模板约束要求 src 和 dst 为同一类型 → **无法表达 src=(N,1) dst=(N,C)** |

**无法表达的场景**：TROWEXPAND 的核心用途就是 src 的 ValidCol=1、dst 的 ValidCol=C，但当前模板约束强制 src/dst 同类型，导致此场景编译不过。（此为 §0.3 系统性约束问题的具体体现，此处特别标注因为它是该指令的核心用例。）

**需修改**：
1. 重命名为 `TROWEXPAND`
2. 拆分模板约束，允许 src/dst 不同 ValidCol

---

### 2.7 TDIVS — 标量整除

**使用算子**：`broadcast_pto.hpp`（`idx = idx / out_dim`）

| 维度 | PTO ISA | 当前编译器 |
|---|---|---|
| **函数名** | `TDIVS` | `TDIVS` — 一致 |
| **签名** | `TDIVS(dst, src0, scalar, events...)` | `TDIVS(tile_shape &dst, tile_shape &src, DType s)` |
| **`PrecisionType`** | `DivAlgorithm::DEFAULT` / `HIGH_PRECISION` | 无 |
| **反向重载** | 有：`TDIVS(dst, scalar, src0, events...)`（scalar ÷ tile） | 无 |
| **核心参数** | `(dst, src, scalar)` | `(dst, src, s)` — 一致 |
| **行为** | `dst[i,j] = src[i,j] / scalar` | 同 — 一致 |

**无法表达的场景**：
1. 精度模式选择（高精度 vs 默认）
2. 反向除法（标量 ÷ tile）

**需修改**：增加 `PrecisionType` 模板参数；增加反向重载。

---

### 2.8 TEXP — 指数运算

**使用算子**：`gelu_pto.hpp`（`exp(t * p)`）

| 维度 | PTO ISA | 当前编译器 |
|---|---|---|
| **函数名** | `TEXP` | `TEXP` — 一致 |
| **签名** | `TEXP(dst, src, events...)` | `TEXP(tile_shape &dst, tile_shape &src)` |
| **`PrecisionType`** | `ExpAlgorithm::DEFAULT`（快，低精度）/ `HIGH_PRECISION`（慢，高精度） | 无 |
| **核心参数** | `(dst, src)` | `(dst, src)` — 一致 |
| **行为** | `dst[i,j] = exp(src[i,j])` | 同 — 一致 |

**无法表达的场景**：精度模式选择。GELU 中可能需要 `HIGH_PRECISION` 保证数值精度。

**需修改**：增加 `PrecisionType` 模板参数。

---

### 2.9 TRECIP — 倒数

**使用算子**：`gelu_pto.hpp`（`1 / (1 + exp)`）

| 维度 | PTO ISA | 当前编译器 |
|---|---|---|
| **函数名** | `TRECIP` | `TRECIP` — 一致 |
| **签名** | `TRECIP(dst, src, events...)` | `TRECIP(tile_shape &dst, tile_shape &src)` |
| **`PrecisionType`** | `RecipAlgorithm::DEFAULT` / `HIGH_PRECISION` | 无 |
| **核心参数** | `(dst, src)` | `(dst, src)` — 一致 |
| **行为** | `dst[i,j] = 1 / src[i,j]` | 同 — 一致 |

**无法表达的场景**：精度模式选择。GELU 中 `1/(1+exp)` 的精度直接影响结果。

**需修改**：增加 `PrecisionType` 模板参数。

---

## 3. 名称和核心参数一致（仅受系统性差异影响）

以下 7 条指令的函数名、核心参数列表、行为语义与 PTO ISA 文档一致，**不需要单条修改**（只需统一补齐 §0 的系统性差异即可）：

| # | 指令 | 使用算子 | PTO ISA 签名 | 当前编译器签名 | 行为 | 差异 |
|---|---|---|---|---|---|---|
| 1 | `TCI` | broadcast_pto | `TCI(TileData &dst, T start, events...)` `<int descending>` | `TCI(tile_shape &dst, T s)` `<int descending>` | `dst[i] = start + i`（升序） | 仅系统性差异 |
| 2 | `TMULS` | broadcast_pto, gelu_pto | `TMULS(dst, src0, scalar, events...)` | `TMULS(tile_shape &dst, tile_shape &src, DType s)` | `dst[i] = src[i] * scalar` | 仅系统性差异 |
| 3 | `TADD` | broadcast_pto, gelu_pto | `TADD(dst, src0, src1, events...)` | `TADD(tile_shape &dst, tile_shape &src0, tile_shape &src1)` | `dst[i] = src0[i] + src1[i]` | 仅系统性差异 |
| 4 | `TMAXS` | gelu_pto | `TMAXS(dst, src, scalar, events...)` | `TMAXS(tile_shape &dst, tile_shape &src, DType s)` | `dst[i] = max(src[i], scalar)` | 仅系统性差异 |
| 5 | `TMINS` | gelu_pto | `TMINS(dst, src, scalar, events...)` | `TMINS(tile_shape &dst, tile_shape &src, DType s)` | `dst[i] = min(src[i], scalar)` | 仅系统性差异 |
| 6 | `TMUL` | gelu_pto | `TMUL(dst, src0, src1, events...)` | `TMUL(tile_shape &dst, tile_shape &src0, tile_shape &src1)` | `dst[i] = src0[i] * src1[i]` | 仅系统性差异 |
| 7 | `TADDS` | gelu_pto | `TADDS(dst, src0, scalar, events...)` | `TADDS(tile_shape &dst, tile_shape &src, DType s)` | `dst[i] = src[i] + scalar` | 仅系统性差异 |

---

## 4. 汇总表

| 类别 | 数量 | 指令 | 修改内容 |
|---|---|---|---|
| **完全缺失** | **1** | `TREMS` | 新增 API（含 `tmp`/`PrecisionType`/uint32 支持） |
| **需要修改** | **9** | `TCVT`, `MGATHER`, `TLOAD`, `TSTORE`, `TEXPANDS`, `TROWEXPAND`, `TDIVS`, `TEXP`, `TRECIP` | 见下 |
| └ 名称不同 | 4 | `TLOAD`→`TCOPYIN`, `TSTORE`→`TCOPYOUT`, `TEXPANDS`→`TEXPANDSCALAR`, `TROWEXPAND`→`TEXPANDROW` | 重命名 |
| └ 参数缺失 | 4 | `TCVT`(缺 RoundMode/SaturationMode/tmp), `MGATHER`(缺 Coalesce/GatherOOB, 索引语义不同), `TDIVS`(缺 PrecisionType/反向重载), `TEXP`+`TRECIP`(缺 PrecisionType) | 补参数 |
| └ 名称+参数 | 1 | `TSTORE`(名不同 + 缺 AtomicType/preQuantScalar) | 重命名+补参数 |
| **一致** | **7** | `TCI`, `TMULS`, `TADD`, `TMAXS`, `TMINS`, `TMUL`, `TADDS` | 无需单条修改 |
| **系统性补齐** | **17** | 全部 | 补 `RecordEvent` 返回 + `WaitEvents` 参数 + 拆分模板约束 |
| **总计** | **17** | | |

## 5. 修改优先级

### P0 — 阻塞算子表达（无法写出 PTO 代码）

| 指令 | 问题 | 阻塞的算子 |
|---|---|---|
| `TREMS` | 完全缺失 | broadcast_pto（无法表达标量取余） |
| `TCVT` | 缺 `RoundMode` | gelu_pto（无法控制 fp32→fp16 舍入） |
| `MGATHER` | 缺 `Coalesce::Elem` + 索引语义不同 | broadcast_pto（无法表达逐元素按索引 gather） |
| 模板约束（§0.3） | 强制 dst/src 同类型 | broadcast_vec_07（TROWEXPAND src=(N,1) dst=(N,C) 无法表达） |

### P1 — 语义不完整（可写但表达力受限）

| 指令 | 问题 |
|---|---|
| `TLOAD`/`TSTORE`/`TEXPANDS`/`TROWEXPAND` | 名称不同，PTO 代码无法直接编译 |
| `TDIVS`/`TEXP`/`TRECIP` | 缺 `PrecisionType`，无法选择精度模式 |
| `TSTORE` | 缺 `AtomicType`/`preQuantScalar` |

### P2 — 系统性增强（全部指令受益）

| 改动 | 影响 |
|---|---|
| 返回 `RecordEvent` | 全部 17 条 — 支持异步流水线 |
| 增加 `WaitEvents` | 全部 17 条 — 支持依赖等待 |
| 拆分模板约束 | 全部 17 条 — 允许 dst/src 不同 Tile 类型 |
