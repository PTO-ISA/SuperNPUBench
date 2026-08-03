# DynamicMxQuant 问题记录

## 问题1：TileSize 大小约束分析

### 硬件约束来源

#### 1. IsValidActiveSize 检查

文件：`tileop-api/jcore/type.hpp`

```cpp
template <typename T>
struct tile_type_traits {
private:
  using PlainT = std::remove_cv_t<std::remove_reference_t<T>>;
  static constexpr std::size_t peLocalBytes = sizeof(PlainT);
  static constexpr std::size_t logicalTileBytes = peLocalBytes * 4;

  static constexpr int mapBytesToEnum(std::size_t b) {
    return
      b == 512   ? __tilesize_512B :   // TSize=1
      b == 1024  ? __tilesize_1KB  :   // TSize=2
      b == 2048  ? __tilesize_2KB  :   // TSize=3
      b == 4096  ? __tilesize_4KB  :   // TSize=4
      b == 8192  ? __tilesize_8KB  :   // TSize=5
      b == 16384 ? __tilesize_16KB :   // TSize=6
      b == 32768 ? __tilesize_32KB :   // TSize=7
      __tilesize_unknown;
  }

public:
  static constexpr int TilesizeCode = mapBytesToEnum(logicalTileBytes);
  static constexpr bool IsValidActiveSize =
      TilesizeCode >= __tilesize_512B && TilesizeCode <= __tilesize_32KB;
};
```

**关键公式**：`logicalTileBytes = sizeof(TileDType) × 4`

#### 2. TileDType 的计算

文件：`tileop-api/common/pto_tile.hpp:427`

```cpp
using TileDType = DType tile_size(Rows * Cols / (sizeof(DType) * 8 / type_traits<DType>::bits));
```

对于常见类型（`type_traits<E>::bits = sizeof(E) * 8`），简化为：

```
N = Rows × Cols
sizeof(TileDType) = Rows × Cols × sizeof(DType)
logicalTileBytes = Rows × Cols × sizeof(DType) × 4
```

#### 3. 哪些操作会触发检查

文件：`tileop-api/jcore/template_asm.hpp`

| 操作 | 检查 IsValidActiveSize | 行号 |
|------|:---:|:---:|
| TLOAD | ✓ | 1697 |
| TSTORE | ✓ | 1721 |
| TCVT | ✗ | 2940 |
| TABS | ✗ | 2711 |
| TROWMAX | ✗ | 3896 |
| TMULS | ✗ | 3000 |
| TCAST | ✗ | — |
| TANDS | ✗ | — |
| TRECIP | ✗ | — |
| TROWEXPANDMUL | ✗ | — |

**只有 TLOAD 和 TSTORE 有 static_assert 检查**，其余 tile 操作不检查 IsValidActiveSize。

### DynamicMxQuant 中的约束分析

#### tail.hpp 中涉及 TLOAD/TSTORE 的 tile

| Tile 定义 | DType | sizeof(TileDType) | logicalTileBytes | 用于 |
|-----------|-------|-------------------|------------------|------|
| `tile_x = Tile<bf16, TileM, BlockSize>` | bf16 | TileM × BS × 2 | TileM × BS × 8 | TLOAD |
| `tile_scale = Tile<u16, TileM, BlockSize>` | u16 | TileM × BS × 2 | TileM × BS × 8 | TSTORE |
| `tile_o = Tile<fp8, TileM, BlockSize>` | fp8 | TileM × BS × 1 | TileM × BS × 4 | TSTORE |

#### tail.hpp 中不涉及 TLOAD/TSTORE 的 tile（无约束）

| Tile 定义 | DType | sizeof(TileDType) | logicalTileBytes | 用于 |
|-----------|-------|-------------------|------------------|------|
| `tile_f = Tile<float, TileM, BlockSize>` | float | TileM × BS × 4 | TileM × BS × 16 | TCVT, TABS, TROWEXPANDMUL |
| `tile_amax_f = Tile<float, TileM, kScaleStride, TileM, 1>` | float | — | — | TROWMAX, TMAXS, TRECIP, TMULS |

#### common.hpp 中的 tile（无约束）

`tile_f32 = Tile<float, TileM, BlockSize>` 仅在 CUBLAS 分支中使用，只参与 TCVT/TABS/TROWMAX/TMULS，不触发检查。

#### 约束公式（TileM=8）

```
TLOAD tile_x:   8 × BS × 8 = 64 × BS ≤ 32768  →  BS ≤ 512
TSTORE scale:   8 × BS × 8 = 64 × BS ≤ 32768  →  BS ≤ 512
TSTORE oq:      8 × BS × 4 = 32 × BS ≤ 32768  →  BS ≤ 1024
```

**瓶颈在 tile_x（TLOAD）和 tile_scale（TSTORE）**，约束为 **BS ≤ 512**。

### 编译验证

| 配置 | BS | 结果 | 说明 |
|------|:--:|------|------|
| OCP | 32 | ✓ | |
| OCP | 64 | ✓ | |
| OCP | 256 | ✓ | |
| OCP | 512 | ✓ | 上限 |
| OCP | 1024 | ✗ | TLOAD dst 和 TSTORE src 均报错 |
| CUBLAS | 512 | ✓ | tile_f32 不受检查 |

### 结论

- **BlockSize 上限 = 512**（TileM=8 时）
- 约束来自 TLOAD/TSTORE 的 `IsValidActiveSize` static_assert
- 约束公式：`TileM × BlockSize × sizeof(DType) × 4 ≤ 32768`
- 中间计算 tile（float 类型）虽然 logicalTileBytes 更大，但不触发检查

---

## 问题2：32 字节对齐约束分析

### 硬件约束来源

#### 1. Tile 构造时的对齐检查

文件：`tileop-api/common/pto_tile.hpp:414`

```cpp
static_assert(
    ((pto::BLayout)0 == BLayout::RowMajor && (pto::SLayout)0 == SLayout::NoneBox && 
     Cols * type_traits<DType>::bits % (32 * 8) == 0) ||
    ((pto::BLayout)0 == BLayout::ColMajor && (pto::SLayout)0 == SLayout::NoneBox && 
     Rows * type_traits<DType>::bits % (32 * 8) == 0) ||
    ((pto::SLayout)0 != SLayout::NoneBox) && 
     (Rows % InnerRows == 0 && Cols % InnerCols == 0),
    "BFractal_ is RowMajor and SFractal_ is NoneBox: Rows must be 32 bytes align, ..."
);
```

**关键公式**（RowMajor + NoneBox）：
```
Cols × sizeof(DType) % 32 == 0
```

即：行优先存储下，Tile 的列数 × 元素字节数必须是 32 的倍数（32 字节 = 256 位）。

#### 2. 不同数据类型的约束

| DType | sizeof | Cols 最小倍数 | 示例 |
|-------|--------|--------------|------|
| bf16 | 2 | 16 | Cols ∈ {16, 32, 48, 64, ...} |
| fp8_e4m3 | 1 | 32 | Cols ∈ {32, 64, 96, 128, ...} |
| float | 4 | 8 | Cols ∈ {8, 16, 24, 32, ...} |
| uint16 | 2 | 16 | Cols ∈ {16, 32, 48, 64, ...} |

#### 3. 触发场景

**与问题1的关键区别**：此约束在 **Tile 构造时** 检查，适用于 **所有 tile**（包括中间计算 tile），而不仅限于 TLOAD/TSTORE。

触发错误示例（nontail.hpp 中 TileN=4）：
```
Tile<bf16, 32, 4, RowMajor>  →  4 × 2 = 8 bytes  ✗ (不满足 32 字节对齐)
Tile<fp8, 32, 4, RowMajor>   →  4 × 1 = 4 bytes  ✗
Tile<uint16, 1, 4, RowMajor> →  4 × 2 = 8 bytes  ✗
```

### DynamicMxQuant 中的约束分析

#### tail.hpp（尾轴量化）

Tile shape: `[TileM, BlockSize]`，RowMajor 布局

| Tile | DType | 约束 | 最小 BlockSize |
|------|-------|------|---------------|
| `tile_x` | bf16 | BlockSize × 2 % 32 == 0 | 16 |
| `tile_o` | fp8 | BlockSize × 1 % 32 == 0 | 32 |
| `tile_scale` | uint16 | BlockSize × 2 % 32 == 0 | 16 |

**约束**：BlockSize 必须是 32 的倍数（由 fp8 输出决定）。

#### nontail.hpp（非尾轴量化）

Tile shape: `[BlockSize, TileN]`，RowMajor 布局

| Tile | DType | 约束 | 最小 TileN |
|------|-------|------|-----------|
| `tile_x` | bf16 | TileN × 2 % 32 == 0 | 16 |
| `tile_o` | fp8 | TileN × 1 % 32 == 0 | 32 |
| `tile_scale` | uint16 | TileN × 2 % 32 == 0 | 16 |
| `tile_f` | float | TileN × 4 % 32 == 0 | 8 |

**约束**：TileN 必须是 32 的倍数（由 fp8 输出决定）。

### 与 TileSize 约束的交互

两个约束共同作用，严重限制 tile 切分方式：

#### tail.hpp 示例（TileM=8）

| BlockSize | 32B 对齐 | TileSize (≤32KB) | 可行 |
|-----------|---------|------------------|------|
| 32 | ✓ | ✓ (2KB) | ✓ |
| 64 | ✓ | ✓ (4KB) | ✓ |
| 128 | ✓ | ✓ (8KB) | ✓ |
| 256 | ✓ | ✓ (16KB) | ✓ |
| 512 | ✓ | ✓ (32KB) | ✓ |
| 1024 | ✓ | ✗ (64KB) | ✗ |

#### nontail.hpp 示例（BlockSize=32）

| TileN | 32B 对齐 | TileSize (≤32KB) | 可行 |
|-------|---------|------------------|------|
| 4 | ✗ | ✓ (1KB) | ✗ |
| 8 | ✗ | ✓ (2KB) | ✗ |
| 16 | ✗ | ✓ (4KB) | ✗ |
| 32 | ✓ | ✓ (8KB) | ✓ |
| 64 | ✓ | ✓ (16KB) | ✓ |
| 128 | ✓ | ✓ (32KB) | ✓ |
| 256 | ✓ | ✗ (64KB) | ✗ |

### 结论

- **32 字节对齐约束**适用于 **所有 tile**（包括中间计算 tile），比 TileSize 约束更严格
- 约束公式（RowMajor）：`Cols × sizeof(DType) % 32 == 0`
- 对于 DynamicMxQuant：
  - 尾轴：BlockSize 必须是 32 的倍数
  - 非尾轴：TileN 必须是 32 的倍数
- **两个约束共同作用**，有效 tile 切分组合非常有限：
  - tail：BlockSize ∈ {32, 64, 128, 256, 512}（TileM=8）
  - nontail：TileN ∈ {32, 64, 128}（BlockSize=32）
