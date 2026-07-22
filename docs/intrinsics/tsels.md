# TSELS - Select between a tile and a scalar with a predicate mask

## Purpose / When to use

Select per element between a source tile and a scalar using a packed predicate mask. Use it for branch-free clamping, thresholding, and fill-on-condition patterns.

## C++ declaration

```cpp
template <typename TileDataDst, typename TileDataMask, typename TileDataSrc, typename TileDataTmp>
PTO_INST void TSELS(TileDataDst &dst, TileDataMask &mask, TileDataSrc &src, TileDataTmp &tmp, typename TileDataSrc::DType scalar);
```

## Parameters table

| Parameter | Description |
| --- | --- |
| `dst` | Output tile. The operation writes only the destination valid region. |
| `mask` | Packed predicate mask tile. A true predicate selects the tile value. |
| `src` | Input tile. It must be shape-compatible with `dst`. |
| `tmp` | Temporary tile used by implementations that need scratch storage. Keep it type- and shape-compatible with the documented constraints. |
| `scalar` | Scalar operand. Its type is the tile element type in the public declaration unless the template accepts a generic `T`. |

## Operation / Semantics

For every element in the destination valid region, copy `src(i, j)` when the corresponding mask predicate is true; otherwise write `scalar`. The mask is a packed predicate tile.

## Constraints

- Supported element types from the catalog/current page: `Template-dependent; see public signature`.
- The mask tile is interpreted as packed predicate bits; a true predicate selects the tile source documented in the semantics section.
- Provide `tmp` when using the overload that declares it. For remainder operations, the current page requires enough temporary columns for `dst` and at least one temporary row on implementations that validate the buffer.
- Tiles should be vector tiles, row-major, and valid-shape compatible unless a narrower page-specific note says otherwise.
- The operation iterates over `dst.GetValidRow()` and `dst.GetValidCol()`; values outside the destination valid region are not part of the operation.

## Example

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;
void example_auto() {
  using TileDst = Tile<TileType::Vec, float, 16, 16>;
  using TileSrc = Tile<TileType::Vec, float, 16, 16>;
  using TileTmp = Tile<TileType::Vec, float, 16, 16>;
  using TileMask = Tile<TileType::Vec, uint8_t, 16, 32, BLayout::RowMajor, -1, -1>;
  TileDst dst;
  TileSrc src;
  TileTmp tmp;
  TileMask mask(16, 2);
  float scalar = 0.0f;
  TSELS(dst, mask, src, tmp, scalar);
}
```

## Common mistakes

- Do not pass a scalar with a different C++ type than the tile element type unless the declaration explicitly accepts a generic scalar type.
- Do not treat the mask tile as one boolean value per full data element; it is packed.
- Keep the mask valid shape consistent with the data tile region being selected.
- Do not omit or alias temporary storage when using an overload that requires `tmp`.

## Used by kernels

Used by the vector microbenchmark kernels for this intrinsic, including `microbenchmark/vector/src/tsels_fp16_16x16.cpp`, `microbenchmark/vector/src/tsels_fp32_16x16.cpp`.
