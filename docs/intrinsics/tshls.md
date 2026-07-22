# TSHLS - Shift a tile left by a scalar count

## Purpose / When to use

Elementwise shift-left a tile by a scalar. Use it inside vector-tile kernels when the operation should run element by element over the destination tile valid region.

## C++ declaration

```cpp
template <typename TileDataDst, typename TileDataSrc>
PTO_INST void TSHLS(TileDataDst &dst, TileDataSrc &src, typename TileDataDst::DType scalar);
```

## Parameters table

| Parameter | Description |
| --- | --- |
| `dst` | Output tile. The operation writes only the destination valid region. |
| `src` | Input tile. It must be shape-compatible with `dst`. |
| `scalar` | Scalar operand. Its type is the tile element type in the public declaration unless the template accepts a generic `T`. |

## Operation / Semantics

For every element in the destination valid region, `dst(i, j) = src(i, j) << scalar`. The scalar shift count must be zero or positive.

## Constraints

- Supported element types from the catalog/current page: `S32`, `S16`, `S8`, `U32`, `U16`, `U8`.
- Use integral element types for bitwise and shift operations.
- The scalar shift count must be zero or positive.
- Tiles should be vector tiles, row-major, and valid-shape compatible unless a narrower page-specific note says otherwise.
- The operation iterates over `dst.GetValidRow()` and `dst.GetValidCol()`; values outside the destination valid region are not part of the operation.

## Example

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;
void example() {
  using TileDst = Tile<TileType::Vec, uint16_t, 16, 16>;
  using TileSrc = Tile<TileType::Vec, uint16_t, 16, 16>;
  TileDst dst;
  TileSrc src;
  TSHLS(dst, src, 0x2);
}
```

## Common mistakes

- Do not pass a scalar with a different C++ type than the tile element type unless the declaration explicitly accepts a generic scalar type.
- Do not use floating-point tiles for bitwise or shift operations.

## Used by kernels

Used by the vector microbenchmark kernels for this intrinsic, including `microbenchmark/vector/src/tshls_fp16_16x16.cpp`.
