# TMINS - Take the minimum of a tile and a scalar

## Purpose / When to use

Elementwise minimum of a tile and a scalar. Use it inside vector-tile kernels when the operation should run element by element over the destination tile valid region.

## C++ declaration

```cpp
template <typename TileDataDst, typename TileDataSrc>
PTO_INST void TMINS(TileDataDst &dst, TileDataSrc &src, typename TileDataSrc::DType scalar);
```

## Parameters table

| Parameter | Description |
| --- | --- |
| `dst` | Output tile. The operation writes only the destination valid region. |
| `src` | Input tile. It must be shape-compatible with `dst`. |
| `scalar` | Scalar operand. Its type is the tile element type in the public declaration unless the template accepts a generic `T`. |

## Operation / Semantics

For every element in the destination valid region, `dst(i, j) = min(src(i, j), scalar)`.

## Constraints

- Supported element types from the catalog/current page: `U8`, `S8`, `U16`, `S16`, `U32`, `S32`, `F16`, `F32`, `BF16`.
- Tiles should be vector tiles, row-major, and valid-shape compatible unless a narrower page-specific note says otherwise.
- The operation iterates over `dst.GetValidRow()` and `dst.GetValidCol()`; values outside the destination valid region are not part of the operation.

## Example

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;
void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src, dst;
  TMINS(dst, src, 0.0f);
}
```

## Common mistakes

- Do not pass a scalar with a different C++ type than the tile element type unless the declaration explicitly accepts a generic scalar type.

## Used by kernels

Used by the vector microbenchmark kernels for this intrinsic, including `microbenchmark/vector/src/tmins_fp16_16x16.cpp`.
