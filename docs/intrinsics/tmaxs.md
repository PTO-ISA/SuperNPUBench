# TMAXS - Take the maximum of a tile and a scalar

## Purpose / When to use

Elementwise max of a tile and a scalar: `max(src, scalar)`. Use it inside vector-tile kernels when the operation should run element by element over the destination tile valid region.

## C++ declaration

```cpp
template <typename TileDataDst, typename TileDataSrc>
PTO_INST void TMAXS(TileDataDst& dst, TileDataSrc& src, typename TileDataSrc::DType scalar);
```

## Parameters table

| Parameter | Description |
| --- | --- |
| `dst` | Output tile. The operation writes only the destination valid region. |
| `src` | Input tile. It must be shape-compatible with `dst`. |
| `scalar` | Scalar operand. Its type is the tile element type in the public declaration unless the template accepts a generic `T`. |

## Operation / Semantics

For every element in the destination valid region, `dst(i, j) = max(src(i, j), scalar)`.

## Constraints

- Supported element types from the catalog/current page: `S32`, `U32`, `F32`, `S16`, `U16`, `F16`, `BF16`, `U8`, `S8`.
- Tiles should be vector tiles, row-major, and valid-shape compatible unless a narrower page-specific note says otherwise.
- The operation iterates over `dst.GetValidRow()` and `dst.GetValidCol()`; values outside the destination valid region are not part of the operation.

## Example

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;
void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT x, out;
  TMAXS(out, x, 0.0f);
}
```

## Common mistakes

- Do not pass a scalar with a different C++ type than the tile element type unless the declaration explicitly accepts a generic scalar type.

## Used by kernels

Used by the vector microbenchmark kernels for this intrinsic, including `microbenchmark/vector/src/tmaxs_fp16_16x16.cpp`.
