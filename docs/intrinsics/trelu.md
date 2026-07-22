# TRELU - Apply ReLU to a tile

## Purpose / When to use

Elementwise ReLU of a tile. Use it inside vector-tile kernels when the operation should run element by element over the destination tile valid region.

## C++ declaration

```cpp
template <typename TileDataDst, typename TileDataSrc>
PTO_INST void TRELU(TileDataDst &dst, TileDataSrc &src);
```

## Parameters table

| Parameter | Description |
| --- | --- |
| `dst` | Output tile. The operation writes only the destination valid region. |
| `src` | Input tile. It must be shape-compatible with `dst`. |

## Operation / Semantics

For every element in the destination valid region, `dst(i, j) = max(src(i, j), 0)`.

## Constraints

- Supported element types from the catalog/current page: `F16`, `F32`, `S32`.
- Tiles should be vector tiles, row-major, and valid-shape compatible unless a narrower page-specific note says otherwise.
- The operation iterates over `dst.GetValidRow()` and `dst.GetValidCol()`; values outside the destination valid region are not part of the operation.

## Example

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;
void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT x, out;
  TRELU(out, x);
}
```

## Common mistakes

- Do not assume the operation changes elements outside the destination valid region.

## Used by kernels

Used by the vector microbenchmark kernels for this intrinsic, including `microbenchmark/vector/src/trelu_fp16_16x16.cpp`, `microbenchmark/vector/src/trelu_fp32_16x16.cpp`.
