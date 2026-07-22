# TNEG - Negate a tile

## Purpose / When to use

Elementwise negation of a tile. Use it inside vector-tile kernels when the operation should run element by element over the destination tile valid region.

## C++ declaration

```cpp
template <typename TileDataDst, typename TileDataSrc>
PTO_INST void TNEG(TileDataDst &dst, TileDataSrc &src);
```

## Parameters table

| Parameter | Description |
| --- | --- |
| `dst` | Output tile. The operation writes only the destination valid region. |
| `src` | Input tile. It must be shape-compatible with `dst`. |

## Operation / Semantics

For every element in the destination valid region, `dst(i, j) = -src(i, j)`.

## Constraints

- Supported element types from the catalog/current page: `S32`, `S16`, `F32`, `F16`, `BF16`.
- Tiles should be vector tiles, row-major, and valid-shape compatible unless a narrower page-specific note says otherwise.
- The operation iterates over `dst.GetValidRow()` and `dst.GetValidCol()`; values outside the destination valid region are not part of the operation.

## Example

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;
void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT x, out;
  TNEG(out, x);
}
```

## Common mistakes

- Do not assume the operation changes elements outside the destination valid region.

## Used by kernels

Used by the vector microbenchmark kernels for this intrinsic, including `microbenchmark/vector/src/tneg_fp16_16x16.cpp`, `microbenchmark/vector/src/tneg_fp32_16x16.cpp`, `microbenchmark/vector/src/tneg_i16_16x16.cpp`, `microbenchmark/vector/src/tneg_i32_16x16.cpp`.
