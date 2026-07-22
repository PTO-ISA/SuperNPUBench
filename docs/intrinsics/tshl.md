# TSHL - Shift a tile left by per-element counts

## Purpose / When to use

Elementwise shift-left of two tiles. Use it inside vector-tile kernels when the operation should run element by element over the destination tile valid region.

## C++ declaration

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INST void TSHL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1);
```

## Parameters table

| Parameter | Description |
| --- | --- |
| `dst` | Output tile. The operation writes only the destination valid region. |
| `src0` | First input tile. It must be shape-compatible with `dst` unless the specific implementation states otherwise. |
| `src1` | Second input tile. For binary arithmetic and bitwise operations it supplies the right-hand operand. |

## Operation / Semantics

For every element in the destination valid region, `dst(i, j) = src0(i, j) << src1(i, j)`. The second source supplies the per-element shift count.

## Constraints

- Supported element types from the catalog/current page: `U32`, `S32`, `U16`, `S16`, `U8`, `S8`.
- Use integral element types for bitwise and shift operations.
- Tiles should be vector tiles, row-major, and valid-shape compatible unless a narrower page-specific note says otherwise.
- The operation iterates over `dst.GetValidRow()` and `dst.GetValidCol()`; values outside the destination valid region are not part of the operation.

## Example

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;
void example() {
  using TileT = Tile<TileType::Vec, uint32_t, 16, 16>;
  TileT x, sh, out;
  TSHL(out, x, sh);
}
```

## Common mistakes

- Do not pass sources with a smaller valid shape than `dst`; the operation uses the destination valid region.
- Do not use floating-point tiles for bitwise or shift operations.

## Used by kernels

Used by the vector microbenchmark kernels for this intrinsic, including `microbenchmark/vector/src/tshl_i16_16x16.cpp`, `microbenchmark/vector/src/tshl_i32_16x16.cpp`.
