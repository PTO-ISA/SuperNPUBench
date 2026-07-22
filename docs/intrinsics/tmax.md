# TMAX - Take the maximum of two tiles

## Purpose / When to use

Elementwise maximum of two tiles. Use it inside vector-tile kernels when the operation should run element by element over the destination tile valid region.

## C++ declaration

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INST void TMAX(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1);
```

## Parameters table

| Parameter | Description |
| --- | --- |
| `dst` | Output tile. The operation writes only the destination valid region. |
| `src0` | First input tile. It must be shape-compatible with `dst` unless the specific implementation states otherwise. |
| `src1` | Second input tile. For binary arithmetic and bitwise operations it supplies the right-hand operand. |

## Operation / Semantics

For every element in the destination valid region, `dst(i, j) = max(src0(i, j), src1(i, j))`.

## Constraints

- Supported element types from the catalog/current page: `U32`, `S32`, `U16`, `S16`, `U8`, `S8`, `F32`, `F16`.
- Tiles should be vector tiles, row-major, and valid-shape compatible unless a narrower page-specific note says otherwise.
- The operation iterates over `dst.GetValidRow()` and `dst.GetValidCol()`; values outside the destination valid region are not part of the operation.

## Example

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;
void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src0, src1, dst;
  TMAX(dst, src0, src1);
}
```

## Common mistakes

- Do not pass sources with a smaller valid shape than `dst`; the operation uses the destination valid region.

## Used by kernels

Used by the vector microbenchmark kernels for this intrinsic, including `microbenchmark/vector/src/tmax_fp16_16x16.cpp`, `microbenchmark/vector/src/tmax_fp32_16x16.cpp`, `microbenchmark/vector/src/tmax_i16_16x16.cpp`, `microbenchmark/vector/src/tmax_i32_16x16.cpp`.
