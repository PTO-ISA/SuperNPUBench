# TDIVS - Divide with a tile and a scalar

## Purpose / When to use

Elementwise division with a scalar (tile/scalar or scalar/tile). Use it inside vector-tile kernels when the operation should run element by element over the destination tile valid region.

## C++ declaration

```cpp
template <typename TileDataDst, typename TileDataSrc>
PTO_INST void TDIVS(TileDataDst &dst, TileDataSrc &src0, typename TileDataSrc::DType scalar);
template <typename TileDataDst, typename TileDataSrc>
PTO_INST void TDIVS(TileDataDst &dst, typename TileDataDst::DType scalar, TileDataSrc &src0);
```

## Parameters table

| Parameter | Description |
| --- | --- |
| `dst` | Output tile. The operation writes only the destination valid region. |
| `src0` | First input tile. It must be shape-compatible with `dst` unless the specific implementation states otherwise. |
| `scalar` | Scalar operand. Its type is the tile element type in the public declaration unless the template accepts a generic `T`. |

## Operation / Semantics

The tile/scalar overload computes `dst(i, j) = src0(i, j) / scalar`. The scalar/tile overload computes `dst(i, j) = scalar / src0(i, j)`. Division by zero is backend-defined; the current page records that one backend maps tile/scalar zero divisors through reciprocal multiplication and uses positive infinity for `scalar == 0`.

## Constraints

- Supported element types from the catalog/current page: `U8`, `S8`, `U16`, `S16`, `U32`, `S32`, `F16`, `F32`.
- Division by zero is not normalized by this reference; follow backend behavior and avoid zero divisors when portable results matter.
- Tiles should be vector tiles, row-major, and valid-shape compatible unless a narrower page-specific note says otherwise.
- The operation iterates over `dst.GetValidRow()` and `dst.GetValidCol()`; values outside the destination valid region are not part of the operation.

## Example

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;
void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src, dst;
  TDIVS(dst, src, 2.0f);
}
```

## Common mistakes

- Do not pass a scalar with a different C++ type than the tile element type unless the declaration explicitly accepts a generic scalar type.
- Do not rely on portable results for zero divisors.

## Used by kernels

Used by the vector microbenchmark kernels for this intrinsic, including `microbenchmark/vector/src/tdivs_fp16_16x16.cpp`.
