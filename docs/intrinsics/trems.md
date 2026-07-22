# TREMS - Compute the remainder of a tile and a scalar

## Purpose / When to use

Elementwise remainder with a scalar: `remainder(src, scalar)`. Use it inside vector-tile kernels when the operation should run element by element over the destination tile valid region.

## C++ declaration

```cpp
template <typename TileDataDst, typename TileDataSrc, typename TileDataTmp>
PTO_INST void TREMS(TileDataDst &dst, TileDataSrc &src, typename TileDataSrc::DType scalar,
                           TileDataTmp &tmp);
```

## Parameters table

| Parameter | Description |
| --- | --- |
| `dst` | Output tile. The operation writes only the destination valid region. |
| `src` | Input tile. It must be shape-compatible with `dst`. |
| `scalar` | Scalar operand. Its type is the tile element type in the public declaration unless the template accepts a generic `T`. |
| `tmp` | Temporary tile used by implementations that need scratch storage. Keep it type- and shape-compatible with the documented constraints. |

## Operation / Semantics

For every element in the destination valid region, `dst(i, j) = src(i, j) % scalar`. Division by zero is backend-defined; the CPU simulator asserts in debug builds.

## Constraints

- Supported element types from the catalog/current page: `F32`, `S32`, `U32`, `F16`, `S16`, `U16`.
- Provide `tmp` when using the overload that declares it. For remainder operations, the current page requires enough temporary columns for `dst` and at least one temporary row on implementations that validate the buffer.
- Division by zero is not normalized by this reference; follow backend behavior and avoid zero divisors when portable results matter.
- For `int32_t` remainder on implementations that convert through float32, keep operands in `[-16777216, 16777216]` when exact conversion is required.
- Tiles should be vector tiles, row-major, and valid-shape compatible unless a narrower page-specific note says otherwise.
- The operation iterates over `dst.GetValidRow()` and `dst.GetValidCol()`; values outside the destination valid region are not part of the operation.

## Example

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;
void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT x, out;
  Tile<TileType::Vec, float, 16, 16> tmp;
  TREMS(out, x, 3.0f, tmp);
}
```

## Common mistakes

- Do not pass a scalar with a different C++ type than the tile element type unless the declaration explicitly accepts a generic scalar type.
- Do not rely on portable results for zero divisors.
- Do not omit or alias temporary storage when using an overload that requires `tmp`.

## Used by kernels

Used by the vector microbenchmark kernels for this intrinsic, including `microbenchmark/vector/src/trems_fp16_16x16.cpp`.
