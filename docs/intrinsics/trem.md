# TREM - Compute the remainder of two tiles

## Purpose / When to use

Elementwise remainder of two tiles. Use it inside vector-tile kernels when the operation should run element by element over the destination tile valid region.

## C++ declaration

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp>
PTO_INST void TREM(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp);
```

## Parameters table

| Parameter | Description |
| --- | --- |
| `dst` | Output tile. The operation writes only the destination valid region. |
| `src0` | First input tile. It must be shape-compatible with `dst` unless the specific implementation states otherwise. |
| `src1` | Second input tile. For binary arithmetic and bitwise operations it supplies the right-hand operand. |
| `tmp` | Temporary tile used by implementations that need scratch storage. Keep it type- and shape-compatible with the documented constraints. |

## Operation / Semantics

For every element in the destination valid region, `dst(i, j) = src0(i, j) % src1(i, j)`. The current page records that the result has the same sign as the divider. Division by zero is backend-defined; the CPU simulator asserts in debug builds.

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
  using TileT = Tile<TileType::Vec, int32_t, 16, 16>;
  TileT out, a, b;
  Tile<TileType::Vec, int32_t, 16, 16> tmp;
  TREM(out, a, b, tmp);
}
```

## Common mistakes

- Do not pass sources with a smaller valid shape than `dst`; the operation uses the destination valid region.
- Do not rely on portable results for zero divisors.
- Do not omit or alias temporary storage when using an overload that requires `tmp`.

## Used by kernels

Used by the vector microbenchmark kernels for this intrinsic, including `microbenchmark/vector/src/trem_fp16_16x16.cpp`, `microbenchmark/vector/src/trem_fp32_16x16.cpp`, `microbenchmark/vector/src/trem_i32_16x16.cpp`.
