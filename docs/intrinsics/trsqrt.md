# TRSQRT - Compute reciprocal square root of a tile

## Purpose / When to use

Elementwise reciprocal square root. Use it inside vector-tile kernels when the operation should run element by element over the destination tile valid region.

## C++ declaration

```cpp
template <typename TileDataDst, typename TileDataSrc>
PTO_INST void TRSQRT(TileDataDst &dst, TileDataSrc &src);
template <typename TileDataDst, typename TileDataSrc, typename TileDataTmp>
PTO_INST void TRSQRT(TileDataDst &dst, TileDataSrc &src, TileDataTmp &tmp);
```

## Parameters table

| Parameter | Description |
| --- | --- |
| `dst` | Output tile. The operation writes only the destination valid region. |
| `src` | Input tile. It must be shape-compatible with `dst`. |
| `tmp` | Temporary tile used by implementations that need scratch storage. Keep it type- and shape-compatible with the documented constraints. |

## Operation / Semantics

For every element in the destination valid region, `dst(i, j) = 1 / sqrt(src(i, j))`. Zero and negative-input behavior is backend-defined. The overload with `tmp` selects the high-precision form recorded by the current docs.

## Constraints

- Supported element types from the catalog/current page: `F32`, `F16`.
- Use floating-point tile element types (`float` or `half`) as recorded by the current page.
- Provide `tmp` when using the overload that declares it. For remainder operations, the current page requires enough temporary columns for `dst` and at least one temporary row on implementations that validate the buffer.
- Results for negative inputs are backend-defined.
- Tiles should be vector tiles, row-major, and valid-shape compatible unless a narrower page-specific note says otherwise.
- The operation iterates over `dst.GetValidRow()` and `dst.GetValidCol()`; values outside the destination valid region are not part of the operation.

## Example

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;
void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src, dst;
  TRSQRT(dst, src);
}
```

## Common mistakes

- Do not assume IEEE exception or NaN behavior is identical across backends for out-of-domain inputs.
- Do not omit or alias temporary storage when using an overload that requires `tmp`.

## Used by kernels

Used by the vector microbenchmark kernels for this intrinsic, including `microbenchmark/vector/src/trsqrt_fp16_16x16.cpp`.
