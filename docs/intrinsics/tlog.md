# TLOG - Compute natural log of a tile

## Purpose / When to use

Elementwise natural logarithm of a tile. Use it inside vector-tile kernels when the operation should run element by element over the destination tile valid region.

## C++ declaration

```cpp
template <typename TileDataDst, typename TileDataSrc>
PTO_INST void TLOG(TileDataDst &dst, TileDataSrc &src);
```

## Parameters table

| Parameter | Description |
| --- | --- |
| `dst` | Output tile. The operation writes only the destination valid region. |
| `src` | Input tile. It must be shape-compatible with `dst`. |

## Operation / Semantics

For every element in the destination valid region, `dst(i, j) = log(src(i, j))`. Behavior for non-positive inputs is backend-defined.

## Constraints

- Supported element types from the catalog/current page: `F32`, `F16`.
- Use floating-point tile element types (`float` or `half`) as recorded by the current page.
- Results for zero or negative inputs are backend-defined.
- Tiles should be vector tiles, row-major, and valid-shape compatible unless a narrower page-specific note says otherwise.
- The operation iterates over `dst.GetValidRow()` and `dst.GetValidCol()`; values outside the destination valid region are not part of the operation.

## Example

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;
void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT x, out;
  TLOG(out, x);
}
```

## Common mistakes

- Do not assume IEEE exception or NaN behavior is identical across backends for out-of-domain inputs.

## Used by kernels

Used by the vector microbenchmark kernels for this intrinsic, including `microbenchmark/vector/src/tlog_fp16_16x16.cpp`, `microbenchmark/vector/src/tlog_fp32_16x16.cpp`.
