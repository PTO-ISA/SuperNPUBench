# TAXPY - Accumulate a scaled tile into the destination

## Purpose / When to use

Perform an AXPY-style fused update that scales a source tile by a scalar and accumulates into `dst`. Use it when the destination already contains the accumulation value.

## C++ declaration

```cpp
template <typename TileDataDst, typename TileDataSrc>
PTO_INST void TAXPY(TileDataDst &dst, TileDataSrc &src0, typename TileDataSrc::DType scalar);
```

## Parameters table

| Parameter | Description |
| --- | --- |
| `dst` | Output tile. The operation writes only the destination valid region. |
| `src0` | Input tile to scale before accumulation. |
| `scalar` | Scale factor applied to `src0` before accumulating into `dst`. |

## Operation / Semantics

For every element in the destination valid region, update the destination with an AXPY-style operation: multiply the source tile by `scalar` and accumulate it into `dst`. The current page describes this as `dst += src0 * scalar`.

## Constraints

- Supported element types from the catalog/current page: `Template-dependent; see public signature`.
- The current page does not publish a complete public C++ declaration or type legality list; use the same tile/scalar type pattern as existing `taxpy_fp16_16x16` microbenchmark code.
- Tiles should be vector tiles, row-major, and valid-shape compatible unless a narrower page-specific note says otherwise.
- The operation iterates over `dst.GetValidRow()` and `dst.GetValidCol()`; values outside the destination valid region are not part of the operation.

## Example

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;
void example() {
  using TileT = Tile<TileType::Vec, half, 16, 16>;
  TileT x, acc;
  TAXPY(acc, x, half(0.5f));
}
```

## Common mistakes

- Do not pass a scalar with a different C++ type than the tile element type unless the declaration explicitly accepts a generic scalar type.
- Do not initialize `dst` as if this were a pure multiply; `dst` is the accumulation destination.

## Used by kernels

Used by the vector microbenchmark kernels for this intrinsic, including `microbenchmark/vector/src/taxpy_fp16_16x16.cpp`.
