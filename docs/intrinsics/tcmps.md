# TCMPS - Compare a tile with a scalar into a predicate mask

## Purpose / When to use

Compare each element of a vector tile with one scalar and write a mask-like result. Use it to build predicates such as threshold masks.

## C++ declaration

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename T>
PTO_INST void TCMPS(TileDataDst& dst, TileDataSrc0& src0, T src1, CmpMode cmpMode);
```

## Parameters table

| Parameter | Description |
| --- | --- |
| `dst` | Output tile. The operation writes only the destination valid region. |
| `src0` | First input tile. It must be shape-compatible with `dst` unless the specific implementation states otherwise. |
| `src1` | Scalar operand in the `TCMPS` declaration. |
| `cmpMode` | Comparison selector. Supported modes are `CmpMode::EQ`, `CmpMode::NE`, `CmpMode::LT`, `CmpMode::GT`, `CmpMode::LE`, and `CmpMode::GE` where recorded by the current docs. |

## Operation / Semantics

For every element in the destination valid region, evaluate `src0(i, j) cmpMode src1` and write comparison results to `dst`. The output is mask-like and backend-defined.

## Constraints

- Supported element types from the catalog/current page: `Input: S32`, `F32`, `F16`, `U16`, `S16; output mask: U32`.
- The destination is a predicate/mask tile; its element type and packing follow the backend mask representation described by the current page.
- Tiles should be vector tiles, row-major, and valid-shape compatible unless a narrower page-specific note says otherwise.
- The operation iterates over `dst.GetValidRow()` and `dst.GetValidCol()`; values outside the destination valid region are not part of the operation.

## Example

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;
void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, uint8_t, 16, 32, BLayout::RowMajor, -1, -1>;
  SrcT src;
  DstT dst(16, 2);
  TCMPS(dst, src, 0.0f, CmpMode::GT);
}
```

## Common mistakes

- Do not pass a scalar with a different C++ type than the tile element type unless the declaration explicitly accepts a generic scalar type.
- Do not treat the mask tile as one boolean value per full data element; it is packed.

## Used by kernels

Used by the vector microbenchmark kernels for this intrinsic, including `microbenchmark/vector/src/tcmps_fp16_16x16.cpp`.
