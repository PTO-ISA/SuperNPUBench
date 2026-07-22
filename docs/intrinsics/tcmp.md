# TCMP - Compare two tiles into a predicate mask

## Purpose / When to use

Compare two vector tiles and materialize the result as a packed predicate mask. Use it before `TSEL`/`TSELS` or other mask-consuming code that needs per-element condition results.

## C++ declaration

```cpp
template <typename TileDataDst, typename TileDataSrc>
PTO_INST void TCMP(TileDataDst &dst, TileDataSrc &src0, TileDataSrc &src1, CmpMode cmpMode);
```

## Parameters table

| Parameter | Description |
| --- | --- |
| `dst` | Output tile. The operation writes only the destination valid region. |
| `src0` | First input tile. It must be shape-compatible with `dst` unless the specific implementation states otherwise. |
| `src1` | Second input tile. For binary arithmetic and bitwise operations it supplies the right-hand operand. |
| `cmpMode` | Comparison selector. Supported modes are `CmpMode::EQ`, `CmpMode::NE`, `CmpMode::LT`, `CmpMode::GT`, `CmpMode::LE`, and `CmpMode::GE` where recorded by the current docs. |

## Operation / Semantics

For every element in the destination valid region, evaluate `src0(i, j) cmpMode src1(i, j)` and write the predicate bits into the mask tile. The mask packing is backend-defined.

## Constraints

- Supported element types from the catalog/current page: `Input: U32`, `S32`, `U16`, `S16`, `U8`, `S8`, `F32`, `F16; output mask: U32`.
- The destination is a predicate/mask tile; its element type and packing follow the backend mask representation described by the current page.
- Tiles should be vector tiles, row-major, and valid-shape compatible unless a narrower page-specific note says otherwise.
- The operation iterates over `dst.GetValidRow()` and `dst.GetValidCol()`; values outside the destination valid region are not part of the operation.

## Example

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;
void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using MaskT = Tile<TileType::Vec, uint8_t, 16, 32, BLayout::RowMajor, -1, -1>;
  SrcT src0, src1;
  MaskT mask(16, 2);
  TCMP(mask, src0, src1, CmpMode::GT);
}
```

## Common mistakes

- Do not treat the mask tile as one boolean value per full data element; it is packed.

## Used by kernels

Used by the vector microbenchmark kernels for this intrinsic, including `microbenchmark/vector/src/tcmp_fp16_16x16.cpp`, `microbenchmark/vector/src/tcmp_fp32_16x16.cpp`, `microbenchmark/vector/src/tcmp_i32_16x16.cpp`.
