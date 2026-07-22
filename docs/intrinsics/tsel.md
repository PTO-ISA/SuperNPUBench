# TSEL - Select between two tiles with a predicate mask

## Purpose / When to use

Select per element between two same-typed source tiles using a packed predicate mask. Use it to implement branch-free conditional assignment in vector kernels.

## C++ declaration

```cpp
template <typename TileData, typename MaskTile, typename TmpTile>
PTO_INST void TSEL(TileData &dst, MaskTile &selMask, TileData &src0, TileData &src1, TmpTile &tmp);
```

## Parameters table

| Parameter | Description |
| --- | --- |
| `dst` | Output tile. The operation writes only the destination valid region. |
| `selMask` | Packed predicate mask tile. A true predicate selects the first data source. |
| `src0` | First input tile. It must be shape-compatible with `dst` unless the specific implementation states otherwise. |
| `src1` | Second input tile. For binary arithmetic and bitwise operations it supplies the right-hand operand. |
| `tmp` | Temporary tile used by implementations that need scratch storage. Keep it type- and shape-compatible with the documented constraints. |

## Operation / Semantics

For every element in the destination valid region, copy `src0(i, j)` when the corresponding mask predicate is true; otherwise copy `src1(i, j)`. The mask is a packed predicate tile.

## Constraints

- Supported element types from the catalog/current page: `Template-dependent; see public signature`.
- The mask tile is interpreted as packed predicate bits; a true predicate selects the tile source documented in the semantics section.
- Provide `tmp` when using the overload that declares it. For remainder operations, the current page requires enough temporary columns for `dst` and at least one temporary row on implementations that validate the buffer.
- Tiles should be vector tiles, row-major, and valid-shape compatible unless a narrower page-specific note says otherwise.
- The operation iterates over `dst.GetValidRow()` and `dst.GetValidCol()`; values outside the destination valid region are not part of the operation.

## Example

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;
void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  using MaskT = Tile<TileType::Vec, uint8_t, 16, 32, BLayout::RowMajor, -1, -1>;
  using TmpT = Tile<TileType::Vec, uint32_t, 1, 16>;
  TileT src0, src1, dst;
  MaskT mask(16, 2);
  TmpT tmp;
  TSEL(dst, mask, src0, src1, tmp);
}
```

## Common mistakes

- Do not treat the mask tile as one boolean value per full data element; it is packed.
- Keep the mask valid shape consistent with the data tile region being selected.
- Do not omit or alias temporary storage when using an overload that requires `tmp`.

## Used by kernels

No dedicated `microbenchmark/vector/src/tsel_*_16x16.cpp` file is present in this checkout; use it from vector-tile kernels that need the operation described above.
