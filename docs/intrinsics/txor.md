# TXOR - Bitwise-XOR two tiles

## Purpose / When to use

Elementwise bitwise XOR of two tiles. Use it inside vector-tile kernels when the operation should run element by element over the destination tile valid region.

## C++ declaration

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp>
PTO_INST void TXOR(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp);
```

## Parameters table

| Parameter | Description |
| --- | --- |
| `dst` | Output tile. The operation writes only the destination valid region. |
| `src0` | First input tile. It must be shape-compatible with `dst` unless the specific implementation states otherwise. |
| `src1` | Second input tile. For binary arithmetic and bitwise operations it supplies the right-hand operand. |
| `tmp` | Temporary tile used by implementations that need scratch storage. Keep it type- and shape-compatible with the documented constraints. |

## Operation / Semantics

For every element in the destination valid region, `dst(i, j) = src0(i, j) ^ src1(i, j)`. The `tmp` tile is temporary storage required by some implementations.

## Constraints

- Supported element types from the catalog/current page: `U32`, `S32`, `U16`, `S16`, `U8`, `S8`.
- Use integral element types for bitwise and shift operations.
- Provide `tmp` when using the overload that declares it. For remainder operations, the current page requires enough temporary columns for `dst` and at least one temporary row on implementations that validate the buffer.
- Tiles should be vector tiles, row-major, and valid-shape compatible unless a narrower page-specific note says otherwise.
- The operation iterates over `dst.GetValidRow()` and `dst.GetValidCol()`; values outside the destination valid region are not part of the operation.

## Example

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;
void example() {
  using TileDst = Tile<TileType::Vec, uint32_t, 16, 16>;
  using TileSrc0 = Tile<TileType::Vec, uint32_t, 16, 16>;
  using TileSrc1 = Tile<TileType::Vec, uint32_t, 16, 16>;
  using TileTmp = Tile<TileType::Vec, uint32_t, 16, 16>;
  TileDst dst;
  TileSrc0 src0;
  TileSrc1 src1;
  TileTmp tmp;
  TXOR(dst, src0, src1, tmp);
}
```

## Common mistakes

- Do not pass sources with a smaller valid shape than `dst`; the operation uses the destination valid region.
- Do not use floating-point tiles for bitwise or shift operations.
- Do not omit or alias temporary storage when using an overload that requires `tmp`.

## Used by kernels

Used by the vector microbenchmark kernels for this intrinsic, including `microbenchmark/vector/src/txor_i16_16x16.cpp`, `microbenchmark/vector/src/txor_i32_16x16.cpp`.
