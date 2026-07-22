# TCOLEXPANDMAX Column Broadcast Maximum

## Purpose / When to use

Use `TCOLEXPANDMAX` to combine each element of `src0` with one scalar from the matching column of `src1`. It is useful for applying per-column normalization, scaling, bounds, or exponent transforms without materializing a full scalar matrix.

## C++ declaration

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INST void TCOLEXPANDMAX(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1);
```

## Parameters

| Parameter | Description |
| --- | --- |
| `dst` | Output tile. Only the valid region described by the operation is written. |
| `src0` | Elementwise input tile for the valid destination region. |
| `src1` | Column scalar tile. It supplies one logical scalar for each valid destination column. |

## Operation

Let `R = dst.GetValidRow()` and `C = dst.GetValidCol()`. Let `s_j` be the scalar supplied by `src1` for the current column. For each valid element `(i, j)`, `TCOLEXPANDMAX` writes `dst[i,j] = max(src0[i,j], s_j)`. The operation covers `dst.GetValidRow()` by `dst.GetValidCol()` elements. Values outside the valid destination region are not part of the contract.

## Constraints

- `dst` and data operands are vector tiles using compatible layouts for the selected backend.
- `src1` must provide one readable scalar for each valid destination column.
- `dst` and the elementwise input tile must cover the same valid shape.
- Broadcast arithmetic forms use matching `half` or `float` operands unless a backend documents additional supported types.
- Use the overload with `tmp` only when the implementation path requires scratch storage; it does not change the mathematical result.

## Example

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void apply_column_scalars() {
  using TileT = Tile<TileType::Vec, half, 16, 64>;
  using Scalars = Tile<TileType::Vec, half, 1, 64>;
  TileT src0, dst;
  Scalars scalars;

  TCOLEXPANDMAX(dst, src0, scalars);
}
```

## Common mistakes

- Passing a scalar tile laid out along the wrong axis.
- Expecting rows or columns outside the destination valid region to be written.
- Mixing element types across `dst`, `src0`, and `src1` for arithmetic forms.
- Using `TROWEXPAND*` where the scalar varies by column, or `TCOLEXPAND*` where the scalar varies by row.

## Used by kernels

- No generated benchmark catalog entry currently lists `TCOLEXPANDMAX`.
