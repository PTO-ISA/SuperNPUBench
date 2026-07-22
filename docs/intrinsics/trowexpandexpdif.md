# TROWEXPANDEXPDIF Row Broadcast Exponential Difference

## Purpose / When to use

Use `TROWEXPANDEXPDIF` to combine each element of `src0` with one scalar from the matching row of `src1`. It is useful for applying per-row normalization, scaling, bounds, or exponent transforms without materializing a full scalar matrix.

## C++ declaration

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INST void TROWEXPANDEXPDIF(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1);
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp>
PTO_INST void TROWEXPANDEXPDIF(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp);
```

## Parameters

| Parameter | Description |
| --- | --- |
| `dst` | Output tile. Only the valid region described by the operation is written. |
| `src0` | Elementwise input tile for the valid destination region. |
| `src1` | Row scalar tile. It supplies one logical scalar for each valid destination row. Some overloads also accept a widened per-row representation for implementation storage. |
| `tmp` | Temporary vector tile used by implementations that need scratch storage. Keep it type- and shape-compatible with `src` unless the overload documents otherwise. |

## Operation

Let `R = dst.GetValidRow()` and `C = dst.GetValidCol()`. Let `s_i` be the scalar supplied by `src1` for the current row. For each valid element `(i, j)`, `TROWEXPANDEXPDIF` writes `dst[i,j] = exp(src0[i,j] - s_i)`. The operation covers `dst.GetValidRow()` by `dst.GetValidCol()` elements. Values outside the valid destination region are not part of the contract.

## Constraints

- `dst` and data operands are vector tiles using compatible layouts for the selected backend.
- `src1` must provide one readable scalar for each valid destination row.
- `dst` and the elementwise input tile must cover the same valid shape.
- Broadcast arithmetic forms use matching `half` or `float` operands unless a backend documents additional supported types.
- Use the overload with `tmp` only when the implementation path requires scratch storage; it does not change the mathematical result.

## Example

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void apply_row_scalars() {
  using TileT = Tile<TileType::Vec, half, 16, 64>;
  using Scalars = Tile<TileType::Vec, half, 16, 1, BLayout::ColMajor>;
  TileT src0, dst;
  Scalars scalars;

  TROWEXPANDEXPDIF(dst, src0, scalars);
}
```

## Common mistakes

- Passing a scalar tile laid out along the wrong axis.
- Expecting rows or columns outside the destination valid region to be written.
- Mixing element types across `dst`, `src0`, and `src1` for arithmetic forms.
- Using `TROWEXPAND*` where the scalar varies by column, or `TCOLEXPAND*` where the scalar varies by row.

## Used by kernels

- No generated benchmark catalog entry currently lists `TROWEXPANDEXPDIF`.
