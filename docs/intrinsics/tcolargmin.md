# TCOLARGMIN Column Argmin Reduction

## Purpose / When to use

Use `TCOLARGMIN` to return the row index of the minimum element in each column. It is useful for indices of column minima while keeping the result in a tile.

## C++ declaration

```cpp
template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INST void TCOLARGMIN(TileDataOut& dst, TileDataIn& src, TileDataTmp& tmp);
```

## Parameters

| Parameter | Description |
| --- | --- |
| `dst` | Output tile. Only the valid region described by the operation is written. |
| `src` | Input tile. The operation reads `src.GetValidRow()` and `src.GetValidCol()`. |
| `tmp` | Temporary vector tile used by implementations that need scratch storage. Keep it type- and shape-compatible with `src` unless the overload documents otherwise. |

## Operation

Let `R = src.GetValidRow()` and `C = src.GetValidCol()`. For every valid column `j`, `TCOLARGMIN` writes `dst[0,j] = argmin_{0 <= i < R} src[i,j]`. Only elements in the valid input region participate in the reduction. The destination stores integer index type (`uint32_t` or `int32_t`).

For equal maximum or minimum values, tie-breaking follows the intrinsic implementation. Do not rely on a specific tied index unless the backend contract documents one.

## Constraints

- `src` and `dst` are vector tiles.
- `src.GetValidRow()` and `src.GetValidCol()` must describe a non-empty reduction domain for defined reduction results.
- Destination valid shape must match the reduced axis: one output value per input row for row reductions, or one output value per input column for column reductions.
- Value reductions require `dst` and `src` to use the same element type. Arg reductions write an integer index tile while reading a supported numeric source tile.
- Floating-point reductions support `half` and `float`; column value reductions also support supported integer tile types where the implementation provides them.
- Provide `tmp` with enough scratch capacity for the implementation path, especially for arg reductions and wide row reductions.

## Example

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void reduce_columns() {
  using Src = Tile<TileType::Vec, float, 16, 64>;
  using Dst = Tile<TileType::Vec, uint32_t, 1, 64>;
  Src src;
  Src tmp;
  Dst dst;

  TCOLARGMIN(dst, src, tmp);
}
```

## Common mistakes

- Passing a destination whose valid column count does not match `src.GetValidCol()`.
- Treating inactive rows or columns outside the valid region as reduction inputs.
- Reusing an undersized temporary tile when the overload requires `tmp`.
- Assuming arg reduction ties are portable across implementations.

## Used by kernels

- No generated benchmark catalog entry currently lists `TCOLARGMIN`.
