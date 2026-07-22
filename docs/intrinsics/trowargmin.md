# TROWARGMIN Row Argmin Reduction

## Purpose / When to use

Use `TROWARGMIN` to return the column index of the minimum element in each row. It is useful for indices of row minima while preserving tile-local execution.

## C++ declaration

```cpp
template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INST void TROWARGMIN(TileDataOut& dst, TileDataIn& src, TileDataTmp& tmp);
```

## Parameters

| Parameter | Description |
| --- | --- |
| `dst` | Output tile. Only the valid region described by the operation is written. |
| `src` | Input tile. The operation reads `src.GetValidRow()` and `src.GetValidCol()`. |
| `tmp` | Temporary vector tile used by implementations that need scratch storage. Keep it type- and shape-compatible with `src` unless the overload documents otherwise. |

## Operation

Let `R = src.GetValidRow()` and `C = src.GetValidCol()`. For every valid row `i`, `TROWARGMIN` writes `dst[i,0] = argmin_{0 <= j < C} src[i,j]`. Only elements in the valid input region participate in the reduction. The destination stores integer index type (`uint32_t` or `int32_t`).

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

void reduce_rows() {
  using Src = Tile<TileType::Vec, float, 16, 64>;
  using Dst = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor>;
  Src src;
  Dst dst;
  Src tmp;

  TROWARGMIN(dst, src, tmp);
}
```

## Common mistakes

- Passing a destination whose valid row count does not match `src.GetValidRow()`.
- Treating inactive rows or columns outside the valid region as reduction inputs.
- Reusing an undersized temporary tile when the overload requires `tmp`.
- Assuming arg reduction ties are portable across implementations.

## Used by kernels

- No generated benchmark catalog entry currently lists `TROWARGMIN`.
