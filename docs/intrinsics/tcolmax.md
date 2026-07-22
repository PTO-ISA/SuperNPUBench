# TCOLMAX Column Maximum Reduction

## Purpose / When to use

Use `TCOLMAX` to take the maximum value in each column. It is useful for column maxima for normalization or bounds work.

## C++ declaration

```cpp
template <typename TileDataOut, typename TileDataIn>
PTO_INST void TCOLMAX(TileDataOut &dst, TileDataIn &src);
```

## Parameters

| Parameter | Description |
| --- | --- |
| `dst` | Output tile. Only the valid region described by the operation is written. |
| `src` | Input tile. The operation reads `src.GetValidRow()` and `src.GetValidCol()`. |

## Operation

Let `R = src.GetValidRow()` and `C = src.GetValidCol()`. For every valid column `j`, `TCOLMAX` writes `dst[0,j] = max_{0 <= i < R} src[i,j]`. Only elements in the valid input region participate in the reduction. The destination stores same element type as `src`.


## Constraints

- `src` and `dst` are vector tiles.
- `src.GetValidRow()` and `src.GetValidCol()` must describe a non-empty reduction domain for defined reduction results.
- Destination valid shape must match the reduced axis: one output value per input row for row reductions, or one output value per input column for column reductions.
- Value reductions require `dst` and `src` to use the same element type. Arg reductions write an integer index tile while reading a supported numeric source tile.
- Floating-point reductions support `half` and `float`; column value reductions also support supported integer tile types where the implementation provides them.

## Example

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void reduce_columns() {
  using Src = Tile<TileType::Vec, float, 16, 64>;
  using Dst = Tile<TileType::Vec, float, 1, 64>;
  Src src;
  Dst dst;

  TCOLMAX(dst, src);
}
```

## Common mistakes

- Passing a destination whose valid column count does not match `src.GetValidCol()`.
- Treating inactive rows or columns outside the valid region as reduction inputs.
- Reusing an undersized temporary tile when the overload requires `tmp`.

## Used by kernels

- [`reducemax-col-1c7de9cb`](../benchmarks/catalog/one-level/reduction-reducemax-col/reducemax-col-1c7de9cb.md) - `benchmark/one-level-arch/kernels/reduction/reducemax_colvec_pto.hpp`
- [`fa-hif4-86b1f297`](../benchmarks/catalog/one-level/fa/fa-hif4-86b1f297.md) - `benchmark/one-level-arch/kernels/fa/fa_hif4.hpp`
