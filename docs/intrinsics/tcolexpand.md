# TCOLEXPAND Column Broadcast

## Purpose / When to use

Use `TCOLEXPAND` to broadcast one scalar from each source column across the corresponding destination column. It is useful for expanding reduced column values back to matrix shape before an elementwise operation.

## C++ declaration

```cpp
template <typename TileDataDst, typename TileDataSrc>
PTO_INST void TCOLEXPAND(TileDataDst &dst, TileDataSrc &src);
```

## Parameters

| Parameter | Description |
| --- | --- |
| `dst` | Output tile. Only the valid region described by the operation is written. |
| `src` | Input tile. The operation reads `src.GetValidRow()` and `src.GetValidCol()`. |

## Operation

Let `R = dst.GetValidRow()` and `C = dst.GetValidCol()`. For each valid element `(i, j)`, `TCOLEXPAND` writes `dst[i,j] = src[0,j]`. The operation covers `dst.GetValidRow()` by `dst.GetValidCol()` elements. Values outside the valid destination region are not part of the contract.

## Constraints

- `dst` and data operands are vector tiles using compatible layouts for the selected backend.
- `src` must provide one readable scalar for each valid destination column.
- `dst` and the elementwise input tile must cover the same valid shape.
- Broadcast-only forms support the standard vector tile element types used by the backend.
- Use the overload with `tmp` only when the implementation path requires scratch storage; it does not change the mathematical result.

## Example

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void broadcast_columns() {
  using Dst = Tile<TileType::Vec, half, 16, 64>;
  using Src = Tile<TileType::Vec, half, 1, 64>;
  Src src;
  Dst dst;

  TCOLEXPAND(dst, src);
}
```

## Common mistakes

- Passing a scalar tile laid out along the wrong axis.
- Expecting rows or columns outside the destination valid region to be written.
- Mixing element types across `dst`, `src0`, and `src1` for arithmetic forms.
- Using `TROWEXPAND*` where the scalar varies by column, or `TCOLEXPAND*` where the scalar varies by row.

## Used by kernels

- [`hif4-hif4-f0fd84bb`](../benchmarks/catalog/one-level/matmul/hif4-hif4-f0fd84bb.md) - `benchmark/one-level-arch/kernels/matmul/matmul_mx.hpp`
- [`a16w4-f1f175ff`](../benchmarks/catalog/one-level/matmul/a16w4-f1f175ff.md) - `benchmark/one-level-arch/kernels/matmul/matmul_mx.hpp`
