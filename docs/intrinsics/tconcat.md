# TCONCAT: Concatenate tiles by columns

## Purpose / When to use

Use `TCONCAT` to build a wider tile by placing two source tiles next to each other along the column dimension.

## C++ Declaration

```cpp
template <typename DstTileData, typename Src0TileData, typename Src1TileData>
PTO_INST void TCONCAT(DstTileData &dst, Src0TileData &src0, Src1TileData &src1);
```

## Parameters

| Parameter | Meaning |
| --- | --- |
| `dst` | Destination tile with column count equal to the sum of source columns. |
| `src0` | Left source tile. |
| `src1` | Right source tile. |

## Operation / Semantics

- Copies `src0` into the left columns of `dst`.
- Copies `src1` into the following columns of `dst`.
- Rows are preserved; the destination valid column count is the sum of valid source columns.

## Constraints

- Source tiles must have compatible row counts, dtype, layout, and valid rows.
- Destination columns must be large enough for both source valid regions.
- The operation concatenates columns only; use other reshape operations for row stacking.
- Alignment requirements follow the tile type and dtype.

## Example

```cpp
TCONCAT(dst, lhs, rhs);
```

## Common mistakes

- Allocating a destination with only one source width.
- Expecting row-wise concatenation.
- Concatenating tiles with different valid row counts.

## Used by kernels

- `microbenchmark/vector/vector_bench.hpp` contains a `bench_concat` helper.
- Generated vector microbenchmark source files cover fp16 and fp32 concatenation cases.
