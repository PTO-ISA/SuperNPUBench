# TPARTMUL: Multiply partial tiles

## Purpose / When to use

Use `TPARTMUL` to combine two partial tiles element-by-element and produce another partial tile.

## C++ Declaration

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INST void TPARTMUL(TileDataDst &dst, TileDataSrc0 &src0,
                       TileDataSrc1 &src1);
```

## Parameters

| Parameter | Meaning |
| --- | --- |
| `dst` | Destination partial tile. |
| `src0` | First source partial tile. |
| `src1` | Second source partial tile. |

## Operation / Semantics

- For each valid element, computes `dst[i,j] = src0[i,j] * src1[i,j]` conceptually.
- The destination valid region defines the result domain.
- Partial-validity handling follows the implementation for the selected tile layout.

## Constraints

- Source and destination tiles must have compatible shapes, dtype, layout, and valid regions.
- Element type and layout legality follow backend checks.
- Destination valid rows and columns must describe the intended result region.

## Example

```cpp
TPARTMUL(dst, lhs, rhs);
```

## Common mistakes

- Mixing full-tile and partial-tile shapes accidentally.
- Assuming unsupported dtype combinations are promoted automatically.
- Leaving destination valid shape inconsistent with the source tiles.

## Used by kernels

- Vector microbenchmarks include partial add, multiply, max, and min cases.
- Reduction and elementwise kernels can use these operations for tiled partial results.
