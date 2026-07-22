# TCI: Create a contiguous index tile

## Purpose / When to use

Use `TCI` to fill a tile with a row/column index sequence starting at a scalar value.

## C++ Declaration

```cpp
template <typename TileData, typename T, int descending>
PTO_INST void TCI(TileData &dst, T start);
```

## Parameters

| Parameter | Meaning |
| --- | --- |
| `dst` | Destination index tile. |
| `start` | First index value. |
| `descending` | Template flag; `0` creates ascending values, nonzero creates descending values. |

## Operation / Semantics

- Ascending form writes a contiguous increasing sequence beginning at `start`.
- Descending form writes a contiguous decreasing sequence beginning at `start`.
- The sequence covers the destination valid region in the implementation-defined tile traversal order.

## Constraints

- Destination dtype must be integer-compatible or otherwise accepted by the implementation.
- `descending` is a compile-time template value.
- Destination valid rows and columns must describe a non-empty region.

## Example

```cpp
TCI<TileT, int32_t, 0>(idx, 0);
```

## Common mistakes

- Expecting `descending` to be a runtime parameter.
- Using `TCI` indexes with a gather/scatter that expects a different traversal order.
- Using a tile dtype too small for the generated index range.

## Used by kernels

- DeepSeek hash and mapping notes use `TCI`-style index generation with scalar arithmetic.
- Index-generation patterns feed gather, compare, selection, and scatter kernels.
