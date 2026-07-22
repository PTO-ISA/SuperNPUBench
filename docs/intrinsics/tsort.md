# TSORT: Sort tile values with indexes

## Purpose / When to use

Use `TSORT` to sort values in a tile while carrying an index tile alongside the values.

## C++ Declaration

```cpp
template <typename DstTileData, typename SrcTileData, typename IdxTileData>
PTO_INST void TSORT(DstTileData &dst, SrcTileData &src, IdxTileData &idx);

template <typename DstTileData, typename SrcTileData, typename IdxTileData,
          typename TmpTileData>
PTO_INST void TSORT(DstTileData &dst, SrcTileData &src, IdxTileData &idx,
                    TmpTileData &tmp);
```

## Parameters

| Parameter | Meaning |
| --- | --- |
| `dst` | Destination tile containing sorted value-index pairs or sorted representation. |
| `src` | Source value tile. |
| `idx` | Input index tile permuted with `src`. |
| `tmp` | Optional temporary tile. |

## Operation / Semantics

- Sorts valid source values and carries each element's input index from `idx`.
- The CPU simulation sorts descending by value and keeps smaller indexes first for equal values.
- `idx` is an input operand, not an output-only destination.

## Constraints

- `idx` must be initialized and shape-compatible with `src`.
- Source, destination, index, and temporary tile shapes must satisfy implementation legality checks.
- Destination valid region follows the sorted source region.
- Tie-breaking beyond the documented simulation behavior should be treated as implementation-defined unless the backend specifies it.

## Example

```cpp
TSORT(sorted, values, indexes, tmp);
```

## Common mistakes

- Passing an uninitialized index tile.
- Expecting ascending order from the current CPU simulation.
- Assuming `dst` contains only values when the selected representation carries indexes too.

## Used by kernels

- DeepSeek mapping notes mention sort operations as unavailable in current migration paths.
- Use with merge sort primitives for tiled top-k or ordering kernels when exposed.
