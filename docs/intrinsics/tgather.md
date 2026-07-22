# TGATHER: Gather elements within a tile

## Purpose / When to use

Use `TGATHER` for tile-local gather or mask-pattern gather when source data is already in tile registers.

## C++ Declaration

```cpp
template <typename TileDataD, typename TileDataS0,
          typename TileDataS1, typename TileDataTmp>
PTO_INST void TGATHER(TileDataD &dst, TileDataS0 &src0,
                      TileDataS1 &src1, TileDataTmp &tmp);

template <typename DstTileData, typename SrcTileData, MaskPattern maskPattern>
PTO_INST void TGATHER(DstTileData &dst, SrcTileData &src);
```

## Parameters

| Parameter | Meaning |
| --- | --- |
| `dst` | Destination tile. |
| `src0` | Source data tile for index-based gather. |
| `src1` | Index tile for index-based gather. |
| `tmp` | Temporary tile required by selected implementations. |
| `src` | Source tile for mask-pattern gather. |
| `maskPattern` | Compile-time mask pattern selector. |

## Operation / Semantics

- Index-based form reads elements from `src0` selected by `src1` into `dst`.
- Mask-pattern form gathers lanes from `src` according to the compile-time mask pattern.
- The destination valid region defines produced lanes.

## Constraints

- Index, temporary, source, and destination tiles must use compatible shapes for the selected form.
- Index values must be valid for the source tile layout.
- Mask pattern is a compile-time selector.
- Use `MGATHER` instead when indexes address global memory.

## Example

```cpp
TGATHER(dst, src, indexes, tmp);
```

## Common mistakes

- Using global-memory indexes with tile-local `TGATHER`.
- Leaving the temporary tile unallocated for forms that require it.
- Confusing mask-pattern gather with a runtime mask tile.

## Used by kernels

- Vector benchmark helper documents tile-local gather patterns.
- Assigned pages distinguish global `MGATHER` from tile-local `TGATHER`.
