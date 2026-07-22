# TMRGSORT: Merge sorted tile streams

## Purpose / When to use

Use `TMRGSORT` to merge one or more sorted tile streams, or to sort fixed-length blocks within one tile.

## C++ Declaration

```cpp
template <typename DstTileData, typename TmpTileData, typename Src0TileData,
          typename Src1TileData, typename Src2TileData, typename Src3TileData>
PTO_INST void TMRGSORT(DstTileData &dst,
                       MrgSortExecutedNumList &executedNumList,
                       TmpTileData &tmp, Src0TileData &src0,
                       Src1TileData &src1, Src2TileData &src2,
                       Src3TileData &src3);

template <typename DstTileData, typename TmpTileData, typename Src0TileData,
          typename Src1TileData, typename Src2TileData>
PTO_INST void TMRGSORT(DstTileData &dst,
                       MrgSortExecutedNumList &executedNumList,
                       TmpTileData &tmp, Src0TileData &src0,
                       Src1TileData &src1, Src2TileData &src2);

template <typename DstTileData, typename TmpTileData, typename Src0TileData,
          typename Src1TileData, bool exhausted>
PTO_INST void TMRGSORT(DstTileData &dst,
                       MrgSortExecutedNumList &executedNumList,
                       TmpTileData &tmp, Src0TileData &src0,
                       Src1TileData &src1);

template <typename DstTileData, typename SrcTileData>
PTO_INST void TMRGSORT(DstTileData &dst, SrcTileData &src, uint32_t blockLen);
```

## Parameters

| Parameter | Meaning |
| --- | --- |
| `dst` | Destination tile for merged or block-sorted output. |
| `executedNumList` | Output/metadata object tracking consumed counts for multi-list forms. |
| `tmp` | Temporary tile. |
| `src0` | First sorted input tile. |
| `src1` | Second sorted input tile. |
| `src2` | Optional third sorted input tile. |
| `src3` | Optional fourth sorted input tile. |
| `src` | Single-list source tile. |
| `blockLen` | Block length for the single-list variant. |
| `exhausted` | Compile-time flag describing whether an input stream is exhausted. |

## Operation / Semantics

- Single-list form sorts or merges fixed-size blocks of length `blockLen` within `src`.
- Multi-list forms merge two to four sorted input tiles into `dst`.
- `executedNumList` records how many elements were consumed from each input list where required.

## Constraints

- Inputs to multi-list forms must already satisfy the expected sorted order.
- `blockLen` must be valid for the source valid region.
- Temporary tile and destination shape must be compatible with the selected merge width.
- Exhaustion metadata must match the producer state.

## Example

```cpp
TMRGSORT(dst, executed, tmp, a, b);
TMRGSORT(dst, src, 64);
```

## Common mistakes

- Using unsorted input streams with multi-list forms.
- Choosing a `blockLen` that does not partition the valid region as expected.
- Ignoring `executedNumList` when feeding the next merge step.

## Used by kernels

- DeepSeek mapping notes mention `TMRGSORT` as unavailable in current migration paths.
- Use for tiled sorting and top-k pipelines when the backend exposes the full merge-sort surface.
