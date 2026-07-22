# TINSERT: Insert a sub-tile region

## Purpose / When to use

Use `TINSERT` to write a smaller source tile into a destination tile at an element row and column offset.

## C++ Declaration

```cpp
template <typename DstTileData, typename SrcTileData>
PTO_INST void TINSERT(DstTileData &dst, SrcTileData &src,
                      uint16_t indexRow, uint16_t indexCol);

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode>
PTO_INST void TINSERT(DstTileData &dst, SrcTileData &src,
                      uint16_t indexRow, uint16_t indexCol);

template <typename DstTileData, typename SrcTileData,
          ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INST void TINSERT(DstTileData &dst, SrcTileData &src,
                      uint64_t preQuantScalar,
                      uint16_t indexRow, uint16_t indexCol);

template <TInsertMode mode, typename DstTileData, typename SrcTileData>
PTO_INST void TINSERT(DstTileData &dst, SrcTileData &src,
                      uint32_t indexRow = 0, uint32_t indexCol = 0);
```

## Parameters

| Parameter | Meaning |
| --- | --- |
| `dst` | Destination tile updated in place. |
| `src` | Source tile inserted into `dst`. |
| `indexRow` | Destination row offset. |
| `indexCol` | Destination column offset. |
| `preQuantScalar` | Optional scalar for quantized accumulator insertion paths. |
| `reluMode` | Optional ReLU preprocessing mode. |
| `mode` | Optional insertion mode for implementations that expose mode-specific forms. |

## Operation / Semantics

- Copies the source valid region into `dst` beginning at `(indexRow, indexCol)`.
- Destination elements outside the inserted region retain their previous value.
- Optional modes may apply ReLU or quantization preprocessing on supported paths.

## Constraints

- The inserted region must fit within the destination valid region expected by the implementation.
- Source and destination element sizes or conversion modes must be supported.
- Offsets are element coordinates.
- Mode-specific insertion forms are available only for supported location pairs.

## Example

```cpp
TINSERT(tile, fragment, 0, 16);
```

## Common mistakes

- Assuming `dst` is cleared before insertion; initialize it explicitly when needed.
- Using byte offsets instead of row/column element offsets.
- Inserting a source tile whose valid shape overruns the destination.

## Used by kernels

- Reduction single-tree kernels use `TINSERT` to collect partial reductions.
- DeepSeek group-count and uniqueness kernels use `TINSERT` to build histogram or updated index tiles.
