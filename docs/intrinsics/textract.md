# TEXTRACT: Extract a sub-tile region

## Purpose / When to use

Use `TEXTRACT` to copy a tile window or fragment from a larger source tile into a destination tile.

## C++ Declaration

```cpp
template <typename DstTileData, typename SrcTileData>
PTO_INST void TEXTRACT(DstTileData &dst, SrcTileData &src,
                       uint16_t indexRow = 0, uint16_t indexCol = 0);

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode>
PTO_INST void TEXTRACT(DstTileData &dst, SrcTileData &src,
                       uint16_t indexRow, uint16_t indexCol);

template <typename DstTileData, typename SrcTileData,
          ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INST void TEXTRACT(DstTileData &dst, SrcTileData &src,
                       uint64_t preQuantScalar,
                       uint16_t indexRow, uint16_t indexCol);
```

## Parameters

| Parameter | Meaning |
| --- | --- |
| `dst` | Destination tile receiving the extracted region. |
| `src` | Source tile. |
| `indexRow` | Source row offset of the extracted region. |
| `indexCol` | Source column offset of the extracted region. |
| `preQuantScalar` | Optional scalar for quantized accumulator extraction paths. |
| `reluMode` | Optional ReLU preprocessing mode. |

## Operation / Semantics

- Copies the destination-sized valid region from `src` starting at `(indexRow, indexCol)`.
- Optional modes may apply ReLU or quantization preprocessing on supported accumulator paths.
- The destination valid region defines how much data is produced.

## Constraints

- The extracted region must fit within the source valid region expected by the implementation.
- Source and destination element sizes or conversion modes must be supported.
- Row and column offsets are element offsets, not byte offsets.
- Quantized overloads require a compatible accumulator/source path.

## Example

```cpp
TEXTRACT(fragment, tile, 0, 16);
```

## Common mistakes

- Swapping row and column offsets.
- Extracting beyond the source valid region.
- Using `TEXTRACT` for a thread-owned shared-tile fragment; use shared `TMOV` forms for that contract.

## Used by kernels

- Group-programming tutorials use `TEXTRACT` to read peer rows after shared-tile broadcast.
- DeepSeek uniqueness kernels use `TEXTRACT` for pairwise column comparisons.
