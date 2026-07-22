# THISTOGRAM: Count values into bins

## Purpose / When to use

Use `THISTOGRAM` to accumulate per-bin counts from source values and an index tile.

## C++ Declaration

```cpp
template <typename DstTileData, typename SrcTileData, typename IdxTileData>
PTO_INST void THISTOGRAM(DstTileData &dst, SrcTileData &src,
                         IdxTileData &idx, int byteId);
```

## Parameters

| Parameter | Meaning |
| --- | --- |
| `dst` | Destination tile for bin counts. |
| `src` | Source values. |
| `idx` | Index or bin-selection tile. |
| `byteId` | Byte lane selector for packed index formats. |

## Operation / Semantics

- For each valid source element, chooses a bin using `idx` and `byteId`.
- Accumulates counts into `dst` according to the selected bin layout.
- Destination shape represents bins, not a pointwise copy of the source.

## Constraints

- `byteId` must select a valid byte lane for packed forms.
- Source, index, and destination dtypes must be supported together.
- Destination must be initialized or defined according to the accumulation mode expected by the implementation.
- When direct support is unavailable, build histograms with compare, select, row sum, and insert operations.

## Example

```cpp
THISTOGRAM(hist, values, indexes, 0);
```

## Common mistakes

- Forgetting that `dst` contains counts rather than selected source values.
- Passing an invalid byte lane selector.
- Using it in toolchain configurations where direct histogram lowering is unavailable.

## Used by kernels

- Vector microbenchmarks include histogram helpers and fp16/fp32 source files.
- DeepSeek MoE group-count kernels currently simulate histograms with compare, select, row reduction, and insert operations.
