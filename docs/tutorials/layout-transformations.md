# Layout transformations without scalar copies

Tile layout operations change how logical tensor coordinates map to tile
elements. Use them when a producer and consumer require different axis order,
packing, interleave, or partial-tile placement.

## Choose the operation by intent

| Intent | Intrinsic |
| --- | --- |
| Transpose tile axes | [`TTRANS`](../intrinsics/ttrans.md) |
| Change a logical view without moving elements | [`TRESHAPE`](../intrinsics/treshape.md) |
| Select a rectangular region | [`TEXTRACT`](../intrinsics/textract.md) |
| Place a region into a destination tile | [`TINSERT`](../intrinsics/tinsert.md) |
| Join tile regions | [`TCONCAT`](../intrinsics/tconcat.md) |
| Reorder by tile indices | [`TGATHER`](../intrinsics/tgather.md) and [`TSCATTER`](../intrinsics/tscatter.md) |
| Convert image windows to matrix panels | [`TIMG2COL`](../intrinsics/timg2col.md) |
| Change interleaving | [`TINTERLEAVE`](../intrinsics/tinterleave.md) and [`TDEINTERLEAVE`](../intrinsics/tdeinterleave.md) |

```cpp
template <typename InputTile, typename OutputTile>
void transpose_panel(OutputTile& output, const InputTile& input) {
  TTRANS(output, input);
}
```

The destination type states the resulting shape and layout. The compiler can
therefore reject incompatible transformations before code generation.

## Valid regions and padding

A transformation moves the valid-region metadata with the logical values.
When a destination has a larger physical extent, initialize or fill the padding
according to the next consumer's identity requirement. Never assume padded
lanes are zero unless an earlier operation established that value.

## Thread ownership

Threads normally transform disjoint input and output fragments. If a transform
crosses thread boundaries, stage the producer fragments in block-shared tile
storage and use tile movement operations to build the consumer-visible layout.
Do not exchange the tensor through scalar arrays.

The [tiled tensor transpose benchmark](../benchmarks/catalog/one-level/transpose/index.md)
shows the source, build variants, and intrinsic surface for a complete layout
kernel.
