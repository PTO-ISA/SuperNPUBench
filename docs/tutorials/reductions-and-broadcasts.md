# Reductions and broadcasts across tile axes

Reductions turn one tile axis into a summary value. Broadcasts move that
summary back across an axis without routing the tensor data path through
scalar C++ code. Together they implement normalization, softmax, statistics,
and many gating kernels.

## Reduce, transform, broadcast

The common pattern is:

1. Load a thread-owned tensor fragment with [`TLOAD`](../intrinsics/tload.md).
2. Reduce rows or columns, for example with
   [`TROWMAX`](../intrinsics/trowmax.md) or
   [`TCOLSUM`](../intrinsics/tcolsum.md).
3. Transform the reduced tile when the algorithm requires it.
4. Broadcast and combine it with the original tile using an operation such as
   [`TROWEXPANDSUB`](../intrinsics/trowexpandsub.md).
5. Store only the valid result region.

```cpp
template <typename Tile, typename Summary>
void normalize_rows(Tile& out, const Tile& input, Summary& row_max) {
  TROWMAX(row_max, input);
  TROWEXPANDSUB(out, input, row_max);
}
```

The reduced tile's orientation is part of its type contract. A row reduction
and a column reduction are not interchangeable even when they contain the same
number of scalar values.

## Partitioning work

Each thread reduces its own non-overlapping fragment. A block-wide reduction
first produces thread-local partials, moves those partials into block-shared
tile storage, and then lets the designated combining operation consume the
shared versions. Tile versions express both readiness and the producer-consumer
relationship in the kernel API.

For migrated examples, every participating thread owns at least 128 bytes of
valid tensor data. Boundary tiles may carry padding, but reduction operations
must ignore values outside the declared valid region.

## Common mistakes

- Broadcasting along the wrong axis changes the mathematical operation.
- Reducing padded elements without an identity fill corrupts boundary results.
- Reading a block-shared partial before its producing tile version is available
  creates an invalid dependency graph.
- Copying tile elements through a scalar pointer loop bypasses the tile data
  path and is not a supported replacement for a missing intrinsic.

See [Online softmax for tiled FlashAttention](flash-attention.md) for a complete
max-reduce, exponentiate, sum-reduce, and normalize pipeline.
