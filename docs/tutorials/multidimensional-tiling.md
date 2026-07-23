# Multidimensional Tiling in C++

A tile operation works on a two-dimensional value. A model tensor may have
more dimensions: batch, sequence, head, channel, row, and column. Kernel C++
selects the outer coordinates and walks a two-dimensional tile grid; tile
operations perform the calculation inside each selected region.

The same loop structure applies to compact `4 x 8` FP32 regions. The
[fine-grained tile guide](fine-grained-tiles.md) develops that 128-byte case
and compares it with one-dimensional vector-style tiles.

```text
scalar C++ loops:  batch -> head -> tile row -> tile column
tile data path:                         TLOAD -> compute -> TSTORE
```

## Start with a Two-Dimensional Matrix

Declare the complete global shape and the smaller local shape separately:

```cpp
constexpr int kRows = 128;
constexpr int kCols = 256;
constexpr int kTileRows = 32;
constexpr int kTileCols = 32;

using Tile = LocalTile<float, kTileRows, kTileCols>;
using Matrix = global_tensor<float, RowMajor<kRows, kCols>>;
using Tiles = global_iterator<Matrix, Tile>;
```

The iterator receives tile coordinates. Its `(1, 3)` region begins at element
coordinate `(32, 96)`.

```cpp
Tiles input_tiles(input_ptr);
Tiles output_tiles(output_ptr);

for (int tr = 0; tr < kRows / kTileRows; ++tr) {
  for (int tc = 0; tc < kCols / kTileCols; ++tc) {
    Tile input;
    Tile output;
    TLOAD(input, input_tiles(tr, tc));
    TRELU(output, input);
    TSTORE(output_tiles(tr, tc), output);
  }
}
```

The loops move between regions. `TRELU` applies to all valid elements in one
region. There is no scalar loop over the tile's rows and columns.

## Add a Batch Dimension

Treat a dense `[batch, rows, columns]` tensor as a sequence of matrices. The
batch loop selects a base address, and the inner loops tile that matrix:

```cpp
constexpr int kMatrixElements = kRows * kCols;

for (int batch = 0; batch < kBatch; ++batch) {
  Tiles input_tiles(input_ptr + batch * kMatrixElements);
  Tiles output_tiles(output_ptr + batch * kMatrixElements);

  for (int tr = 0; tr < kRows / kTileRows; ++tr) {
    for (int tc = 0; tc < kCols / kTileCols; ++tc) {
      Tile x;
      Tile y;
      TLOAD(x, input_tiles(tr, tc));
      TADDS(y, x, bias[batch]);
      TSTORE(output_tiles(tr, tc), y);
    }
  }
}
```

The scalar `batch` index controls addressing and selects a scalar parameter.
The tensor calculation remains `TADDS`, not a pointer-indexed element loop.

## Add Batch and Head Dimensions

Attention commonly stores `[batch, head, sequence, width]`. Flatten the two
outer dimensions when each `(batch, head)` plane is contiguous:

```cpp
constexpr int kPlaneElements = kSequence * kWidth;

for (int batch = 0; batch < kBatch; ++batch) {
  for (int head = 0; head < kHeads; ++head) {
    const int plane = batch * kHeads + head;
    float *q_plane = q + plane * kPlaneElements;
    float *o_plane = output + plane * kPlaneElements;

    QueryTiles queries(q_plane);
    OutputTiles outputs(o_plane);

    for (int query_tile = 0; query_tile < kSequence / kQueryRows;
         ++query_tile) {
      QueryTile q_fragment;
      OutputTile result;
      TLOAD(q_fragment, queries(query_tile, 0));
      attention_for_query_tile(result, q_fragment, key, value);
      TSTORE(outputs(query_tile, 0), result);
    }
  }
}
```

Use a typed strided view instead of pointer offsets when planes are not
contiguous. The ownership rule is unchanged: scalar code chooses one plane;
tile code transforms its regions.

## Walk Three Matrix Dimensions

Tiled matrix multiplication uses one loop for each output-grid dimension and a
third loop for the reduction dimension:

```cpp
for (int m = 0; m < kM / kTM; ++m) {
  for (int n = 0; n < kN / kTN; ++n) {
    AccumulatorTile<float, kTM, kTN> accum;

    for (int k = 0; k < kK / kTK; ++k) {
      MatrixLeftTile<half, kTM, kTK> left;
      MatrixRightTile<half, kTK, kTN> right;
      TLOAD(left, lhs_tiles(m, k));
      TLOAD(right, rhs_tiles(k, n));

      if (k == 0) {
        TMATMUL(accum, left, right);
      } else {
        TMATMUL_ACC(accum, accum, left, right);
      }
    }

    LocalTile<float, kTM, kTN> result;
    TCVT(result, accum);
    TSTORE(output_tiles(m, n), result);
  }
}
```

The `k` loop is different from the output loops: every iteration contributes
to the same accumulator version chain. Different `(m, n)` output tiles are
independent and may overlap in the superscalar schedule.

## Reuse a Shared Right Tile

When every thread in a block uses the same right panel, define it once as a
shared value and keep each left panel local:

```cpp
SharedTile<half, kTK, kTN> shared_right;

for (int k = 0; k < kK / kTK; ++k) {
  if (thread == 0) {
    TLOAD(shared_right, rhs_tiles(k, output_column));
  }

  MatrixLeftTile<half, kRowsPerThread, kTK> local_left;
  TLOAD(local_left, lhs_tiles(local_row, k));

  if (k == 0) {
    TMATMUL(local_accum, local_left, shared_right);
  } else {
    TMATMUL_ACC(local_accum, local_accum, local_left, shared_right);
  }
}
```

Every loop iteration defines a new shared-right version. The matrix operation
names the version it consumes; source code does not wait on an event or a
physical register.

## Partition Outer Work Across Blocks and Threads

Map a flattened outer index to batch and head after block/thread partitioning:

```cpp
const uint32_t worker =
    get_block_idx() * kThreadsPerBlock + get_thread_idx();
const uint32_t worker_count = block_count * kThreadsPerBlock;

for (uint32_t plane = worker; plane < kBatch * kHeads;
     plane += worker_count) {
  const uint32_t batch = plane / kHeads;
  const uint32_t head = plane % kHeads;
  process_plane(batch, head);
}
```

This grid-stride form gives each worker disjoint planes without hard-coding the
number of blocks. When several threads collaborate on one plane, partition its
rows again and use shared tiles only for values that cross thread ownership.

## Handle Tails Without Scalar Tensor Math

When a dimension is not divisible by its tile size, keep a full physical tile
and set a smaller valid region:

```cpp
const int valid_rows = min(kTileRows, kRows - row_origin);
const int valid_cols = min(kTileCols, kCols - col_origin);

LocalTile<float, kTileRows, kTileCols,
          BLayout::RowMajor, DYNAMIC, DYNAMIC> tail(valid_rows, valid_cols);
TLOAD(tail, input_tail);
TRELU(tail_result, tail);
TSTORE(output_tail, tail_result);
```

Padding remains storage, not tensor data. Reductions must exclude it, and
stores must not overwrite elements outside the logical tensor.

## Review Checklist

- Outer C++ loops identify batches, heads, channels, or tile coordinates.
- Tile operations implement the tensor data path inside each region.
- Each output region has one owner unless an explicit reduction combines it.
- Matrix reduction loops carry an accumulator tile version.
- Shared tiles represent genuinely reused or exchanged block-local values.
- Tail tiles preserve the physical allocation and narrow only the valid region.
