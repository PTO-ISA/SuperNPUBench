# Fine-Grained 128-Byte Tile Kernels

Fine-grained tiles are useful when a kernel has little data reuse, follows a
vector-style access pattern, or needs several independent operations in
flight. A 128-byte tile gives the programmer a familiar SIMD-sized unit of
work while preserving tile semantics: the value still has rows, columns, a
layout, and a valid region.

This chapter uses **128 bytes as the logical source payload**. The compiler may
lower that value through a larger internal carrier. Kernel code should reason
about the declared element count and shape, not the size of a C++ tile object
or an implementation register.

## Choose a Shape

The payload size is the product of the physical dimensions and element size:

```text
logical bytes = rows * columns * sizeof(element)
```

Several shapes can describe the same 128-byte payload:

| Element type | Elements | Example shapes | Typical use |
| --- | ---: | --- | --- |
| `float` or `int32_t` | 32 | `1 x 32`, `2 x 16`, `4 x 8`, `8 x 4` | vector arithmetic, small reductions |
| 16-bit element | 64 | `1 x 64`, `4 x 16`, `8 x 8` | conversion, activation, compact matrix fragments |
| 8-bit element | 128 | `1 x 128`, `8 x 16`, `16 x 8` | quantization, lookup, byte transforms |

Enforce the intended payload at compile time:

```cpp
constexpr int kFineTileBytes = 128;
using VectorTile = LocalTile<float, 1, 32>;
using MatrixTile = LocalTile<float, 4, 8>;

template <typename Tile>
constexpr int logical_tile_bytes() {
  return Tile::Rows * Tile::Cols * sizeof(typename Tile::DType);
}

static_assert(logical_tile_bytes<VectorTile>() == kFineTileBytes);
static_assert(logical_tile_bytes<MatrixTile>() == kFineTileBytes);
```

Choose `1 x N` when the operation is naturally linear. Choose a 2D shape when
rows and columns matter to addressing, broadcasting, reduction, transpose, or
layout conversion. Do not flatten a matrix merely to make it resemble a
vector loop.

## Write a SIMD-Style Vector Loop

A `1 x 32` FP32 tile covers 32 adjacent elements. Eight iterations process a
256-element vector without scalar pointer-indexed tensor arithmetic:

```cpp
using F32VectorTile = LocalTile<float, 1, 32>;
using Vector = global_tensor<float, RowMajor<1, 256>>;
using VectorTiles = global_iterator<Vector, F32VectorTile>;

extern "C" void fine_vector_add(float *out, float *lhs, float *rhs) {
  VectorTiles out_tiles(out);
  VectorTiles lhs_tiles(lhs);
  VectorTiles rhs_tiles(rhs);

  for (int tile_col = 0; tile_col < 8; ++tile_col) {
    F32VectorTile left;
    F32VectorTile right;
    F32VectorTile sum;
    TLOAD(left, lhs_tiles(0, tile_col));
    TLOAD(right, rhs_tiles(0, tile_col));
    TADD(sum, left, right);
    TSTORE(out_tiles(0, tile_col), sum);
  }
}
```

The C++ loop advances by one tile, not by one element. `TADD` performs the
32-lane data path and defines a new tile version for `sum`.

## Keep Two-Dimensional Meaning

The same FP32 payload can be a `4 x 8` region. This is useful for a small image,
token-by-channel, or matrix fragment:

```cpp
using F32MatrixTile = LocalTile<float, 4, 8>;
using Matrix = global_tensor<float, RowMajor<16, 16>>;
using MatrixTiles = global_iterator<Matrix, F32MatrixTile>;

for (int tile_row = 0; tile_row < 4; ++tile_row) {
  for (int tile_col = 0; tile_col < 2; ++tile_col) {
    F32MatrixTile values;
    F32MatrixTile lower_bounded;
    F32MatrixTile clamped;
    TLOAD(values, input_tiles(tile_row, tile_col));
    TMAXS(lower_bounded, values, low);
    TMINS(clamped, lower_bounded, high);
    TSTORE(out_tiles(tile_row, tile_col), clamped);
  }
}
```

The outer loops select regions of the global matrix. The tile operations
calculate all 32 elements inside each region. Retaining the `4 x 8` shape also
makes a later row reduction or column broadcast explicit.

## Expose Independent Tile Chains

One short tile chain may not provide enough independent work for a
superscalar scheduler. Keep several regions live when they do not depend on
one another:

```cpp
F32VectorTile x0, x1, x2, x3;
F32VectorTile y0, y1, y2, y3;

TLOAD(x0, input_tiles(0, 0));
TLOAD(x1, input_tiles(0, 1));
TLOAD(x2, input_tiles(0, 2));
TLOAD(x3, input_tiles(0, 3));
TADDS(y0, x0, bias);
TADDS(y1, x1, bias);
TADDS(y2, x2, bias);
TADDS(y3, x3, bias);
TSTORE(out_tiles(0, 0), y0);
TSTORE(out_tiles(0, 1), y1);
TSTORE(out_tiles(0, 2), y2);
TSTORE(out_tiles(0, 3), y3);
```

Each `yN` depends only on the corresponding `xN`. Tile IDs and versions make
those relationships visible, allowing independent loads and arithmetic to
overlap. Keep the number of live tiles bounded; excessive unrolling can
increase tile pressure and reduce performance.

## Reduce a Small Tile

A fine-grained tile can produce an even smaller logical result. Here each row
of a `4 x 8` FP32 tile is reduced to one value:

```cpp
using InputTile = LocalTile<float, 4, 8>;
using RowSums =
    LocalTile<float, 4, 8, BLayout::RowMajor, 4, 1>;

InputTile values;
RowSums sums;
TLOAD(values, input_tiles(0, 0));
TROWSUM(sums, values);
TSTORE(output_tiles(0, 0), sums);
```

`RowSums` retains a 128-byte physical shape but declares only `4 x 1` elements
valid. `TSTORE` writes the valid result region. This pattern avoids a scalar
loop and keeps the reduction in the tile data path.

## Handle Short Tails

Do not switch to scalar tensor arithmetic for the final partial vector. Keep
the physical `1 x 32` tile and narrow its runtime valid region:

```cpp
using TailTile = LocalTile<float, 1, 32, BLayout::RowMajor,
                           DYNAMIC, DYNAMIC>;

TailTile values(1, 19);
TailTile result(1, 19);
TLOAD(values, input_tiles(0, 0));
TADDS(result, values, bias);
TSTORE(output_tiles(0, 0), result);
```

Only 19 elements participate. The remaining physical lanes are padding and
must not affect reductions or stores. Construct a tail with positive valid
dimensions that do not exceed its physical dimensions.

## Use Fine Tiles in Multidimensional Loops

For a tensor shaped `[batch, head, sequence, width]`, scalar loops choose the
outer plane and tile operations handle its inner regions:

```cpp
for (int batch = 0; batch < kBatch; ++batch) {
  for (int head = 0; head < kHeads; ++head) {
    PlaneTiles input_tiles(input_plane(batch, head));
    PlaneTiles output_tiles(output_plane(batch, head));

    for (int tr = 0; tr < kSequence / 4; ++tr) {
      for (int tc = 0; tc < kWidth / 8; ++tc) {
        LocalTile<float, 4, 8> x;
        LocalTile<float, 4, 8> y;
        TLOAD(x, input_tiles(tr, tc));
        TRELU(y, x);
        TSTORE(output_tiles(tr, tc), y);
      }
    }
  }
}
```

This separation is important: C++ controls coordinates and specialization;
tile operations implement the tensor data path. Partition independent planes
or tile rows across threads when the launch contains multiple workers.

## Decide Between Fine and Coarse Tiles

Prefer a 128-byte tile when:

- work is element-wise, reduction-oriented, or has limited reuse;
- a small valid region avoids unnecessary computation;
- several independent tile chains can overlap;
- tensor dimensions naturally divide into compact row or column fragments.

Prefer a larger tile when:

- matrix multiplication benefits from reuse across rows and columns;
- each load feeds substantial computation;
- a larger region reduces loop and address-generation overhead;
- the kernel would otherwise keep too many fine tiles live at once.

Fine granularity is a scheduling choice, not a universal optimization. Measure
representative shapes and account for global-memory alignment, contiguous
access, instruction count, and live tile pressure.

## Complete Compile-Checked Source

The complete example includes vector addition, a 2D clamp, independent tile
chains, a runtime tail, and a row reduction:

??? code "fine_grained_tiles.cpp"

    ```cpp
    --8<-- "docs/examples/fine_grained_tiles.cpp"
    ```

Compile it with the active target compiler:

```bash
bash docs/examples/check.sh fine-grained-tiles
```

## Review Checklist

- Assert `rows * columns * sizeof(element) == 128` for an exact fine tile.
- Use `1 x N` for linear work and a 2D shape when axes carry meaning.
- Advance global addresses by tile coordinates, never by scalar lane loops.
- Keep independent tile chains visible when additional overlap is useful.
- Represent tails with a valid region inside the fixed physical shape.
- Use shared tiles only when a value crosses thread ownership.
- Compare fine and coarse variants for kernels with substantial data reuse.
