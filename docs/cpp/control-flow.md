# Loops, Branches, and Scalar Control

NPU kernels are still C++ programs. Use scalar loops and branches to choose
tensor regions and tile operations, but keep the tensor data path in tile
operations.

## Tile-Grid Loops

Use nested loops to walk a two-dimensional tile grid:

```cpp
for (int tile_row = 0; tile_row < row_tiles; ++tile_row) {
  for (int tile_col = 0; tile_col < col_tiles; ++tile_col) {
    LocalTile<float, kTileRows, kTileCols> x;
    LocalTile<float, kTileRows, kTileCols> y;
    TLOAD(x, input(tile_row, tile_col));
    TRELU(y, x);
    TSTORE(output(tile_row, tile_col), y);
  }
}
```

Outer logical dimensions add outer C++ loops. A `[batch, head, row, column]`
tensor commonly uses batch and head loops around this two-dimensional grid.

## Reduction Loops

Matrix multiplication carries one accumulator version across the reduction
dimension:

```cpp
for (uint32_t k0 = 0; k0 < kK; k0 += kTileK) {
  TLOAD(a_fragment, a.slice_k(k0, kTileK));
  if (thread == 0) {
    TLOAD(shared_b, b.slice_k(k0, kTileK));
  }
  if (k0 == 0) {
    TMATMUL(accum, a_fragment, shared_b);
  } else {
    TMATMUL_ACC(accum, accum, a_fragment, shared_b);
  }
}
```

Each iteration defines new operand versions. `accum` creates the intended
loop-carried dependency. Different output tiles have independent chains and
may overlap in the superscalar schedule.

## Branches

```cpp
if (has_bias) {
  TADD(out, accum, bias);
} else {
  TMOV(out, accum);
}
```

Every path that reaches a later consumer must define the value consumed by
that later operation. Branches may select operations, layouts, or tensor
regions; they should not implement an element-by-element tensor fallback.

## Scalar Values

Use scalar values for indices, bounds, strides, mode selection, and predicates.
Do not derive correctness from physical tile-register numbers or operation
latency.

Use C++ to answer **where** a tile comes from and **which** operation runs. Do
not use pointer-indexed scalar loops to answer **what happens to each tensor
element**; that calculation belongs in tile operations.

For complete loop nests, read
[Multidimensional tiling](../tutorials/multidimensional-tiling.md).
