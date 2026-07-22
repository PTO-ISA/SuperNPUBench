# First Tile Program

This kernel adds two `64 x 64` matrices as four `32 x 32` tiles. It is small
enough to show the complete C++ shape without exposing the compiler backend.
The public `pto_kernel/tile.hpp` header provides typed tiles and tile
operations; target-specific namespaces stay behind that header.

```cpp title="docs/examples/tile_add.cpp" linenums="1" hl_lines="14-16 26-40"
--8<-- "docs/examples/tile_add.cpp"
```

## Define the Tensor and Tile

`global_tensor<int, RowMajor<Rows, Cols>>` describes the complete matrix in
global memory. `LocalTile<int, TileRows, TileCols>` describes one thread-local
working value. Element type, physical shape, and layout are compile-time
properties.

| Name | Meaning |
| --- | --- |
| `Matrix` | Global `64 x 64` row-major tensor type |
| `ValueTile` | Local `32 x 32` tile type |
| `MatrixTiles` | Iterator mapping `(tile_row, tile_col)` to a tensor region |
| `TLOAD` | Define a local tile from a global tensor region |
| `TADD` | Define a local tile from two compatible sources |
| `TSTORE` | Store a local tile through a global tensor region |

## Read the Two-Dimensional Loop

The loop variables count **tiles**, not elements. Each loop has two iterations,
so the body executes four times:

```text
(tile_row, tile_col) = (0,0) (0,1) (1,0) (1,1)
global origin        = (0,0) (0,32) (32,0) (32,32)
```

The intrinsic sequence in each iteration is a small value graph:

```text
lhs region -> TLOAD -> left  --\
                              TADD -> sum -> TSTORE -> output region
rhs region -> TLOAD -> right --/
```

No C++ element loop performs the calculation. C++ chooses tensor regions;
tile operations process every valid element in those regions.

## Compile It

```bash
cd "$SUPERNPU_ROOT"
bash docs/examples/check.sh tile-add
```

The check invokes the target `clang++` with C++20, matrix support, the public
tile include root, and the active target triple.

## Connect the Entry Point

The `extern "C"` entry gives the harness a stable symbol. Production kernels
usually template the matrix and tile dimensions, then instantiate concrete
shapes in their benchmark entry points.

Next, [iterate over multidimensional tensors](../tutorials/multidimensional-tiling.md),
[inspect the object](build-run.md), or move to the [GEMM tutorial](../tutorials/gemm.md).
