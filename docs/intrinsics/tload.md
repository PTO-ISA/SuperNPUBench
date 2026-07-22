# TLOAD: Load a global tensor view into a tile

## Purpose / When to use

Use `TLOAD` to move a rectangular valid region from a `GlobalTensor` or iterator-produced global view into a local tile. Use the shared-tile overload when one selected PE loads a value that becomes visible to the PE group.

## C++ Declaration

```cpp
template <typename TileData, typename GlobalData>
PTO_INST void TLOAD(TileData &dst, GlobalData &src);

template <typename SharedTileData, typename GlobalData>
PTO_INST void TLOAD(SharedTileData &shared_dst, GlobalData &src);
```

## Parameters

| Parameter | Meaning |
| --- | --- |
| `dst` | Tile destination. Its valid rows and columns define the transfer size. |
| `src` | Global tensor view with the same element size as the destination tile. |
| `shared_dst` | Versioned shared tile destination for group-visible loads. |

## Operation / Semantics

- Copies `src` into `dst` over `dst.GetValidRow()` by `dst.GetValidCol()`.
- For normal tile destinations, the result is private to the issuing PE.
- For shared tile destinations, exactly one statically identifiable thread issues the memory transfer; the ready shared tile version is visible to participating threads.
- Layout conversion is limited to implementation-supported tile/global layout pairs.

## Constraints

- Element sizes of `TileData::DType` and `GlobalData::DType` must match.
- Destination tile location is `Vec` or `Mat`.
- All runtime source dimensions and destination valid extents must be greater than zero.
- 64-bit integer loads are limited to direct row-major or column-major layout-preserving forms.
- Shared tile loads are an architectural contract; compiler lowering support may lag the C++ surface.

## Example

```cpp
TLOAD(tile, global_tile);
```

## Common mistakes

- Passing an iterator expression directly when the intrinsic requires an lvalue global view; bind it to `auto view = it(0, 0);` first.
- Expecting invalid or padded elements outside the tile valid region to be loaded.
- Using the shared overload from more than one PE for the same shared destination version.

## Used by kernels

- Memory microbenchmarks use `TLOAD` in `microbenchmark/memory/memory_bench.hpp`.
- Tile compute microbenchmarks use `TLOAD` before arithmetic in `microbenchmark/vector/vector_bench.hpp` and `microbenchmark/cube/cube_bench.hpp`.
- DeepSeek, reduction, GEMM, and tutorial kernels use `TLOAD` as the normal global-to-tile ingress.
