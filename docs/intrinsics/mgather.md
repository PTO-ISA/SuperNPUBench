# MGATHER: Gather global memory elements into a tile

## Purpose / When to use

Use `MGATHER` when each output element reads from a global memory element selected by an index tile.

## C++ Declaration

```cpp
template <typename TileDst, typename GlobalData, typename TileInd>
PTO_INST void MGATHER(TileDst &dst, GlobalData &src, TileInd &indexes);

template <typename TileDst, typename GlobalData, typename TileInd, typename TileMask>
PTO_INST void MGATHER_MASK(TileDst &dst, GlobalData &src, TileInd &indexes, TileMask &mask);
```

## Parameters

| Parameter | Meaning |
| --- | --- |
| `dst` | Destination tile. |
| `src` | Global tensor base or view. |
| `indexes` | Index tile. Each valid element selects a global element. |
| `mask` | Optional mask tile for masked compatibility form. |

## Operation / Semantics

- For each valid destination element, reads `src[indexes[i,j]]` into `dst[i,j]`.
- The masked form only updates lanes enabled by `mask`; disabled lanes keep the implementation-defined preserved value.
- Index interpretation is linear unless the selected global view defines a more specific addressing mode.

## Constraints

- Destination and index tiles must have compatible valid shapes.
- Index element type must be supported by the implementation, commonly 32-bit integer-compatible values.
- Gathered addresses must be within the global view region expected by the kernel.
- Masked gathers require a mask tile with the same valid region.

## Example

```cpp
MGATHER(dst, global_base, indexes);
```

## Common mistakes

- Confusing `MGATHER` with tile-local `TGATHER`.
- Supplying byte offsets when the API expects element indexes.
- Leaving index lanes uninitialized in the destination valid region.

## Used by kernels

- `microbenchmark/memory/memory_bench.hpp` contains gather and masked-gather kernels.
- Memory microbenchmark source files cover fp16, fp32, and i32 gather cases.
