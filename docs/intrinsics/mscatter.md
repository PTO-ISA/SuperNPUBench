# MSCATTER: Scatter tile elements into global memory

## Purpose / When to use

Use `MSCATTER` when each valid source tile element writes to a global memory element selected by an index tile.

## C++ Declaration

```cpp
template <typename GlobalData, typename TileSrc, typename TileInd>
PTO_INST void MSCATTER(GlobalData &dst, TileSrc &src, TileInd &indexes);

template <typename GlobalData, typename TileSrc, typename TileInd, typename TileMask>
PTO_INST void MSCATTER_MASK(GlobalData &dst, TileSrc &src, TileInd &indexes, TileMask &mask);
```

## Parameters

| Parameter | Meaning |
| --- | --- |
| `dst` | Global tensor base or view. |
| `src` | Source tile. |
| `indexes` | Index tile. Each valid element selects a global destination element. |
| `mask` | Optional mask tile for masked compatibility form. |

## Operation / Semantics

- For each valid source element, writes `src[i,j]` to `dst[indexes[i,j]]`.
- The masked form writes only lanes enabled by `mask`.
- If multiple lanes address the same destination without an atomic mode, the final value is not portable.

## Constraints

- Source and index tiles must have compatible valid shapes.
- Index element type must be supported by the implementation.
- Scattered addresses must be within the global view region expected by the kernel.
- Masked scatters require a mask tile with the same valid region.

## Example

```cpp
MSCATTER(global_base, src, indexes);
```

## Common mistakes

- Confusing `MSCATTER` with tile-local `TSCATTER`.
- Assuming duplicate destination indexes are resolved deterministically.
- Passing indexes computed for bytes instead of elements.

## Used by kernels

- `microbenchmark/memory/memory_bench.hpp` contains scatter and masked-scatter kernels.
- Memory microbenchmark source files cover fp16, fp32, and i32 scatter cases.
