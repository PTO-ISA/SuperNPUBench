# TSCATTER: Scatter elements within a tile

## Purpose / When to use

Use `TSCATTER` to write source tile elements into positions in another tile selected by an index tile.

## C++ Declaration

```cpp
template <typename TileDataD, typename TileDataS, typename TileDataI>
PTO_INST void TSCATTER(TileDataD &dst, TileDataS &src, TileDataI &indexes);
```

## Parameters

| Parameter | Meaning |
| --- | --- |
| `dst` | Destination tile updated by scatter. |
| `src` | Source tile. |
| `indexes` | Index tile selecting destination positions. |

## Operation / Semantics

- Views `dst` as a linear tile storage sequence.
- For each valid source element, writes to the flattened destination element selected by the corresponding index.
- Does not interpret an index as a destination row selector unless the layout-specific implementation says so.

## Constraints

- Source and index tiles must have compatible valid shapes.
- Indexes must be valid flattened destination element positions.
- Duplicate indexes have implementation-defined resolution unless a higher-level algorithm rules them out.
- Destination dtype and source dtype must be compatible.

## Example

```cpp
TSCATTER(dst, src, indexes);
```

## Common mistakes

- Treating indexes as row numbers instead of flattened element indexes.
- Expecting deterministic behavior for duplicate destination indexes.
- Using `TSCATTER` when the destination is global memory; use `MSCATTER`.

## Used by kernels

- Useful for tile-local permutation kernels; no direct benchmark kernel use was found in assigned sources.
