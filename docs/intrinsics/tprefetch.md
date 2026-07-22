# TPREFETCH: Hint a future global-to-tile load

## Purpose / When to use

Use `TPREFETCH` to issue a best-effort hint that a global tensor view will be loaded soon. It is a performance hint, not a correctness requirement.

## C++ Declaration

```cpp
template <typename TileData, typename GlobalData>
PTO_INST void TPREFETCH(TileData &dst, GlobalData &src);
```

## Parameters

| Parameter | Meaning |
| --- | --- |
| `dst` | Tile whose shape describes the future load region. |
| `src` | Global tensor view expected to be read later. |

## Operation / Semantics

- Requests early movement or caching for `src` using the tile shape of `dst`.
- The visible program result is unchanged.
- Implementations may ignore the hint.

## Constraints

- `dst` and `src` should be compatible with a later `TLOAD`.
- Code must remain correct if the hint has no effect.
- Do not rely on prefetch for ordering or synchronization.

## Example

```cpp
TPREFETCH(tile, global_tile);
TLOAD(tile, global_tile);
```

## Common mistakes

- Using prefetch to make an otherwise invalid `TLOAD` legal.
- Assuming the hint allocates or initializes `dst`.
- Adding prefetches on paths where the data may not be used.

## Used by kernels

- No required kernel dependency was found; use it to tune memory-bound tiled kernels after correctness is established.
