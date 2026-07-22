# TPUSH: Push a tile into a pipe

## Purpose / When to use

Use `TPUSH` on producer paths that send a tile through a pipe or FIFO-style object.

## C++ Declaration

```cpp
template <typename Pipe, typename TileData>
PTO_INST void TPUSH(Pipe &pipe, TileData &src);
```

## Parameters

| Parameter | Meaning |
| --- | --- |
| `pipe` | Pipe or FIFO producer endpoint. |
| `src` | Source tile to enqueue or publish. |

## Operation / Semantics

- Makes the source tile available to the pipe consumer according to pipe ordering rules.
- The tile valid region defines the payload region.
- Producer progress may depend on pipe capacity.

## Constraints

- `pipe` must have been associated with storage using `TALLOC` where that protocol is required.
- `src` shape and dtype must match the pipe element contract.
- Producer and consumer code must agree on push/pop ordering.
- Do not use `TPUSH` for shared tile registers; use shared `TMOV` contracts instead.

## Example

```cpp
TPUSH(pipe, tile);
```

## Common mistakes

- Pushing before allocating or binding the pipe.
- Changing the tile valid shape between producer and consumer assumptions.
- Assuming the push copies padding outside the valid region.

## Used by kernels

- No direct repository kernel use was found; retained for pipe-based producer/consumer kernels.
