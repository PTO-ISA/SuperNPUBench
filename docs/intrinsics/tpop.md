# TPOP: Pop a tile from a pipe

## Purpose / When to use

Use `TPOP` on consumer paths that receive a tile from a pipe or FIFO-style object.

## C++ Declaration

```cpp
template <typename TileData, typename Pipe>
PTO_INST void TPOP(TileData &dst, Pipe &pipe);
```

## Parameters

| Parameter | Meaning |
| --- | --- |
| `dst` | Destination tile receiving the popped payload. |
| `pipe` | Pipe or FIFO consumer endpoint. |

## Operation / Semantics

- Reads the next pipe payload into `dst` according to pipe ordering rules.
- The destination valid region describes the expected payload shape.
- Consumer progress may depend on producer availability.

## Constraints

- `pipe` must have been associated with storage using `TALLOC` where that protocol is required.
- `dst` shape and dtype must match the pipe element contract.
- Producer and consumer code must agree on push/pop ordering.
- Do not use `TPOP` for shared tile registers; use shared `TMOV` contracts instead.

## Example

```cpp
TPOP(tile, pipe);
```

## Common mistakes

- Popping from an unallocated or unbound pipe.
- Mismatching tile shape with the producer payload.
- Assuming pop initializes padding outside the valid region.

## Used by kernels

- No direct repository kernel use was found; retained for pipe-based producer/consumer kernels.
