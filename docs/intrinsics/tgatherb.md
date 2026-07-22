# TGATHERB: Gather tile elements by offsets

## Purpose / When to use

Use `TGATHERB` for tile-local gather when each lane carries an offset into the source tile storage.

## C++ Declaration

```cpp
template <typename TileDataDst, typename TileDataSrc, typename TileDataOffset>
PTO_INST void TGATHERB(TileDataDst &dst, TileDataSrc &src, TileDataOffset &offset);
```

## Parameters

| Parameter | Meaning |
| --- | --- |
| `dst` | Destination tile. |
| `src` | Source tile. |
| `offset` | Offset tile selecting source elements. |

## Operation / Semantics

- For each valid destination element, reads from `src` at the corresponding offset.
- Offsets are interpreted in the source tile storage order used by the implementation.
- The destination valid region defines the number of gathered elements.

## Constraints

- Destination, source, and offset valid regions must be compatible.
- Offsets must be in range for the source tile storage.
- Offset dtype must be supported by the implementation.
- Use `MGATHER` when offsets address global memory.

## Example

```cpp
TGATHERB(dst, src, offsets);
```

## Common mistakes

- Using byte offsets when element offsets are expected.
- Using `TGATHERB` for global memory gather.
- Leaving offset lanes uninitialized.

## Used by kernels

- `microbenchmark/vector/vector_bench.hpp` includes a gather helper for `TGATHERB`.
- Vector microbenchmark cases cover fp16 and fp32 gather-by-offset flows.
