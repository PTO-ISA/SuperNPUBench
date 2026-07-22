# TINTERLEAVE: Pack two tiles into interleaved columns

## Purpose / When to use

Use `TINTERLEAVE` to combine separate even and odd streams into packed interleaved destination tiles.

## C++ Declaration

```cpp
template <typename DstTile, typename SrcTile>
PTO_INST void TINTERLEAVE(DstTile &packed_dst0, DstTile &packed_dst1,
                          SrcTile &even_src, SrcTile &odd_src);
```

## Parameters

| Parameter | Meaning |
| --- | --- |
| `packed_dst0` | First packed destination tile. |
| `packed_dst1` | Second packed destination tile. |
| `even_src` | Source tile for even positions. |
| `odd_src` | Source tile for odd positions. |

## Operation / Semantics

- Alternates elements from `even_src` and `odd_src` into the packed destination stream.
- Destination positions 0, 2, 4, ... come from `even_src`.
- Destination positions 1, 3, 5, ... come from `odd_src`.

## Constraints

- All operands must use the same element type and row-major layout.
- Source and destination valid column counts must be even and mutually compatible.
- The destination stream must be large enough for both source streams.
- Supported dtypes include signed and unsigned 8-, 16-, and 32-bit integers plus common floating types where exposed.

## Example

```cpp
TINTERLEAVE(packed0, packed1, even, odd);
```

## Common mistakes

- Expecting concatenation instead of alternating positions.
- Using odd valid column counts.
- Mixing layouts between sources and destinations.

## Used by kernels

- No direct repository kernel use was found; use it for channel or field packing before stores or communication.
