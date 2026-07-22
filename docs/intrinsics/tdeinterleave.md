# TDEINTERLEAVE: Split interleaved tile columns

## Purpose / When to use

Use `TDEINTERLEAVE` to split packed even/odd positions into separate destination tiles.

## C++ Declaration

```cpp
template <typename DstTile, typename SrcTile>
PTO_INST void TDEINTERLEAVE(DstTile &even_dst, DstTile &odd_dst,
                            SrcTile &packed_src);

template <typename DstTile, typename SrcTile>
PTO_INST void TDEINTERLEAVE(DstTile &even_dst, DstTile &odd_dst,
                            SrcTile &packed_src0, SrcTile &packed_src1);
```

## Parameters

| Parameter | Meaning |
| --- | --- |
| `even_dst` | Destination tile receiving even-position elements. |
| `odd_dst` | Destination tile receiving odd-position elements. |
| `packed_src` | Packed source tile. |
| `packed_src0` | First packed source tile for two-source forms. |
| `packed_src1` | Second packed source tile for two-source forms. |

## Operation / Semantics

- Writes even-position elements from the packed source stream into `even_dst`.
- Writes odd-position elements into `odd_dst`.
- Two-source forms continue the packed stream across both source tiles.

## Constraints

- All operands must use the same element type and row-major layout.
- The packed valid column count must be even.
- Source and destination valid regions must describe the same total element count.
- Supported dtypes include signed and unsigned 8-, 16-, and 32-bit integers plus common floating types where exposed.

## Example

```cpp
TDEINTERLEAVE(even, odd, packed);
```

## Common mistakes

- Calling it with an odd valid column count.
- Expecting row splitting instead of even/odd position splitting.
- Mixing dtypes between packed and output tiles.

## Used by kernels

- No direct repository kernel use was found; use it for layouts that alternate channels or packed fields.
