# TINTERLEAVE

## Introduction

Interleave even and odd source sequences into two destination tiles. This is
the inverse of `TDEINTERLEAVE`.

## Math Interpretation

For flattened sources `even` and `odd`, the logical result satisfies
`result[2*i] = even[i]` and `result[2*i + 1] = odd[i]`. The result is split at
its midpoint across `dst0` and `dst1`.

## C++ Intrinsic

```cpp
template <typename DstTile, typename SrcTile>
PTO_INST void TINTERLEAVE(
    DstTile &dst0, DstTile &dst1, SrcTile &even, SrcTile &odd);
```

## Constraints

- All operands have the same element type and row-major layout.
- Source and destination valid column counts are even and mutually compatible.
- LinxISA 0.57 lists signed/unsigned 8-, 16-, and 32-bit integers, `float`,
  `half`, and `bfloat16_t`.
- The 0.57 workbook marks the operation as A5-only among its listed targets.

## Examples

```cpp
TINTERLEAVE(packed0, packed1, even, odd);
```

## Assembly Status

The 0.57 workbook defines the PTO operation but does not provide a standalone
DavinciOO PTO-AS page. The selected backend supplies its block-template
lowering.
