# TPARTARGMIN

## Introduction

Select the minimum values and their indices across two partial tile inputs.

## Math Interpretation

For every valid element `i`, compare `(src0[i], src0_idx[i])` with
`(src1[i], src1_idx[i])`. Write the lesser value to `dst[i]` and copy the
matching index to `dst_idx[i]`.

## C++ Intrinsic

```cpp
template <typename DstTile, typename SrcTile, typename DstIndexTile,
          typename SrcIndexTile>
PTO_INST void TPARTARGMIN(
    DstTile &dst, SrcTile &src0, SrcTile &src1,
    DstIndexTile &dst_idx, SrcIndexTile &src0_idx,
    SrcIndexTile &src1_idx);
```

## Constraints

- Value operands share element type, shape, layout, and valid region.
- Index operands share shape and valid region with their value operands.
- Tie-breaking follows the selected target's stable-index rule.

## Examples

```cpp
TPARTARGMIN(values, lhs, rhs, indices, lhs_indices, rhs_indices);
```

## Assembly Status

LinxISA 0.57 defines this public PTO operation. Its exact block-template
lowering is target-defined.
