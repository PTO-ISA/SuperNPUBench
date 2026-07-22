# TPARTARGMIN: Select partial minima and indexes

## Purpose / When to use

Use `TPARTARGMIN` to combine two partial value/index streams and keep the smaller value with its associated index.

## C++ Declaration

```cpp
template <typename DstTile, typename SrcTile, typename DstIndexTile,
          typename SrcIndexTile>
PTO_INST void TPARTARGMIN(DstTile &values, SrcTile &lhs, SrcTile &rhs,
                          DstIndexTile &indices, SrcIndexTile &lhs_indices,
                          SrcIndexTile &rhs_indices);
```

## Parameters

| Parameter | Meaning |
| --- | --- |
| `values` | Destination values tile. |
| `lhs` | Left values tile. |
| `rhs` | Right values tile. |
| `indices` | Destination index tile. |
| `lhs_indices` | Indexes associated with `lhs`. |
| `rhs_indices` | Indexes associated with `rhs`. |

## Operation / Semantics

- Compares each valid `lhs` and `rhs` value.
- Writes the selected minimum value to `values`.
- Writes the corresponding source index to `indices`.
- Tie-breaking follows the stable-index rule exposed by the selected implementation.

## Constraints

- Value operands must share element type, shape, layout, and valid region.
- Index operands must share shape and valid region with their value operands.
- Index dtype must be supported by the consumer kernel.
- Tie behavior must be validated when deterministic ordering matters.

## Example

```cpp
TPARTARGMIN(values, lhs, rhs, indices, lhs_idx, rhs_idx);
```

## Common mistakes

- Passing value tiles and index tiles with different valid regions.
- Ignoring tie behavior when equal values are common.
- Updating values without carrying indexes through the same comparison.

## Used by kernels

- Useful in argmin reductions and ranking kernels; no direct repository kernel use was found in assigned sources.
