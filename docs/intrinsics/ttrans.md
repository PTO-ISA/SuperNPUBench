# TTRANS: Transpose a tile

## Purpose / When to use

Use `TTRANS` to swap rows and columns of a tile without going through global memory.

## C++ Declaration

```cpp
template <typename TileDataDst, typename TileDataSrc, typename TileDataTmp>
PTO_INST void TTRANS(TileDataDst &dst, TileDataSrc &src, TileDataTmp &tmp);
```

## Parameters

| Parameter | Meaning |
| --- | --- |
| `dst` | Destination tile with transposed shape. |
| `src` | Source tile. |
| `tmp` | Temporary tile required by the implementation. |

## Operation / Semantics

- Writes `dst[j,i] = src[i,j]` over the source valid region.
- Destination valid rows become source valid columns; destination valid columns become source valid rows.
- The temporary tile carries implementation scratch state and has no user-visible result.

## Constraints

- Destination shape must be compatible with the transposed source shape.
- Source, destination, and temporary dtypes/layouts must be supported together.
- Temporary tile sizing must follow the selected tile shape requirements.
- Convolution-tile sources have additional implementation-specific restrictions.

## Example

```cpp
TTRANS(dst, src, tmp);
```

## Common mistakes

- Forgetting that destination rows and columns are swapped.
- Reusing an unrelated temporary tile with incompatible shape.
- Using `TMOV` or `TRESHAPE` when a mathematical transpose is required.

## Used by kernels

- DeepSeek transpose migration notes use `TLOAD`, `TTRANS`, then `TSTORE`.
- Tile transform microbenchmarks and docs classify `TTRANS` as a register-local transform.
