# TTRI: Build a triangular mask tile

## Purpose / When to use

Use `TTRI` to create lower- or upper-triangular masks for tiled algorithms such as causal attention or triangular matrix work.

## C++ Declaration

```cpp
template <typename TileData, int isUpperOrLower>
PTO_INST void TTRI(TileData &dst, int diagonal);
```

## Parameters

| Parameter | Meaning |
| --- | --- |
| `dst` | Destination tile that receives the mask values. |
| `diagonal` | Diagonal offset used when deciding whether an element is inside the triangle. |
| `isUpperOrLower` | Template flag; `0` selects lower-triangular, `1` selects upper-triangular behavior. |

## Operation / Semantics

- For each valid element, writes the implementation-defined in-mask value when the element is on the selected side of `diagonal`.
- Writes the out-of-mask value elsewhere.
- The row and column coordinates are evaluated within the destination tile valid region.

## Constraints

- `isUpperOrLower` must be `0` or `1`.
- Some implementations require row-major destination tiles.
- The destination dtype must support the mask values expected by the consuming operation.

## Example

```cpp
TTRI<TileT, 0>(mask, 0);
```

## Common mistakes

- Treating `diagonal` as a byte offset.
- Using lower/upper template values inconsistently with the consumer comparison.
- Expecting padding elements outside the valid region to be initialized.

## Used by kernels

- Useful for attention-style kernels that need causal or triangular masks; no direct repository kernel use was found in the assigned source set.
