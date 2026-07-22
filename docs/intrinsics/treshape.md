# TRESHAPE: Reinterpret tile shape with equal storage size

## Purpose / When to use

Use `TRESHAPE` when the same tile bytes need a different tile shape without mathematical transposition or dtype conversion.

## C++ Declaration

```cpp
template <typename TileDataOut, typename TileDataIn>
PTO_INST void TRESHAPE(TileDataOut &dst, TileDataIn &src);
```

## Parameters

| Parameter | Meaning |
| --- | --- |
| `dst` | Destination tile with the new shape. |
| `src` | Source tile. |

## Operation / Semantics

- Copies source storage into destination storage byte-for-byte.
- Preserves data order; only the programmer-visible tile shape changes.
- Destination valid shape is derived from the source valid shape where the implementation supports it.

## Constraints

- Source and destination tile locations must match.
- Total byte size of source and destination tiles must match.
- Boxed and non-boxed layout forms are not converted.
- Use `TTRANS` for mathematical transpose and `TCVT` for dtype conversion.

## Example

```cpp
TRESHAPE(reshaped, flat_tile);
```

## Common mistakes

- Using it to transpose rows and columns.
- Changing total element byte count.
- Expecting layout conversion between boxed and non-boxed storage.

## Used by kernels

- Reduction kernels use `TRESHAPE` to reinterpret temporary tiles before row/column reductions.
- Vector helper code exposes `TRESHAPE` as a unary tile transform.
