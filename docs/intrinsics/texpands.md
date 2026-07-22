# TEXPANDS: Broadcast a scalar into a tile

## Purpose / When to use

Use `TEXPANDS` to initialize or fill a tile valid region with one scalar value.

## C++ Declaration

```cpp
template <typename TileData>
PTO_INST void TEXPANDS(TileData &dst, typename TileData::DType scalar);
```

## Parameters

| Parameter | Meaning |
| --- | --- |
| `dst` | Destination tile. |
| `scalar` | Scalar value converted to `TileData::DType` and written to every valid element. |

## Operation / Semantics

- Writes `scalar` to each element in the destination valid region.
- Elements outside the valid region are not part of the operation contract.
- The destination valid shape is preserved.

## Constraints

- Destination tile location, dtype, and shape must be supported by the implementation.
- The scalar must be representable by `TileData::DType` or follow the language conversion used by the intrinsic wrapper.
- Valid rows and columns must describe a non-empty region.

## Example

```cpp
TEXPANDS(acc, 0.0f);
```

## Common mistakes

- Using it to clear padding bytes outside the valid region; use `TFILLPAD` for padding semantics.
- Passing a scalar with unintended narrowing conversion.
- Forgetting to set the destination valid shape before broadcasting into a partially valid tile.

## Used by kernels

- Vector microbenchmarks use `TEXPANDS` scalar-broadcast cases.
- DeepSeek reduction and mapping kernels use `TEXPANDS` to initialize accumulators and fill constant tiles.
