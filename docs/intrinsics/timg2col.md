# TIMG2COL: Materialize image windows as columns

## Purpose / When to use

Use `TIMG2COL` in convolution lowering when an image-like source tile must be presented as column-major patches for matrix-style compute.

## C++ Declaration

```cpp
template <typename TileData, typename ConvTileData>
PTO_INST void TIMG2COL(TileData &dst, ConvTileData &src,
                       uint16_t posM = 0, uint16_t posK = 0);
```

## Parameters

| Parameter | Meaning |
| --- | --- |
| `dst` | Destination tile containing the lowered patch data. |
| `src` | Convolution tile source. |
| `posM` | Patch row/block position. |
| `posK` | Patch channel/kernel position. |

## Operation / Semantics

- Reads a patch window from `src` selected by `posM` and `posK`.
- Writes the flattened patch representation into `dst`.
- The exact patch traversal follows the convolution tile descriptor.

## Constraints

- `src` must be a convolution tile type accepted by the implementation.
- `dst` shape must match the lowered patch extent.
- `posM` and `posK` are element/block positions, not bytes.
- Use regular `TLOAD` for non-convolution global-to-tile movement.

## Example

```cpp
TIMG2COL(patch_tile, conv_tile, pos_m, pos_k);
```

## Common mistakes

- Using `TIMG2COL` as a general reshape.
- Passing ordinary tile sources instead of convolution tile descriptors.
- Mismatching destination shape with kernel/window dimensions.

## Used by kernels

- Convolution-style kernels can use this intrinsic before matrix operations; no direct repository kernel use was found in assigned sources.
