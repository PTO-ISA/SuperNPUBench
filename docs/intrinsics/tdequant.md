# TDEQUANT: Dequantize tile values

## Purpose / When to use

Use `TDEQUANT` when a backend exposes a direct integer or low-precision tile to floating-point tile conversion with scale and optional offset.

## C++ Declaration

```cpp
template <typename TileDataOut, typename TileDataSrc, typename TileDataScale>
PTO_INST void TDEQUANT(TileDataOut &dst, TileDataSrc &src,
                       TileDataScale &scale, TileDataScale *offset = nullptr);
```

## Parameters

| Parameter | Meaning |
| --- | --- |
| `dst` | Floating-point destination tile. |
| `src` | Quantized source tile. |
| `scale` | Scale tile or scalar-tile parameter. |
| `offset` | Optional zero-point or offset parameter. |

## Operation / Semantics

- Converts each valid source element to the destination dtype.
- Applies scale and, when supplied, offset according to the selected quantization format.
- The result valid region follows the source valid region.

## Constraints

- The current benchmark kernels often use explicit `TCVT` plus row/column broadcast multiply instead of direct `TDEQUANT`.
- Source format, scale format, and destination dtype must be supported together.
- Scale and offset shapes must match the quantization granularity.
- Treat this page as the C++ reference shape for implementations that expose the intrinsic.

## Example

```cpp
TDEQUANT(fp32_tile, fp8_tile, scale_tile);
```

## Common mistakes

- Assuming `TDEQUANT` is available in every toolchain configuration.
- Using a per-channel scale tile where the kernel expects per-token scale.
- Forgetting to apply offset when the quantized format uses a nonzero zero point.

## Used by kernels

- DeepSeek quantization docs mention `TDEQUANT` as unavailable in current paths and use explicit `TCVT` plus broadcast multiply instead.
