# TCVT: Convert tile element type

## Purpose / When to use

Use `TCVT` to materialize a tile in another dtype before arithmetic or store paths that require matching element types.

## C++ Declaration

```cpp
template <typename TileDataD, typename TileDataS>
PTO_INST void TCVT(TileDataD &dst, TileDataS &src, RoundMode mode,
                   SaturationMode satMode);

template <typename TileDataD, typename TileDataS>
PTO_INST void TCVT(TileDataD &dst, TileDataS &src, RoundMode mode);
```

## Parameters

| Parameter | Meaning |
| --- | --- |
| `dst` | Destination tile with the desired dtype. |
| `src` | Source tile. |
| `mode` | Rounding mode for conversions that need rounding. |
| `satMode` | Saturation mode for conversions that can overflow. |

## Operation / Semantics

- Converts each valid source element to the destination dtype.
- Rounding and saturation are controlled by the selected modes when applicable.
- The destination valid region follows the converted source region.

## Constraints

- Source and destination shapes or effective valid regions must be compatible.
- The source-to-destination dtype pair must be supported.
- Use explicit rounding and saturation modes when relying on non-default behavior.
- No implicit global-memory access occurs; combine with `TLOAD` and `TSTORE` as needed.

## Example

```cpp
TCVT(fp32_tile, bf16_tile, RoundMode::RNE);
```

## Common mistakes

- Assuming `TSTORE` will perform vector dtype conversion automatically.
- Ignoring saturation behavior when narrowing values.
- Using `TCVT` as a layout transform; use `TRESHAPE`, `TTRANS`, or load/store layout forms as appropriate.

## Used by kernels

- Vector microbenchmarks include fp16 and fp32 conversion cases.
- DeepSeek quantization, dequantization workarounds, and reductions use `TCVT` for explicit precision changes.
