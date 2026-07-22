# TQUANT: Quantize tile values

## Purpose / When to use

Use `TQUANT` to convert higher-precision tile values into a quantized tile representation using scale, exponent, max, or offset metadata.

## C++ Declaration

```cpp
template <auto quant_type, typename TileDataOut, typename TileDataSrc,
          typename TileDataExp, typename TileDataMax>
PTO_INST void TQUANT(TileDataOut &dst, TileDataSrc &src, TileDataExp *exp,
                     TileDataMax *max, TileDataSrc *scaling);

template <auto quant_type, auto store_mode, typename TileDataOut,
          typename TileDataSrc, typename TileDataExp, typename TileDataMax,
          typename TileDataIdx>
PTO_INST void TQUANT(TileDataOut &dst, TileDataSrc &src, TileDataExp *exp,
                     TileDataMax *max, TileDataSrc *scaling,
                     TileDataExp *exp_zz, TileDataIdx *vgather_idx);

template <auto quant_type, typename TileDataOut, typename TileDataSrc,
          typename TileDataPara>
PTO_INST void TQUANT(TileDataOut &dst, TileDataSrc &src, TileDataPara &scale,
                     TileDataPara *offset = nullptr);
```

## Parameters

| Parameter | Meaning |
| --- | --- |
| `dst` | Quantized destination tile. |
| `src` | Source tile to quantize. |
| `exp` | Optional exponent metadata tile pointer. |
| `max` | Optional maximum-value metadata tile pointer. |
| `scaling` | Optional scaling tile pointer. |
| `scale` | Scale tile for parameterized forms. |
| `offset` | Optional offset tile pointer. |
| `exp_zz` | Optional packed exponent metadata for store-mode variants. |
| `vgather_idx` | Optional vector gather index tile for store-mode variants. |

## Operation / Semantics

- Computes quantized output from `src` according to `quant_type`.
- Metadata operands provide scale, exponent, maximum, offset, or gather information depending on the selected overload.
- Output dtype, rounding, and clipping behavior are selected by the quantization mode.

## Constraints

- The quantization mode determines legal source and destination dtypes.
- Pointer metadata operands must be non-null when required by the selected mode.
- Scale/offset shapes must match the quantization granularity used by the kernel.
- Use explicit conversion sequences when the backend does not expose the requested quantization mode.

## Example

```cpp
TQUANT<QuantType::PerTensor>(q, fp, scale);
```

## Common mistakes

- Mixing per-token and per-channel scale shapes.
- Passing optional metadata pointers that are null for a mode that requires them.
- Assuming all FP8/FP4 formats use the same exponent and scale layout.

## Used by kernels

- DeepSeek quantization kernels document explicit quantization sequences and fallback paths.
- Microbenchmarks primarily cover lower-level `TCVT` and broadcast operations used to build quantization flows.
