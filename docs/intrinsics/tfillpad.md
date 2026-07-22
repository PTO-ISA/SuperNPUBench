# TFILLPAD: Fill invalid tile padding

## Purpose / When to use

Use `TFILLPAD` after a tile operation leaves padding lanes that must contain a defined pad value before a later store, compare, or reduction.

## C++ Declaration

```cpp
template <typename TileData, PadValue PadVal = PadValue::Zero>
PTO_INST void TFILLPAD(TileData &dst, TileData &src);

template <typename DstTileData, typename SrcTileData>
PTO_INST void TFILLPAD(DstTileData &dst, SrcTileData &src);
```

## Parameters

| Parameter | Meaning |
| --- | --- |
| `dst` | Destination tile with padding policy. |
| `src` | Source tile whose valid elements are copied. |
| `PadVal` | Optional compile-time pad value selector. |

## Operation / Semantics

- Copies valid elements from `src` to `dst`.
- Fills destination padding lanes according to `PadVal` or `DstTileData::PadVal`.
- Does not change the logical valid region.

## Constraints

- Destination pad value must not be `PadValue::Null` when padding is filled.
- Source and destination element sizes must match and are limited to supported byte widths.
- Source and destination shapes must match for the same-tile overload.
- Matrix-tile padding support is limited to implementation-supported layouts and pad values.

## Example

```cpp
TFILLPAD<PadValue::Zero>(dst, src);
```

## Common mistakes

- Using `TEXPANDS` to initialize padding instead of using the pad-aware operation.
- Expecting dtype conversion while filling padding.
- Forgetting that valid elements are copied from `src`, not recomputed.

## Used by kernels

- Used as a support operation for kernels that carry partially valid tiles; no direct benchmark kernel use was found.
