# Tiles, Layouts, and Distribution

## One Logical Tile

Source code describes complete logical tiles. A distribution states how a
logical value maps to the four PEs or to a shared tile register.

```cpp
Tile<half, 64, 64> logical_a;
```

With M-axis sharding, each PE owns a `16 x 64` local quarter.

## Distribution Kinds

| Distribution | Meaning |
| --- | --- |
| `LinearShard4` | Four equal linear element ranges |
| `AxisShard4<D>` | Four equal slices along dimension `D` |
| `MShard4` | Matrix rows split across four PEs |
| `Replicated4` | Same value in every PE-local tile |
| `Partial<Mask>` | Only the selected PE regions are defined |
| `Shared` | One core-local shared tile version |

Elementwise operations normally preserve distribution. Reshape and transpose
preserve ownership only when the mapping does not move elements between PE
regions.

## Full and Local Shapes

The type's full shape describes the logical tile. The local shape is derived
from distribution:

```text
full_shape(A)  = [64, 64]
distribution  = MShard4
local_shape(A) = [16, 64]
```

Runtime PE activity never changes the full shape or quarter size.

## Valid Regions

The valid region identifies logical elements for the current operation. Padded
storage may be larger. Unless an intrinsic defines padding behavior, an
implementation must not treat padding as valid input or visible output.

## Layouts

Layout describes element-to-byte mapping. Common forms include row-major,
column-major, and target matrix fractal layouts. A movement between local and
shared tile registers is byte preserving; it does not silently convert layout.

## Matrix Distribution

For group matrix multiplication:

- `A` is `MShard4` in PE-local tile registers;
- `B` is one fully defined shared tile register;
- `C` is `MShard4` in PE-local accumulator or destination registers.

Each PE computes:

```text
C.quarter[pe] = A.quarter[pe] x SharedB
```

The model does not use split-K for this operation.

## Partial Values

A partial shared value may be moved or stored by defined region. It cannot be
used as the right operand of group matrix multiplication. A new definition is
required to change the defined-region mask; a published version is immutable.
