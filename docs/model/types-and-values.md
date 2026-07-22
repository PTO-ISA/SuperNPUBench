# Types and Values

## Scalars

Scalars include integer, floating-point, pointer, shape, stride, and mode
values. Scalars are PE-local unless they are compile-time constants. The two
identity intrinsics return `uint32_t` values.

## Global Tensors

A global tensor carries:

- base address;
- element type;
- logical shape;
- byte or element strides;
- layout and padding rules;
- valid region.

`get_block_idx()` commonly selects a disjoint tensor slice before `TLOAD`.

## PE-Local Tiles

A PE-local tile is directly visible only to its owning PE. Its static type
includes element type, logical shape, layout, location, and distribution.

```cpp
using AQuarter = PETile<half, M / 4, K, Layout::RowMajor>;
```

The name denotes a logical value. The compiler assigns a tile ID and creates a
new version for each definition.

## Shared Tiles

A shared tile register is visible to all four PEs in one core and has four
fixed regions. Its descriptor includes:

- 8-bit architectural shared tile ID;
- logical byte size;
- element type, shape, and layout;
- defined-region mask;
- physical version tag and readiness state.

```cpp
using SharedB = SharedTile<half, K, N, Layout::ColMajor>;
```

Shared tiles are not pointers. They cannot be indexed by address, cast to a
global tensor, or manually freed.

## Tile Sizes

The group model supports logical tile sizes:

```text
512 B, 1 KB, 2 KB, 4 KB, 8 KB, 16 KB, 32 KB
```

A distributed logical tile uses one quarter of its payload in each PE. A
shared tile version retains the full logical size and fixed four-region
mapping.

## Definedness

Definedness is tracked per PE region. A value with only PE0 and PE1 regions
defined has mask `1100` using the manual's `{PE0, PE1, PE2, PE3}` display
order. A consumer that requires a complete tile may read only mask `1111`.

## Type Compatibility

Two tile operands are compatible only when the intrinsic permits their:

- element types and accumulator promotion;
- shapes and valid regions;
- layouts and locations;
- PE distributions;
- local/shared register classes.

Equal byte size alone does not make two tile types interchangeable.
