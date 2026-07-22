# Types and Values

The C++ model has scalar values, global tensor views, local tiles, and shared
tiles. Treat all tiles as typed values. Do not treat tile IDs as pointers or
physical register numbers.

## Named Constants

Examples use these names for profile-shaped quantities:

```cpp
static constexpr uint32_t kThreadsPerBlock = /* profile value */;
static constexpr uint32_t kMinimumThreadFragmentBytes = /* profile value */;
```

The mapping page defines why those values hold for the current profile.

## Scalars

Scalars include integers, floating-point values, pointers, shapes, strides,
layout tags, and mode tags. Scalars follow ordinary C++ control flow. When a
scalar is derived from `get_thread_idx()`, its value may differ by thread.

## Global Tensors

A global tensor is a typed view of external memory. Its descriptor carries:

- base address;
- element type;
- logical shape;
- byte or element strides;
- layout and padding rules;
- valid region.

`TLOAD` reads a global tensor view and defines a tile. `TSTORE` consumes a tile
and writes through a global tensor view. The view must describe the intended
logical elements; padding bytes are not valid output unless the operation
explicitly says so.

## Local Tiles

A local tile is visible only to the current thread. Its static type carries
element type, shape, layout, location, and distribution.

```cpp
using AFragment = LocalTile<half, kRowsPerThread, kK, Layout::RowMajor>;
```

The C++ variable denotes a logical tile value. The compiler assigns a tile ID
and creates a new version each time an operation defines that value.

## Shared Tiles

A shared tile is a block-local tile value visible to participating threads.
Its descriptor carries:

- architectural shared tile ID;
- logical byte size;
- element type, shape, and layout;
- defined-region mask;
- physical version tag and readiness state managed by the implementation.

```cpp
using SharedRight = SharedTile<half, kK, kN, Layout::ColMajor>;
```

Shared tiles are not addressable memory. They cannot be indexed by pointer,
cast to global tensors, or manually freed.

## Tile Sizes and Fragments

A full logical tile may be distributed into per-thread fragments or held as one
shared tile value. The current minimum local fragment is
`kMinimumThreadFragmentBytes`. If a distributed tile has one fragment per
thread, its minimum block-wide payload is:

```cpp
static constexpr uint32_t kMinimumBlockTileBytes =
    kThreadsPerBlock * kMinimumThreadFragmentBytes;
```

The implementation may support larger tile sizes such as `512 B`, `1 KB`,
`2 KB`, `4 KB`, `8 KB`, `16 KB`, and `32 KB`. Element type, shape, and layout
still decide whether two operands are compatible; equal byte size alone is not
enough.

## Definedness

Definedness is tracked per shared-tile region. A partial shared tile may have
only a subset of regions defined. A consumer that requires a complete shared
tile, such as shared-right matrix multiplication, may read only a fully defined
shared version.

Published shared tile versions are immutable. To change the defined-region
mask or payload, define a new version.

## Type Compatibility

Two tile operands are compatible only when the operation permits their:

- element types and accumulator promotion;
- full shape and valid region;
- layout and location;
- local or shared tile class;
- distribution across participating threads.
