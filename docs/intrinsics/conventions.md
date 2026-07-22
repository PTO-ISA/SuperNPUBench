# Operand and tile conventions

The intrinsic API extends ordinary C++ with typed tile values. Kernel code keeps
loops, branches, templates, and address calculations in C++; an intrinsic
expresses one data-parallel operation over a tile's valid region.

## Destination-first calls

Most calls name the destination before their sources:

```cpp
TADD(dst, lhs, rhs);
TSTORE(output_view, dst);
```

The destination is a new logical tile value. It may reuse C++ storage syntax,
but consumers depend on the newly defined tile version rather than on an
instruction-completion token.

## Tile types

A tile type fixes its ownership, element type, physical rows and columns,
layout, and optionally a smaller valid region:

```cpp
#include <pto_kernel/tile.hpp>

using InputTile = pto::LocalTile<float, 16, 16>;
```

Two tiles with equal byte size are not interchangeable when their element type,
shape, layout, or location differs. Intrinsic pages state any additional
relationships required between source and destination types.

## Valid regions

The valid region identifies elements that belong to the logical tensor. Padding
outside that region is storage, not input data. Arithmetic, reductions, loads,
and stores operate only on valid elements unless an intrinsic explicitly defines
padding behavior.

For migrated kernels, every thread-local fragment occupies at least 128 bytes.
Tail processing uses a valid region inside that allocation instead of creating a
smaller physical fragment.

## Global tensor views

Global memory is described through a typed view rather than pointer arithmetic in
the tile data path:

```cpp
using Shape = RowMajor<Rows, Cols>;
using Tensor = global_tensor<float, Shape>;
using Iterator = global_iterator<Tensor, InputTile>;
```

An iterator selects a tensor region. `TLOAD` transfers that region into a tile;
`TSTORE` writes a tile's valid region back through a destination view.

## Thread-local and shared tiles

Thread-local tiles are visible only to their owning thread. A shared tile is a
block-scoped, versioned value used to exchange or reuse tile data across the
threads mapped to one core. Shared forms of `TLOAD`, `TSTORE`, and `TMOV` define
or consume shared versions. Matrix operations consume a fully defined shared
right-hand operand when the group form is used.

## Aliasing

Do not assume that an intrinsic permits a destination to alias one of its sources.
Use a distinct destination unless the operation page explicitly defines an
in-place form. Global views written by different threads must be disjoint unless
the operation explicitly provides an atomic update.

## Data-type notation

Reference tables use compact names such as `F32`, `F16`, `BF16`, `S32`, and
`U32`. C++ examples use the corresponding public element types supplied by the
tile headers. Supported combinations are part of each operation's contract;
ordinary C++ implicit conversion does not widen an intrinsic's accepted types.

## Dependency rule

Write dependencies through C++ values:

```cpp
TLOAD(a, input_a);
TLOAD(b, input_b);
TADD(sum, a, b);
TMUL(out, sum, scale);
```

The implementation may issue independent operations out of source order, but it
cannot consume a tile version before the corresponding producer makes that
version ready.
