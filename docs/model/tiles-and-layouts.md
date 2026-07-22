# Tiles, Layouts, and Distribution

Source code describes logical tiles. A distribution states how a logical tile
maps to local thread fragments or to a shared tile value.

## Logical and Local Shape

The full shape is the logical shape used by the operation. The local shape is
the fragment shape visible to one thread when the tile is distributed.

```text
full_shape(A)  = [kM, kK]
distribution  = RowShard<kThreadsPerBlock>
local_shape(A) = [kM / kThreadsPerBlock, kK]
```

Runtime activity does not change the full shape or the per-thread fragment
shape. Tail handling belongs in the valid region or in explicit scalar code.

## Distribution Kinds

| Distribution | Meaning |
| --- | --- |
| `LinearShard<kThreadsPerBlock>` | Equal linear element ranges |
| `AxisShard<Dim, kThreadsPerBlock>` | Equal slices along one dimension |
| `RowShard<kThreadsPerBlock>` | Matrix rows split across threads |
| `Replicated<kThreadsPerBlock>` | Same value in each local tile |
| `Partial<Mask>` | Only selected shared regions are defined |
| `Shared` | One block-local shared tile version |

Elementwise operations normally preserve distribution. Reshape and transpose
preserve ownership only when the element mapping does not move data between
thread fragments.

## Minimum Fragment Bytes

Each local fragment must carry at least
`kMinimumThreadFragmentBytes` for the current profile. For distributed tiles,
check the fragment size, not only the full tile size:

```cpp
static_assert(kLocalRows * kCols * sizeof(Element) >=
              kMinimumThreadFragmentBytes);
```

When a tail has fewer valid elements, keep the tile storage padded and describe
the valid subset separately.

## Layout

Layout describes how logical elements map to bytes. Common forms include
row-major, column-major, and target-specific matrix layouts. Movement between
local and shared tiles is byte preserving unless an operation explicitly
converts layout.

## Global, Local, and Shared Tiles

Global tensors are memory views. Local tiles are per-thread values. Shared
tiles are block-local values used for exchange and grouped operations.

```cpp
TLOAD(local_a, global_a_slice);       // global -> local
TMOV<SharedMove::Insert>(shared_x, local_a);
TMOV<SharedMove::Extract>(local_b, shared_x);
TSTORE(global_out_slice, local_b);    // local -> global
```

Use shared tiles for register-sized exchange. Use global memory for persistent
kernel inputs and outputs.

## Matrix Distribution

For shared-right matrix multiplication:

- left operand: row-sharded local tile;
- right operand: fully defined shared tile;
- result operand: row-sharded local tile.

Each thread computes its own result fragment:

```text
C.fragment[thread] = A.fragment[thread] x SharedB
```

The operation does not imply split-K accumulation. Use an explicit reduction
pattern when multiple fragments contribute to the same output element.
