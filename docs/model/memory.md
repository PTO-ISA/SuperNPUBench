# Memory and Tile Movement

The model has three tile-related storage spaces: global tensor views, local
tiles, and shared tiles. Movement operations define or consume versions across
those spaces.

## Global Tensor Views

A global tensor view names external memory plus element type, shape, stride,
layout, and valid region. A view is not loaded until a tile operation consumes
it.

```cpp
auto rows = partition_rows(total_rows, block, block_count,
                           thread, kThreadsPerBlock);
auto input_slice = input.slice_rows(rows.begin, rows.count);
```

Views used by different blocks or threads should be non-overlapping unless the
kernel deliberately implements a reduction or scatter pattern with defined
conflict behavior.

## `TLOAD`

`TLOAD` reads a global tensor view and defines a tile version.

```cpp
TLOAD(local_tile, input_slice);   // global -> local
TLOAD(shared_tile, rhs_view);     // global -> shared, selected thread
```

A local load defines one local version. A shared load defines one shared
version. Padding bytes are not valid inputs unless the operation defines a
padding behavior.

## `TSTORE`

`TSTORE` consumes a tile version and writes a global tensor view.

```cpp
TSTORE(output_slice, local_tile);
TSTORE(rhs_copy, shared_tile);
```

A store proves that the source payload has been accepted by the target memory
contract. It does not by itself create a general cross-block synchronization
point.

## `TMOV`

`TMOV` moves bytes among compatible local and shared tile classes:

```cpp
TMOV<SharedMove::Insert>(shared_parts, local_part);
TMOV<SharedMove::Broadcast>(local_full, shared_full);
```

`TMOV` preserves bytes. It does not reinterpret element type, convert layout,
or resize a tile.

## Partitioning

Use `get_block_idx()` for the block slice and `get_thread_idx()` for the
thread fragment:

```cpp
const uint32_t block = get_block_idx();
const uint32_t thread = get_thread_idx();

RowSlice rows = partition_rows(total_rows, block, block_count, thread);
TLOAD(local_input, input.slice_rows(rows.begin, rows.count));
```

The launch configuration supplies `block_count` and global shape. The kernel
must keep computed ranges inside the tensor's valid region.

## Reordering

The compiler may reorder non-conflicting operations when tile-version,
global-memory, and grouped-operation rules permit it. Programs must avoid data
races between blocks and overlapping writes between threads unless the selected
operation defines the conflict behavior.

## Cross-Thread and Cross-Block Data

Use shared tile versions for register-sized data exchange inside one block.
Use global memory for persistent inputs and outputs. Communication between
blocks belongs to the runtime or launch protocol rather than the tile dataflow.
