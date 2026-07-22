# Global Memory and Tile Movement

## Memory Spaces

The one-level profile has three programmer-visible value locations:

1. global memory through `GlobalTensor` descriptors;
2. PE-local tile registers;
3. versioned shared tile registers within one core.

Shared tile registers are not a memory space. They have no address and cannot
alias global memory.

## PE-Local `TLOAD` and `TSTORE`

```cpp
TLOAD(local_tile, global_slice);
TSTORE(global_slice, local_tile);
```

The transfer uses the calling PE's global pointer and local tile version.
`get_block_idx()` and `get_thread_idx()` may select disjoint tensor offsets.

## Shared-Destination `TLOAD`

```cpp
if (get_thread_idx() == 0) {
  TLOAD(shared_tile, global_full_tile);
}
```

The issuing PE transfers the full logical shared payload. The compiler must
prove exactly one issuer for that dynamic definition. The new shared version is
fully defined.

## Shared-Source `TSTORE`

```cpp
if (get_thread_idx() == 0) {
  TSTORE(global_full_tile, shared_tile);
}
```

For a partial shared value, each defined PE region may be stored to a
non-overlapping global range. The source version itself is not modified.

## `TMOV`

`TMOV` moves bytes among compatible tile-register classes. Local-to-shared
movement defines a new shared version. Shared-to-local movement defines a new
local version. It does not change datatype or layout.

## Tensor Partitioning by Core and PE

For a tensor split along rows:

```cpp
const uint32_t core = get_block_idx();
const uint32_t pe = get_thread_idx();

const uint32_t core_row = core * rows_per_core;
const uint32_t pe_row = core_row + pe * rows_per_pe;
auto slice = input.slice_rows(pe_row, rows_per_pe);
TLOAD(local_input, slice);
```

The launch configuration supplies the number of cores and per-core work size.
The identity intrinsics do not infer tensor geometry.

## Global-Memory Conflicts

Two operations conflict when their byte ranges overlap and at least one writes.
The compiler may reorder non-conflicting operations subject to tile-version
dependencies. Programs must avoid data races between PEs and cores.

## Cross-PE Data

Use shared tile versions for register-sized cross-PE data. Global memory may be
used for larger or persistent data, but visibility across independently
executing cores requires a launch/runtime protocol outside the current PTO
intrinsic set.
