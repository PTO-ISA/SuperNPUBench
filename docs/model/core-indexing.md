# Core Indexing and Tensor Partitioning

## Core Identity

`get_block_idx()` returns the launch-defined core ID. All four PEs in one core
observe the same value.

```cpp
const uint32_t core = get_block_idx();
```

The intrinsic does not return the number of cores. Launch parameters or kernel
arguments provide the global work geometry.

## One-Dimensional Split

For `rows` divided across `core_count` cores:

```cpp
const uint32_t core = get_block_idx();
const uint32_t begin = rows * core / core_count;
const uint32_t end = rows * (core + 1) / core_count;
auto block_rows = input.slice_rows(begin, end - begin);
```

This quotient-based split handles non-divisible row counts without overlap.

## Core and PE Split

The core first selects a coarse slice. The PE then selects one quarter inside
that slice:

```cpp
const uint32_t core = get_block_idx();
const uint32_t pe = get_thread_idx();

const uint32_t core_begin = core * rows_per_core;
const uint32_t pe_begin = core_begin + pe * rows_per_pe;
auto local_rows = input.slice_rows(pe_begin, rows_per_pe);
TLOAD(local_tile, local_rows);
```

## Matrix Split

Grouped matrix multiplication normally partitions output M across cores, then
partitions each core's M slice across its four PEs. The right-hand B tile is
loaded once into a core-local shared tile version.

## Restrictions

- Different cores do not share tile-register IDs or versions.
- `get_block_idx()` does not establish memory visibility.
- Core slices must be disjoint unless atomic updates are intentional.
- Cross-core collective operations are outside the current one-level model.
