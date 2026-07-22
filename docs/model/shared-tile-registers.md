# Shared Tile Operations

Shared tiles carry block-local tile data between participating threads without
exposing peer-local physical registers or addressable scratchpad memory. A
shared tile is a versioned value, not a pointer.

## Declare Local and Shared Values

Use the type name to make ownership visible at the declaration:

```cpp title="docs/examples/shared_tile_types.cpp" linenums="1"
--8<-- "docs/examples/shared_tile_types.cpp"
```

The shape of `LocalRows` is one thread's region. The shape of `SharedMatrix` is
the logical value visible to the block. Neither declaration
names a physical register. Use `LocalTile` for private intermediate values and
`SharedTile` only when another thread consumes or reuses the value.

## IDs and Versions

Shared architectural IDs name logical shared tiles. Every definition creates a
new physical version:

```text
source definition S#17.v0 -> physical version p31
new definition    S#17.v1 -> physical version p44
```

A consumer binds to one version during rename. A later definition of the same
shared ID cannot change an older consumer's source.

## Fixed Regions

A shared version has one fixed region per participating thread slot:

```text
region i <-> thread i, for 0 <= i < kThreadsPerBlock
```

The mapping never compresses when fewer threads produce data. If only a subset
of regions is defined, the undefined regions remain unavailable to consumers
that require a complete tile.

## Defined and Ready State

`defined_region_mask` is immutable metadata for a shared version. It states
which regions contain valid payload. The implementation tracks readiness as
producers finish. A shared version publishes atomically after every defined
region is ready.

Source code cannot inspect or modify internal readiness state. It can only
create and consume tile versions.

## Lifetime

Physical allocation, version rename, and reclamation are automatic. An old
version remains live while any bound consumer may still read it. Source code
does not allocate, push, pop, free, or mutate a shared tile version in place.

## Shared `TLOAD`

A shared `TLOAD` reads a global tensor view and defines a fully populated
shared version. Exactly one statically identifiable thread should issue the
load for that shared destination.

```cpp
SharedTile<half, kK, kN> shared_b;

if (thread == 0) {
  TLOAD(shared_b, global_b);
}
```

Consumers can use `shared_b` only after the full transfer has defined the
shared version.

## Shared `TSTORE`

A shared `TSTORE` consumes a shared version and writes it to a global tensor
view. A selected thread may store a fully defined shared value to one global
range, or a partitioned store may write defined regions to non-overlapping
thread-provided ranges when the operation supports that form.

```cpp
if (thread == 0) {
  TSTORE(global_b_copy, shared_b);
}
```

Do not use shared stores as a cross-block synchronization mechanism. Global
memory handoff belongs to the runtime or launch protocol.

## `TMOV`

`TMOV` moves bytes between local and shared tile classes. It does not convert
element type, shape, or layout.

| Mode | Destination | Source | Result |
| --- | --- | --- | --- |
| `Insert` | Shared | Local fragment | Current thread defines its fixed region |
| `Publish` | Shared | One local full tile | Selected thread defines all regions |
| `Extract` | Local fragment | Shared | Current thread reads its fixed region |
| `Broadcast` | Local full tile | Shared | Current thread copies the full shared value |

```cpp
TMOV<SharedMove::Insert>(shared_parts, local_partial);
TMOV<SharedMove::Extract>(local_copy, shared_parts);
```

Every shared destination defines a new shared version.

## Shared-Right `TMATMUL`

The `TMATMUL` family uses a local left matrix and a fully defined shared right
matrix:

```cpp
TMATMUL(local_c, local_a, shared_b);
```

Each thread computes its own row-sharded result fragment. The right operand
must be complete before the operation can consume it.

For a reduction-dimension loop, each shared load defines the right panel for
that iteration:

```cpp
for (int k = 0; k < kKTiles; ++k) {
  if (thread == 0) {
    TLOAD(shared_b, rhs_tiles(k, output_column));
  }

  TLOAD(local_a, lhs_tiles(local_row, k));
  if (k == 0) {
    TMATMUL(local_accum, local_a, shared_b);
  } else {
    TMATMUL_ACC(local_accum, local_accum, local_a, shared_b);
  }
}
```

The next shared load does not overwrite a value already bound by an older
matrix operation; it creates the next shared tile version.

## Exchange a Two-Dimensional Fragment

Each thread can publish one row region, then read the complete shared matrix:

```cpp
LocalTile<float, kRowsPerThread, kCols> my_rows;
SharedTile<float, kRowsPerThread * kThreadsPerBlock, kCols> shared_rows;
LocalTile<float, kRowsPerThread * kThreadsPerBlock, kCols> all_rows;

TLOAD(my_rows, input_rows_for(thread));
TMOV<SharedMove::Insert>(shared_rows, my_rows);
TMOV<SharedMove::Broadcast>(all_rows, shared_rows);
```

This is useful for small transpose, halo, and cross-thread reduction stages.
If every thread needs only its own region, keep the value local instead.

## Illegal Uses

- Reading a shared version before all defined regions are ready.
- Using a partial shared version as the right operand of `TMATMUL`.
- Treating a shared ID as an address or physical register ID.
- Depending on thread arrival time to select the latest version.
- Mutating a published version in place.
