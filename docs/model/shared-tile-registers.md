# Versioned Shared Tile Registers

## Purpose

A shared tile register carries tile data between the four PEs without exposing
peer-local physical registers or using addressable scratchpad memory. It is a
core-local register class with true definition/use semantics.

Compiler support is not required yet. The contracts on this page define the
architecture-level behavior that compiler lowering must eventually implement.

## Namespace and Versions

Shared architectural IDs are 8-bit values `S[0]` through `S[255]`. Every
definition creates a new physical version:

```text
source definition S#17.v0 -> physical version p31
new definition    S#17.v1 -> physical version p44
```

A consumer binds to a specific version during rename. A later definition of
the same architectural ID cannot change an older consumer's source.

## Four Fixed Regions

Every shared version has four equal regions:

```text
region 0 <-> PE0
region 1 <-> PE1
region 2 <-> PE2
region 3 <-> PE3
```

The region mapping never compresses when fewer PEs produce data.

## Defined and Ready Masks

`defined_region_mask` is immutable metadata for a version. It states which
regions contain valid payload. Hardware tracks readiness internally as
producers finish. A shared version publishes atomically after every defined
region is ready.

Source code cannot inspect or modify the internal readiness state.

## Lifetime

Physical allocation, version rename, and reclamation are automatic. An old
version remains live while any bound consumer may still read it. Source code
does not allocate, push, pop, or free a shared tile version.

## `TMOV` Forms

`TMOV` is byte preserving and provides four local/shared modes.

| Mode | Destination | Source | Result |
| --- | --- | --- | --- |
| `Insert` | Shared | PE-local quarter | Current PE defines its fixed region |
| `Publish` | Shared | One PE-local full tile | Selected PE defines all regions |
| `Extract` | PE-local quarter | Shared | PE reads its fixed region |
| `Broadcast` | PE-local full tile | Shared | Participating PE copies full shared value |

```cpp
TMOV<SharedMove::Insert>(shared_parts, local_partial);
TMOV<SharedMove::Extract>(local_copy, shared_parts);
```

Changing movement mode does not convert element type, shape, or layout.

## Shared `TLOAD`

Exactly one statically identifiable PE issues a full global-memory load into a
new shared version:

```cpp
SharedTile<half, K, N> shared_b;
if (get_thread_idx() == 0) {
  TLOAD(shared_b, global_b);
}
```

The definition is fully defined (`1111`) and becomes consumable only when the
full transfer completes.

## Shared `TSTORE`

A selected PE may store a fully defined shared value to one global region, or
the group may store non-overlapping fixed quarters from a partial shared value.

```cpp
if (get_thread_idx() == 0) {
  TSTORE(global_b_copy, shared_b);
}
```

## Shared-Right Matrix Multiply

The `TMATMUL` family requires the right matrix to be a fully defined shared
tile version:

```cpp
TMATMUL(local_c, local_a, shared_b);
```

The shared operand is delivered to all four matrix engines. Each PE computes
its own M-axis result quarter.

## Illegal Uses

- Reading a shared version before all defined regions are ready.
- Using a partial shared version as matrix RHS.
- Treating a shared ID as an address or physical register ID.
- Depending on PE arrival time to select the latest version.
- Mutating a published version in place.
