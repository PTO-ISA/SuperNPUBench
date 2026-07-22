# Tile Memory Semantics

This page states the ordering rules for global, local, and shared tile values.

## Architectural State

For each thread, the machine tracks scalar registers, a program counter, and a
map from local tile IDs to versions. For each block, it tracks a map from
shared tile IDs to versions.

An operation that writes local tile ID `t` in thread `p` creates:

```text
LocalVersion(block, p, t, sequence)
```

An operation that writes shared tile ID `s` in block `b` creates:

```text
SharedVersion(b, s, sequence, defined_region_mask)
```

Version identifiers are conceptual. Source code names C++ values, not physical
register versions.

## Binding Sources

A tile source binds to one local or shared version before issue. Issue requires
the bound source versions to be ready. Completion of a later write to the same
architectural ID cannot change the version already bound by an older consumer.

## Loads

A local `TLOAD` reads a global byte range and defines one local tile version. A
shared `TLOAD` is issued by one statically identifiable thread, reads the
global byte range once, and defines one fully populated shared version.

## Stores

A local `TSTORE` reads one local tile version and writes a global range. A
shared `TSTORE` reads one shared version. A full shared store writes the full
shared payload from a selected thread. A partition store writes defined fixed
regions to non-overlapping ranges when the operation supports that form.

## `TMOV`

`TMOV` between local and shared classes is byte preserving. Every shared
destination creates a new shared version. Every local destination creates a new
local version for the current thread.

## Shared Publication

For each thread slot `p`:

1. A local-to-shared insert records whether region `p` is defined.
2. The shared version waits until every defined region is ready.
3. The shared version publishes atomically.
4. Consumers bind the published version.

Undefined regions remain invalid. A consumer that requires a complete shared
tile may not read a partial version.

## Ordering

The machine may reorder operations when doing so preserves tile-version
dependencies and required global-memory conflict order. Non-overlapping global
accesses may overlap with tile computation.

## Lifetime

Local and shared physical storage is allocated, renamed, and reclaimed by the
implementation. Source code has no physical reclamation operation for shared
tiles.

## Invalid Programs

- Missing participant at a required grouped operation.
- Reading an undefined shared region.
- Using a partial shared right operand for `TMATMUL`.
- Overlapping global stores without a defined conflict rule.
- Relying on an assumed cycle delay instead of tile versions for correctness.
