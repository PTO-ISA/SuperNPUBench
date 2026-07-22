# Complete One-Level Memory Semantics

## State

For each PE `p`, architectural state contains scalar registers, a PC, and a map
from PE-local tile IDs to versions. Each core contains a map from shared tile
IDs to shared versions. Global memory is a byte-addressed map.

## Local Tile Definition

An operation that writes local tile ID `t` in PE `p` creates version
`L[p,t,v+1]`. Existing readers of `L[p,t,v]` continue to read the old version.

## Shared Tile Definition

An operation that writes shared ID `s` creates `S[s,v+1]` with immutable:

- logical size;
- type, shape, and layout;
- defined-region mask;
- four fixed region identities.

The new version is not readable until every defined region is ready.

## Reads

A tile source binds to one local or shared version before issue. Issue requires
that version to be ready and compatible with the intrinsic's operand class.

## Global Loads

A local `TLOAD` reads a global byte range and defines one local tile version. A
shared `TLOAD` is issued by exactly one statically identifiable PE, reads the
full global range, and defines one fully defined shared version.

## Global Stores

A local `TSTORE` reads one local tile version and writes a global range. A
shared `TSTORE` reads one shared version. Full store writes the full shared
payload from one selected PE. Partition store writes defined fixed quarters to
non-overlapping PE-provided ranges.

## Movement

`TMOV` between local and shared classes is byte preserving. Every destination
is a new version. No movement form grants direct access to a peer's local
physical register.

## Matrix Multiply

For each PE `p`:

```text
C[p] = A[p] x B_shared
```

`A[p]` and `C[p]` are M-axis local quarters. `B_shared` is one fully defined
shared version bound by all four participants.

## Reordering

The machine may reorder operations when doing so preserves tile-version
dependencies, group dynamic order, and conflicting global-memory accesses.
Non-overlapping global accesses may overlap with tile computation.

## Reclamation

A physical local or shared version may be reclaimed only after no future
instruction can bind it and every already-bound reader has completed. Source
code has no physical reclamation operation for shared tiles.

## Undefined Behavior

- Multiple dynamic issuers for a single-owner shared definition.
- Missing participant at a four-PE group operation.
- Read of an undefined shared region.
- Partial shared RHS for `TMATMUL`.
- Incompatible type, shape, layout, size, or distribution.
- Global-memory data race.
