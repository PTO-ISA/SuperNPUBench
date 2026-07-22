# Dependency and Consistency Rules

## Register Values

PE-local and shared tile registers follow versioned register semantics. A
consumer reads the exact source version bound during rename. It cannot observe
a later write to the same architectural tile ID.

## Shared Publication

A shared definition publishes atomically when all regions selected by its
defined mask are ready. A consumer never observes a half-published shared
version. Published metadata and payload are immutable.

## Program Order and Issue Order

Program order defines control flow and version selection. Hardware may issue
independent operations out of order. It must preserve:

- source-version dependencies;
- destination-version allocation;
- group dynamic order;
- global-memory conflict order required by the program;
- architectural exceptions and retirement rules.

## Memory Visibility

A ready tile load means its destination tile version is available. A completed
tile store means the source payload has been accepted according to the target
memory contract; it does not establish a general cross-core handoff.

Within one core, shared tile registers are the preferred mechanism for
register-sized inter-PE exchange. Cross-core memory handoff belongs to the
launch/runtime memory protocol and is outside the two identity intrinsics.

## Races

The following are data races or undefined programs:

- two PEs write overlapping non-atomic global ranges without an ordering
  protocol;
- two cores write the same tensor slice unintentionally;
- one PE uses a partial shared region that it does not own;
- a group operation is reached in a different dynamic order on different PEs.

## Determinism

Out-of-order issue does not change deterministic tile results when the value
graph and global-memory conflict rules are well formed. Floating-point
reductions may still vary if the program explicitly permits a different
association order.
