# Dependency and Consistency Rules

Correct kernels express ordering through tile versions, global-memory effects,
and grouped-operation participation.

## Register Values

Local and shared tiles follow versioned register semantics. A consumer reads
the exact source version selected during rename. It cannot observe a later
write to the same architectural tile ID.

## Shared Publication

A shared definition publishes atomically when every defined region is ready. A
consumer never observes a half-published shared version. Published metadata and
payload are immutable.

## Program Order and Issue Order

Program order defines control flow and version selection. The implementation
may issue independent operations out of order. It must preserve:

- source-version dependencies;
- destination-version allocation;
- grouped-operation dynamic order;
- global-memory conflict order required by the program;
- architectural exceptions and retirement rules.

## Memory Visibility

A ready tile load means its destination tile version is available. A completed
tile store means the source payload has been accepted according to the target
memory contract. It does not establish a general cross-block handoff.

Within one block, shared tile versions are the register-sized exchange
mechanism. Cross-block memory handoff belongs to the runtime or launch
protocol.

## Race Avoidance

Two blocks or threads may write different global ranges without coordination.
If they may write the same location, the kernel must use an operation with
defined conflict semantics or restructure the algorithm so ownership is
unique.

## Dependency Ordering

Use tile values to carry dependencies:

- produce a tile before passing it to a consumer;
- publish a shared version before a cross-thread read;
- keep global-memory handoff outside the kernel unless the runtime defines it.
