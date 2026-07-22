# One-Level PTO Programming Model

The LinxISA PTO 0.57 programming model runs one program image on a group of
four processing elements (PEs) inside each core. All four PEs begin at the same
entry and initial program counter. After launch, each PE has its own program
counter, frontend, reorder buffer, scalar state, and PE-local tile registers.

## Execution Hierarchy

```text
launch
  core 0 -> PE 0, PE 1, PE 2, PE 3
  core 1 -> PE 0, PE 1, PE 2, PE 3
  ...
```

`get_block_idx()` identifies the core. `get_thread_idx()` identifies the PE
within that core. A PE-ID branch may send the four PEs down different control
paths.

```cpp
const uint32_t core = get_block_idx();
const uint32_t pe = get_thread_idx();

if (pe == 0) {
  load_group_metadata(core);
} else {
  compute_partition(core, pe);
}
```

This is not a lockstep warp. Divergent PEs may fetch, issue, and commit
different instructions in the same cycle.

## Architectural Values

| Value | Visibility | Lifetime |
| --- | --- | --- |
| Scalar register | One PE | PE instruction stream |
| PE-local tile register | One PE | Tile-version lifetime |
| Shared tile register | Four PEs in one core | Shared tile-version lifetime |
| Global tensor | Addressable by the program | Memory allocation lifetime |

A logical tile may be distributed across four PE-local quarter fragments or
materialized as one versioned shared tile register. Shared tile registers are
register values, not addressable scratchpad storage.

## Intrinsic Surface

The public instruction set is the 111-operation allowlist in
`PTO-ISA-LinxISA 0.57.numbers`, plus:

- [`get_thread_idx()`](../intrinsics/get_thread_idx.md)
- [`get_block_idx()`](../intrinsics/get_block_idx.md)

The complete set is in the [PTO 0.57 intrinsic manual](../intrinsics/index.md).
Operations outside that set are not part of this programming profile.

## Dependency Model

The compiler assigns logical tile IDs. Every destination defines a new tile
version, and every source binds to a specific version. The superscalar machine
may issue independent operations out of program order, but a consumer cannot
issue until all bound source versions are ready.

```text
T#4.v1 = TLOAD(global_a)
T#7.v2 = TLOAD(global_b)
T#9.v1 = TADD(T#4.v1, T#7.v2)
TSTORE(global_c, T#9.v1)
```

The source language does not wire completion tokens. Ordering follows tile
versions, memory conflicts, and the intrinsic's group-participation contract.

## Group Operations

Most elementwise operations execute independently on each PE's quarter. A
group operation has a convergent dynamic instance: all four PEs eventually
reach the same operation in the same dynamic order, although they need not
arrive in the same cycle.

The `TMATMUL` family is the primary group operation. Its left operand and
destination are PE-local M-axis quarters; its right operand is a fully defined
shared tile version.

## Compiler Status

The four-PE execution model and identity intrinsics are normative. Versioned
shared tile registers and their `TLOAD`, `TSTORE`, `TMOV`, and `TMATMUL` forms
are also normative architecture contracts, but current compilers are not yet
required to lower them. Examples clearly distinguish architecture-level code
from compile-checked one-level benchmark code.

## Next Reading

1. [Group execution](group-execution.md)
2. [Tiles and distribution](tiles-and-layouts.md)
3. [Shared tile registers](shared-tile-registers.md)
4. [Tile-ID superscalar execution](execution.md)
5. [Group programming examples](../tutorials/group-programming.md)
