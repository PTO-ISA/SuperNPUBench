# Four-PE Group Execution

## Program Instance

A core program instance contains one program image, four PE execution
contexts, and one core-local shared tile-register namespace. Shared IDs in two
different cores denote unrelated values.

At launch:

```text
PC(PE0) = PC(PE1) = PC(PE2) = PC(PE3) = entry
```

After launch, each PE advances its own PC. The architecture does not require
cycle-by-cycle lockstep.

## Identity Intrinsics

```cpp
const uint32_t pe = get_thread_idx();  // 0, 1, 2, or 3
const uint32_t core = get_block_idx(); // launch-defined core ID
```

All four PEs in a core observe the same core ID. Each PE observes a distinct,
immutable PE ID.

## Independent Regions

An independent region may execute on one PE, a static subset, or all PEs.

```cpp
if (get_thread_idx() < 2) {
  TADD(local_out, local_lhs, local_rhs);
}
```

Only PE0 and PE1 define `local_out` on this path. PE2 and PE3 retain their
previous local versions or an undefined value if no previous definition
exists.

## Convergent Regions

A convergent operation requires all four PEs to reach the same dynamic
instance with compatible operands. Arrival may be staggered. The group begins
execution only after all participants and source tile versions are ready.

```cpp
TMATMUL(local_c, local_a, shared_b);
```

This operation is legal only when:

- `shared_b` is one fully defined shared tile version;
- every PE reaches this dynamic `TMATMUL` in the same group order;
- `local_a` and `local_c` describe compatible M-axis quarters;
- all four PEs agree on shape, datatype, and layout.

The following is invalid because only two PEs reach the group operation:

```cpp
if (get_thread_idx() < 2) {
  TMATMUL(local_c, local_a, shared_b); // invalid group participation
}
```

## Static and Dynamic Divergence

Branches on `get_thread_idx()` are statically analyzable and may be lowered to
PE masks. Data-dependent branches are legal for PE-local work, but the compiler
must prove that control paths reconverge with compatible tile-version state
before the next group operation.

## Logical Tile Distribution

A logical tile has four fixed PE regions. A common M-axis distribution is:

```text
PE0 owns rows [0, M/4)
PE1 owns rows [M/4, M/2)
PE2 owns rows [M/2, 3M/4)
PE3 owns rows [3M/4, M)
```

The PE ID does not act as a tile-register address. A PE cannot directly name a
peer's local tile payload. Cross-PE tile data must pass through a shared tile
version.

## Group Progress Rules

- Independent operations do not wait for other PEs.
- A shared definition publishes when all defined regions are ready.
- A group consumer waits for its exact shared source version.
- A convergent instance must have the same dynamic order on all four PEs.
- Leaving one PE behind a never-taken path can prevent group progress.
