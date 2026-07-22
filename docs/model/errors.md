# Errors and Undefined Behavior

## Group Participation Errors

- Fewer than four PEs reach a required group operation.
- PEs reach group operations in different dynamic orders.
- Participants disagree on datatype, shape, layout, or shared source version.

## Shared Tile Errors

- A partial shared value is used where mask `1111` is required.
- A PE reads a shared region that is undefined.
- Source code treats a shared tile ID as an address.
- A program relies on PE arrival time to select a shared version.
- A published shared version is modified in place.

## Matrix Errors

- The right operand of a `TMATMUL` family operation is PE-local rather than
  shared.
- M, N, or K dimensions are incompatible.
- A, B, and C types violate an intrinsic's accumulation rules.
- A or C is not distributed as the required PE-local M-axis quarter.

## Identity Errors

- `get_thread_idx()` is used as a peer tile-register address.
- `get_block_idx()` is used without launch geometry to infer a tensor extent.
- A computed core or PE slice falls outside the tensor's valid region.

## Tile-ID Errors

- A source refers to a tile version outside the target's live producer window.
- A consumer reads a version that is undefined on its control-flow path.
- Physical tile-register numbers are embedded in source-level correctness.

## Memory Errors

- Multiple PEs or cores write overlapping non-atomic global ranges.
- A load or store violates alignment, layout, or valid-region constraints.
- A pointer is derived from an out-of-range core or PE partition.

## Compiler Diagnostics

The compiler should reject statically provable violations, especially group
non-convergence, incompatible shared matrix operands, impossible shape
relations, and invalid identity-based slices. Dynamic violations are undefined
program behavior unless the launch/runtime layer defines a trap.
