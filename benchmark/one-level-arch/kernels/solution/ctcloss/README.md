# CTC loss forward (Ctcloss_v2 profile)

FP32 forward-only CTC with an unreduced loss and complete log-alpha output.
The single-PE implementation uses public ElementTile / element-if for indexed
reads and predicates, and native FP32 TileOps for the recurrence.

- ctcloss_single_pe.hpp: ctc_loss_forward::forward(args).
- ctcloss_four_pe.hpp: ctc_loss_forward_4pe::forward(args), an SPMD entry for
  four actual contexts. It partitions complete sample pairs into contiguous,
  disjoint batch ranges and reuses the single-PE core.
- ctcloss_params.hpp: low-level U32 metadata, stride and capacity validation.
- ctcloss_partition.hpp: batch ownership and argument rebasing.
- ctcloss_v2_i32.hpp: contiguous padded INT32 input adapter. Call
  ctcloss_v2::prepare(...) before launching either entry. Negative signed
  labels/lengths and insufficient storage are rejected before an unsigned view
  is passed to the core; the external arrays remain INT32.

The supported public attribute combination is blank in range, reduction=none
and zero_infinity=false. Log-probs must already be log-softmax values (finite
or -Inf). This implementation does not compute log-softmax, reductions,
backward, FP64, or zero_infinity=true.

All strides/capacities are in elements. Outputs must not alias inputs or each
other. The caller owns batch-sized target-offset storage until completion.
The low-level API supports nonoverlapping output strides; the INT32 adapter
constructs contiguous [T,N,C] input, padded [N,U] targets and [N,T,2U+1] alpha.

## Four-PE launch contract

Supply PE IDs 0..3, independent stacks/Local Tile state, immutable inputs, and
disjoint sample-owned outputs. Each PE computes all time/state steps for its
samples, so no cross-PE DP reads or per-time-step barrier is needed. The caller
must join the entire launch before consuming results. This function is not a
launcher and cannot be emulated by four calls on PE0. For the shipped N=256
case, each PE owns 64 samples.

The empty asm in the core is a compiler ordering boundary, not a hardware
barrier. Frozen-backend strided writes use native TSTORE to a private transfer
buffer followed by scalar layout copies; numerical DP remains Tile arithmetic.
The currently supported Tile mapping is 16 states x 2 batch slots.

## Compatibility and current validation

The supplied test is solely the requested network specification; see
../../../test/solution/ctcloss/README.md relative to this directory's location,
or benchmark/one-level-arch/test/solution/ctcloss in the repository.

This relies on the historical generic element-region compiler/API profile.
Native exp arguments <= -104 are mapped to -Inf: FP32 exp is exactly zero in
that range. This avoids frozen-model TEXP(-1000)=Inf without changing goldens
or tolerances (SuperScalarModel #1074).

The complete network inputs/oracle and host validation pass; both target ELFs
build. Full-size target numerical and timing runs remain unverified.
Historical model-side problems are tracked in SuperScalarModel #1075 and #982.
The latter's GPR routing fix is upstream in PR #1000; the local frozen test
environment includes only a routing backport and still deadlocks later in CTC.
No speedup or complete network simulator PASS is claimed.
