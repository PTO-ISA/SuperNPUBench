# CTC loss forward

The public entry points follow the solution interface used by normalization:
input pointers, a host-computed tiling_info pointer, output pointers, and a
caller-owned offset workspace.

- ctcloss_v2_single_pe(...) in ctcloss_single_pe.hpp.
- ctcloss_v2_four_pe(...) in ctcloss_four_pe.hpp.
- CtcLossTilingData / make_ctcloss_tiling(...) in ctcloss_tiling.hpp.
- ctcloss_v2::prepare(...) in ctcloss_v2_i32.hpp validates signed INT32 inputs
  and fills the target-offset workspace before launch.
- ctcloss_params.hpp and ctcloss_partition.hpp retain the low-level layout,
  capacity checks and sample-pair partition helpers.

The math core is shared by the two entry points. It uses public ElementTile /
TPARTELEMENT / element-if for predicated indexed reads and native FP32 TileOps
for log-sum-exp. The mapping remains 16 states x 2 batch slots. The frozen
compiler's strided output restriction requires a private TSTORE transfer
followed by scalar layout copies; numerical recurrence is not scalar DP.

## Contract

log_probs is FP32 [T,N,C] log-softmax, targets is padded INT32 [N,U], and both
length arrays are INT32 [N]. Validate signed values and capacities with
ctcloss_v2::prepare before launch. Unsigned internal views are corresponding
C++ types of the original signed arrays; input dtype/storage is unchanged.
Outputs must be disjoint from inputs and each other.

Supported attributes are reduction=none and zero_infinity=false. Loss is
unreduced, and full log_alpha [N,T,2U+1] is retained. Unused alpha states are
-Inf. There is no backward, log-softmax, FP64, reduction, or zero_infinity=true
implementation in this change.

Tiling is serialized as 24 little-endian uint32 words in the order declared by
CtcLossTilingData. It includes dimensions, strides and begin/end ownership for
each PE. The host computes these ranges; the four-PE kernel only applies them.
For the sole N=256 network fixture each of four PEs owns 64 complete samples.

Four-PE launch requires real PE IDs 0..3, private stacks/Local Tile state, and
shared immutable inputs. Each PE computes all time/state steps of its samples,
without cross-PE DP reads or per-step synchronization. The launcher/test must
join all participants before reading outputs. The empty asm in the core is
compiler ordering, not a hardware barrier.

## Test integration and limits

See benchmark/one-level-arch/test/solution/ctcloss. Its Makefile includes the
public test/common/Makefile.common; compile.all builds the two testcases.
RES_CHECK exports ordinary files checked by the independent comparison script.
No input image embedding, private startup assembly or memory-dump acceptance
path is needed.

This implementation uses the historical generic element-region profile:
LLVM 482b4207, TileOP b223de6, model base 8e8d8ac7. Native exp(x) inputs at or
below -104 map to -Inf because FP32 exp is exactly zero in that domain; this
avoids the frozen-model bug tracked in SuperScalarModel #1074 without changing
the oracle or tolerances.

Full-size input/reference validation and all single/four-PE RES_CHECK on/off
builds pass. Full-network target numerical/timing acceptance remains unresolved.
Historical model blockers are SuperScalarModel #1075 and #982; the local model
has only the scalar-owner routing backport associated with upstream PR #1000.
No complete CTC simulator PASS or performance speedup is claimed.
