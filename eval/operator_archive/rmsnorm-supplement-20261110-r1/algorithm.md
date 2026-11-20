# RMSNorm timing snapshot scope

Target: y[a,r] = x[a,r] * rsqrt(sum_r x[a,r]^2 / 8192 + epsilon) * gamma[r]. The archived tree-reduction kernel distributes 128 rows over four PEs, with FP16 input/output, FP32 reduction intermediates and a single I/O buffer with 512-byte alignment. See the immutable operator source snapshot for exact reduction order/epsilon and conversion implementation.

A-001 correctness premise: contiguous fixed [128,8192], PE4, no tail/alias contract beyond the archived driver. Verified only as compile-time specialization, not a numerical proof. Other shapes are unsupported by this archive and are not guaranteed to be rejected safely.

A-002 input specialization: shipped ELF embeds static zeros including gamma. This is verified from the archived driver; it cannot exercise nonzero reduction accuracy, NaN/Inf, rounding or output correctness. Numerical status remains not_checked.

A-003 implementation dependency: historical model base598638a8 plus local tail/shared128 patch; runtime6467f7f4 and frozen configuration. Verified by hashes for timing replay. Upstream mainline equivalence is unknown.

A-004 performance premise: single-buffer placement, synchronization, initialization and drain are included in Total Cycles43694. No pure-kernel bound, bandwidth-bound conclusion or general statistical stability is claimed. The deterministic historical replay is one case; performance acceptance awaits nonzero validation and same-ELF mainline control.
