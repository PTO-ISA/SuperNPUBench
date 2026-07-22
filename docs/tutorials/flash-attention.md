# FlashAttention on the One-Level Architecture

## Current Benchmark Sources

| Role | Path |
| --- | --- |
| FP16 entry | `benchmark/one-level-arch/test/kernel/fa/src/fa_2d_unroll.cpp` |
| HIF4 entry | `benchmark/one-level-arch/test/kernel/fa/src/fa_hif4.cpp` |
| Softmax probe | `benchmark/one-level-arch/test/kernel/fa/src/fa_softmax_pto.cpp` |
| Kernels | `benchmark/one-level-arch/kernels/fa/` |
| Build matrix | `benchmark/one-level-arch/test/kernel/fa/compile.all` |

The generated [FlashAttention catalog](../benchmarks/catalog/one-level/fa/index.md)
shows complete source and supported 0.57 intrinsics.

## Build

```bash
cd benchmark/one-level-arch/test/kernel/fa
PLAT=linx COMPILER_DIR=/path/to/linx-isa/compiler/llvm/build-linxisa-clang/bin \
  make TESTCASE=fa_2d_unroll
```

Use an exact command from `compile.all` when shape parameters are required.

## Four-PE Mapping

Partition the query M dimension across the four PEs. K and V blocks that are
reused by every query quarter are candidates for shared tile versions.

```cpp
const uint32_t pe = get_thread_idx();
TLOAD(q_quarter, q.slice_rows(pe * q_rows_per_pe, q_rows_per_pe));

if (pe == 0) {
  TLOAD(shared_k, k_block);
  TLOAD(shared_v, v_block);
}

TMATMUL(score_quarter, q_quarter, shared_k);
// Local max, exponential, and sum operations update this PE's online state.
TMATMUL(output_quarter, probability_quarter, shared_v);
```

Both matrix operations require a fully defined shared RHS. The intermediate
softmax state is PE-local because each PE owns disjoint query rows.

## Reduction

Row reductions stay local when each query row belongs to one PE. A reduction
that crosses the PE distribution must first publish local partials through a
shared tile version, as shown in [group programming](group-programming.md).

## Compiler Status

The current one-level FlashAttention benchmark is compile checked. Shared K/V
register lowering is architecture-level code until compiler support lands.
