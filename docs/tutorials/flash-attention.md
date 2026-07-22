# FlashAttention Tutorial

FlashAttention uses the same building blocks as GEMM: block/thread row
partitioning, shared right-hand tiles, local reductions, and stores back to
global memory.

## Current Benchmark Sources

| Role | Path |
| --- | --- |
| FP16 entry | `benchmark/one-level-arch/test/kernel/fa/src/fa_2d_unroll.cpp` |
| HIF4 entry | `benchmark/one-level-arch/test/kernel/fa/src/fa_hif4.cpp` |
| Softmax probe | `benchmark/one-level-arch/test/kernel/fa/src/fa_softmax_pto.cpp` |
| Kernels | `benchmark/one-level-arch/kernels/fa/` |
| Build matrix | `benchmark/one-level-arch/test/kernel/fa/compile.all` |

The generated [FlashAttention catalog](../benchmarks/catalog/one-level/fa/index.md)
shows complete source and exact supported commands.

## Build

```bash
cd benchmark/one-level-arch/test/kernel/fa
PLAT=linx COMPILER_DIR=/path/to/linx-isa/compiler/llvm/build-linxisa-clang/bin \
  make TESTCASE=fa_2d_unroll
```

Use an exact command from `compile.all` when shape parameters are required.

## Compile-Checked QK/PV Loop

The compact probe below shows the multidimensional structure without exposing
the implementation namespace. The outer loop selects query-row tiles; the
inner loop walks key/value tiles and accumulates each contribution.

```cpp title="docs/examples/flash_attention.cpp" linenums="1"
--8<-- "docs/examples/flash_attention.cpp"
```

The probe intentionally leaves out online-softmax rescaling so the QK/PV tile
flow stays visible. The benchmark implementation below is the numerical
reference for the complete algorithm.

## Row Partition

Partition query rows by block and thread. Key and value blocks reused by every
query fragment are candidates for shared tiles.

```cpp
const uint32_t thread = get_thread_idx();
RowSlice q_rows = partition_rows(kQueryRows, block, block_count, thread);

TLOAD(q_fragment, q.slice_rows(q_rows.begin, q_rows.count));

if (thread == 0) {
  TLOAD(shared_k, k_block);
  TLOAD(shared_v, v_block);
}
```

## Score and Output Matrices

```cpp
TMATMUL(score_fragment, q_fragment, shared_k);
// Local max, exponential, and sum operations update this thread's online state.
TMATMUL(output_fragment, probability_fragment, shared_v);
```

Both matrix operations require a fully defined shared right operand. The
intermediate softmax state is local because each thread owns disjoint query
rows.

## Local Reductions

Row reductions stay local when each query row belongs to one thread. A
reduction that crosses thread fragments must publish local partials through a
shared tile version before the final reduction consumes them.

```cpp
TROWMAX(local_max, score_fragment);
TMOV<SharedMove::Insert>(shared_max_partials, local_max);
```

Use the all-reduce pattern when every participant needs the final scalar.

## Fragment Size and Tails

Each query fragment used as a local tile must satisfy
`kMinimumThreadFragmentBytes`. For short tail rows, use valid-region metadata
or a scalar tail path. Do not let padded attention scores participate in the
softmax denominator.

## Compiler Boundary

The checked benchmark sources are the current compile gate. Shared K/V tile
examples describe the C++ NPU contract for grouped matrix operations; rely on
the catalog for exact source that compiles in this repository revision.
