# GEMM Tutorial

GEMM combines row partitioning, shared right-hand tiles, matrix tile versions,
and stores back to global memory.

## Current Benchmark Sources

| Role | Path |
| --- | --- |
| Generic entry | `benchmark/one-level-arch/test/kernel/matmul/src/matmul.cpp` |
| A16W4 entry | `benchmark/one-level-arch/test/kernel/matmul/src/A16W4.cpp` |
| HIF4 entry | `benchmark/one-level-arch/test/kernel/matmul/src/HiF4_HiF4.cpp` |
| Kernels | `benchmark/one-level-arch/kernels/matmul/` |
| Build matrix | `benchmark/one-level-arch/test/kernel/matmul/compile.all` |

The generated [matmul catalog](../benchmarks/catalog/one-level/matmul/index.md)
shows each active source implementation and exact build command.

## Build a Current Case

```bash
cd benchmark/one-level-arch/test/kernel/matmul
PLAT=linx COMPILER_DIR=/path/to/linx-isa/compiler/llvm/build-linxisa-clang/bin \
  make TESTCASE=matmul TYPE=MASK MODE=VEC M=64 N=64 K=64 tM=64 tN=64 tK=64
```

## Compile-Checked Tiled Baseline

This complete example walks output tile rows, output tile columns, and the
matrix reduction dimension. Its code uses only the public tile header and tile
operations:

```cpp title="docs/examples/gemm_tile.cpp" linenums="1"
--8<-- "docs/examples/gemm_tile.cpp"
```

The `(m, n)` loops select independent output tiles. The `k` loop contributes
successive products to one `AccumulatorTile`, so it is the only loop-carried
tile dependency.

## Grouped Source Contract

The grouped source form keeps the left matrix and accumulator local while the
right matrix is shared:

```cpp
const uint32_t thread = get_thread_idx();
static constexpr uint32_t kRowsPerThread = kM / kThreadsPerBlock;

MatrixLeftTile<half, kRowsPerThread, kK> a_fragment;
AccumulatorTile<float, kRowsPerThread, kN> c_fragment;
SharedTile<half, kK, kN> shared_b;

TLOAD(a_fragment, global_a.slice_rows(thread * kRowsPerThread,
                                      kRowsPerThread));
if (thread == 0) {
  TLOAD(shared_b, global_b);
}

TMATMUL(c_fragment, a_fragment, shared_b);
LocalTile<float, kRowsPerThread, kN> result;
TCVT(result, c_fragment);
TSTORE(global_c.slice_rows(thread * kRowsPerThread, kRowsPerThread), result);
```

The shared right operand must be fully defined. Each participant computes and
stores a disjoint row fragment.

## Tile Versions

The value graph has four important definitions:

1. `a_fragment` from the global left matrix;
2. `shared_b` from the global right matrix;
3. `c_fragment` from `TMATMUL`;
4. the global store that consumes `c_fragment`.

The local left and result fragments are independent per thread. All
participants bind the same shared right version.

## Fragment Size

Check the local fragment size before selecting a tile shape:

```cpp
static_assert(kRowsPerThread * kK * sizeof(half) >=
              kMinimumThreadFragmentBytes);
static_assert(kRowsPerThread * kN * sizeof(float) >=
              kMinimumThreadFragmentBytes);
```

When `kM` is not divisible by `kThreadsPerBlock`, use valid regions or a tail
path rather than overlapping rows.

## Correctness Checklist

- Block slices are disjoint.
- Thread fragments are disjoint and cover the intended block rows.
- The right matrix shared tile is fully defined before `TMATMUL`.
- Matrix operand element types, layouts, and accumulator types match the
  selected intrinsic.
- Stores write only valid output rows.
