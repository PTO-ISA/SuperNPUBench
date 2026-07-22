# GEMM on the One-Level Architecture

## Current Benchmark Sources

| Role | Path |
| --- | --- |
| Generic entry | `benchmark/one-level-arch/test/kernel/matmul/src/matmul.cpp` |
| A16W4 entry | `benchmark/one-level-arch/test/kernel/matmul/src/A16W4.cpp` |
| HIF4 entry | `benchmark/one-level-arch/test/kernel/matmul/src/HiF4_HiF4.cpp` |
| Kernels | `benchmark/one-level-arch/kernels/matmul/` |
| Build matrix | `benchmark/one-level-arch/test/kernel/matmul/compile.all` |

The generated [matmul catalog](../benchmarks/catalog/one-level/matmul/index.md)
shows each active source implementation, exact build commands, and PTO 0.57
surface.

## Build a Current Case

```bash
cd benchmark/one-level-arch/test/kernel/matmul
PLAT=linx COMPILER_DIR=/path/to/linx-isa/compiler/llvm/build-linxisa-clang/bin \
  make TESTCASE=matmul TYPE=MASK MODE=VEC M=64 N=64 K=64 tM=64 tN=64 tK=64
```

## Group-Level Matrix Contract

The architecture-level four-PE form uses a shared right matrix:

```cpp
const uint32_t pe = get_thread_idx();

PETile<half, M / 4, K> a_quarter;
PETile<float, M / 4, N> c_quarter;
SharedTile<half, K, N> shared_b;

TLOAD(a_quarter, global_a.slice_rows(pe * (M / 4), M / 4));
if (pe == 0) {
  TLOAD(shared_b, global_b);
}
TMATMUL(c_quarter, a_quarter, shared_b);
TSTORE(global_c.slice_rows(pe * (M / 4), M / 4), c_quarter);
```

The shared RHS and four-PE group formation are not yet required compiler
features. The current benchmark remains the compile gate while the shared
lowering is developed.

## Tile-ID Dataflow

The compiler assigns versions for A, shared B, accumulator C, and stored C.
Each PE's local A and C versions are independent; all four group participants
bind the same shared B version.

## Correctness Checks

- M is partitioned across cores and then four PEs without overlap.
- B is fully defined before use as matrix RHS.
- All four PEs reach the same dynamic matrix operation.
- Accumulator and input type combinations satisfy the selected intrinsic page.
- Tail rows use a valid-region form rather than reading padding as data.
