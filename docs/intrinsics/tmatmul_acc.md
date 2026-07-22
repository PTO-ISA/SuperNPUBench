# TMATMUL_ACC Matrix Multiply With Accumulation

## Purpose / When to use

Use `TMATMUL_ACC` to add a tile matrix product to an existing accumulator. It is the C++ reference form for GEMM-style tile kernels that keep the right-hand-side matrix in a shared tile.

## C++ declaration

```cpp
template <typename TileRes, typename TileLeft, typename SharedTileRight>
PTO_INST void TMATMUL_ACC(TileRes &cOutMatrix, TileRes &cInMatrix, TileLeft &aMatrix, SharedTileRight &bMatrix);
template <AccPhase Phase, typename TileRes, typename TileLeft, typename SharedTileRight>
PTO_INST void TMATMUL_ACC(TileRes &cOutMatrix, TileRes &cInMatrix, TileLeft &aMatrix, SharedTileRight &bMatrix);
template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename SharedTileRight>
PTO_INST void TMATMUL_ACC(TileRes &cMatrix, TileLeft &aMatrix, SharedTileRight &bMatrix);
```

## Parameters

| Parameter | Description |
| --- | --- |
| `cOutMatrix` | Output accumulator tile. |
| `cInMatrix` | Existing accumulator tile read by accumulation overloads. |
| `cMatrix` | In-place accumulator/output tile for the overload that reads and writes the same accumulator. |
| `aMatrix` | Left operand tile. Its valid rows and columns define `M` and `K`. |
| `bMatrix` | Shared right-hand-side matrix tile. Its valid columns define `N`; all participants consume the same logical RHS tile version. |

## Operation

Let `M = aMatrix.GetValidRow()`, `K = aMatrix.GetValidCol()`, and `N = bMatrix.GetValidCol()`. For every valid output element `(i, j)`, `cOutMatrix[i,j] = cInMatrix[i,j] + sum_{k=0}^{K-1} aMatrix[i,k] * bMatrix[k,j]`. The in-place overload accumulates into `cMatrix`.

The RHS operand is a shared matrix form: the logical right-hand-side tile is fully defined once and consumed by the participating computation lanes. Readiness is carried by tile data dependencies; no explicit synchronization operand is part of this intrinsic.

## Constraints

- `aMatrix`, `bMatrix`, and the accumulator/output tile must satisfy the GEMM shape relation `A[M,K] * B[K,N] -> C[M,N]`.
- Runtime valid sizes `M`, `K`, and `N` must be non-zero and within the implementation-supported tile range.
- Accumulator/output type must match a supported `(C, A, B)` tuple. Common portable tuples are `float` accumulation for `half`, `bfloat16_t`, or `float` inputs, and `int32_t` accumulation for `int8_t` inputs.
- The right-hand side uses the shared RHS matrix form (`SharedTileRight`); do not substitute a non-shared RHS tile for these overloads.

## Example

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void gemm_tile() {
  using A = TileLeft<half, 16, 32>;
  using B = SharedTile<half, 32, 64>;
  using C = TileAcc<float, 16, 64>;
  C c0, c1;
  A a;
  B shared_b;

  TMATMUL_ACC(c1, c0, a, shared_b);
}
```

## Common mistakes

- Passing a non-shared RHS matrix to a `SharedTileRight` overload.
- Mismatching `K` between `aMatrix.GetValidCol()` and the RHS matrix rows.
- Assuming accumulator promotion without checking the selected backend contract.

## Used by kernels

- [`pto-mamulb-e9f16fd1`](../benchmarks/catalog/one-level/pto-kernels/pto-mamulb-e9f16fd1.md) - `benchmark/one-level-arch/kernels/pto_kernels/matmul/mamulb.cpp`
- [`pto-gemm-basic-a2ba2c9c`](../benchmarks/catalog/one-level/pto-kernels/pto-gemm-basic-a2ba2c9c.md) - `benchmark/one-level-arch/kernels/pto_kernels/matmul/gemm_basic.cpp`
- [`pto-gemm-performance-ffe72cfc`](../benchmarks/catalog/one-level/pto-kernels/pto-gemm-performance-ffe72cfc.md) - `benchmark/one-level-arch/kernels/pto_kernels/matmul/gemm_performance.cpp`
- [`pto-gemm-demo-85ba9afc`](../benchmarks/catalog/one-level/pto-kernels/pto-gemm-demo-85ba9afc.md) - `benchmark/one-level-arch/kernels/pto_kernels/matmul/gemm_demo.cpp`
- [`pto-tmatmul-acc-e3e7d902`](../benchmarks/catalog/one-level/pto-kernels/pto-tmatmul-acc-e3e7d902.md) - `benchmark/one-level-arch/kernels/pto_kernels/matmul/tmatmul_acc.cpp`
- [`pto-gemm-eb591839`](../benchmarks/catalog/one-level/pto-kernels/pto-gemm-eb591839.md) - `benchmark/one-level-arch/kernels/pto_kernels/matmul/gemm.cpp`
- [`matmul-a008ac76`](../benchmarks/catalog/one-level/matmul/matmul-a008ac76.md) - `benchmark/one-level-arch/kernels/matmul/matmul.hpp`
- [`hif4-hif4-f0fd84bb`](../benchmarks/catalog/one-level/matmul/hif4-hif4-f0fd84bb.md) - `benchmark/one-level-arch/kernels/matmul/matmul_mx.hpp`
