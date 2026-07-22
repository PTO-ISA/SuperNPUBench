# TMATMUL_BIAS Matrix Multiply With Bias

## Purpose / When to use

Use `TMATMUL_BIAS` to compute a tile matrix product and add a column bias. It is the C++ reference form for GEMM-style tile kernels that keep the right-hand-side matrix in a shared tile.

## C++ declaration

```cpp
template <typename TileRes, typename TileLeft, typename SharedTileRight, typename TileBias>
PTO_INST void TMATMUL_BIAS(TileRes &cMatrix, TileLeft &aMatrix, SharedTileRight &bMatrix, TileBias &biasData);
template <AccPhase Phase, typename TileRes, typename TileLeft, typename SharedTileRight, typename TileBias>
PTO_INST void TMATMUL_BIAS(TileRes &cMatrix, TileLeft &aMatrix, SharedTileRight &bMatrix, TileBias &biasData);
```

## Parameters

| Parameter | Description |
| --- | --- |
| `cMatrix` | Accumulator/output tile for the computed product. |
| `aMatrix` | Left operand tile. Its valid rows and columns define `M` and `K`. |
| `bMatrix` | Shared right-hand-side matrix tile. Its valid columns define `N`; all participants consume the same logical RHS tile version. |
| `biasData` | Bias tile, logically a single row broadcast across the output columns. |

## Operation

Let `M = aMatrix.GetValidRow()`, `K = aMatrix.GetValidCol()`, and `N = bMatrix.GetValidCol()`. For every valid output element `(i, j)`, `cMatrix[i,j] = biasData[0,j] + sum_{k=0}^{K-1} aMatrix[i,k] * bMatrix[k,j]`.

The RHS operand is a shared matrix form: the logical right-hand-side tile is fully defined once and consumed by the participating computation lanes. Readiness is carried by tile data dependencies; no explicit synchronization operand is part of this intrinsic.

## Constraints

- `aMatrix`, `bMatrix`, and the accumulator/output tile must satisfy the GEMM shape relation `A[M,K] * B[K,N] -> C[M,N]`.
- Runtime valid sizes `M`, `K`, and `N` must be non-zero and within the implementation-supported tile range.
- Accumulator/output type must match a supported `(C, A, B)` tuple. Common portable tuples are `float` accumulation for `half`, `bfloat16_t`, or `float` inputs, and `int32_t` accumulation for `int8_t` inputs.
- The right-hand side uses the shared RHS matrix form (`SharedTileRight`); do not substitute a non-shared RHS tile for these overloads.
- Bias overloads require a single-row bias tile with an element type compatible with the accumulator/output tile.

## Example

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void gemm_tile() {
  using A = TileLeft<half, 16, 32>;
  using B = SharedTile<half, 32, 64>;
  using C = TileAcc<float, 16, 64>;
  using Bias = Tile<TileType::Bias, float, 1, 64>;
  Bias bias;
  C c;
  A a;
  B shared_b;

  TMATMUL_BIAS(c, a, shared_b, bias);
}
```

## Common mistakes

- Passing a non-shared RHS matrix to a `SharedTileRight` overload.
- Mismatching `K` between `aMatrix.GetValidCol()` and the RHS matrix rows.
- Using a bias tile with more than one logical row.
- Assuming accumulator promotion without checking the selected backend contract.

## Used by kernels

- No generated benchmark catalog entry currently lists `TMATMUL_BIAS`.
