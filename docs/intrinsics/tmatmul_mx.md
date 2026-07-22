# TMATMUL_MX Mixed-Precision Matrix Multiply

## Purpose / When to use

Use `TMATMUL_MX` to compute a scaled mixed-precision or quantized matrix product. It is the C++ reference form for GEMM-style tile kernels that keep the right-hand-side matrix in a shared tile.

## C++ declaration

```cpp
template <typename TileRes, typename TileLeft, typename TileLeftScale, typename SharedTileRight, typename TileRightScale>
PTO_INST void TMATMUL_MX(TileRes &cMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix, SharedTileRight &bMatrix, TileRightScale &bScaleMatrix);
template <AccPhase Phase, typename TileRes, typename TileLeft, typename TileLeftScale, typename SharedTileRight,
          typename TileRightScale>
PTO_INST void TMATMUL_MX(TileRes &cMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix, SharedTileRight &bMatrix, TileRightScale &bScaleMatrix);
template <typename TileRes, typename TileLeft, typename TileLeftScale, typename SharedTileRight, typename TileRightScale>
PTO_INST void TMATMUL_MX(TileRes &cOutMatrix, TileRes &cInMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix, SharedTileRight &bMatrix, TileRightScale &bScaleMatrix);
template <AccPhase Phase, typename TileRes, typename TileLeft, typename TileLeftScale, typename SharedTileRight,
          typename TileRightScale>
PTO_INST void TMATMUL_MX(TileRes &cOutMatrix, TileRes &cInMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix, SharedTileRight &bMatrix, TileRightScale &bScaleMatrix);
template <typename TileRes, typename TileLeft, typename TileLeftScale, typename SharedTileRight, typename TileRightScale,
          typename TileBias>
PTO_INST void TMATMUL_MX(TileRes &cMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix, SharedTileRight &bMatrix, TileRightScale &bScaleMatrix, TileBias &biasData);
template <AccPhase Phase, typename TileRes, typename TileLeft, typename TileLeftScale, typename SharedTileRight,
          typename TileRightScale, typename TileBias>
PTO_INST void TMATMUL_MX(TileRes &cMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix, SharedTileRight &bMatrix, TileRightScale &bScaleMatrix, TileBias &biasData);
```

## Parameters

| Parameter | Description |
| --- | --- |
| `cMatrix` | Accumulator/output tile for the computed product. |
| `cOutMatrix` | Output accumulator tile. |
| `cInMatrix` | Existing accumulator tile read by accumulation overloads. |
| `aMatrix` | Left operand tile. Its valid rows and columns define `M` and `K`. |
| `aScaleMatrix` | Scale tile associated with `aMatrix` for mixed-precision reconstruction. |
| `bMatrix` | Shared right-hand-side matrix tile. Its valid columns define `N`; all participants consume the same logical RHS tile version. |
| `bScaleMatrix` | Scale tile associated with `bMatrix` for mixed-precision reconstruction. |
| `biasData` | Bias tile, logically a single row broadcast across the output columns. |

## Operation

Let `M = aMatrix.GetValidRow()`, `K = aMatrix.GetValidCol()`, and `N = bMatrix.GetValidCol()`. For every valid output element `(i, j)`, the output is the matrix product of `aMatrix` and the shared RHS `bMatrix` under the scaling semantics supplied by `aScaleMatrix` and `bScaleMatrix`. Accumulation and bias overloads add `cInMatrix` or `biasData` respectively.

The RHS operand is a shared matrix form: the logical right-hand-side tile is fully defined once and consumed by the participating computation lanes. Readiness is carried by tile data dependencies; no explicit synchronization operand is part of this intrinsic.

## Constraints

- `aMatrix`, `bMatrix`, and the accumulator/output tile must satisfy the GEMM shape relation `A[M,K] * B[K,N] -> C[M,N]`.
- Runtime valid sizes `M`, `K`, and `N` must be non-zero and within the implementation-supported tile range.
- Accumulator/output type must match a supported `(C, A, B)` tuple. Common portable tuples are `float` accumulation for `half`, `bfloat16_t`, or `float` inputs, and `int32_t` accumulation for `int8_t` inputs.
- The right-hand side uses the shared RHS matrix form (`SharedTileRight`); do not substitute a non-shared RHS tile for these overloads.
- Bias overloads require a single-row bias tile with an element type compatible with the accumulator/output tile.
- MX overloads require scale tiles whose formats match the selected mixed-precision operand formats.

## Example

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void gemm_tile() {
  using A = TileLeft<half, 16, 32>;
  using B = SharedTile<half, 32, 64>;
  using C = TileAcc<float, 16, 64>;
  using Scale = Tile<TileType::Vec, float, 16, 16>;
  Scale a_scale, b_scale;
  C c;
  A a;
  B shared_b;

  TMATMUL_MX(c, a, a_scale, shared_b, b_scale);
}
```

## Common mistakes

- Passing a non-shared RHS matrix to a `SharedTileRight` overload.
- Mismatching `K` between `aMatrix.GetValidCol()` and the RHS matrix rows.
- Using a bias tile with more than one logical row.
- Assuming accumulator promotion or MX scale interpretation without checking the selected backend contract.

## Used by kernels

- [`matmul-a008ac76`](../benchmarks/catalog/one-level/matmul/matmul-a008ac76.md) - `benchmark/one-level-arch/kernels/matmul/matmul.hpp`
- [`fa-hif4-86b1f297`](../benchmarks/catalog/one-level/fa/fa-hif4-86b1f297.md) - `benchmark/one-level-arch/kernels/fa/fa_hif4.hpp`
