# TGEMV_MX Mixed-Precision GEMV

## Purpose / When to use

Use `TGEMV_MX` to compute a scaled mixed-precision or quantized GEMV result. It is the C++ reference form for one-row GEMM/GEMV work where `M` is `1`.

## C++ declaration

```cpp
template <typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale>
PTO_INST void TGEMV_MX(TileRes &cMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix, TileRight &bMatrix, TileRightScale &bScaleMatrix);
template <AccPhase Phase, typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight,
          typename TileRightScale>
PTO_INST void TGEMV_MX(TileRes &cMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix, TileRight &bMatrix, TileRightScale &bScaleMatrix);
template <typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale>
PTO_INST void TGEMV_MX(TileRes &cOutMatrix, TileRes &cInMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix, TileRight &bMatrix, TileRightScale &bScaleMatrix);
template <AccPhase Phase, typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight,
          typename TileRightScale>
PTO_INST void TGEMV_MX(TileRes &cOutMatrix, TileRes &cInMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix, TileRight &bMatrix, TileRightScale &bScaleMatrix);
template <typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale,
          typename TileBias>
PTO_INST void TGEMV_MX(TileRes &cMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix, TileRight &bMatrix, TileRightScale &bScaleMatrix, TileBias &biasData);
template <AccPhase Phase, typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight,
          typename TileRightScale, typename TileBias>
PTO_INST void TGEMV_MX(TileRes &cMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix, TileRight &bMatrix, TileRightScale &bScaleMatrix, TileBias &biasData);
```

## Parameters

| Parameter | Description |
| --- | --- |
| `cMatrix` | Accumulator/output tile for the computed product. |
| `cOutMatrix` | Output accumulator tile. |
| `cInMatrix` | Existing accumulator tile read by accumulation overloads. |
| `aMatrix` | Left operand tile. Its valid rows and columns define `M` and `K` for GEMM, or the vector length for GEMV. |
| `aScaleMatrix` | Scale tile associated with `aMatrix` for mixed-precision reconstruction. |
| `bMatrix` | Right operand matrix tile. Its valid rows and columns define `K` and `N`. |
| `bScaleMatrix` | Scale tile associated with `bMatrix` for mixed-precision reconstruction. |
| `biasData` | Bias tile, logically a single row broadcast across the output columns. |

## Operation

Let `K = bMatrix.GetValidRow()` and `N = bMatrix.GetValidCol()`. The left operand is a single valid row. For every valid output column `j`, the output is the GEMV product under the scaling semantics supplied by `aScaleMatrix` and `bScaleMatrix`. Accumulation and bias overloads add `cInMatrix` or `biasData` respectively.

Exact accumulator precision and mixed-precision reconstruction are defined by the selected implementation, but the tile shape contract remains `A[1,K] * B[K,N] -> C[1,N]`.

## Constraints

- `TileLeft::Rows`, `TileRes::Rows`, and the runtime valid row count for the result must represent one GEMV row.
- Static tile shapes must satisfy `TileLeft::Cols == TileRight::Rows` and `TileRight::Cols == TileRes::Cols`.
- Runtime valid `K` and `N` must be non-zero and within the implementation-supported tile range.
- Accumulator/output type must match a supported `(C, A, B)` tuple. Common portable tuples are `float` accumulation for `half`, `bfloat16_t`, or `float` inputs, and `int32_t` accumulation for `int8_t` inputs.
- Bias overloads require a single-row bias tile with an element type compatible with the accumulator/output tile.
- MX overloads require scale tiles whose formats match the selected mixed-precision operand formats.

## Example

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void gemv_tile() {
  using A = TileLeft<half, 1, 32>;
  using B = TileRight<half, 32, 64>;
  using C = TileAcc<float, 1, 64>;
  using Scale = Tile<TileType::Vec, float, 1, 16>;
  Scale a_scale, b_scale;
  C c;
  A a;
  B b;

  TGEMV_MX(c, a, a_scale, b, b_scale);
}
```

## Common mistakes

- Using `TGEMV*` with a left operand that has more than one logical row.
- Mismatching `K` between the left vector and the RHS matrix rows.
- Passing a bias tile that is not a single logical row.
- Assuming MX scale layout or accumulator promotion without checking the selected backend contract.

## Used by kernels

- No generated benchmark catalog entry currently lists `TGEMV_MX`.
