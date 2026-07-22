# TGEMV_ACC GEMV With Accumulation

## Purpose / When to use

Use `TGEMV_ACC` to add a GEMV result to an existing accumulator. It is the C++ reference form for one-row GEMM/GEMV work where `M` is `1`.

## C++ declaration

```cpp
template <typename TileRes, typename TileLeft, typename TileRight>
PTO_INST void TGEMV_ACC(TileRes &cOutMatrix, TileRes &cInMatrix, TileLeft &aMatrix, TileRight &bMatrix);
template <AccPhase Phase, typename TileRes, typename TileLeft, typename TileRight>
PTO_INST void TGEMV_ACC(TileRes &cOutMatrix, TileRes &cInMatrix, TileLeft &aMatrix, TileRight &bMatrix);
```

## Parameters

| Parameter | Description |
| --- | --- |
| `cOutMatrix` | Output accumulator tile. |
| `cInMatrix` | Existing accumulator tile read by accumulation overloads. |
| `aMatrix` | Left operand tile. Its valid rows and columns define `M` and `K` for GEMM, or the vector length for GEMV. |
| `bMatrix` | Right operand matrix tile. Its valid rows and columns define `K` and `N`. |

## Operation

Let `K = bMatrix.GetValidRow()` and `N = bMatrix.GetValidCol()`. The left operand is a single valid row. For every valid output column `j`, `cOutMatrix[0,j] = cInMatrix[0,j] + sum_{k=0}^{K-1} aMatrix[0,k] * bMatrix[k,j]`.

Exact accumulator precision is defined by the selected implementation, but the tile shape contract remains `A[1,K] * B[K,N] -> C[1,N]`.

## Constraints

- `TileLeft::Rows`, `TileRes::Rows`, and the runtime valid row count for the result must represent one GEMV row.
- Static tile shapes must satisfy `TileLeft::Cols == TileRight::Rows` and `TileRight::Cols == TileRes::Cols`.
- Runtime valid `K` and `N` must be non-zero and within the implementation-supported tile range.
- Accumulator/output type must match a supported `(C, A, B)` tuple. Common portable tuples are `float` accumulation for `half`, `bfloat16_t`, or `float` inputs, and `int32_t` accumulation for `int8_t` inputs.

## Example

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void gemv_tile() {
  using A = TileLeft<half, 1, 32>;
  using B = TileRight<half, 32, 64>;
  using C = TileAcc<float, 1, 64>;
  C c0, c1;
  A a;
  B b;

  TGEMV_ACC(c1, c0, a, b);
}
```

## Common mistakes

- Using `TGEMV*` with a left operand that has more than one logical row.
- Mismatching `K` between the left vector and the RHS matrix rows.
- Assuming accumulator promotion without checking the selected backend contract.

## Used by kernels

- No generated benchmark catalog entry currently lists `TGEMV_ACC`.
