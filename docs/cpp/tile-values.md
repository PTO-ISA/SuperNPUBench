# Tile Values and Valid Regions

Tile variables name logical values. The compiler assigns tile IDs and creates
versions for each definition.

## The Public Tile Header

Kernel examples include one reader-facing header:

```cpp
#include <pto_kernel/tile.hpp>
```

The header exports tile types, tensor views, iterators, and intrinsic names.
Backend-specific headers and namespaces are implementation details and should
not appear in kernel source.

## Local Tiles

```cpp
LocalTile<float, kTileRows, kTileCols> input;
LocalTile<float, kTileRows, kTileCols> output;

TLOAD(input, input_view);
TEXP(output, input);
```

The template arguments are element type, physical rows, and physical columns.
An optional layout and valid shape can follow them:

```cpp
using TailTile = LocalTile<float, 32, 32,
                           BLayout::RowMajor, 11, 23>;
```

`TailTile` reserves `32 x 32` elements but only its leading `11 x 23` region is
logically valid. `output` consumes the version defined by `TLOAD`; a later
definition of `input` does not change that already-bound source.

## Matrix Operand Tiles

Matrix multiplication uses semantic aliases because each operand has a
different role and layout contract:

```cpp
using Left = MatrixLeftTile<half, kM, kK>;
using Right = MatrixRightTile<half, kK, kN>;
using Accum = AccumulatorTile<float, kM, kN>;
using Result = LocalTile<float, kM, kN>;
```

`MatrixLeftTile` and `MatrixRightTile` are inputs to `TMATMUL`.
`AccumulatorTile` receives the matrix result. Convert or move the accumulator
to a local tile before an operation that expects a local vector tile.

## Shared Tiles

```cpp
SharedTile<half, kK, kN> shared_rhs;

if (thread == 0) {
  TLOAD(shared_rhs, rhs_view);
}
```

`SharedTile` declares one block-scoped value, not an array of peer-local
registers. The selected producer defines the shared version, and grouped
consumers refer to that version through ordinary C++ value flow. Shared
lowering is a programming-model contract; consult a benchmark page before
assuming a particular compiler revision accepts a shared form.

| Type | Owner | Typical role |
| --- | --- | --- |
| `LocalTile<T, R, C>` | One thread | elementwise work, reductions, result fragments |
| `MatrixLeftTile<T, M, K>` | One thread | left matrix fragment |
| `MatrixRightTile<T, K, N>` | One thread | local right matrix fragment |
| `AccumulatorTile<T, M, N>` | One thread | matrix accumulation |
| `SharedTile<T, R, C>` | One block | reused right operand or exchanged fragments |

## Valid Regions

The tile shape describes storage. The valid region describes which logical
elements participate in the operation. Tail tiles often have padded storage
and a smaller valid region.

Do not read padding as input or store padding as output unless the selected
operation explicitly defines that behavior.

For a multidimensional tensor, the tile still has two physical dimensions.
Scalar C++ loops select outer batch, head, or channel indices. Read
[Multidimensional tiling](../tutorials/multidimensional-tiling.md) for complete
2D, 3D, and matrix examples.

## Fragment Size

Local tile fragments must satisfy `kMinimumThreadFragmentBytes` in the current
profile. Use padding, packing, or valid-region tails when the logical fragment
is smaller than the minimum allocation.
