# Transpose Operator Test Cases

This directory contains make targets for compiling and testing the **transpose** operator across different tensor dimensionalities. Each case is invoked via `make TESTCASE=transpose` with shape, dimension, and permutation parameters.

## Prerequisites

Set the compiler directory before running any target:

```bash
export COMPILER_DIR=/path/to/your/compiler/bin
```

## Usage

```bash
bash compile.all
```

Or run individual targets directly, e.g.:

```bash
make TESTCASE=transpose COMPILER_DIR=$COMPILER_DIR DType=__half MAX_DIM=8 tM=512 \
    IN_SHAPE=1476,32 OUT_SHAPE=32,1476 \
    IN_SHAPE_NAME=1476_32 OUT_SHAPE_NAME=32_1476 \
    IN_DIM=2 OUT_DIM=2 \
    TRANSPOSE_DIM1=0 TRANSPOSE_DIM0=1 \
    gIM='1476*32' gOM='32*1476'
```

---

## Test Cases

### 1. 2D Transpose

| Field | Value |
|---|---|
| Input Shape | `{1476, 32}` |
| Output Shape | `{32, 1476}` |
| Dimensions | 2 → 2 |
| Permutation | Swap dim 0 and dim 1 |

A standard 2D matrix transpose. Rows become columns and vice versa.

---

### 2. 4D Transpose (Case A)

| Field | Value |
|---|---|
| Input Shape | `{1, 8, 4096, 3}` |
| Output Shape | `{1, 8, 3, 4096}` |
| Dimensions | 4 → 4 |
| Permutation | Swap dim 2 and dim 3 |

Swaps the last two dimensions of a 4D tensor. The first two dimensions (batch and channel) remain unchanged.

---

### 3. 4D Transpose (Case B)

| Field | Value |
|---|---|
| Input Shape | `{1, 32, 8, 32}` |
| Output Shape | `{1, 8, 32, 32}` |
| Dimensions | 4 → 4 |
| Permutation | Swap dim 1 and dim 2 |

Swaps the middle two dimensions of a 4D tensor.

---

### 4. 6D Transpose

> **Important Note:** The "6D transpose" test case is internally specified as a **5D shape** (`IN_DIM=5, OUT_DIM=5`), but it is **semantically equivalent to a 6D transpose**. A leading dimension of size 1 is implicitly prepended, turning `{8, 64, 4, 16, 7}` into `{1, 8, 64, 4, 16, 7}`. A transpose on a 5D tensor is equivalent to a transpose on the corresponding 6D tensor with a trivial leading batch dimension of 1, because the leading 1 does not alter the memory layout or the permutation logic.

| Field | Value |
|---|---|
| 5D Input Shape | `{8, 64, 4, 16, 7}` |
| 6D Equivalent Input | `{1, 8, 64, 4, 16, 7}` |
| 5D Output Shape | `{8, 16, 4, 64, 7}` |
| 6D Equivalent Output | `{1, 8, 16, 4, 64, 7}` |
| Dimensions | 5 → 5 (≡ 6 → 6) |
| Permutation | Swap dim 1 and dim 3 (5D) ≡ Swap dim 2 and dim 4 (6D) |

The transposition swaps the dimension of size 64 (dim 1 in 5D / dim 2 in 6D) with the dimension of size 16 (dim 3 in 5D / dim 4 in 6D). The equivalence holds because:

- Prepending a dimension of size 1 does not change the underlying data layout in memory.
- The permutation of non-trivial dimensions is identical whether viewed in 5D or 6D.
- Therefore, a **5D transpose with a leading batch of 1 is exactly equivalent to a 6D transpose**.

---

## Summary

| # | Dimensionality | Input Shape | Output Shape | Swap |
|---|---|---|---|---|
| 1 | 2D | `{1476, 32}` | `{32, 1476}` | dim 0 ↔ dim 1 |
| 2 | 4D (A) | `{1, 8, 4096, 3}` | `{1, 8, 3, 4096}` | dim 2 ↔ dim 3 |
| 3 | 4D (B) | `{1, 32, 8, 32}` | `{1, 8, 32, 32}` | dim 1 ↔ dim 2 |
| 4 | 6D (≡5D) | `{1, 8, 64, 4, 16, 7}` | `{1, 8, 16, 4, 64, 7}` | dim 2 ↔ dim 4 |
