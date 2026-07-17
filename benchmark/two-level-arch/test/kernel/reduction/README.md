# Reduction Operators Documentation

This directory contains benchmark implementations for various reduction operations on NPU (Neural Processing Unit). The operators are optimized for different reduction patterns and data layouts.

## Overview

The reduction operators are categorized into three main types based on their reduction direction and optimization strategy:

- **Column Reduction (col)**: Reduces along the column dimension (M-axis)
- **Row Reduction (row)**: Reduces along the row dimension (N-axis)
- **3D Column Reduction (3dcol)**: Batch column reduction for 3D tensors with layout transformation to maximize vector lane utilization

## Directory Structure

```
reduction/
├── reducesum_col/          # Column sum reduction (basic)
├── reducesum_row/          # Row sum reduction (optimized)
├── reducesum_3dcol/        # 3D column sum reduction (layout-transformed for vector efficiency)
├── reducemax_col/          # Column max reduction (basic)
├── reducemax_row/          # Row max reduction (optimized)
└── reducemax_3dcol/        # 3D column max reduction (layout-transformed for vector efficiency)
```

## Operator Categories

### 1. Column Reduction Operators

Column reduction operators perform reduction along the M-axis (vertical direction), producing a 1×N output from an M×N input.

#### Basic Implementation: `reducesum_colvec.hpp` / `reducemax_colvec.hpp`

**Header Location**: `kernels/reduction/reducesum_colvec.hpp`

**Key Features**:
- 8-way tree unrolling for vectorized accumulation
- Handles both aligned and corner cases (remainder tiles)
- Uses `Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor>` for tile management

**Core Algorithm**:
```cpp
// 8-way tree reduction within each tile
for(size_t j=0; j<tileSrc::ValidRow; j+=8) {
    // Compute 8 pairwise sums/maxes
    sum_01 = src[idx_0] + src[idx_1];
    sum_23 = src[idx_2] + src[idx_3];
    sum_45 = src[idx_4] + src[idx_5];
    sum_67 = src[idx_6] + src[idx_7];
    // Tree reduction
    sum_0123 = sum_01 + sum_23;
    sum_4567 = sum_45 + sum_67;
    sum_tmp = sum_0123 + sum_4567;
    upd_sum += sum_tmp;
}
```

**Template Parameters**:
- `dtype`: Data type (int32_t, float, etc.)
- `gIM`, `gIN`: Global input dimensions (M×N)
- `tM`, `tN`: Tile dimensions for processing

#### Optimized Single Tree: `reducesum_colvec_single_tree.hpp`

**Header Location**: `kernels/reduction/reducesum_colvec_single_tree.hpp`

**Optimization Strategy**:
- Multi-stage tree reduction within tiles
- First stage: 8-way reduction to intermediate results
- Second stage: 64-way reduction (8×8)
- Final stage: Progressive reduction using `__builtin_ctz` for iteration count

**Two-Phase Approach**:
1. **Phase 1**: Each tile reduces to intermediate partial results stored in a temporary tile
2. **Phase 2**: Final kernel reduces all partial results to final output

**Key Functions**:
- `reducesum_col_kernel()`: Performs per-tile reduction with multi-level tree
- `reducesum_col_final_kernel()`: Reduces intermediate results to final output

### 2. Row Reduction Operators

Row reduction operators perform reduction along the N-axis (horizontal direction), producing an M×1 output from an M×N input.

#### Basic Implementation: `reducesum_rowvec.hpp` / `reducemax_rowvec.hpp`

**Header Location**: `kernels/reduction/reducesum_rowvec.hpp`

**Key Features**:
- 8-way tree unrolling for horizontal reduction
- Processes tiles column-by-column
- Maintains running sum/max across tile boundaries

**Core Algorithm**:
```cpp
// Each vector unit processes one row
for(size_t i=0; i<tileSrc::ValidCol; i+=8) {
    // 8-way tree reduction across columns
    sum_01 = src[idx0] + src[idx1];
    sum_23 = src[idx2] + src[idx3];
    sum_45 = src[idx4] + src[idx5];
    sum_67 = src[idx6] + src[idx7];
    sum_0123 = sum_01 + sum_23;
    sum_4567 = sum_45 + sum_67;
    sum_tmp = sum_0123 + sum_4567;
    upd_sum += sum_tmp;
}
```

#### Optimized Single Tree: `reducesum_rowvec_single_tree.hpp` / `reducemax_rowvec_single_tree.hpp`

**Header Location**: `kernels/reduction/reducesum_rowvec_single_tree.hpp`

**Optimization Strategy**:
- Uses column-major intermediate storage for better memory access
- Two-phase reduction similar to column optimized version
- Leverages `tile_shapeDataCol` with `BLayout::ColMajor` for intermediate results

**Key Improvements**:
- Reduces memory bandwidth by storing intermediate results efficiently
- Uses 4-way parallel processing with `blkv_get_index_y()` for z-dimension
- Progressive tree reduction with stride doubling

### 3. 3D Column Reduction Operators

These operators perform column-wise reduction on batches of 3D tensor slices (`Nums × gIM × gIN`), producing `Nums × 1 × gIN` outputs. The typical target shape is **120×8** (e.g., 381 slices of 120×8 for `__half`).

#### Implementation: `reducesum_colvec_unalign_120_8.hpp` / `reducemax_colvec_unalign_120_8.hpp`

**Header Location**: `kernels/reduction/reducesum_colvec_unalign_120_8.hpp`

##### Why Layout Transformation Is Needed

NPU `__vec__` functions execute vector operations across **at most 64 lanes**. If the original data layout (120×8, RowMajor) were used directly, each vector row would contain only 8 elements, activating just 8 out of 64 lanes — a **lane utilization of only 12.5%**.

To maximize vector efficiency, the data undergoes a **logical layout transformation** that reshapes each 2D slice from `gIM × gIN` (120×8) into `(gIM/8) × (gIN*8)` (15×64):

```
Original layout (120 × 8):            Transformed layout (15 × 64):
┌──┬──┬──┬──┬──┬──┬──┬──┐            ┌────────────────── 64 elements ──────────────────┐
│  │  │  │  │  │  │  │  │ row 0      │ row0: cols of orig_row0 | orig_row1 |...| orig_row7 │
├──┼──┼──┼──┼──┼──┼──┼──┤            ├────────────────────────────────────────────────┤
│  │  │  │  │  │  │  │  │ row 1      │ row1: cols of orig_row8 | orig_row9 |...| orig_r15 │
├──┼──┼──┼──┼──┼──┼──┼──┤            ├────────────────────────────────────────────────┤
│              ...                    │                     ...                        │
├──┼──┼──┼──┼──┼──┼──┼──┤            ├────────────────────────────────────────────────┤
│  │  │  │  │  │  │  │  │ row 119    │ row14: cols of orig_row112|...| orig_row119(+pad)│
└──┴──┴──┴──┴──┴──┴──┴──┘            └────────────────────────────────────────────────┘
      8 lanes used (12.5%)                  64 lanes used (100%)
```

Each original row's 8 elements are interleaved across 8 consecutive "virtual columns" in the transformed 64-wide row. This ensures **all 64 vector lanes are active**, achieving 100% lane utilization.

##### Layout Transformation in Code

```cpp
// Original: gIM × gIN (e.g., 120 × 8)
// Transformed: (gIM/8) × (gIN*8) → 15 × 64
using gm_shapeIn   = global_tensor<dtype, RowMajor<gIM/8, gIN*8>>;
using tile_shapeData = Tile<Location::Vec, dtype, tM/8, tN*8, BLayout::RowMajor, tM_VLD/8, tN*8>;
//   tile shape: 16×64, valid rows=15, valid cols=64 (tM=128, tN=8, tM_VLD=120)
using tile_shapeTmp = Tile<Location::Vec, dtype, 1, tN*8, BLayout::RowMajor>;          // 1×64
using tile_shapeSum = Tile<Location::Vec, dtype, 1, tN*8, BLayout::RowMajor, 1, tN>;   // 1×64, valid=8
```

Note: `tM/8 = 128/8 = 16` rows in the tile, but only `tM_VLD/8 = 120/8 = 15` are valid. The 16th row is zero-padded during DMA load (`TLOAD`) so that the row count remains a multiple of 8, enabling a complete 8-way tree reduction without remainder handling.

##### Two-Stage Reduction

Because the layout transformation "spreads" each original column across 8 virtual columns, the reduction must be completed in two stages:

**Stage 1 — `col_tmp` (per-column partial reduction):**
- Launched with **64 lanes** (`tile_shapeTmp::ValidCol = 64`), one lane per virtual column.
- Each lane processes one column of the 16×64 tile (15 valid rows + 1 zero-padded row), performing 8-way tree reduction along the row dimension.
- Output: a `1×64` intermediate tile of partial results.

**Stage 2 — `col_final` (merge virtual columns):**
- Launched with **8 lanes** (`tile_shapeSum::ValidCol = tN = 8`), one lane per original column.
- Lane `i` reads 8 values from the intermediate tile at stride `tN` (positions `i`, `i+8`, `i+16`, ..., `i+56`), which correspond to the 8 virtual columns belonging to original column `i`.
- An 8-way tree reduction (sum or max) merges these into the final scalar result for column `i`.

```
Stage 1 (col_tmp, 64 lanes):          Stage 2 (col_final, 8 lanes):
┌──────────── 64 cols ────────────┐   ┌──── 8 virtual cols per original col ────┐
│ ↕ reduce 16 rows per column      │   │ lane 0: merge tmp[0,8,16,...,56] → out[0]│
│ → 64 partial results (1×64)      │   │ lane 1: merge tmp[1,9,17,...,57] → out[1]│
└──────────────────────────────────┘   │ ...                                       │
                                        │ lane 7: merge tmp[7,15,23,...,63]→ out[7]│
                                        └───────────────────────────────────────────┘
```

##### reducesum vs reducemax Differences

Both operators share the same structure but differ in the reduction operation:
- **reducesum**: Uses `+` for accumulation, zero-padding is semantically neutral.
- **reducemax**: Uses `blkv_max()` for accumulation; zero-padding is valid under the assumption that all data values are ≥ 0.

##### Template Parameters

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `dtype` | Data type | `__half` |
| `gIM` | Original row count per slice | 120 |
| `gIN` | Original column count per slice | 8 |
| `tM` | Tile M dimension (pre-transform) | 128 |
| `tN` | Tile N dimension (pre-transform) | 8 |
| `tM_VLD` | Valid row count (≤ gIM, for unaligned dimensions) | 120 |

The host code loops over `Nums` slices, invoking the kernel independently for each:
```cpp
for(int i = 0; i < Nums; i++){
    reducesum_colsum_rand<dtype, gIMs, gINs, tMs, tNs, tM_VLDs>(
        &input[i * gIMs * gINs], &output[i * gINs]);
}
```

## Common Optimization Techniques

### 1. 8-Way Tree Reduction

All operators use 8-way unrolled tree reduction to maximize instruction-level parallelism:

```cpp
// Level 1: 8 pairwise operations
sum_01 = a + b;  sum_23 = c + d;  sum_45 = e + f;  sum_67 = g + h;
// Level 2: 4 pairwise operations
sum_0123 = sum_01 + sum_23;  sum_4567 = sum_45 + sum_67;
// Level 3: Final reduction
sum_tmp = sum_0123 + sum_4567;
```

### 2. Tile-Based Processing

Uses the PTO (Parallel Tile Operations) framework:
- `global_tensor`: Defines global memory layout
- `Tile`: Defines tile shape and memory location (Vec/Scalar)
- `global_iterator`: Iterates over tiles in global memory
- `TLOAD`/`TSTORE`: DMA transfers between global and tile memory

### 3. Corner Case Handling

All operators handle non-aligned dimensions:
- `rmd_M = gIM % tM`: Remainder in M dimension
- `rmd_N = gIN % tN`: Remainder in N dimension
- Specialized tile shapes for corner tiles with `ValidRow`/`ValidCol` parameters

### 4. Vector Instructions

Uses built-in vector operations:
- `blkv_get_index_x()`: Get vector lane index
- `blkv_get_tile_ptr()`: Get pointer to tile memory
- `blkv_max()`: Vectorized max operation (for reducemax)
- `__builtin_ctz()`: Count trailing zeros for iteration calculation

## Configuration Parameters

### Compile-Time Macros

- `DType`: Data type (default: `int32_t`)
- `gIMs`, `gINs`: Global input dimensions
- `tMs`, `tNs`: Tile dimensions
- `Nums`: Number of 3D slices (for 3dcol operators)
- `tM_VLDs`: Valid row count for unaligned tiles

### Example Configuration

```cpp
// For reducesum_col (8192×1024 input)
#define DType int32_t
#define gIMs 8192    // Global M dimension
#define gINs 1024    // Global N dimension
#define tMs 32       // Tile M dimension
#define tNs 128      // Tile N dimension
```

## Usage Examples

### Column Sum Reduction

```cpp
#include "reduction/reducesum_colvec.hpp"

dtype input[8192 * 1024];
dtype output[1 * 1024];

reducesum_colsum_rand<dtype, 8192, 1024, 32, 128>(input, output);
```

### Row Sum Reduction (Optimized)

```cpp
#include "reduction/reducesum_rowvec_single_tree.hpp"

dtype input[1024 * 8192];
dtype output[1024 * 1];

reducesum_trowsum_rand<dtype, 1024, 8192, 128, 64>(input, output);
```

### 3D Column Sum Reduction

```cpp
#include "reduction/reducesum_colvec_unalign_120_8.hpp"

dtype input[381 * 120 * 8];  // 381 batches of 120×8
dtype output[381 * 1 * 8];

for(int i = 0; i < 381; i++) {
    reducesum_colsum_rand<dtype, 120, 8, 32, 128, 120>(
        &input[i * 120 * 8],
        &output[i * 8]
    );
}
```

## Header File Naming Convention

- **Basic**: `xxx_colvec.hpp` / `xxx_rowvec.hpp`
  - Standard tile-based reduction with 8-way tree unrolling

- **Optimized**: `xxx_colvec_single_tree.hpp` / `xxx_rowvec_single_tree.hpp`
  - Multi-stage tree reduction with intermediate storage
  - Better for large reduction dimensions

- **3D Unaligned**: `xxx_colvec_unalign_120_8.hpp`
   - Batch column reduction for 3D tensors (Nums × gIM × gIN)
   - Logical layout transformation (gIM×gIN → (gIM/8)×(gIN*8)) to maximize __vec__ lane utilization (8/64 → 64/64)
   - Two-stage reduction: per-virtual-column partial reduction (64 lanes) + cross-virtual-column merge (8 lanes)
   - Zero-padding to keep row count a multiple of 8 for clean 8-way tree reduction

## Performance Considerations

1. **Tile Size Selection**: Choose `tM` and `tN` to balance parallelism and memory usage
2. **Memory Alignment**: Aligned dimensions enable more efficient vectorization
3. **Reduction Direction**: Column reduction is generally more efficient for tall matrices
4. **Batch Processing**: 3D operators process multiple slices in a loop; kernel invocation overhead is amortized across `Nums` slices
5. **Vector Lane Utilization (3dcol)**: The layout transformation reshapes `gIM×gIN` into `(gIM/8)×(gIN*8)` so that each vector row spans 64 elements, filling all 64 `__vec__` lanes. Without this, only `gIN` (e.g., 8) lanes would be active, wasting over 87% of vector capacity
6. **Zero-Padding Trade-off (3dcol)**: Padding rows to a multiple of 8 enables clean 8-way tree reduction but introduces a small number of dummy operations; for 120→128 (padded to tM=128), the overhead is only 1/16 ≈ 6.25%

## Dependencies

- `common/pto_tileop.hpp`: PTO framework for tile operations
- `template_asm.h`: Assembly-level vector operations
- `fileop.h`: File I/O utilities for testing

## References

- PTO (Parallel Tile Operations) Framework Documentation
- NPU Vector Architecture Specification
- Tree Reduction Optimization Techniques
