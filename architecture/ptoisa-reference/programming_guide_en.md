# PTO ISA Tile Programming Guide

## 1. Overview

PTO ISA (Pervasive Tile Operations) is an instruction set architecture designed for Tile (tensor) operations. This guide is intended for programmers and introduces how to perform tensor programming using PTO ISA Tile instructions.

### 1.1 Core Concepts

- **Tile**: A 2D data block, the basic operation unit in PTO ISA
- **TileType**: The type of Tile, determining its storage location and execution pipeline
  - `Vec`: General vector operations (UB, 256 KB)
  - `Mat`: Matrix operations (L1, 512 KB)
  - `Left`: Matrix multiplication left operand (L0A, 64 KB)
  - `Right`: Matrix multiplication right operand (L0B, 64 KB)
  - `Acc`: Matrix multiplication accumulator (L0C, 256 KB)
- **Valid Region**: The valid area of a Tile, defining the iteration domain of operations
- **Auto/Manual Mode**: Automatic mode vs manual mode

### 1.2 Data Type Support

| Type | Bit Width | Description |
|------|-----------|-------------|
| FP64 | 64-bit | Double-precision floating point |
| FP32 | 32-bit | Single-precision floating point |
| TF32 | 19-bit | TensorFloat-32 |
| HF32 | 19-bit | High-precision Float 32 |
| FP16 | 16-bit | Half-precision floating point |
| BF16 | 16-bit | Brain Float 16 |
| FP8 | 8-bit | FP8 (e4m3/e5m2) |
| INT64/32/16/8 | 64/32/16/8-bit | Signed integers |
| UINT64/32/16/8 | 64/32/16/8-bit | Unsigned integers |

## 2. Programming Model

### 2.1 Tile and Valid Region

Tile is the core data structure in PTO ISA:

```cpp
// Tile definition
Tile<Vec, float, 16, 16> myTile;  // 16x16 float Tile

// Valid Region
myTile.GetValidRow()  // Number of valid rows
myTile.GetValidCol()  // Number of valid columns
```

**Valid Region** defines the actual iteration domain of Tile operations. Only elements within the Valid Region will be processed.

### 2.2 TileType and Memory Hierarchy

| TileType | Memory Location | Capacity | Alignment Requirement | Usage |
|----------|-----------------|----------|----------------------|-------|
| Vec | UB | 256 KB | 32 B | General element-wise operations |
| Mat | L1 | 512 KB | 32 B | Matrix operations |
| Left | L0A | 64 KB | 32 B | Matrix multiplication A operand |
| Right | L0B | 64 KB | 32 B | Matrix multiplication B operand |
| Acc | L0C | 256 KB | 32 B | Matrix multiplication accumulator |

### 2.3 Auto vs Manual Mode

**Auto Mode**: The compiler automatically manages Tile allocation and synchronization

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void auto_mode() {
    Tile<Vec, float, 16, 16> a, b, c;
    // Compiler automatically inserts TASSIGN and TSYNC
    TADD(c, a, b);
}
```

**Manual Mode**: Programmers explicitly manage Tile allocation and synchronization

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void manual_mode() {
    Tile<Vec, float, 16, 16> a, b, c;
    
    // Explicitly allocate Tiles to hardware resources
    TASSIGN(a, 0x1000);
    TASSIGN(b, 0x2000);
    TASSIGN(c, 0x3000);
    
    // Explicit synchronization
    RecordEvent e0 = TLOAD(a, ga);
    RecordEvent e1 = TLOAD(b, gb);
    TSYNC(e0, e1);
    
    TADD(c, a, b);
    TSYNC();
    
    TSTORE(gc, c);
}
```

## 3. Tile Instruction Categories

### 3.1 View and Tile Buffer

| Instruction | Function |
|-------------|----------|
| `pto.make_tensor_view` | Create Tensor View |
| `pto.get_tensor_view_dim` | Get View dimensions |
| `pto.get_tensor_view_stride` | Get View stride |
| `pto.tensor_view_addr` | Get View address |
| `pto.partition_view` | Partition View |
| `pto.alloc_tile` | Allocate Tile |
| `pto.subset` | Create subset |
| `pto.set_validshape` | Set Valid Shape |

### 3.2 Synchronization and Configuration

| Instruction | Function |
|-------------|----------|
| `pto.tsync` | Tile synchronization |
| `pto.syncall` | Global synchronization |
| `pto.tassign` | Allocate Tile to hardware resources |
| `pto.talias` | Create Tile alias |

### 3.3 Element-wise Operations

| Instruction | Function |
|-------------|----------|
| `pto.tadd` | Tile addition |
| `pto.tsub` | Tile subtraction |
| `pto.tmul` | Tile multiplication |
| `pto.tdiv` | Tile division |
| `pto.tmax` | Tile maximum |
| `pto.tmin` | Tile minimum |
| `pto.tcmp` | Tile comparison |
| `pto.tabs` | Tile absolute value |
| `pto.texp` | Tile exponential |
| `pto.tsqrt` | Tile square root |
| `pto.trelu` | Tile ReLU activation |
| `pto.tcvt` | Tile type conversion |

### 3.4 Tile-Scalar Operations

| Instruction | Function |
|-------------|----------|
| `pto.tadds` | Tile + scalar |
| `pto.tsubs` | Tile - scalar |
| `pto.tmuls` | Tile * scalar |
| `pto.tdivs` | Tile / scalar |
| `pto.tmaxs` | Tile max with scalar |
| `pto.tmins` | Tile min with scalar |
| `pto.taxpy` | AXPY operation (a*x + y) |

### 3.5 Reduce and Expand

| Instruction | Function |
|-------------|----------|
| `pto.trowsum` | Row-wise sum |
| `pto.tcolsum` | Column-wise sum |
| `pto.trowmax` | Row-wise maximum |
| `pto.tcolmax` | Column-wise maximum |
| `pto.trowmin` | Row-wise minimum |
| `pto.tcolmin` | Column-wise minimum |
| `pto.trowexpand` | Row expansion |
| `pto.tcolexpand` | Column expansion |

### 3.6 Memory and Data Transfer

| Instruction | Function |
|-------------|----------|
| `pto.tload` | Load from global memory to Tile |
| `pto.tstore` | Store from Tile to global memory |
| `pto.tprefetch` | Tile prefetch |
| `pto.mgather` | Masked Gather |
| `pto.mscatter` | Masked Scatter |

### 3.7 Matrix Operations

| Instruction | Function |
|-------------|----------|
| `pto.tmatmul` | Matrix multiplication (GEMM) |
| `pto.tmatmul_acc` | Matrix multiply-accumulate |
| `pto.tmatmul_bias` | Matrix multiply with bias |
| `pto.tgemv` | Matrix-vector multiplication |
| `pto.tgemv_acc` | Matrix-vector multiply-accumulate |

### 3.8 Layout and Rearrangement

| Instruction | Function |
|-------------|----------|
| `pto.tmov` | Tile move |
| `pto.textract` | Extract sub-Tile |
| `pto.tinsert` | Insert sub-Tile |
| `pto.tconcat` | Tile concatenation |
| `pto.treshape` | Tile reshape |
| `pto.ttrans` | Tile transpose |
| `pto.tfillpad` | Fill padding |
| `pto.tpack` | Tile pack |

## 4. Programming Examples

### 4.1 Matrix Addition

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void matrix_add() {
    // Define 16x16 float Tiles
    Tile<Vec, float, 16, 16> a, b, c;
    
    // Auto mode: compiler manages automatically
    TADD(c, a, b);
}
```

### 4.2 Matrix Multiplication

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void matrix_multiply() {
    // Matrix multiplication operands require specific TileTypes
    Tile<Left, half, 16, 16> a;    // L0A
    Tile<Right, half, 16, 16> b;   // L0B
    Tile<Acc, float, 16, 16> c;    // L0C
    
    // Execute C = A × B
    TMATMUL(c, a, b);
}
```

### 4.3 Data Load and Compute

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void load_compute_store() {
    Tile<Vec, float, 16, 16> a, b, c;
    
    // Manual mode
    TASSIGN(a, 0x1000);
    TASSIGN(b, 0x2000);
    TASSIGN(c, 0x3000);
    
    // Load from global memory
    RecordEvent e0 = TLOAD(a, ptr_a);
    RecordEvent e1 = TLOAD(b, ptr_b);
    TSYNC(e0, e1);
    
    // Compute c = a + b
    TADD(c, a, b);
    TSYNC();
    
    // Store to global memory
    TSTORE(ptr_c, c);
}
```

### 4.4 Reduction Operations

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void row_sum() {
    Tile<Vec, float, 16, 64> input;
    Tile<Vec, float, 16, 1> output;
    
    // Sum each row
    TROWSUM(output, input);
}
```

### 4.5 Flash Attention Pattern

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void flash_attention() {
    // Q, K, V matrices
    Tile<Left, half, 16, 16> Q, K;
    Tile<Right, half, 16, 16> V;
    Tile<Acc, float, 16, 16> S, O;
    
    // 1. Compute attention scores: S = Q × K^T / sqrt(d)
    TMATMUL(S, Q, K);
    
    // 2. Softmax (requires additional scalar operations)
    // ...
    
    // 3. Output: O = P × V
    TMATMUL(O, P, V);
}
```

## 5. Performance Optimization

### 5.1 Tile Reuse

```cpp
// Good practice: reuse Tiles
Tile<Vec, float, 16, 16> temp;
for (int i = 0; i < N; i++) {
    TLOAD(temp, ptr[i]);
    TADD(result, result, temp);
}

// Bad practice: frequently allocate new Tiles
for (int i = 0; i < N; i++) {
    Tile<Vec, float, 16, 16> temp;  // Allocates every iteration
    TLOAD(temp, ptr[i]);
    TADD(result, result, temp);
}
```

### 5.2 Pipeline Overlap

```cpp
// Overlap computation with data transfer
RecordEvent load_event = TLOAD(tile_next, ptr_next);
TCOMPUTE(tile_current);
TSYNC(load_event);
// Swap tile_current and tile_next
```

### 5.3 Choosing the Right TileType

```cpp
// Matrix multiplication: use dedicated TileTypes
Tile<Left, half, 16, 16> a;    // For matrix multiplication A
Tile<Right, half, 16, 16> b;   // For matrix multiplication B
Tile<Acc, float, 16, 16> c;    // For accumulator

// Element-wise operations: use Vec
Tile<Vec, float, 16, 16> x;    // For element-wise operations
```

### 5.4 Layout Optimization

```cpp
// RowMajor layout typically performs better
Tile<Vec, float, 16, 64, RowMajor> tile;

// Avoid unnecessary transposes
// If possible, operate directly on the original layout
```

## 6. Synchronization Mechanism

### 6.1 RecordEvent

`RecordEvent` is the synchronization token in PTO ISA:

```cpp
RecordEvent e0 = TLOAD(a, ptr_a);
RecordEvent e1 = TLOAD(b, ptr_b);

// Wait for both loads to complete
TSYNC(e0, e1);

// Now safe to use a and b
TADD(c, a, b);
```

### 6.2 Implicit Synchronization

In Auto mode, the compiler automatically inserts necessary synchronization:

```cpp
// Auto mode: compiler synchronizes automatically
TADD(c, a, b);  // Compiler ensures a, b are loaded
```

### 6.3 Explicit Synchronization

In Manual mode, explicit synchronization management is required:

```cpp
// Manual mode: explicit synchronization
TLOAD(a, ptr_a);
TSYNC();  // Wait for load to complete
TADD(c, a, b);
TSYNC();  // Wait for computation to complete
TSTORE(ptr_c, c);
```

## 7. Constraints and Limitations

### 7.1 Type Matching

All operands must have the same element type:

```cpp
// Correct: types match
Tile<Vec, float, 16, 16> a, b, c;
TADD(c, a, b);

// Incorrect: type mismatch
Tile<Vec, float, 16, 16> a;
Tile<Vec, half, 16, 16> b;
TADD(c, a, b);  // Compilation error
```

### 7.2 Shape Constraints

Matrix multiplication has strict shape constraints:

```cpp
// C[M,N] = A[M,K] × B[K,N]
// Constraints:
// - A.GetValidRow() = C.GetValidRow() = M
// - A.GetValidCol() = B.GetValidRow() = K
// - B.GetValidCol() = C.GetValidCol() = N
```

### 7.3 TileType Constraints

Different instructions have specific TileType requirements:

```cpp
// Matrix multiplication
Tile<Left, half, 16, 16> a;    // Must be Left
Tile<Right, half, 16, 16> b;   // Must be Right
Tile<Acc, float, 16, 16> c;    // Must be Acc
TMATMUL(c, a, b);

// Element-wise operations
Tile<Vec, float, 16, 16> x, y, z;  // Must be Vec
TADD(z, x, y);
```

## 8. Debugging and Verification

### 8.1 Simulation Tools

- **gfrun**: Functional model, fast correctness verification
- **gfsim**: Cycle-accurate model, performance evaluation

### 8.2 Common Issues

1. **Type mismatch**: Ensure all operand types are consistent
2. **Shape mismatch**: Check matrix dimension constraints
3. **TileType errors**: Use the correct TileType
4. **Missing synchronization**: Ensure proper synchronization in Manual mode
5. **Valid Region out-of-bounds**: Do not access elements outside the Valid Region

### 8.3 Performance Analysis

Use gfsim to analyze performance:

```bash
gfsim -f program.elf --stop_cycle 1000000
```

Review performance statistics in the output:
- Cycle count
- Instruction throughput
- Resource utilization

## 9. Reference Resources

- [PTO ISA Official Documentation](https://pto-isa.github.io/docs/isa/tile/)
- [PTO ISA GitHub](https://github.com/PTO-ISA/pto-isa)
- SuperNPUBench operator examples

---

**Document Version**: 1.0  
**Last Updated**: 2026-01-08
