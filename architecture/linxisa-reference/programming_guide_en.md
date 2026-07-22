# LinxISA Tensor Programming Guide

## 1. Overview

LinxISA (Linx Instruction Set Architecture) adopts a **block-structured instruction** design, encapsulating tensor computation operations in different block types. This guide is intended for programmers and introduces how to perform tensor programming using LinxISA.

### 1.1 Core Concepts

- **Block Instruction**: The basic execution unit of LinxISA, containing a block header and block body
- **Tensor Block Types**:
  - `VPAR` / `VSEQ`: Vector data blocks (parallel/serial)
  - `CUBE`: Matrix data block
  - `TMA`: Data transfer block
  - `TEPL`: Template data block

### 1.2 Data Type Support

| Type | Bit Width | Description |
|------|-----------|-------------|
| FP64 | 64-bit | Double-precision floating point |
| FP32 | 32-bit | Single-precision floating point |
| TF32 | 19-bit | TensorFloat-32 |
| FP16 | 16-bit | Half-precision floating point |
| BF16 | 16-bit | Brain Float 16 |
| FP8 | 8-bit | FP8 (e4m3/e5m2) |
| INT64/32/16/8 | 64/32/16/8-bit | Signed integers |
| UINT64/32/16/8 | 64/32/16/8-bit | Unsigned integers |

## 2. Programming Model

### 2.1 Block Instruction Structure

```
┌─────────────────────────────────┐
│         Block Header            │
│  - Block Type                   │
│  - Branch Info                  │
│  - Loop Bounds                  │
│  - Scheduling Hints             │
└─────────────────────────────────┘
┌─────────────────────────────────┐
│         Block Body              │
│  - Micro-instructions           │
│  - Actual computation/memory    │
│    operations                   │
└─────────────────────────────────┘
```

### 2.2 Execution Model

LinxISA adopts a **main core + dedicated cores** heterogeneous architecture:

- **BCC (Block Control Core)**: Main core, responsible for scheduling and scalar computation
- **Cube Core**: Dedicated core for matrix computation
- **Vector Core**: Dedicated core for vector computation (64 lanes)
- **MTC/TMA**: Dedicated cores for data transfer

## 3. Vector Programming

### 3.1 Vector Block Types

- **VPAR**: Vector parallel block, supports multi-lane parallel execution
- **VSEQ**: Vector serial block, executes sequentially

### 3.2 Vector Arithmetic Operations

```cpp
// Vector addition
V.ADD vd, vs1, vs2      // vd = vs1 + vs2
V.ADDI vd, vs1, imm     // vd = vs1 + imm

// Vector subtraction
V.SUB vd, vs1, vs2      // vd = vs1 - vs2

// Vector multiplication
V.MUL vd, vs1, vs2      // vd = vs1 * vs2
V.MADD vd, vs1, vs2, vs3 // vd = vs1 * vs2 + vs3

// Vector division
V.DIV vd, vs1, vs2      // vd = vs1 / vs2
```

### 3.3 Vector Comparison and Selection

```cpp
// Comparison operations
V.CMP.EQ vd, vs1, vs2   // vd = (vs1 == vs2)
V.CMP.LT vd, vs1, vs2   // vd = (vs1 < vs2)

// Conditional selection
V.CSEL vd, vs1, vs2, vm // vd = vm ? vs1 : vs2
```

### 3.4 Vector Memory Access

```cpp
// Normal memory access
V.LW vd, offset(base)   // Vector load word
V.SW vd, offset(base)   // Vector store word

// Bridge memory access (cross-lane)
V.LW.BRG vd, offset(base)
V.SW.BRG vd, offset(base)
```

## 4. Matrix Programming

### 4.1 CUBE Block

The CUBE block is used for matrix multiplication (GEMM) operations:

```
C = A × B + C (optional accumulation)
```

### 4.2 Matrix Multiplication Example

```cpp
// Matrix multiplication: C[M,N] = A[M,K] × B[K,N]
// Using CUBE block
bstart.cube
  // Load A and B into L0 buffer
  // Perform matrix multiplication
  // Store result C
bstop
```

### 4.3 Matrix Data Types

Data type combinations supported by CUBE block:

| Accumulator Type | A Type | B Type |
|------------------|--------|--------|
| INT32 | INT8 | INT8 |
| FP32 | FP16 | FP16 |
| FP32 | BF16 | BF16 |
| FP32 | FP32 | FP32 |

## 5. Data Transfer

### 5.1 TMA Block

The TMA (Tile Memory Access) block is used for efficient data transfer:

```asm
BSTART.TLOAD INT32
  // B.ARG / B.DIM / B.IOR / B.IOT descriptors
```

### 5.2 Transfer Modes

- **Continuous transfer**: Continuous memory regions
- **Tiled transfer**: Supports stride and tiling
- **Gather/Scatter**: Non-contiguous memory access

## 6. Programming Best Practices

### 6.1 Data Layout

- Row-Major layout is recommended for matrix data
- Pay attention to alignment requirements for vector data (32-byte alignment)
- Use TEPL blocks for data rearrangement

### 6.2 Performance Optimization

1. **Data reuse**: Fully utilize Tile Register space
2. **Pipeline overlap**: Overlap computation with data transfer
3. **Vectorization**: Use VPAR blocks instead of VSEQ whenever possible
4. **Tiled computation**: Use tiling strategy for large matrices

### 6.3 Synchronization Mechanism

- Instructions within a block execute in program order
- Dependencies between blocks are guaranteed by the scheduler
- Use `BSTOP` for explicit synchronization points

## 7. Common Patterns

### 7.1 Matrix Multiplication + Activation Function

```cpp
// 1. Matrix multiplication (CUBE)
C = A × B

// 2. Activation function (VPAR)
C = ReLU(C)  // or GELU, Sigmoid, etc.
```

### 7.2 Reduction Operations

```cpp
// Row reduction: sum each row
result[i] = sum(C[i, :])

// Column reduction: sum each column
result[j] = sum(C[:, j])
```

### 7.3 Flash Attention

```cpp
// Q, K, V matrices
// 1. Compute attention scores: S = Q × K^T / sqrt(d)
// 2. Softmax: P = softmax(S)
// 3. Output: O = P × V
```

## 8. Debugging and Verification

### 8.1 Simulation Tools

- **gfrun**: Functional model, fast correctness verification
- **gfsim**: Cycle-accurate model, performance evaluation

### 8.2 Common Issues

1. **Data type mismatch**: Check CUBE block type constraints
2. **Alignment errors**: Ensure data meets alignment requirements
3. **Tile Register overflow**: Allocate Tile resources reasonably

## 9. Reference Resources

- [LinxISA Official Documentation](https://linxisa.github.io/linx-isa/)
- [LinxISA GitHub](https://github.com/LinxISA/linx-isa)
- SuperNPUBench operator examples

---

**Document Version**: 1.0
**Last Updated**: 2026-01-08
