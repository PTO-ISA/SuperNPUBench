# Test Kernel - Operator Test Suites

This directory contains test code and build scripts for each operator in SuperNPUBench.

## Directory Structure

```
test/kernel/
├── broadcast/           # Broadcast operator tests
├── concat/              # Concat operator tests
├── control/             # Control flow tests (hash table)
├── element_wise/        # Element-wise operator tests (GELU)
├── fa/                  # Flash Attention tests
├── gather/              # Gather operator tests
├── matmul/              # Matrix multiplication tests
├── reduction/           # Reduction operator tests
│   ├── reducemax_col/
│   ├── reducemax_row/
│   ├── reducesum_col/
│   └── reducesum_row/
├── sort/                # Sort operator tests (topk)
└── transpose/           # Transpose operator tests
```

## Operator Test Status

| Operator | ELF Count | Status | Description |
|----------|-----------|--------|-------------|
| matmul | 13 | ✓ Success | FP4/BF16/FP32/FP16/FP8 matrix multiplication |
| fa | 9 | ✓ Success | Flash Attention with various configs |
| transpose | 8 | ✓ Success | 3D~6D tensor transpose |
| reduction | 8 | ✓ Success | Row/column max/sum reduction |
| gelu | 8 | ✓ Success | GELU activation (exact/approximate) |
| broadcast | 5 | ✓ Success | 2D~5D broadcast operations |
| gather | 4 | ✓ Success | Data gathering operations |
| concat | 4 | ✓ Success | Data concatenation (gather/scatter) |
| control | 0 | ✗ Failed | Missing data files |
| sort | 0 | ✗ Failed | Missing data files |

**Total**: 59 ELF files compiled successfully

## Build System

### Common Structure

Each operator test directory contains:
- `Makefile` - Build configuration
- `compile.all` - Batch compilation script with typical configurations
- `src/` - Source code directory

### Build Commands

#### Single Configuration

```bash
cd test/kernel/matmul

# FP4 x FP4 matmul
make TESTCASE=matmul TYPE=HIF4_HIF4 M=256 N=2048 K=2048 tM=128 tN=128 tK=128

# BF16 x FP4 mixed precision
make TESTCASE=matmul TYPE=A16W4 M=256 N=2048 K=2048 tM=128 tN=128 tK=128

# FP32 mask matmul
make TESTCASE=matmul TYPE=MASK MODE=MASK_FP32 M=256 N=256 K=256 tM=64 tN=64 tK=64
```

#### Batch Compilation

```bash
cd test/kernel/matmul && bash compile.all
cd test/kernel/fa && bash compile.all
cd test/kernel/broadcast && bash compile.all
```

#### Full Compilation

```bash
cd /path/to/SuperNPUBench
./compile_all.sh
```

### Makefile Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `TESTCASE` | Test case name | `matmul`, `fa_2d_unroll` |
| `TYPE` | Operator type (matmul only) | `HIF4_HIF4`, `A16W4`, `MASK` |
| `MODE` | Operator mode | `MASK_FP32`, `BF16x2_NOGATHER` |
| `M/N/K` | Matrix dimensions | `M=256 N=2048 K=2048` |
| `tM/tN/tK` | Tile sizes | `tM=128 tN=128 tK=128` |
| `COMPILER_DIR` | Compiler path | `/path/to/linx/bin` |

### Build Targets

```bash
make TESTCASE=<case> all      # Compile
make TESTCASE=<case> diss     # Generate disassembly
make TESTCASE=<case> sim      # Run in simulator
make TESTCASE=<case> debug    # Debug mode
make clean                    # Clean current operator
make clean_all                # Clean all
```

## Operator Details

### 1. Matmul (13 configs)

**Test Files**:
- `src/HiF4_HiF4.cpp` - FP4 x FP4 quantized matmul
- `src/A16W4.cpp` - BF16 x FP4 mixed precision matmul
- `src/matmul.cpp` - General matmul (multiple data types)

**Supported Types**:
- `TYPE=HIF4_HIF4`: FP4 x FP4 quantized (4 configs)
- `TYPE=A16W4`: BF16 x FP4 mixed precision (3 configs)
- `TYPE=MASK`: General matmul with various modes (6 configs)
  - `MASK_FP32`, `MASK_FP32_REUSEA`, `MASK_FP32_DYNAMIC`
  - `MASK_FP16`, `MASK_FP16_REUSEA`
  - `MASK_FP8`, `MASK_FP8_REUSEA`, `MX_FP8`

### 2. Flash Attention (9 configs)

**Test Files**:
- `src/fa_2d_unroll.cpp` - 2D unroll test (8 configs)
- `src/fa_hif4.cpp` - HIF4 quantized test (1 config)

**Supported Configs**:
- `X=1, Y=2/4`
- `X=2, Y=2/4`
- Sequence lengths: 256, 512

**Known Issues**:
- `X=1, Y=1` and `X=2, Y=1` cause compiler crash (see Issue #6)

### 3. Transpose (8 configs)

**Test File**: `src/transpose.cpp`

**Supported Configs**:
- 3D transpose (2 configs)
- 4D transpose (2 configs)
- 5D transpose (1 config)
- 6D transpose (2 configs)
- Different data types (__half, int32_t)

### 4. Reduction (8 configs)

**Test Directories**:
- `reducemax_col/` - Column-wise max (2 configs)
- `reducemax_row/` - Row-wise max (2 configs)
- `reducesum_col/` - Column-wise sum (2 configs)
- `reducesum_row/` - Row-wise sum (2 configs)

**Data Types**: int32_t, __half

### 5. GELU (8 configs)

**Test File**: `src/gelu.cpp`

**Supported Configs**:
- Data types: __bf16, __half
- Shapes: 24_8_1024 (3D), 128_1024 (2D)
- Approximation modes: false (exact), true (tanh)

### 6. Broadcast (5 configs)

**Test Files**: Various broadcast_*.cpp files

**Supported Configs**:
- 2D broadcast (1 config)
- 3D broadcast (1 config)
- 4D broadcast (2 configs)
- 5D broadcast (1 config)

### 7. Gather (4 configs)

**Test File**: `src/gather.cpp`

**Supported Configs**:
- Large-scale input (200000, 875000)
- Medium-scale input (754)
- Power-of-2 dimensions (131072)

### 8. Concat (4 configs)

**Test Files**:
- `src/concat_gather.cpp` - Gather-based concat (2 configs)
- `src/concat_scatter.cpp` - Scatter-based concat (2 configs)

**Data Types**: int32_t, __half

## Known Issues

### 1. Compiler Crash (Issue #6)

`fa_2d_unroll` triggers compiler assertion failure with `X=1, Y=1` and `X=2, Y=1` configurations:

```
Assertion failed: (Reg != 0 && "LinxV5 CallingConv Fail!")
```

**Workaround**: Avoid using `Ydim=1` configurations.

### 2. Control/Sort Operators

These operators require additional data files (`.data`) which are not currently included in the repository.

## Adding New Tests

1. Create test directory: `test/kernel/<operator>/`
2. Create `Makefile` (refer to existing operators)
3. Create `compile.all` with typical configurations
4. Add test source code in `src/` directory
5. Update this README with new operator information

## Related Documentation

- [Top-level README](../README.md) - Project overview
- [Operator Implementations](../kernels/README.md) - Kernel implementation details
- [Compilation Report](../COMPILATION_REPORT.md) - Compilation statistics
