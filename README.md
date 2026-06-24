# SuperNPUBench

SuperNPUBench is a high-performance operator library and benchmark platform for LinxISA/PTO-style tile programming. This repository contains reusable kernel implementations and make-driven test suites.

## Repository Structure

```
SuperNPUBench/
├── kernels/           # Operator implementations (header-only)
├── test/
│   ├── common/        # Common build system
│   └── kernel/        # Operator test suites
└── output/            # Build artifacts (not tracked)
```

## Operator Overview

This repository implements 8 core operator categories with 59 typical configurations:

| Operator | Count | Typical Scenarios | Description |
|----------|-------|-------------------|-------------|
| **matmul** | 13 | FP4/BF16/FP32/FP16/FP8 matrix multiplication | Supports quantization, mixed precision, various reuse strategies |
| **fa** | 9 | Flash Attention | 2D unroll, HIF4 quantization, various X/Y configs |
| **transpose** | 8 | 3D~6D tensor transpose | Supports multiple data types and dimension configs |
| **reduction** | 8 | Row/column reduction (max/sum) | Supports int32_t and __half data types |
| **gelu** | 8 | GELU activation | Supports exact mode and tanh approximation |
| **broadcast** | 5 | 2D~5D broadcast | Supports vectorized versions |
| **gather** | 4 | Data gathering | Large-scale input, power-of-2 dimensions |
| **concat** | 4 | Data concatenation | gather/scatter modes |

See [`kernels/README.md`](kernels/README.md) for detailed operator implementation details.

## Quick Start

### 1. Environment Setup

Requires Linx toolchain (linx_blockisa_llvm_musl):

```bash
export COMPILER_DIR=/path/to/linx_blockisa_llvm_musl/bin
```

### 2. Compile Single Operator

```bash
cd test/kernel/matmul

# FP4 x FP4 matrix multiplication
make TESTCASE=matmul TYPE=HIF4_HIF4 M=256 N=2048 K=2048 tM=128 tN=128 tK=128

# BF16 x FP4 mixed precision
make TESTCASE=matmul TYPE=A16W4 M=256 N=2048 K=2048 tM=128 tN=128 tK=128

# FP32 mask matmul
make TESTCASE=matmul TYPE=MASK MODE=MASK_FP32 M=256 N=256 K=256 tM=64 tN=64 tK=64
```

### 3. Batch Compilation

Each operator directory has a `compile.all` script for batch compiling typical configurations:

```bash
cd test/kernel/matmul && bash compile.all
cd test/kernel/fa && bash compile.all
cd test/kernel/broadcast && bash compile.all
```

### 4. Full Compilation

Use the `compile_all.sh` script in the root directory to compile all operators:

```bash
./compile_all.sh
```

Build artifacts are output to `output/kernel/<operator>/elf/` directory.

## Build System

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

See [`test/kernel/README.md`](test/kernel/README.md) for detailed build system documentation.

## Directory Structure

### kernels/ - Operator Implementations

Contains header-only operator implementations organized by function:

- [`kernels/matmul/`](kernels/matmul/) - Matrix multiplication (general, quantized, mixed precision)
- [`kernels/fa/`](kernels/fa/) - Flash Attention (2D unroll, quantized versions)
- [`kernels/broadcast/`](kernels/broadcast/) - Broadcast operations (2D~5D)
- [`kernels/reduction/`](kernels/reduction/) - Reduction operations (max/sum)
- [`kernels/gather/`](kernels/gather/) - Data gathering
- [`kernels/concat/`](kernels/concat/) - Data concatenation
- [`kernels/transpose/`](kernels/transpose/) - Transpose operations (3D~6D)
- [`kernels/element_wise/`](kernels/element_wise/) - Element-wise operations (GELU)

### test/kernel/ - Test Suites

Contains test code and build scripts for each operator:

- Each operator directory contains `Makefile`, `compile.all`, `src/`
- `compile.all` defines typical scenario compilation configurations
- Build artifacts output to `output/kernel/<operator>/elf/`

See [`test/kernel/README.md`](test/kernel/README.md) for detailed test documentation.

## Known Issues

### 1. Compiler Crash (Issue #6)

`fa_2d_unroll` triggers compiler assertion failure with `X=1, Y=1` and `X=2, Y=1` configurations:

```
Assertion failed: (Reg != 0 && "LinxV5 CallingConv Fail!")
```

**Workaround**: Avoid using `Ydim=1` configurations.

### 2. Control/Sort Operators

These operators require additional data files (`.data`) which are not currently included in the repository.

## Development Guide

### Adding New Operators

1. Add header-only implementation in `kernels/<operator>/`
2. Create test directory in `test/kernel/<operator>/`
3. Write `Makefile` (refer to existing operators)
4. Write `compile.all` (define typical scenarios)
5. Add test code in `test/kernel/<operator>/src/`

### Code Standards

- Kernel implementations use header-only approach
- Use PTO tile programming paradigm
- Follow existing directory structure and naming conventions
- Build artifacts are not tracked (already in `.gitignore`)

## Toolchain

- **Compiler**: linx_blockisa_llvm_musl (clang-15, linx64v5-musl)
- **Compiler flags**: `-mlxbc -fenable-matrix -O2 -mllvm -enable-all-vector-as-tilereg=true -std=c++20`
- **Target architecture**: Linx64 V5

## Compilation Report

See [`COMPILATION_REPORT.md`](COMPILATION_REPORT.md) for detailed compilation statistics and operator descriptions.

Current statistics:
- **Total**: 59 ELF files
- **Operators**: 8 categories
- **Status**: All compiled successfully

## Related Links

- [Compilation Report](COMPILATION_REPORT.md) - Detailed compilation statistics
- [Operator Implementations](kernels/README.md) - Kernel implementation details
- [Test Suites](test/kernel/README.md) - Test system documentation
