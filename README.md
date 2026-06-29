# SuperNPUBench

SuperNPUBench is a high-performance operator library and benchmark platform supporting both **LinxISA** and **PTO ISA** tile programming paradigms. This repository contains reusable kernel implementations, test suites, and comprehensive documentation for both ISA architectures.

## Repository Structure

```
SuperNPUBench/
├── benchmark-linxisa/        # LinxISA benchmark suite
│   ├── kernels/              # Operator implementations (header-only)
│   ├── test/                 # Test suites and build system
│   │   ├── common/           # Common build infrastructure
│   │   └── kernel/           # Operator test cases
│   ├── output/               # Build artifacts (ELF files, disassembly)
│   └── compile_all.sh        # Batch compilation script
│
├── benchmark-ptoisa/         # PTO ISA benchmark suite
│   ├── kernels/              # Operator implementations (header-only)
│   ├── test/                 # Test suites and build system
│   │   ├── common/           # Common build infrastructure
│   │   └── kernel/           # Operator test cases
│   ├── output/               # Build artifacts (ELF files, disassembly)
│   └── compile_all.sh        # Batch compilation script
│
├── common/                   # Shared resources
│   ├── linxisa-reference/    # LinxISA programming guides & ISA reference
│   ├── ptoisa-reference/     # PTO ISA programming guides & ISA reference
│   ├── scripts/              # Utility scripts
│   ├── tools/                # Analysis and comparison tools
│   └── data/                 # Shared test data
│
└── compile_all.sh            # Top-level build script (supports both ISAs)
```

## Supported ISA Architectures

### LinxISA
- **Architecture**: Block-structured instruction set with heterogeneous cores
- **Core Types**: BCC (main core), Cube Core (matrix), Vector Core (vector), MTC/TMA (data transfer)
- **Programming Model**: Block instructions (VPAR/VSEQ, CUBE, TMA, TEPL)
- **Documentation**: See [`common/linxisa-reference/`](common/linxisa-reference/)

### PTO ISA
- **Architecture**: Tile-centric instruction set with explicit memory hierarchy
- **Tile Types**: Vec (UB), Mat (L1), Left (L0A), Right (L0B), Acc (L0C)
- **Programming Model**: Tile operations with Auto/Manual modes
- **Documentation**: See [`common/ptoisa-reference/`](common/ptoisa-reference/)

## Operator Overview

Both ISA implementations support 8 core operator categories with 59+ typical configurations:

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

## Quick Start

### 1. Environment Setup

Requires Linx toolchain (linx_blockisa_llvm_musl):

```bash
export COMPILER_DIR=/path/to/linx_blockisa_llvm_musl/bin
```

### 2. Compile for LinxISA

```bash
# Navigate to LinxISA benchmark directory
cd benchmark-linxisa/test/kernel/matmul

# FP4 x FP4 matrix multiplication
make TESTCASE=matmul TYPE=HIF4_HIF4 M=256 N=2048 K=2048 tM=128 tN=128 tK=128

# BF16 x FP4 mixed precision
make TESTCASE=matmul TYPE=A16W4 M=256 N=2048 K=2048 tM=128 tN=128 tK=128

# FP32 mask matmul
make TESTCASE=matmul TYPE=MASK MODE=MASK_FP32 M=256 N=256 K=256 tM=64 tN=64 tK=64
```

### 3. Compile for PTO ISA

```bash
# Navigate to PTO ISA benchmark directory
cd benchmark-ptoisa/test/kernel/matmul

# Similar commands as LinxISA
make TESTCASE=matmul TYPE=HIF4_HIF4 M=256 N=2048 K=2048 tM=128 tN=128 tK=128
```

### 4. Batch Compilation

Each operator directory has a `compile.all` script for batch compiling typical configurations:

```bash
# LinxISA
cd benchmark-linxisa/test/kernel/matmul && bash compile.all

# PTO ISA
cd benchmark-ptoisa/test/kernel/matmul && bash compile.all
```

### 5. Full Compilation

Use the top-level `compile_all.sh` script to compile all operators for both ISAs:

```bash
# Compile all operators for both ISAs
./compile_all.sh all

# Compile only LinxISA operators
./compile_all.sh linx

# Compile only PTO ISA operators
./compile_all.sh pto
```

Build artifacts are output to:
- LinxISA: `benchmark-linxisa/output/kernel/<operator>/elf/`
- PTO ISA: `benchmark-ptoisa/output/kernel/<operator>/elf/`

## Running on the Models

LinxCoreModel provides two simulators:

- `gfrun` — functional model (verifies instruction correctness)
- `gfsim` — cycle-accurate model (verifies timing behavior)

### Running LinxISA Binaries

```bash
# Functional run
bin/gfrun -f benchmark-linxisa/output/kernel/<op>/elf/<name>.elf

# Cycle-accurate run
bin/gfsim -f benchmark-linxisa/output/kernel/<op>/elf/<name>.elf
```

### Running PTO ISA Binaries

```bash
# Functional run
bin/gfrun -f benchmark-ptoisa/output/kernel/<op>/elf/<name>.elf

# Cycle-accurate run
bin/gfsim -f benchmark-ptoisa/output/kernel/<op>/elf/<name>.elf
```

### Tile-op kernels: run gfsim in single-tier mode

Kernels written purely with tile ops that use TEPL template instructions (e.g.
`TCVT`, as in `control/hashtable_lookup_simd`) execute on the **VectorLite**
tile/template engine. gfsim only steps that engine in **single-tier mode**, so these
kernels must be run with:

```bash
bin/gfsim -f <elf> -s core.singleTierMode=true
```

Without this flag (gfsim's default multi-tier mode) the VectorLite engine is inert,
TEPL blocks are dispatched but never consumed, and the run deadlocks (res-verify /
deadlock abort). `gfrun` (functional) does not need the flag.

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

## Documentation

### Programming Guides

- **LinxISA**: [`common/linxisa-reference/programming_guide.md`](common/linxisa-reference/programming_guide.md) (Chinese) | [`programming_guide_en.md`](common/linxisa-reference/programming_guide_en.md) (English)
- **PTO ISA**: [`common/ptoisa-reference/programming_guide.md`](common/ptoisa-reference/programming_guide.md) (Chinese) | [`programming_guide_en.md`](common/ptoisa-reference/programming_guide_en.md) (English)

### ISA Reference

- **LinxISA**: [`common/linxisa-reference/isa_reference.md`](common/linxisa-reference/isa_reference.md)
- **PTO ISA**: [`common/ptoisa-reference/isa_reference.md`](common/ptoisa-reference/isa_reference.md)

### Architecture Reports

- **LinxCoreModel Architecture**: [`~/Documents/LinxCoreModel_Architecture_Report.md`](~/Documents/LinxCoreModel_Architecture_Report.md) (local)
- **ISA Migration Plan**: [`~/Documents/SuperNPUBench_ISA_Migration_Plan.md`](~/Documents/SuperNPUBench_ISA_Migration_Plan.md) (local)

## Directory Structure Details

### benchmark-linxisa/ and benchmark-ptoisa/

Each benchmark directory contains:

- **kernels/**: Header-only operator implementations organized by function
  - `matmul/` - Matrix multiplication (general, quantized, mixed precision)
  - `fa/` - Flash Attention (2D unroll, quantized versions)
  - `broadcast/` - Broadcast operations (2D~5D)
  - `reduction/` - Reduction operations (max/sum)
  - `gather/` - Data gathering
  - `concat/` - Data concatenation
  - `transpose/` - Transpose operations (3D~6D)
  - `element_wise/` - Element-wise operations (GELU)

- **test/kernel/**: Test code and build scripts for each operator
  - Each operator directory contains `Makefile`, `compile.all`, `src/`
  - `compile.all` defines typical scenario compilation configurations
  - Build artifacts output to `output/kernel/<operator>/elf/`

### common/

Shared resources for both ISA implementations:

- **linxisa-reference/**: LinxISA programming guides and ISA reference documentation
- **ptoisa-reference/**: PTO ISA programming guides and ISA reference documentation
- **scripts/**: Utility scripts for compilation, testing, and analysis
- **tools/**: Analysis and comparison tools
- **data/**: Shared test data and configurations

## Known Issues

### 1. Compiler Crash (Issue #6)

`fa_2d_unroll` triggers compiler assertion failure with `X=1, Y=1` and `X=2, Y=1` configurations:

```
Assertion failed: (Reg != 0 && "LinxV5 CallingConv Fail!")
```

**Workaround**: Avoid using `Ydim=1` configurations.

### 2. Control/Sort Operators

`control/hashtable_lookup_simd` is implemented with **pure tile ops** (no SIMT lane
programming; the SIMT variant has been removed). Its `.data` files are generated
automatically by the Makefile (`gen_data.py`) on first build. Run it on **gfsim** with
`-s core.singleTierMode=true` — see [Running on the Models](#running-on-the-models).

## Development Guide

### Adding New Operators

1. Add header-only implementation in both:
   - `benchmark-linxisa/kernels/<operator>/`
   - `benchmark-ptoisa/kernels/<operator>/`
2. Create test directory in both:
   - `benchmark-linxisa/test/kernel/<operator>/`
   - `benchmark-ptoisa/test/kernel/<operator>/`
3. Write `Makefile` (refer to existing operators)
4. Write `compile.all` (define typical scenarios)
5. Add test code in `test/kernel/<operator>/src/`

### Code Standards

- Kernel implementations use header-only approach
- Use PTO tile programming paradigm
- Follow existing directory structure and naming conventions
- Build artifacts are not tracked (already in `.gitignore`)
- Maintain both LinxISA and PTO ISA implementations in parallel

## Toolchain

- **Compiler**: linx_blockisa_llvm_musl (clang-15, linx64v5-musl)
- **Compiler flags**: `-mlxbc -fenable-matrix -O2 -mllvm -enable-all-vector-as-tilereg=true -std=c++20`
- **Target architecture**: Linx64 V5

## Compilation Statistics

Current statistics (LinxISA):
- **Total**: 45+ ELF files
- **Operators**: 8 categories
- **Status**: Most compiled successfully, some have known issues

See [`common/COMPILATION_REPORT.md`](common/COMPILATION_REPORT.md) for detailed compilation statistics and operator descriptions.

## Related Links

- [LinxISA Official Documentation](https://linxisa.github.io/linx-isa/)
- [PTO ISA Official Documentation](https://pto-isa.github.io/docs/isa/tile/)
- [LinxISA GitHub](https://github.com/LinxISA/linx-isa)
- [PTO ISA GitHub](https://github.com/PTO-ISA/pto-isa)
- [Compilation Report](common/COMPILATION_REPORT.md) - Detailed compilation statistics
- [LinxISA Programming Guide](common/linxisa-reference/programming_guide_en.md) - LinxISA programming documentation
- [PTO ISA Programming Guide](common/ptoisa-reference/programming_guide_en.md) - PTO ISA programming documentation

## License

See LICENSE file for details.

## Contact

For questions and support, please open an issue on GitHub.
